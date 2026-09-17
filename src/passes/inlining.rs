//! Function inlining: one clone routine, two eligibility rules.
//!
//! * A function called exactly once is SPLICED: its statements, locals and
//!   trailing `Return` value move into the caller at the call's position
//!   and the definition is compacted away (LLVM's last-call bonus: deleting
//!   the definition is the whole gain, so no size cap applies).  Eligible is
//!   any body whose only `Return` is the trailing top-level statement.
//!   Nothing is duplicated, so no effect count or convergence position
//!   changes.
//! * A function called at several sites is cloned into each - every site
//!   or none, a definition kept beside its clones being pure loss - only
//!   when its body is an expression DAG (`[Emit*, Return { value }]` with
//!   no locals, stores or calls) and the copies render shorter than the
//!   definition plus the calls (`paying_clones`: the generator's own
//!   prices shortlist, rendering the module with the clone applied
//!   confirms); `max_node_count` caps the DAG, `max_call_sites` the reuse.
//!
//! Every site substitutes the caller's argument handles for the parameter
//! reads and the callee's returned value for the call's result, and passes
//! the same gates in both directions: a `-0.0` literal argument, and a
//! const-ness crossing into a sign-sensitive operator or into a static-error
//! slot (an integer divisor, a shift amount, a bounded index) - the callee's
//! by an argument, the caller's by the value.

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::pipeline::{Pass, PassContext};

use super::const_fold::{
    ConstantLiterals, Role, constant_literals, evaluates_to_negative_zero, operand_is_static_error,
    static_error_slots,
};
use super::expr_util::{
    LEAF_FLOAT, LEAF_FLOAT_ZERO, access_static_lengths, const_expression_leaf, const_sign_changes,
    expression_needs_emit, float_leaf_bits, has_negative_zero_leaf_through,
    is_disallowed_inline_expression, is_library_module, is_sign_sensitive_op, zero_value_is_float,
};
use crate::generator::price::{FunctionPricer, Pricer};
use crate::ir::rewrite::{
    Rewrite, clone_expression_handle, follow, push_emit_runs, rebuild_block_expressions,
    rebuild_function_expressions, rewrite_block,
};
use crate::ir::visit::{
    Scope, all_functions, contains_return, for_each_function_mut, for_each_statement,
    nested_blocks, try_map_expression_handles_in_place, visit_expression_children,
    visit_statement_expression_handles,
};
use crate::name_gen::{function_local_names, module_scope_names, type_names};
use std::collections::HashSet;

/// Const-ness of a caller argument and the [`float_leaf_bits`] its cone
/// reaches, read through `replacements`: a call result an earlier splice
/// replaced counts as the value it becomes, not as the runtime read the
/// statement still spells (`h(f() + 0.0)` with `f` spliced first hands `h`
/// a const `0.0 + 0.0`).
fn argument_facts(
    caller: &naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
    root: naga::Handle<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> (bool, u8) {
    let (mut is_const, mut leaves) = (true, 0);
    let mut seen = HandleSet::default();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        let h = follow(replacements, h);
        if !seen.insert(h) {
            continue;
        }
        let expr = &caller[h];
        leaves |= float_leaf_bits(expr, types);
        match const_expression_leaf(expr) {
            Some(true) => {}
            Some(false) => is_const = false,
            None => visit_expression_children(expr, |c| stack.push(c)),
        }
    }
    (is_const, leaves)
}

/// Const-ness of every callee expression with the parameter reads left
/// runtime (`before`, the call's own meaning) and with the arguments
/// substituted (`after`), plus the float leaves each cone reaches.  A
/// parameter read is runtime by construction, so inlining is the moment
/// const-ness can appear; both crossing gates read this one walk.
struct SubstitutionFacts {
    before: Vec<bool>,
    after: Vec<bool>,
    leaves: Vec<u8>,
}

impl SubstitutionFacts {
    fn new(
        callee: &naga::Arena<naga::Expression>,
        argument_types: &[naga::Handle<naga::Type>],
        view: &SpliceView<'_>,
        types: &naga::UniqueArena<naga::Type>,
    ) -> Self {
        let n = callee.len();
        let (mut before, mut after) = (vec![false; n], vec![false; n]);
        let mut leaves = vec![0u8; n];
        for (handle, expr) in callee.iter() {
            let i = handle.index();
            match expr {
                naga::Expression::FunctionArgument(idx) => {
                    let (is_const, reached) = view
                        .arguments
                        .get(*idx as usize)
                        .map_or((true, LEAF_FLOAT_ZERO), |&a| {
                            argument_facts(view.caller, types, a, view.replacements)
                        });
                    // "Float" comes from the parameter's declared type: an
                    // argument can spell a float value without a float leaf.
                    let typed_float = argument_types
                        .get(*idx as usize)
                        .is_none_or(|&ty| zero_value_is_float(types, ty));
                    (before[i], after[i]) = (false, is_const);
                    leaves[i] =
                        (reached & LEAF_FLOAT_ZERO) | if typed_float { LEAF_FLOAT } else { 0 };
                }
                _ => match const_expression_leaf(expr) {
                    Some(known) => {
                        (before[i], after[i]) = (known, known);
                        leaves[i] = float_leaf_bits(expr, types);
                    }
                    None => {
                        let (mut b, mut a, mut l) = (true, true, float_leaf_bits(expr, types));
                        visit_expression_children(expr, |c| {
                            b &= before[c.index()];
                            a &= after[c.index()];
                            l |= leaves[c.index()];
                        });
                        (before[i], after[i], leaves[i]) = (b, a, l);
                    }
                },
            }
        }
        Self {
            before,
            after,
            leaves,
        }
    }

    /// `h` reads as a const-expression only once the arguments are in.
    fn crosses(&self, h: naga::Handle<naga::Expression>) -> bool {
        self.after[h.index()] && !self.before[h.index()]
    }
}

/// One site's clone of the two arenas a [`SpliceView`] reads, with the
/// constant table its evaluations need: shared by the three gates, so a
/// node any of them reaches is cloned once, and the table - O(module
/// constants) - is built at the first slot that actually crosses, which
/// most sites never reach.
#[derive(Default)]
struct SiteScratch {
    arena: naga::Arena<naga::Expression>,
    maps: SpliceViewMaps,
    const_literals: Option<ConstantLiterals>,
}

/// `true` when substituting the arguments turns one of the callee's
/// `sign_sensitive_ops` ([`is_sign_sensitive_op`] nodes) from runtime into
/// a const-expression that reads differently ([`const_sign_changes`]);
/// rooted at the operator as the predicate is, so `fn g(a: f32) -> f32 {
/// return a * 0.; }` called as `g(-1.)` is seen through the sibling zero.
fn crosses_sign_sensitive(
    facts: &SubstitutionFacts,
    view: &SpliceView<'_>,
    sign_sensitive_ops: &[naga::Handle<naga::Expression>],
    module: &naga::Module,
    scratch: &mut SiteScratch,
) -> bool {
    let SiteScratch {
        arena,
        maps,
        const_literals,
    } = scratch;
    sign_sensitive_ops.iter().any(|&op| {
        facts.crosses(op)
            && const_sign_changes(&view.callee[op], facts.leaves[op.index()], || {
                let cloned = view.callee_node(op, arena, maps);
                let const_literals =
                    const_literals.get_or_insert_with(|| constant_literals(module));
                evaluates_to_negative_zero(&module.types, const_literals, arena, cloned)
            })
    })
}

/// `true` when substituting the arguments makes one of the callee's `slots`
/// ([`static_error_slots`]) read as a const-expression that is a
/// shader-creation error: [`operand_is_static_error`] asked of the operand
/// as the [`SpliceView`] clones it.
fn crosses_failable_slot(
    facts: &SubstitutionFacts,
    view: &SpliceView<'_>,
    slots: &[(naga::Handle<naga::Expression>, Role)],
    module: &naga::Module,
    scratch: &mut SiteScratch,
) -> bool {
    let SiteScratch {
        arena,
        maps,
        const_literals,
    } = scratch;
    slots.iter().any(|&(operand, role)| {
        if !facts.crosses(operand) {
            return false;
        }
        let cloned = view.callee_node(operand, arena, maps);
        let const_literals = const_literals.get_or_insert_with(|| constant_literals(module));
        operand_is_static_error(&module.types, const_literals, arena, role, cloned)
    })
}

/// The caller's own sign-sensitive operators and static-error slots a
/// splice can make const, keyed by the `CallResult` each one's cone reads:
/// `(root, None)` for an [`is_sign_sensitive_op`] node, `(operand, role)`
/// for a slot.  The returned value is the other direction const-ness enters
/// by (`a[idx()]`, `-nz()`), which the argument gates never see: they walk
/// the callee.
type ResultSlots = HandleMap<naga::Expression, Vec<(naga::Handle<naga::Expression>, Option<Role>)>>;

/// [`ResultSlots`] of `caller`.  Only a cone whose every runtime leaf is a
/// call result can turn const, so one bottom-up pass prunes the rest and
/// the collecting walk runs on the few that remain.
fn caller_result_slots(caller: &naga::Function, module: &naga::Module) -> ResultSlots {
    let arena = &caller.expressions;
    // Per handle: the cone reads a call result / its runtime leaves are
    // all call results.
    let (mut reads_call, mut only_calls) = (vec![false; arena.len()], vec![true; arena.len()]);
    for (h, expr) in arena.iter() {
        let i = h.index();
        match expr {
            naga::Expression::CallResult(_) => reads_call[i] = true,
            _ => match const_expression_leaf(expr) {
                Some(true) => {}
                Some(false) => only_calls[i] = false,
                None => visit_expression_children(expr, |c| {
                    reads_call[i] |= reads_call[c.index()];
                    only_calls[i] &= only_calls[c.index()];
                }),
            },
        }
    }
    let mut slots = ResultSlots::default();
    let mut file = |root: naga::Handle<naga::Expression>, role: Option<Role>| {
        if !(reads_call[root.index()] && only_calls[root.index()]) {
            return;
        }
        let mut seen = HandleSet::default();
        let mut stack = vec![root];
        while let Some(h) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            if matches!(arena[h], naga::Expression::CallResult(_)) {
                slots.entry(h).or_default().push((root, role));
            } else {
                visit_expression_children(&arena[h], |c| stack.push(c));
            }
        }
    };
    for (h, expr) in arena.iter() {
        if is_sign_sensitive_op(expr) {
            file(h, None);
        }
    }
    for (operand, role) in static_error_slots(arena, &access_static_lengths(caller, module)) {
        file(operand, Some(role));
    }
    slots
}

/// The two arenas of one call as the splice will leave them, read through
/// the original handles: in the callee a parameter read stands for its
/// argument, in the caller an earlier splice's result is the value it was
/// replaced by, and the call's own `result` is the callee's `value`.  Every
/// gate clones its root through here into a scratch arena for the const
/// evaluator, so the two directions and the nested case share one reading.
struct SpliceView<'a> {
    caller: &'a naga::Arena<naga::Expression>,
    replacements: &'a HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    /// The call's result and the callee's returned value, when a caller
    /// slot reads the result.
    result: Option<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    )>,
    callee: &'a naga::Arena<naga::Expression>,
    arguments: &'a [naga::Handle<naga::Expression>],
}

/// The two memo maps of a [`SpliceView`] clone, one per source arena.
type SpliceViewMaps = (
    HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
);

impl SpliceView<'_> {
    fn caller_node(
        &self,
        handle: naga::Handle<naga::Expression>,
        scratch: &mut naga::Arena<naga::Expression>,
        maps: &mut SpliceViewMaps,
    ) -> naga::Handle<naga::Expression> {
        let handle = follow(self.replacements, handle);
        if let Some((result, value)) = self.result
            && handle == result
        {
            return self.callee_node(value, scratch, maps);
        }
        if let Some(&mapped) = maps.0.get(handle) {
            return mapped;
        }
        let mut expression = self.caller[handle].clone();
        let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
            Some(self.caller_node(child, scratch, maps))
        });
        let mapped = scratch.append(expression, naga::Span::UNDEFINED);
        maps.0.insert(handle, mapped);
        mapped
    }

    fn callee_node(
        &self,
        handle: naga::Handle<naga::Expression>,
        scratch: &mut naga::Arena<naga::Expression>,
        maps: &mut SpliceViewMaps,
    ) -> naga::Handle<naga::Expression> {
        if let naga::Expression::FunctionArgument(k) = self.callee[handle]
            && let Some(&argument) = self.arguments.get(k as usize)
        {
            return self.caller_node(argument, scratch, maps);
        }
        if let Some(&mapped) = maps.1.get(handle) {
            return mapped;
        }
        let mut expression = self.callee[handle].clone();
        let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
            Some(self.callee_node(child, scratch, maps))
        });
        let mapped = scratch.append(expression, naga::Span::UNDEFINED);
        maps.1.insert(handle, mapped);
        mapped
    }
}

/// `true` when the returned value makes one of the caller's `slots` (the
/// [`caller_result_slots`] entry for this call's result) a const-expression
/// that changes meaning: a sign-sensitive operator that reads differently
/// ([`const_sign_changes`]), a static-error slot evaluating to the error.
/// Each root reads as runtime before the splice - its cone holds this call's
/// result - so const-ness after is the crossing, judged with the argument
/// gates' tests on the [`SpliceView`].
fn result_crosses(
    slots: &[(naga::Handle<naga::Expression>, Option<Role>)],
    view: &SpliceView<'_>,
    module: &naga::Module,
    scratch: &mut SiteScratch,
) -> bool {
    let SiteScratch {
        arena,
        maps,
        const_literals,
    } = scratch;
    let no_replacements = HandleMap::default();
    slots.iter().any(|&(root, role)| {
        let cloned = view.caller_node(root, arena, maps);
        let (is_const, leaves) = argument_facts(arena, &module.types, cloned, &no_replacements);
        if !is_const {
            return false;
        }
        let const_literals = const_literals.get_or_insert_with(|| constant_literals(module));
        match role {
            None => const_sign_changes(&arena[cloned], leaves, || {
                evaluates_to_negative_zero(&module.types, const_literals, arena, cloned)
            }),
            Some(role) => {
                operand_is_static_error(&module.types, const_literals, arena, role, cloned)
            }
        }
    })
}

/// A `--preserve-symbol` function: its declaration is an external contract
/// (a `--preamble` stub whose real body the consumer supplies, a
/// host-callable helper) honoured only through intact call sites, so no
/// call to one is deleted or spliced.
fn is_preserved(function: &naga::Function, preserve: &[String]) -> bool {
    function
        .name
        .as_deref()
        .is_some_and(|n| preserve.iter().any(|p| p == n))
}

/// Default inlining budgets (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_NODE_COUNT: usize = 24;
/// Default inlining call-site budget (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_CALL_SITES: usize = 3;
/// Widened node budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_NODE_COUNT: usize = 48;
/// Widened call-site budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_CALL_SITES: usize = 6;

/// Inlining pass with per-run node and call-site budgets.
#[derive(Debug)]
pub struct InliningPass {
    max_node_count: usize,
    max_call_sites: usize,
}

impl Default for InliningPass {
    fn default() -> Self {
        Self {
            max_node_count: DEFAULT_MAX_INLINE_NODE_COUNT,
            max_call_sites: DEFAULT_MAX_INLINE_CALL_SITES,
        }
    }
}

impl InliningPass {
    /// A pass with explicit budgets; the profile picks among the
    /// `DEFAULT_*` / `MAX_PROFILE_*` constants.
    pub fn new(max_node_count: usize, max_call_sites: usize) -> Self {
        Self {
            max_node_count,
            max_call_sites,
        }
    }
}

impl Pass for InliningPass {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        // A call to an empty function is a no-op, but a void function with
        // several call sites is never cloned, so without this `fn e(){} ...
        // e(); e();` survives every pass, pinned alive by its own call
        // sites.
        let mut changed = delete_calls_to_empty_functions(module, &ctx.config.preserve_symbols);
        changed |= splice_calls(module, ctx, self.max_node_count, self.max_call_sites) > 0;
        Ok(changed)
    }
}

/// Remove every `Call` to a function whose body is empty (or a lone bare
/// `return;`), so the next compaction can drop the callee itself.
fn delete_calls_to_empty_functions(module: &mut naga::Module, preserve: &[String]) -> bool {
    let empty: HandleSet<naga::Function> = module
        .functions
        .iter()
        .filter(|(_, f)| {
            let body_is_empty = matches!(
                f.body.iter().collect::<Vec<_>>().as_slice(),
                [] | [naga::Statement::Return { value: None }]
            );
            body_is_empty && f.result.is_none() && !is_preserved(f, preserve)
        })
        .map(|(h, _)| h)
        .collect();
    if empty.is_empty() {
        return false;
    }

    let mut changed = false;
    for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
        changed |= drop_empty_calls_in_block(&mut f.body, &empty);
    });
    changed
}

fn drop_empty_calls_in_block(block: &mut naga::Block, empty: &HandleSet<naga::Function>) -> bool {
    let mut changed = false;
    rewrite_block(block, Scope::default(), &mut |stmt, span, out, _| {
        if let naga::Statement::Call {
            function,
            result: None,
            ..
        } = &stmt
            && empty.contains(function)
        {
            changed = true;
            return;
        }
        out.push(stmt, span);
    });
    changed
}

// MARK: Splice driver

/// Splice every eligible call - a single-call function into its one
/// caller, a clonable expression body into each of its callers; the number
/// of splices.  Definitions left without a call are compacted away before
/// returning: left for the next sweep's `compact`, every later pass of this
/// sweep would see two copies of each body (`const_hoist` counted their
/// literals twice and hoisted single-use vectors).
fn splice_calls(
    module: &mut naga::Module,
    ctx: &PassContext<'_>,
    max_node_count: usize,
    max_call_sites: usize,
) -> usize {
    // A library module keeps every function: a splice would only add a
    // second copy of the body.
    if is_library_module(module) {
        return 0;
    }
    let preserve = &ctx.config.preserve_symbols;
    let call_sites = collect_call_sites(module);
    let callers: Vec<&naga::Function> = all_functions(module).collect();
    // Lazily: only a caller with a multi-site candidate's call pays the
    // typifier walk.
    let mut result_slots: Vec<Option<ResultSlots>> = callers.iter().map(|_| None).collect();
    let no_replacements = HandleMap::default();
    let mut candidates: HandleSet<naga::Function> = HandleSet::default();
    let mut clonable: Vec<Clonable> = Vec::new();
    for (h, f) in module.functions.iter() {
        let Some(sites) = call_sites.get(h) else {
            continue;
        };
        if is_preserved(f, preserve) {
            continue;
        }
        // A body is checked under its `@diagnostic` filter scope; moved
        // under another (its own filter, or a caller's), a derivative or
        // texture builtin can turn an accepted input into a uniformity
        // error.  naga hands every function the module's leaf, so the
        // usual case is equal leaves and the splice is free.
        if sites
            .iter()
            .any(|site| callers[site.caller].diagnostic_filter_leaf != f.diagnostic_filter_leaf)
        {
            continue;
        }
        if sites.len() == 1 {
            candidates.insert(h);
            continue;
        }
        // Multi-site: every site must admit the clone, or the definition
        // stays and the clones made are pure loss against the pricing.
        if sites.len() > max_call_sites {
            continue;
        }
        let Some(value) = clonable_body(f, max_node_count) else {
            continue;
        };
        let admitted = sites.iter().all(|site| {
            let slots = result_slots[site.caller]
                .get_or_insert_with(|| caller_result_slots(callers[site.caller], module));
            site_admits(
                f,
                module,
                &site.arguments,
                &callers[site.caller].expressions,
                &no_replacements,
                Some(value),
                site.result
                    .and_then(|result| slots.get(result).map(|slots| ResultSite { result, slots })),
            )
        });
        if admitted {
            clonable.push(Clonable { function: h, value });
        }
    }
    // Every name in scope at a splice point besides the caller's own.
    let scope_names: HashSet<String> = module_scope_names(module)
        .chain(type_names(module))
        .map(str::to_string)
        .chain(preserve.iter().cloned())
        .collect();
    if !clonable.is_empty() {
        candidates.extend(paying_clones(
            module,
            &clonable,
            &call_sites,
            &callers,
            &scope_names,
            ctx,
        ));
    }
    if candidates.is_empty() {
        return 0;
    }

    // The pre-check judged each multi-site clone on the caller before any
    // splice, the gate at the site reads the caller through its earlier
    // splices, and the two can disagree (`dot(f(), f())` turns const at the
    // second site): such a candidate is withdrawn and the run redone from
    // the snapshot - rare enough that the copy beats a per-site undo, and
    // finite, since withdrawing a splice only makes the other gates' cones
    // less const.
    let mut multi: HandleSet<naga::Function> = candidates
        .iter()
        .filter(|&&h| call_sites[h].len() > 1)
        .copied()
        .collect();
    loop {
        let snapshot =
            (!multi.is_empty()).then(|| (module.functions.clone(), module.entry_points.clone()));
        let spliced = splice_into_callers(module, &candidates, &scope_names);
        let withdrawn: Vec<_> = if multi.is_empty() {
            Vec::new()
        } else {
            // Every site of a candidate is attempted, so a call left is a
            // site that declined.
            let remaining = collect_call_sites(module);
            multi
                .iter()
                .filter(|&&h| remaining.contains_key(h))
                .copied()
                .collect()
        };
        if withdrawn.is_empty() {
            if spliced > 0 {
                super::compact::compact_behind_anchor(module, &|name| {
                    preserve.iter().any(|p| p == name)
                });
            }
            return spliced;
        }
        let (functions, entry_points) = snapshot.expect("a multi-site candidate declined");
        module.functions = functions;
        module.entry_points = entry_points;
        for h in withdrawn {
            candidates.remove(h);
            multi.remove(h);
        }
        if candidates.is_empty() {
            return 0;
        }
    }
}

/// One splicing run over every caller; the number of splices.  Callers go
/// in arena order, entry points last: naga orders callees before callers,
/// so a helper chain flattens bottom-up in one run, each callee read live
/// after its own splices.
fn splice_into_callers(
    module: &mut naga::Module,
    candidates: &HandleSet<naga::Function>,
    scope_names: &HashSet<String>,
) -> usize {
    let mut spliced = 0;
    let callers: Vec<_> = module.functions.iter().map(|(h, _)| h).collect();
    for h in callers {
        // Out of the arena while rewritten, so the callee bodies can be
        // read from the module.
        let mut caller = std::mem::take(&mut module.functions[h]);
        spliced += splice_calls_in_function(&mut caller, module, candidates, scope_names);
        module.functions[h] = caller;
    }
    for i in 0..module.entry_points.len() {
        let mut caller = std::mem::take(&mut module.entry_points[i].function);
        spliced += splice_calls_in_function(&mut caller, module, candidates, scope_names);
        module.entry_points[i].function = caller;
    }
    spliced
}

/// Splice the candidate calls of `caller` (taken out of `module`); the
/// count.  Results are mapped once over the whole function - a call's
/// result flows into later statements at any depth - and the arena is
/// rebuilt: clones are appended behind the consumers of the results they
/// replace.
fn splice_calls_in_function(
    caller: &mut naga::Function,
    module: &naga::Module,
    candidates: &HandleSet<naga::Function>,
    scope_names: &HashSet<String>,
) -> usize {
    let mut calls_candidate = false;
    for_each_statement(&caller.body, &mut |stmt| {
        if let naga::Statement::Call { function, .. } = stmt {
            calls_candidate |= candidates.contains(*function);
        }
    });
    if !calls_candidate {
        return 0;
    }

    let mut used_names = scope_names.clone();
    used_names.extend(function_local_names(caller).map(str::to_string));
    let mut splicer = Splicer {
        module,
        candidates,
        used_names,
        // Over the arena before any splice: a spliced body reads only the
        // arguments it was handed, never a later call's result.
        result_slots: caller_result_slots(caller, module),
        replacements: HandleMap::default(),
    };
    let mut body = std::mem::take(&mut caller.body);
    let spliced = splicer.splice_block(&mut body, caller, Scope::default());
    caller.body = body;
    if spliced > 0 {
        // The replaced results go dead for the arena rebuild.
        Rewrite {
            map: &splicer.replacements,
            backward_only: false,
            retire_replaced: false,
        }
        .apply(caller);
        rebuild_function_expressions(caller);
        caller.named_expressions.clear();
    }
    spliced
}

/// The state of one caller's splices: the names taken at its splice
/// points, its [`caller_result_slots`], and the result-to-value map of the
/// splices made so far.
struct Splicer<'a> {
    module: &'a naga::Module,
    candidates: &'a HandleSet<naga::Function>,
    used_names: HashSet<String>,
    result_slots: ResultSlots,
    replacements: HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
}

impl Splicer<'_> {
    /// Replace each candidate `Call` in `block` (nested blocks included)
    /// by its callee's body; `scope` says whether `block` runs more than
    /// once per caller invocation.  Cloned statements are not rescanned: a
    /// callee's own single-call helpers were spliced when the callee was
    /// the caller.
    fn splice_block(
        &mut self,
        block: &mut naga::Block,
        caller: &mut naga::Function,
        scope: Scope,
    ) -> usize {
        let mut spliced = 0;
        rewrite_block(block, scope, &mut |statement, span, out, scope| {
            if let naga::Statement::Call {
                function,
                arguments,
                result,
            } = &statement
                && self.candidates.contains(*function)
                // Each argument as this function's earlier splices leave
                // it: `f(g(1))` hands `f` the value `g` spliced, not its
                // `CallResult`, so the gates judge the const-ness that
                // arrives rather than the runtime read the rewrite replaces.
                && let Some((body, value)) = self.splice_body(
                    &self.module.functions[*function],
                    &arguments
                        .iter()
                        .map(|&a| follow(&self.replacements, a))
                        .collect::<Vec<_>>(),
                    *result,
                    caller,
                    scope.in_loop(),
                )
            {
                if let (Some(result), Some(value)) = (*result, value) {
                    self.replacements.insert(result, value);
                }
                out.extend_block(body);
                spliced += 1;
                return;
            }
            out.push(statement, span);
        });
        spliced
    }

    /// [`splice_body`] for the call of `callee` whose result is `result`,
    /// with this caller's result slots and replacements.
    fn splice_body(
        &mut self,
        callee: &naga::Function,
        arguments: &[naga::Handle<naga::Expression>],
        result: Option<naga::Handle<naga::Expression>>,
        caller: &mut naga::Function,
        in_loop: bool,
    ) -> Option<(naga::Block, Option<naga::Handle<naga::Expression>>)> {
        let site = result.and_then(|result| {
            self.result_slots
                .get(result)
                .map(|slots| ResultSite { result, slots })
        });
        splice_body(
            callee,
            self.module,
            arguments,
            &self.replacements,
            site,
            caller,
            in_loop,
            &mut self.used_names,
        )
    }
}

/// The number of leading statements to splice and the returned value of a
/// body whose only `Return` is the trailing top-level one - required for a
/// function with a result, an optional bare `return;` otherwise; `None` for
/// every other shape.
fn splice_shape(
    callee: &naga::Function,
) -> Option<(usize, Option<naga::Handle<naga::Expression>>)> {
    let (keep, value) = match callee.body.last() {
        Some(naga::Statement::Return { value }) => (callee.body.len() - 1, *value),
        _ => (callee.body.len(), None),
    };
    if callee.result.is_some() != value.is_some() {
        return None;
    }
    (!contains_return(&callee.body[..keep])).then_some((keep, value))
}

/// `name`, or `name_<n>` for the first free `n`, entered into `used`.
/// `rename` reassigns local names in every profile, but a preserved name
/// survives it, and two preserved `t`s in one function would declare twice;
/// the module-scope names keep a trace dump readable.
fn unique_local_name(name: &str, used: &mut HashSet<String>) -> String {
    let mut candidate = name.to_string();
    let mut n = 1usize;
    while used.contains(&candidate) {
        candidate = format!("{name}_{n}");
        n += 1;
    }
    used.insert(candidate.clone());
    candidate
}

/// The caller side of a call whose result some caller slot reads: the
/// `CallResult` handle and its [`caller_result_slots`] entry.
struct ResultSite<'a> {
    result: naga::Handle<naga::Expression>,
    slots: &'a [(naga::Handle<naga::Expression>, Option<Role>)],
}

/// The per-site gates of the module header: `arguments` may stand in for
/// the callee's parameter reads and `value`, the callee's returned
/// expression, for the call's result; `replacements` are the caller's
/// earlier splices (empty at the multi-site pre-check, which precedes every
/// splice), which every read of the caller's arena goes through.
fn site_admits(
    callee: &naga::Function,
    module: &naga::Module,
    arguments: &[naga::Handle<naga::Expression>],
    caller: &naga::Arena<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    value: Option<naga::Handle<naga::Expression>>,
    site: Option<ResultSite<'_>>,
) -> bool {
    if callee.arguments.len() != arguments.len()
        || arguments
            .iter()
            .any(|&a| has_negative_zero_leaf_through(caller, a, &|h| follow(replacements, h)))
    {
        return false;
    }
    let view = SpliceView {
        caller,
        replacements,
        result: match (site.as_ref(), value) {
            (Some(site), Some(value)) => Some((site.result, value)),
            _ => None,
        },
        callee: &callee.expressions,
        arguments,
    };
    let mut scratch = SiteScratch::default();
    if let Some(site) = site
        && view.result.is_some()
        && result_crosses(site.slots, &view, module, &mut scratch)
    {
        return false;
    }
    let sign_sensitive_ops: Vec<_> = callee
        .expressions
        .iter()
        .filter(|(_, expr)| is_sign_sensitive_op(expr))
        .map(|(h, _)| h)
        .collect();
    let slots = static_error_slots(&callee.expressions, &access_static_lengths(callee, module));
    if sign_sensitive_ops.is_empty() && slots.is_empty() {
        return true;
    }
    let argument_types: Vec<_> = callee.arguments.iter().map(|a| a.ty).collect();
    let facts = SubstitutionFacts::new(&callee.expressions, &argument_types, &view, &module.types);
    !crosses_sign_sensitive(&facts, &view, &sign_sensitive_ops, module, &mut scratch)
        && !crosses_failable_slot(&facts, &view, &slots, module, &mut scratch)
}

/// The callee's body cloned into `caller` for a call passing `arguments`
/// (`replacements`: the caller's earlier splices; `site`: the result's
/// caller side, when a caller slot reads it): the statements to splice and
/// the returned value's caller handle.  `None` (nothing appended) when the
/// shape or a call-site gate declines.
#[allow(clippy::too_many_arguments)]
fn splice_body(
    callee: &naga::Function,
    module: &naga::Module,
    arguments: &[naga::Handle<naga::Expression>],
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    site: Option<ResultSite<'_>>,
    caller: &mut naga::Function,
    in_loop: bool,
    used_names: &mut HashSet<String>,
) -> Option<(naga::Block, Option<naga::Handle<naga::Expression>>)> {
    let (keep, value) = splice_shape(callee)?;
    if !site_admits(
        callee,
        module,
        arguments,
        &caller.expressions,
        replacements,
        value,
        site,
    ) {
        return None;
    }
    // Parameter reads become the caller's argument handles: evaluated once,
    // before the call, exactly as the call did.
    let mut map: HandleMap<naga::Expression, naga::Handle<naga::Expression>> = HandleMap::default();
    for (h, expr) in callee.expressions.iter() {
        if let naga::Expression::FunctionArgument(k) = expr {
            map.insert(h, *arguments.get(*k as usize)?);
        }
    }

    // Everything below appends to the caller; no decline past this point.
    let span = naga::Span::UNDEFINED;
    let mut locals: HandleMap<naga::LocalVariable, naga::Handle<naga::LocalVariable>> =
        HandleMap::default();
    for (lh, local) in callee.local_variables.iter() {
        let name = local
            .name
            .as_deref()
            .map(|n| unique_local_name(n, used_names));
        let moved = caller.local_variables.append(
            naga::LocalVariable {
                name,
                ty: local.ty,
                init: None,
            },
            callee.local_variables.get_span(lh),
        );
        locals.insert(lh, moved);
    }
    let mut local_exprs: HandleMap<naga::LocalVariable, naga::Handle<naga::Expression>> =
        HandleMap::default();
    for (h, expr) in callee.expressions.iter() {
        if let naga::Expression::LocalVariable(l) = expr {
            let moved = caller
                .expressions
                .append(naga::Expression::LocalVariable(locals[*l]), span);
            map.insert(h, moved);
            local_exprs.entry(*l).or_insert(moved);
        }
    }

    let mut body = callee.body.clone();
    body.cull(keep..);
    rebuild_block_expressions(
        &mut body,
        &callee.expressions,
        &mut caller.expressions,
        &mut map,
    );
    let value = value.map(|v| {
        clone_expression_handle(v, &callee.expressions, &mut caller.expressions, &mut map)
    });

    // naga initialises a local once, at function entry.  Outside a loop the
    // splice point runs at most once per invocation and nothing touches the
    // local before it, so the callee's init carries over.  Inside a loop
    // every iteration re-entered the callee, so the init (WGSL's zero value
    // without one) becomes a store - naga's own lowering of a loop-scoped
    // `var`, which it places directly before the first statement touching
    // the variable; `place_local_inits` does the same, because a counter's
    // store is recovered as the `for` initializer only when it is adjacent
    // to its loop, and tint proves a loop finite only in that shape.
    let mut inits: Vec<(naga::Handle<naga::LocalVariable>, naga::Block)> = Vec::new();
    for (lh, local) in callee.local_variables.iter() {
        let moved = locals[lh];
        if !in_loop {
            caller.local_variables[moved].init = local.init.map(|init| {
                clone_expression_handle(
                    init,
                    &callee.expressions,
                    &mut caller.expressions,
                    &mut map,
                )
            });
            continue;
        }
        let pointer = *local_exprs.entry(lh).or_insert_with(|| {
            caller
                .expressions
                .append(naga::Expression::LocalVariable(moved), span)
        });
        let mut init_stmts = naga::Block::new();
        let value = match local.init {
            // A fresh copy: the body walk's clone of an emitted init sits in
            // the callee's own `Emit`, after this store.
            Some(init) => {
                let start = caller.expressions.len();
                let value = clone_expression_handle(
                    init,
                    &callee.expressions,
                    &mut caller.expressions,
                    &mut HandleMap::default(),
                );
                // Split around the declarative leaves the clone appended.
                let emitted: Vec<_> = caller
                    .expressions
                    .range_from(start)
                    .filter(|&h| expression_needs_emit(&caller.expressions[h]))
                    .collect();
                push_emit_runs(&mut init_stmts, &emitted, span);
                value
            }
            None => caller
                .expressions
                .append(naga::Expression::ZeroValue(local.ty), span),
        };
        init_stmts.push(naga::Statement::Store { pointer, value }, span);
        inits.push((moved, init_stmts));
    }
    Some((place_local_inits(inits, body, &caller.expressions), value))
}

/// Weave each spliced local's initialising statements into `body` directly
/// before the first top-level statement that references the local (in its
/// operands, its `Emit` cones or its nested blocks); a local nothing
/// references leads.  Exact because the value is a fresh constant or
/// override clone or a zero value (naga sets a local's `init` only for
/// those), so no skipped statement can observe or feed it.  Locals first
/// touched by one `Loop` keep arena order except the one its `continuing`
/// stores whole - the counter - which goes last: `[Store acc, Store i,
/// Loop]` renders `var acc=0;for(var i=0;...)`.
fn place_local_inits(
    mut inits: Vec<(naga::Handle<naga::LocalVariable>, naga::Block)>,
    body: naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> naga::Block {
    if inits.is_empty() {
        return body;
    }
    let mut first: Vec<Option<usize>> = vec![None; inits.len()];
    let mut locals = HandleSet::default();
    // One memo over the whole body: only the FIRST statement reaching a
    // local matters, so a cone met earlier is already attributed.  A helper
    // chain splices bottom-up into one body of one `Emit` run per splice,
    // each reading the previous; a memo wiped per statement re-walked the
    // whole prefix - cubic in the chain.
    let mut seen = HandleSet::default();
    for (i, stmt) in body.iter().enumerate() {
        let mut walk = |s: &naga::Statement| {
            visit_statement_expression_handles(s, true, &mut |h| {
                cone_locals(h, expressions, &mut seen, &mut locals);
            });
        };
        walk(stmt);
        for nested in nested_blocks(stmt) {
            for_each_statement(nested, &mut walk);
        }
        for (k, (local, _)) in inits.iter().enumerate() {
            if first[k].is_none() && locals.contains(*local) {
                first[k] = Some(i);
            }
        }
    }
    let is_counter = |stmt: &naga::Statement, local| {
        match stmt {
        naga::Statement::Loop { continuing, .. } => continuing.iter().any(|s| {
            matches!(s, naga::Statement::Store { pointer, .. }
                if matches!(expressions[*pointer], naga::Expression::LocalVariable(l) if l == local))
        }),
        _ => false,
    }
    };
    let mut out = naga::Block::with_capacity(body.len() + inits.len());
    for (k, (_, init)) in inits.iter_mut().enumerate() {
        if first[k].is_none() {
            out.extend_block(std::mem::take(init));
        }
    }
    for (i, (stmt, span)) in body.span_into_iter().enumerate() {
        for counter in [false, true] {
            for k in 0..inits.len() {
                if first[k] == Some(i) && is_counter(&stmt, inits[k].0) == counter {
                    out.extend_block(std::mem::take(&mut inits[k].1));
                }
            }
        }
        out.push(stmt, span);
    }
    out
}

/// Add every `LocalVariable` in `h`'s cone to `locals`.
fn cone_locals(
    h: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    seen: &mut HandleSet<naga::Expression>,
    locals: &mut HandleSet<naga::LocalVariable>,
) {
    if !seen.insert(h) {
        return;
    }
    if let naga::Expression::LocalVariable(l) = expressions[h] {
        locals.insert(l);
    }
    visit_expression_children(&expressions[h], |c| {
        cone_locals(c, expressions, seen, locals)
    });
}

// MARK: Multi-site eligibility and pricing

/// The returned value of a body that may be cloned into each of its call
/// sites, `None` otherwise.  The soundness rule is a body of exactly
/// `[Emit*, Return { value }]` with no locals, statement results or other
/// side-effecting statements, so the expression DAG lands in a caller
/// without disturbing any assumption about stores or control flow between
/// statements; `max_node_count` caps the DAG as the render-depth safety.
fn clonable_body(
    function: &naga::Function,
    max_node_count: usize,
) -> Option<naga::Handle<naga::Expression>> {
    if !function.local_variables.is_empty() || function.result.is_none() {
        return None;
    }
    let value = extract_inline_return_expression(&function.body)?;
    // Dense over the callee's arena and re-checked at every node, so a
    // `Vec<bool>` beats a hash set.
    let mut visited = vec![false; function.expressions.len()];
    let node_count = analyze_inline_expression(
        value,
        &function.expressions,
        function.arguments.len(),
        &mut visited,
    )?;
    (node_count > 0 && node_count <= max_node_count).then_some(value)
}

/// How far past its definition's bytes a clone may price and still be
/// tried: what the model cannot see (`paying_clones`) is worth about this
/// much on a small helper.  Calibrated on the corpora: no clone the render
/// confirmed priced further past its definition, and none priced further
/// passed its trial; the ones confirmed past the model were calls the
/// vanished `Call` statements let the emitter stash and a literal the
/// second copy pushed over extraction.
const CLONE_TRIAL_SLACK: isize = 15;

/// A multi-site candidate: the function and the value its body returns.
struct Clonable {
    function: naga::Handle<naga::Function>,
    value: naga::Handle<naga::Expression>,
}

/// One callee as it renders: its declaration, and its body - the `let`s
/// it binds plus the returned value - with each parameter read counted per
/// rendering (a bound root renders once, then as its name).
struct CalleePrice {
    name: usize,
    definition: usize,
    body: usize,
    parameter_reads: Vec<usize>,
    parameter_names: Vec<usize>,
}

/// The reads of each parameter the rendering of `h` spells, `bound` roots
/// standing for their names.
fn parameter_reads(
    h: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    bound: &HandleSet<naga::Expression>,
    reads: &mut [usize],
) {
    if let naga::Expression::FunctionArgument(k) = arena[h] {
        reads[k as usize] += 1;
        return;
    }
    visit_expression_children(&arena[h], |c| {
        if !bound.contains(c) {
            parameter_reads(c, arena, bound, reads);
        }
    });
}

// Out of line: inlined into the candidate loop it is a large copy that
// saves no call.
#[inline(never)]
fn price_callee(
    pricer: &mut FunctionPricer<'_, '_>,
    function: &naga::Function,
    value: naga::Handle<naga::Expression>,
) -> Option<CalleePrice> {
    let definition = pricer.definition_len()?;
    let mut bound = HandleSet::default();
    let mut body = 0;
    for stmt in function.body.iter() {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                body += pricer.emit(h)?;
                if pricer.bound(h) {
                    bound.insert(h);
                }
            }
        }
    }
    body += pricer.expr_len(value)?;
    let mut reads = vec![0; function.arguments.len()];
    let value_root = (!bound.contains(value)).then_some(value);
    for root in bound.iter().copied().chain(value_root) {
        parameter_reads(root, &function.expressions, &bound, &mut reads);
    }
    Some(CalleePrice {
        name: pricer.name_len(),
        definition,
        body,
        parameter_reads: reads,
        parameter_names: (0..function.arguments.len())
            .map(|k| pricer.argument_name_len(k))
            .collect(),
    })
}

/// The candidates whose clones render shorter than the definition plus
/// the calls.  Shortlisted by the generator's own prices - per site, the
/// body as the callee renders it with the caller's arguments in place of
/// the parameter names (an argument read several times at the cheaper of
/// its copies and a `let`) against the call `N(a,b)` and the `let` a call
/// pays that an inlined value does not, summed over the sites against the
/// declaration - and confirmed by rendering the module with the clones
/// applied (`splice_into_callers`, then [`super::shipped_len`]) against
/// the base render, since what the model cannot see is module-wide: the
/// alias or extracted literal the deleted declaration stopped paying for,
/// the literal the copies start paying to extract, the name the slot
/// frees, the calls the vanished `Call` statements let the emitter stash,
/// the parentheses an argument takes in its new place.  A site that
/// declines the clone fails the trial as it would the run; a tie keeps the
/// function; a candidate the model rejects by more than
/// [`CLONE_TRIAL_SLACK`] is not tried (a render is the cost of a function).
fn paying_clones(
    module: &naga::Module,
    clonable: &[Clonable],
    call_sites: &HandleMap<naga::Function, Vec<CallSite>>,
    callers: &[&naga::Function],
    scope_names: &HashSet<String>,
    ctx: &PassContext<'_>,
) -> Vec<naga::Handle<naga::Function>> {
    let preserve: HashSet<String> = ctx.config.preserve_symbols.iter().cloned().collect();
    let plan = super::rename::plan_names(module, &preserve, ctx.config.mangle());
    let renamed = plan.applied(module);
    // The empty-call deletion before this removed statements only, so the
    // pass's `info` still describes every arena.
    let mut pricer = Pricer::new(&renamed, ctx.info, ctx.config, &plan);

    let callees: Vec<Option<CalleePrice>> = clonable
        .iter()
        .map(|c| {
            price_callee(
                &mut pricer.function(c.function),
                &module.functions[c.function],
                c.value,
            )
        })
        .collect();

    // Each caller is priced in one visit, over every site it hosts (a
    // callee hosts none: its body has no `Call`).
    let function_handles: Vec<_> = module.functions.iter().map(|(h, _)| h).collect();
    let mut hosted: Vec<Vec<(usize, usize)>> = callers.iter().map(|_| Vec::new()).collect();
    for (i, c) in clonable.iter().enumerate() {
        for (s, site) in call_sites[c.function].iter().enumerate() {
            hosted[site.caller].push((i, s));
        }
    }
    // Per candidate, what its clones cost beyond its calls; `None` where a
    // site does not price.
    let mut clone_cost: Vec<Option<isize>> =
        callees.iter().map(|b| b.as_ref().map(|_| 0)).collect();
    for (caller, sites) in hosted.iter().enumerate() {
        if sites.is_empty() {
            continue;
        }
        let mut fp = match function_handles.get(caller) {
            Some(&h) => pricer.function(h),
            None => pricer.entry_point(caller - function_handles.len()),
        };
        let let_len = fp.let_name_len();
        let let_cost = let_len + fp.let_boilerplate();
        let arena = &callers[caller].expressions;
        for &(i, s) in sites {
            let Some(callee) = &callees[i] else {
                continue;
            };
            let site = &call_sites[clonable[i].function][s];
            let result_reads = site.result.map_or(0, |r| fp.uses(r));
            // `N(a,b)`, plus the `let` binding its result when the result
            // is read once but not stashed (an inlined value moves to its
            // use instead); a result read more than once is bound either
            // way, one read by nothing dies with the call.
            let mut call = (callee.name + 2 + site.arguments.len().saturating_sub(1)) as isize;
            if result_reads == 1 && site.result.is_some_and(|r| !fp.stashed(r)) {
                call += let_cost as isize;
            }
            let mut clone = if result_reads == 0 {
                0
            } else {
                callee.body as isize
            };
            let mut priced = true;
            for (k, &argument) in site.arguments.iter().enumerate() {
                // Bound in the caller already: both sides spell its name
                // (and its own text, shared, could be long to render).
                let bound = expression_needs_emit(&arena[argument]) && fp.binds(argument);
                let len = if bound {
                    let_len
                } else if let Some(len) = fp.expr_len(argument) {
                    len
                } else {
                    priced = false;
                    break;
                };
                call += len as isize;
                if result_reads == 0 {
                    continue;
                }
                let reads = callee.parameter_reads[k];
                clone += match reads {
                    0 => 0,
                    1 => len,
                    n => (n * len).min(len + let_cost + n * let_len),
                } as isize;
                clone -= (reads * callee.parameter_names[k]) as isize;
            }
            clone_cost[i] = match clone_cost[i] {
                Some(cost) if priced => Some(cost + clone - call),
                _ => None,
            };
        }
    }

    // The candidates the model has paying, and the ones within its slack.
    let (paying, admitted): (Vec<_>, Vec<_>) = clonable
        .iter()
        .zip(&callees)
        .zip(&clone_cost)
        .filter_map(|((c, callee), cost)| {
            let (callee, cost) = (callee.as_ref()?, (*cost)?);
            (cost < callee.definition as isize + CLONE_TRIAL_SLACK)
                .then_some((c.function, cost < callee.definition as isize))
        })
        .partition(|&(_, pays)| pays);
    if paying.is_empty() && admitted.is_empty() {
        return Vec::new();
    }
    let Some(base_len) = super::shipped_len(module.clone(), ctx) else {
        return Vec::new();
    };
    // Whether `module` with `set` cloned renders shorter; a site that
    // declines one of them fails the set.
    let pays = |set: &[naga::Handle<naga::Function>]| {
        let mut trial = module.clone();
        let set: HandleSet<naga::Function> = set.iter().copied().collect();
        let sites: usize = set.iter().map(|&h| call_sites[h].len()).sum();
        splice_into_callers(&mut trial, &set, scope_names) == sites
            && super::shipped_len(trial, ctx).is_some_and(|len| len < base_len)
    };
    // The model's own verdicts are tried together (one render where they
    // all pay, the common case) and each alone otherwise; a candidate
    // admitted on slack is tried alone, so it cannot ride on the others.
    // A wrong verdict can still ride on the rest's surplus in the joint
    // render, which bounds the miss; a render per clone on every sweep
    // would cost more, and would miss the same way in reverse (two clones
    // that each pay alone are not confirmed together).
    let paying: Vec<_> = paying.into_iter().map(|(h, _)| h).collect();
    let mut confirmed = if paying.len() > 1 && pays(&paying) {
        paying
    } else {
        paying.into_iter().filter(|&h| pays(&[h])).collect()
    };
    confirmed.extend(admitted.into_iter().map(|(h, _)| h).filter(|&h| pays(&[h])));
    confirmed
}

/// One call of a candidate: the caller (its position in [`all_functions`]
/// order) and the argument and result handles in the caller's arena.
struct CallSite {
    caller: usize,
    arguments: Vec<naga::Handle<naga::Expression>>,
    result: Option<naga::Handle<naga::Expression>>,
}

/// Per callee, every call site module-wide (so `len()` is the call count).
fn collect_call_sites(module: &naga::Module) -> HandleMap<naga::Function, Vec<CallSite>> {
    let mut sites: HandleMap<naga::Function, Vec<CallSite>> = HandleMap::default();
    for (caller, function) in all_functions(module).enumerate() {
        for_each_statement(&function.body, &mut |stmt| {
            if let naga::Statement::Call {
                function: callee,
                arguments,
                result,
            } = stmt
            {
                sites.entry(*callee).or_default().push(CallSite {
                    caller,
                    arguments: arguments.clone(),
                    result: *result,
                });
            }
        });
    }
    sites
}

/// The value handle of a body shaped exactly `[Emit*, Return { value }]`,
/// `None` for anything else: the pass's single purity gate.
fn extract_inline_return_expression(block: &naga::Block) -> Option<naga::Handle<naga::Expression>> {
    let mut return_value = None;
    let mut seen_return = false;

    for statement in block {
        match statement {
            naga::Statement::Emit(_) if !seen_return => {}
            naga::Statement::Return { value: Some(value) } if !seen_return => {
                return_value = Some(*value);
                seen_return = true;
            }
            _ => return None,
        }
    }

    return_value
}

/// Node count of the DAG under `handle`, or `None` when it holds an
/// inline-disallowed expression or an out-of-range `FunctionArgument`.
fn analyze_inline_expression(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    argument_count: usize,
    visited: &mut [bool],
) -> Option<usize> {
    let slot = visited.get_mut(handle.index())?;
    if *slot {
        return Some(0);
    }
    *slot = true;

    let expr = &expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    if let naga::Expression::FunctionArgument(index) = expr {
        return ((*index as usize) < argument_count).then_some(1);
    }

    // `ok` stops descending siblings after a failure, so the `visited` marks
    // match an early-return walk exactly.
    let mut total = 1usize;
    let mut ok = true;
    visit_expression_children(expr, |child| {
        if !ok {
            return;
        }
        match analyze_inline_expression(child, expressions, argument_count, visited) {
            Some(n) => total += n,
            None => ok = false,
        }
    });
    ok.then_some(total)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = InliningPass::default();
        let config = Config::default();

        let changed = PassContext::run_pass(&mut pass, &mut module, &config)
            .expect("inlining pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn find_function_handle_by_name(
        module: &naga::Module,
        name: &str,
    ) -> naga::Handle<naga::Function> {
        module
            .functions
            .iter()
            .find_map(|(handle, function)| {
                (function.name.as_deref() == Some(name)).then_some(handle)
            })
            .expect("function should exist")
    }

    /// The splice compacts its callee away, so a spliced helper is proven
    /// by its absence: no function of that name, and no call left in `block`.
    fn assert_spliced(module: &naga::Module, name: &str, block: &naga::Block) {
        assert!(
            module
                .functions
                .iter()
                .all(|(_, f)| f.name.as_deref() != Some(name)),
            "{name} must be compacted away"
        );
        let mut calls = 0;
        crate::ir::visit::for_each_statement(block, &mut |s| {
            calls += usize::from(matches!(s, naga::Statement::Call { .. }));
        });
        assert_eq!(calls, 0, "no call may remain after splicing {name}");
    }

    fn count_calls_to_function(block: &naga::Block, target: naga::Handle<naga::Function>) -> usize {
        let mut count = 0usize;
        crate::ir::visit::for_each_statement(block, &mut |statement| {
            if let naga::Statement::Call { function, .. } = statement
                && *function == target
            {
                count += 1;
            }
        });
        count
    }

    #[test]
    fn inlines_simple_return_expression_function() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(2.0);
    return vec4f(y, y, y, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 1, "expected one helper call before inlining");

        let (changed, after) = run_pass(source);
        assert!(changed, "inlining should report a change");
        assert_spliced(&after, "helper", &after.entry_points[0].function.body);
    }

    #[test]
    fn library_module_keeps_call_sites_intact() {
        let src = r#"
fn helper() -> vec4f { return vec4f(0.1, 0.2, 0.3, 1.0); }
fn main_color() -> vec4f { return helper(); }
"#;
        let (changed, after) = run_pass(src);
        assert!(!changed, "a library module must not inline");
        let helper = find_function_handle_by_name(&after, "helper");
        let caller = after
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("main_color"))
            .map(|(_, f)| f)
            .expect("main_color survives");
        assert_eq!(count_calls_to_function(&caller.body, helper), 1);
    }

    /// A call-site literal substituted into a clonable body's runtime
    /// `v[i]` can manufacture a statically out-of-bounds index naga rejects;
    /// the clone must decline (call kept) instead of rolling the whole pass
    /// back.  `run_pass` validates, and the run with both sites in bounds
    /// proves the gate is value-sensitive: the body clones there, while
    /// beside the bad site the good one keeps its call, since a clone next
    /// to a surviving definition is pure loss.
    #[test]
    fn declines_call_site_whose_literal_index_is_out_of_bounds() {
        let source = r#"
fn pick(v: vec4<f32>, i: i32) -> f32 {
    return v[i];
}

@compute @workgroup_size(1)
fn main() {
    let bad = pick(vec4<f32>(2.0, 3.0, 4.0, 5.0), 6);
    let good = pick(vec4<f32>(2.0, 3.0, 4.0, 5.0), 2);
    _ = bad + good;
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed, "one declined site keeps every call");
        let pick = find_function_handle_by_name(&after, "pick");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            2
        );
        let (changed, after) = run_pass(&source.replace("6)", "3)"));
        assert!(changed, "with every site in bounds the body clones");
        assert!(
            after
                .functions
                .iter()
                .all(|(_, f)| f.name.as_deref() != Some("pick")),
            "pick must be cloned into both sites and compacted away"
        );
    }

    /// A `-0.0` literal argument is declined ([`has_negative_zero_leaf`]):
    /// `-(x*x)` is not inlined with `x = -0.0` substituted, and the other
    /// site keeps its call with it.
    #[test]
    fn declines_call_site_passing_a_negative_zero_literal() {
        let source = "fn sq(x: f32) -> f32 { return -(x * x); }\n\
                      @group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
                      @compute @workgroup_size(1) fn main() {\n\
                        out[0] = bitcast<u32>(sq(-0.0));\n\
                        out[1] = bitcast<u32>(sq(2.0));\n\
                      }";
        let (changed, module) = run_pass(source);
        assert!(!changed, "one declined site keeps every call");
        let sq = find_function_handle_by_name(&module, "sq");
        assert_eq!(
            count_calls_to_function(&module.entry_points[0].function.body, sq),
            2
        );
    }

    #[test]
    fn declines_forwarded_argument_base_and_negative_index() {
        let source = r#"
fn pick(v: vec4<f32>, i: i32) -> f32 {
    return v[i];
}

fn outer(v: vec4<f32>) -> f32 {
    return pick(v, 6);
}

@compute @workgroup_size(1)
fn main() {
    var w = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let neg = pick(w, -1);
    _ = outer(vec4<f32>(0.5, 0.5, 0.5, 0.5)) + neg;
}
"#;
        let (_, after) = run_pass(source);
        let pick = find_function_handle_by_name(&after, "pick");
        // `outer` is spliced into `main`, so both sites now sit there.
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            2,
            "both statically-invalid call sites must survive as calls"
        );
    }

    /// A body with locals is not clonable: cloned per site it would
    /// duplicate the local, so only the single-call splice takes it.
    #[test]
    fn a_multi_site_helper_with_locals_keeps_its_calls() {
        let source = r#"
fn helper(x: f32) -> f32 {
    var t: f32;
    t = x + 1.0;
    return t;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(2.0);
    let z = helper(3.0);
    return vec4f(y, z, y, 1.0);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed, "helper with locals should not be inlined");
        let helper = find_function_handle_by_name(&after, "helper");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, helper),
            2
        );
    }

    // MARK: Single-call splice

    #[test]
    fn splices_a_single_call_helper_with_locals_and_a_loop() {
        let source = r#"
fn accumulate(x: f32) -> f32 {
    var total = 0.0;
    for (var i = 0; i < 4; i++) {
        total += x * f32(i);
    }
    return total;
}
@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let s = accumulate(uv.x);
    return vec4f(s, s, s, 1.0);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        assert_spliced(&after, "accumulate", &entry.body);
        assert_eq!(
            entry.local_variables.len(),
            2,
            "`total` and `i` move into the caller"
        );
        assert!(
            entry
                .body
                .iter()
                .any(|s| matches!(s, naga::Statement::Loop { .. })),
            "the helper's loop moves into the caller"
        );
    }

    #[test]
    fn splices_a_void_helper_and_drops_its_bare_return() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn write(v: f32) {
    out[0] = v;
    return;
}
@compute @workgroup_size(1)
fn main() {
    write(2.0);
    out[1] = 3.0;
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        assert_spliced(&after, "write", &entry.body);
        // The entry's own trailing `Return` is the only one left.
        assert!(
            !entry.body[..entry.body.len() - 1]
                .iter()
                .any(|s| matches!(s, naga::Statement::Return { .. })),
            "the helper's bare `return` must not end the caller early"
        );
        assert_eq!(
            entry
                .body
                .iter()
                .filter(|s| matches!(s, naga::Statement::Store { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn a_helper_with_an_early_return_keeps_its_call() {
        let source = r#"
fn safe_sqrt(x: f32) -> f32 {
    if (x < 0.0) {
        return 0.0;
    }
    return sqrt(x);
}
@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(safe_sqrt(uv.x));
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed);
        let helper = find_function_handle_by_name(&after, "safe_sqrt");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, helper),
            1
        );
    }

    /// A helper called from a loop has its locals re-initialised by stores
    /// at the splice point (`splice_body`), the zero value for an
    /// uninitialised one.
    #[test]
    fn a_call_inside_a_loop_reinitialises_the_helpers_locals() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<i32>;
fn bump() -> i32 {
    var n = 0;
    var m: i32;
    n += 1;
    m += n;
    return m;
}
@compute @workgroup_size(1)
fn main() {
    var s = 0;
    for (var i = 0; i < 3; i++) {
        s += bump();
    }
    out[0] = s;
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        let moved: Vec<_> = entry
            .local_variables
            .iter()
            .filter(|(_, l)| matches!(l.name.as_deref(), Some("n" | "m")))
            .collect();
        assert_eq!(moved.len(), 2);
        assert!(
            moved.iter().all(|(_, l)| l.init.is_none()),
            "an init would run once per invocation"
        );
        let body = entry
            .body
            .iter()
            .find_map(|s| match s {
                naga::Statement::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("the caller's loop");
        // Two re-init stores plus the helper's own `n += 1` and `m += n`,
        // inside the `Block` naga wraps a `for` body in.
        let mut stores = 0;
        for_each_statement(body, &mut |s| {
            if let naga::Statement::Store { pointer, .. } = s
                && let naga::Expression::LocalVariable(l) = entry.expressions[*pointer]
                && moved.iter().any(|(h, _)| *h == l)
            {
                stores += 1;
            }
        });
        assert_eq!(stores, 4);
    }

    #[test]
    fn a_chain_of_single_call_helpers_flattens_in_one_run() {
        let source = r#"
fn inner(x: f32) -> f32 {
    var t = x * 2.0;
    return t + 1.0;
}
fn outer(x: f32) -> f32 {
    var u = inner(x);
    return u * 3.0;
}
@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(outer(uv.x));
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        for name in ["inner", "outer"] {
            assert_spliced(&after, name, &entry.body);
        }
        assert_eq!(entry.local_variables.len(), 2);
    }

    /// The bound comes from naga's typifier, so a pointer parameter's base
    /// is sized too: an out-of-bounds literal into `(*p)[i]` is declined
    /// instead of rolling the pass back.
    #[test]
    fn declines_a_literal_index_out_of_bounds_through_a_pointer_parameter() {
        let source = r#"
fn pick(p: ptr<function, array<f32, 4>>, i: i32) -> f32 {
    return (*p)[i];
}
@fragment
fn fs_main() -> @location(0) vec4f {
    var arr = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    return vec4f(pick(&arr, 6));
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed);
        let pick = find_function_handle_by_name(&after, "pick");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            1
        );
    }

    /// An argument that zeroes an integer divisor or puts a shift amount at
    /// or past the bit width - a literal, a literal `const`, or through
    /// arithmetic the evaluator models - is a shader-creation error
    /// wherever it sits, so the site is declined rather than the whole pass
    /// rolled back; a legal literal, a masked amount and a float zero still
    /// splice.
    #[test]
    fn declines_a_literal_argument_that_makes_a_failable_slot_a_static_error() {
        let source = r#"
const Z: i32 = 0;
fn div(n: i32, d: i32) -> i32 { return n / d; }
fn rem(n: i32, d: i32) -> i32 { return n % d; }
fn shl(n: u32, s: u32) -> u32 { return n << s; }
fn sub(n: i32, d: i32) -> i32 { return n / (d - 2); }
fn half(n: i32, d: i32) -> i32 { return n / d; }
fn fdiv(n: f32, d: f32) -> f32 { return n / d; }
fn mask(n: u32, s: u32) -> u32 { return n << (s & 31u); }
@fragment
fn fs_main(@location(0) x: f32) -> @location(0) vec4f {
    let n = i32(x);
    let kept = f32(div(n, 0)) + f32(rem(n, Z)) + f32(shl(u32(x), 40u)) + f32(sub(n, 2));
    return vec4f(kept, f32(half(n, 2)) + fdiv(x, 0.0), f32(mask(u32(x), 40u)), 1.0);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        for name in ["div", "rem", "shl", "sub"] {
            let target = find_function_handle_by_name(&after, name);
            assert_eq!(
                count_calls_to_function(&entry.body, target),
                1,
                "{name} keeps its call"
            );
        }
        for name in ["half", "fdiv", "mask"] {
            assert!(
                after
                    .functions
                    .iter()
                    .all(|(_, f)| f.name.as_deref() != Some(name)),
                "{name} must be spliced and compacted away"
            );
        }
    }

    /// The returned value is the other way const-ness enters a slot, unseen
    /// by the argument gates: a callee returning `8u` into `a[idx()]`, `0u`
    /// into a divisor, `32u` into a shift amount, or `7u` into
    /// `a[seven() + 1u]` would roll the whole pass back - every other splice
    /// in the module with it.  The caller's slots are judged on the value;
    /// a legal value and a runtime sibling still splice.
    #[test]
    fn declines_a_returned_value_that_makes_a_caller_slot_a_static_error() {
        let source = r#"
var<private> a: array<u32, 8>;
fn idx() -> u32 { return 8u; }
fn zero() -> u32 { return 0u; }
fn wide() -> u32 { return 32u; }
fn seven() -> u32 { return 7u; }
fn off(i: u32) -> u32 { return i + 1u; }
fn ok_idx() -> u32 { return 3u; }
fn three() -> u32 { return 3u; }
fn other(x: u32) -> u32 { return x * 3u + 1u; }
@compute @workgroup_size(1)
fn cs_main() {
    let x = other(a[1]);
    a[0] = a[idx()] + x / zero() + (1u << wide()) + a[seven() + 1u] + a[off(7u)]
        + a[ok_idx()] + a[2] % three();
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        for name in ["idx", "zero", "wide", "seven", "off"] {
            let target = find_function_handle_by_name(&after, name);
            assert_eq!(
                count_calls_to_function(&entry.body, target),
                1,
                "{name} keeps its call"
            );
        }
        for name in ["ok_idx", "three", "other"] {
            assert!(
                after
                    .functions
                    .iter()
                    .all(|(_, f)| f.name.as_deref() != Some(name)),
                "{name} must be spliced and compacted away"
            );
        }
    }

    /// A multi-site candidate is judged in the value direction at every
    /// site before any clone: one site's out-of-bounds index keeps the
    /// definition and clones none.
    #[test]
    fn a_returned_value_out_of_bounds_at_one_site_clones_nowhere() {
        let source = r#"
var<private> a: array<u32, 8>;
fn eight() -> u32 { return 8u; }
@compute @workgroup_size(1)
fn cs_main() {
    a[0] = a[eight()];
    a[1] = eight();
}
"#;
        let (changed, after) = run_pass(source);
        assert!(
            !changed,
            "the out-of-bounds site declines, so neither clones"
        );
        let eight = find_function_handle_by_name(&after, "eight");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, eight),
            2
        );
    }

    /// A returned float making a caller's `-x` (a zero) or `x % y` (any
    /// float) a const-expression is the sign-sensitive crossing (Dawn on
    /// Metal computes `-nz()` as `-0.0` and the folded `-(-0.0)` as `0.0`),
    /// with no driver check behind it; the operator that keeps a runtime
    /// operand or reaches no zero still splices.  `-id(z2())` sees `z2`'s
    /// value through the earlier splice's replacement.
    #[test]
    fn declines_a_returned_zero_that_makes_a_sign_sensitive_op_const() {
        let source = r#"
fn nz() -> f32 { return -0.0; }
fn z() -> f32 { return 0.0; }
fn three() -> f32 { return 3.0; }
fn fm() -> f32 { return 3.0; }
fn z2() -> f32 { return 0.0; }
fn id(v: f32) -> f32 { return v; }
@fragment
fn fs_main(@location(0) x: f32) -> @location(0) vec4f {
    return vec4f(-nz() + 4.0 % fm(), x * z(), three() * 2.0, -id(z2()));
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let entry = &after.entry_points[0].function;
        for name in ["nz", "fm", "id"] {
            let target = find_function_handle_by_name(&after, name);
            assert_eq!(
                count_calls_to_function(&entry.body, target),
                1,
                "{name} keeps its call"
            );
        }
        for name in ["z", "three", "z2"] {
            assert!(
                after
                    .functions
                    .iter()
                    .all(|(_, f)| f.name.as_deref() != Some(name)),
                "{name} must be spliced and compacted away"
            );
        }
    }

    /// `a[i + 1]` with `i = 4` substituted passes naga's IR validator
    /// (`index_is_static_error`; tint: `index 5 out of bounds`).  The slot
    /// gate evaluates the substituted cone like a divisor: the OOB sites keep
    /// their call, the in-bounds one splices.  `run_pass` validates through
    /// the driver's own slot check, so an escaped shape panics here rather
    /// than at emission.
    #[test]
    fn declines_a_computed_index_argument_out_of_bounds() {
        let source = r#"
const K: u32 = 5u;
fn next(i: i32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i + 1]; }
fn prev(i: u32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i - 1u]; }
fn named(i: u32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i]; }
fn lane(v: vec2u, i: u32) -> u32 { return v[i * 2u]; }
fn safe(i: i32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i + 1]; }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = next(4);
    out[1] = prev(0u);
    out[2] = named(K);
    out[3] = lane(vec2u(7u, 8u), 1u);
    out[4] = safe(2);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed, "the in-bounds site must splice");
        let entry = &after.entry_points[0].function;
        for name in ["next", "prev", "named", "lane"] {
            let target = find_function_handle_by_name(&after, name);
            assert_eq!(
                count_calls_to_function(&entry.body, target),
                1,
                "{name}: the out-of-bounds site keeps its call"
            );
        }
        assert!(
            after
                .functions
                .iter()
                .all(|(_, f)| f.name.as_deref() != Some("safe")),
            "the in-bounds site splices and its callee is compacted away"
        );
    }

    /// A nested call passes the INNER splice's value, not its `CallResult`:
    /// `f(g(1))` with `g` spliced first hands `f` the const `3` the rewrite
    /// will substitute, so the site gate must judge that, not the runtime
    /// read the statement still spells.  Judged on the `CallResult`, the
    /// out-of-bounds `a[3 + 1]` passed the gate and the driver rolled the
    /// whole pass back, every sweep, losing `g`'s splice with it.
    #[test]
    fn a_nested_call_is_judged_on_the_inner_splices_value() {
        let source = r#"
fn g(i: u32) -> u32 { return i * 2u + 1u; }
fn f(i: u32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i + 1u]; }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = f(g(1u));
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed, "the inner helper splices");
        assert!(
            after
                .functions
                .iter()
                .all(|(_, f)| f.name.as_deref() != Some("g")),
            "g is spliced and compacted away"
        );
        let f = find_function_handle_by_name(&after, "f");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, f),
            1,
            "the outer site, now passing 3 into a[i + 1], keeps its call"
        );
    }

    /// A clone is priced against the definition and every call, so a
    /// multi-site candidate is cloned into all of its sites or none: with
    /// one site declined (here an out-of-bounds literal index) the
    /// definition survives, and a clone beside it is pure loss.
    #[test]
    fn a_multi_site_candidate_with_one_declined_site_is_not_cloned_at_all() {
        let source = r#"
fn pick(i: u32) -> u32 { let a = array<u32,4>(1u, 2u, 3u, 4u); return a[i % 4u] + a[i]; }
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> inp: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = pick(inp[0]);
    out[1] = pick(7u);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed, "neither site may be cloned");
        let pick = find_function_handle_by_name(&after, "pick");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            2,
            "both calls stay"
        );
    }

    /// A local keeps its name into a no-mangle output, so a spliced one
    /// dodges the caller's locals and the module scope it would shadow.
    #[test]
    fn spliced_locals_are_renamed_apart_from_the_callers() {
        let source = r#"
var<private> k: f32;
fn helper(t: f32) -> f32 {
    var k = t * 2.0;
    var t2 = k + 1.0;
    return t2;
}
@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var t2 = 0.5;
    let r = helper(uv.x) + k;
    return vec4f(r + t2);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let mut names: Vec<_> = after.entry_points[0]
            .function
            .local_variables
            .iter()
            .filter_map(|(_, l)| l.name.clone())
            .collect();
        names.sort();
        assert_eq!(names, ["k_1", "t2", "t2_1"]);
    }

    /// The entry point's outer loop body after a splice, with the local
    /// each `Store` targets: `(statement index, local name)` for the whole-
    /// variable stores, plus the index of the inner `Loop`.
    fn outer_loop_body(module: &naga::Module) -> (Vec<(usize, String)>, usize) {
        let entry = &module.entry_points[0].function;
        let outer = entry
            .body
            .iter()
            .find_map(|s| match s {
                naga::Statement::Loop { body, .. } => Some(body),
                _ => None,
            })
            .expect("outer loop");
        // naga lowers a `for` body as one `Block` statement.
        let outer = outer
            .iter()
            .find_map(|s| match s {
                naga::Statement::Block(inner) => Some(inner),
                _ => None,
            })
            .unwrap_or(outer);
        let mut stores = Vec::new();
        let mut inner = None;
        for (i, s) in outer.iter().enumerate() {
            match s {
                naga::Statement::Store { pointer, .. } => {
                    if let naga::Expression::LocalVariable(l) = entry.expressions[*pointer] {
                        stores.push((i, entry.local_variables[l].name.clone().unwrap()));
                    }
                }
                naga::Statement::Loop { .. } if inner.is_none() => inner = Some(i),
                _ => {}
            }
        }
        (stores, inner.expect("inner loop"))
    }

    /// A callee spliced into a loop has its locals' inits stored at the
    /// splice point; each store lands before the FIRST statement touching
    /// the local, so a counter's store is adjacent to its loop and the
    /// emitter recovers the `for` initializer (tint proves the loop finite
    /// only in that shape).
    #[test]
    fn a_spliced_counters_init_lands_beside_its_loop() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn accumulate(x: f32) -> f32 {
    var total = 0.0;
    total = x;
    for (var i = 0; i < 4; i++) { total += f32(i); }
    return total;
}
@compute @workgroup_size(1) fn main() {
    for (var k = 0; k < 2; k++) { out[k] = accumulate(f32(k)); }
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let (stores, inner) = outer_loop_body(&after);
        let names: Vec<&str> = stores.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["total", "total", "i"], "{stores:?}");
        assert_eq!(
            stores[2].0 + 1,
            inner,
            "the counter's store is adjacent to its loop"
        );
    }

    #[test]
    fn two_locals_first_touched_by_one_loop_put_the_counter_last() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn accumulate(x: f32) -> f32 {
    var i = 0;
    var total = 0.0;
    for (; i < 4; i++) { total += x * f32(i); }
    return total;
}
@compute @workgroup_size(1) fn main() {
    for (var k = 0; k < 2; k++) { out[k] = accumulate(f32(k)); }
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let (stores, inner) = outer_loop_body(&after);
        let names: Vec<&str> = stores.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(
            names,
            ["total", "i"],
            "arena order, counter last: {stores:?}"
        );
        assert_eq!(stores[1].0 + 1, inner);
    }

    #[test]
    fn an_unreferenced_spliced_local_keeps_the_front_slot() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn accumulate(x: f32) -> f32 {
    var scratch: f32;
    var total = 0.0;
    for (var i = 0; i < 4; i++) { total += x * f32(i); }
    return total;
}
@compute @workgroup_size(1) fn main() {
    for (var k = 0; k < 2; k++) { out[k] = accumulate(f32(k)); }
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed);
        let (stores, inner) = outer_loop_body(&after);
        let names: Vec<&str> = stores.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["scratch", "total", "i"], "{stores:?}");
        assert_eq!(stores[2].0 + 1, inner);
    }

    #[test]
    fn skips_function_with_too_many_call_sites() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    let c = helper(3.0);
    let d = helper(4.0);
    return vec4f(a + b + c + d, 0.0, 0.0, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(
            before_calls, 4,
            "expected four helper calls before inlining"
        );

        let (changed, after) = run_pass(source);
        assert!(
            !changed,
            "helper should not be inlined when call-site count exceeds threshold"
        );

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(after_calls, 4, "all helper calls should remain");
    }

    #[test]
    fn inlining_with_local_init_preserves_valid_handles() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    var tmp: f32;
    let y = helper(2.0);
    tmp = y;
    return vec4f(tmp, 0.0, 0.0, 1.0);
}
"#;
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let ep_fn = &mut module.entry_points[0].function;

        let init_lit = ep_fn.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            naga::Span::UNDEFINED,
        );
        let local_handle = ep_fn
            .local_variables
            .iter()
            .next()
            .map(|(h, _)| h)
            .expect("expected a local variable");
        ep_fn.local_variables[local_handle].init = Some(init_lit);

        let mut pass = InliningPass::default();
        let config = Config::default();
        let changed =
            PassContext::run_pass(&mut pass, &mut module, &config).expect("inlining should run");
        assert!(changed, "helper should be inlined");

        let ep_fn = &module.entry_points[0].function;
        let local = &ep_fn.local_variables[local_handle];
        assert!(local.init.is_some(), "local init should still be present");
        let init_handle = local.init.unwrap();
        assert!(
            init_handle.index() < ep_fn.expressions.len(),
            "local init handle ({}) should be within rebuilt expression arena (len={})",
            init_handle.index(),
            ep_fn.expressions.len(),
        );
        match ep_fn.expressions[init_handle] {
            naga::Expression::Literal(naga::Literal::F32(v)) => {
                assert!(
                    (v - 0.0).abs() < f32::EPSILON,
                    "init should still be 0.0, got {v}"
                );
            }
            ref other => panic!("init expression should be F32 literal, got {other:?}"),
        }
    }

    /// Three copies of a long body render longer than the definition plus
    /// three calls, so the function stays.
    #[test]
    fn clone_pricing_keeps_a_long_body_at_three_sites() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return sin(x) * cos(x) + sin(x * 2.0) * cos(x * 2.0) + sin(x * 3.0) * cos(x * 3.0);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(helper(uv.x), helper(uv.y), helper(uv.x + uv.y), 1.0);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(!changed, "three copies cost more than the definition");
        let helper = find_function_handle_by_name(&after, "helper");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, helper),
            3
        );
    }

    /// Two copies of `x+1+2+3+4` are shorter than `fn N(a:f32)->f32{return
    /// a+1+2+3+4;}` plus two calls.
    #[test]
    fn clone_pricing_inlines_a_short_body_at_two_sites() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0 + 2.0 + 3.0 + 4.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;
        let (changed, after) = run_pass(source);
        assert!(changed, "two short copies beat the definition");
        assert_spliced(&after, "helper", &after.entry_points[0].function.body);
    }

    /// `module` as the sweep's tail would emit it: compacted, renamed,
    /// rendered.
    fn rendered(module: &naga::Module) -> String {
        let mut module = module.clone();
        let preserve = HashSet::new();
        super::super::compact::compact_module(&mut module, &|_| false);
        super::super::rename::plan_names(&module, &preserve, true).apply(&mut module, None);
        let info = crate::io::validate_module(&module).expect("valid");
        let options = crate::generator::GenerateOptions::from_config(&Config::default());
        crate::generator::generate(&module, &info, options)
            .expect("rendered")
            .source
    }

    /// A call whose single-use result cannot be stashed (a store between
    /// the call and the use) pays a `let` that the inlined value does not:
    /// the definition plus two calls priced at `N()` alone would keep the
    /// function.  The clone is confirmed by the render: the output shrinks.
    #[test]
    fn a_call_result_bound_across_a_store_prices_its_let() {
        let source = r#"
struct U { a: f32, b: f32, c: f32, d: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
fn k() -> f32 { return u.a * 1.5 + u.b * 2.5 - u.c * 3.5 + u.d; }
@compute @workgroup_size(1)
fn main() {
    let a = k(); out[0] = 1.0; out[1] = a;
    let b = k(); out[2] = 2.0; out[3] = b;
}
"#;
        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let (changed, after) = run_pass(source);
        assert!(
            changed,
            "the two copies beat the definition and the bound calls"
        );
        assert_spliced(&after, "k", &after.entry_points[0].function.body);
        let (before, after) = (rendered(&before), rendered(&after));
        assert!(after.len() < before.len(), "{before}\n{after}");
    }

    /// The model prices an argument without the parentheses the emitter
    /// gives an operator child under `.x`, so `cross2`'s clones look four
    /// bytes cheaper than they render; the render decides, and the
    /// function stays.
    #[test]
    fn a_clone_the_render_rejects_is_kept() {
        let source = r#"
fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
fn inside(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> bool {
    let ab = cross2(b - a, p - a);
    let bc = cross2(c - b, p - b);
    let ca = cross2(a - c, p - c);
    return (ab >= 0.0 && bc >= 0.0 && ca >= 0.0) || (ab <= 0.0 && bc <= 0.0 && ca <= 0.0);
}
@fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f {
    return vec4f(f32(inside(p, vec2f(0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0))));
}
"#;
        let (_, after) = run_pass(source);
        let cross2 = find_function_handle_by_name(&after, "cross2");
        let calls: usize = all_functions(&after)
            .map(|f| count_calls_to_function(&f.body, cross2))
            .sum();
        assert_eq!(calls, 3, "every call to cross2 stays");
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_continuing() {
        // naga lets `continuing` reference body-defined expressions; a stale
        // `CallResult` there fails `run_pass`'s validation.
        let source = r#"
fn helper(x: i32) -> i32 {
    return x + 1;
}
@compute @workgroup_size(1)
fn cs_main() {
    var i: i32 = 0;
    loop {
        let x = helper(i);
        if (i > 10) { break; }
        continuing {
            i = i + x;
        }
    }
}
"#;
        let (changed, module) = run_pass(source);
        assert!(changed, "helper call in the loop body should be inlined");
        assert_spliced(&module, "helper", &module.entry_points[0].function.body);
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_break_if() {
        // Same hazard via `break_if`.
        let source = r#"
fn limit(x: i32) -> i32 {
    return x + 5;
}
@compute @workgroup_size(1)
fn cs_main() {
    var i: i32 = 0;
    loop {
        let lim = limit(i);
        i = i + 1;
        continuing {
            break if i >= lim;
        }
    }
}
"#;
        let (changed, module) = run_pass(source);
        assert!(changed, "limit call in the loop body should be inlined");
        assert_spliced(&module, "limit", &module.entry_points[0].function.body);
    }

    /// Float `%` reads differently as a const-expression, so a const
    /// argument into its slot is the crossing; the declined site keeps the
    /// definition, so the runtime-argument site keeps its call with it.
    #[test]
    fn a_const_argument_into_a_float_modulo_is_not_inlined() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
fn helper(a: f32) -> f32 {
    return a % 3.0;
}
@compute @workgroup_size(1)
fn cs_main() {
    out[0] = helper(33554432.0) + helper(out[1]);
}
"#;
        let (changed, module) = run_pass(source);
        assert!(
            !changed,
            "the const-argument site declines, so neither clones"
        );
        let helper = find_function_handle_by_name(&module, "helper");
        assert_eq!(
            count_calls_to_function(&module.entry_points[0].function.body, helper),
            2
        );
    }

    /// A helper with its own `@diagnostic(...)` scope is never spliced: its
    /// body is checked under that filter, not the caller's.
    #[test]
    fn a_helper_with_a_diagnostic_filter_keeps_its_call() {
        let source = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@diagnostic(off, derivative_uniformity)
fn helper(uv: vec2f) -> f32 {
    return textureSample(t, s, uv).x;
}
@fragment
fn fs_main(@location(0) uv: vec2f, @location(1) k: f32) -> @location(0) vec4f {
    if (k > 0.5) {
        return vec4f(helper(uv));
    }
    return vec4f(0.0);
}
"#;
        let (changed, module) = run_pass(source);
        assert!(!changed, "nothing to inline");
        let helper = find_function_handle_by_name(&module, "helper");
        assert_eq!(
            count_calls_to_function(&module.entry_points[0].function.body, helper),
            1
        );
    }
}
