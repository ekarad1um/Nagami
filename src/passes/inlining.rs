//! Function inlining for expression-only helpers (no side-effecting
//! statements; global/texture reads are allowed and keep their call-time
//! timing because the body is re-emitted at the call site).
//!
//! The pass only touches functions whose body is a clean
//! `[Emit*, Return { value }]` sequence with no locals, stores, calls,
//! or other side-effecting statements.  Bodies matching that shape are
//! captured as `InlineTemplate` expression DAGs and cloned into each
//! call site, where the call expression is replaced with the cloned
//! return expression.
//!
//! Two budgets keep output size from regressing: `max_node_count`
//! caps a single template's DAG size, and `max_call_sites` caps how
//! many times a template may be reused.  Above those thresholds, or
//! when duplicating a non-trivial body across multiple sites would
//! exceed `MAX_MULTI_SITE_EXPANSION` net added nodes, the function
//! is left alone.

use std::collections::HashMap;

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

use super::expr_util::{
    expression_needs_emit, is_disallowed_inline_expression, map_atomic_function_handles,
    map_cooperative_data_handles, map_gather_mode_handles, map_ray_pipeline_function_handles,
    map_ray_query_function_handles, nested_blocks, nested_blocks_mut, remap_statement_handles,
    try_map_expression_handles_in_place, visit_expression_children,
};

/// Default inlining budgets (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_NODE_COUNT: usize = 24;
/// Default inlining call-site budget (used by [`super::Profile::Aggressive`]).
pub const DEFAULT_MAX_INLINE_CALL_SITES: usize = 3;
/// Widened node budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_NODE_COUNT: usize = 48;
/// Widened call-site budget when running under [`super::Profile::Max`].
pub const MAX_PROFILE_MAX_INLINE_CALL_SITES: usize = 6;

/// Maximum additional expression nodes tolerated when duplicating a
/// template across several call sites, equal to
/// `node_count * (call_sites - 1)`.
///
/// After minification, call syntax is already 1-2 characters, so
/// cloning a non-trivial body over multiple sites is almost always a
/// net size regression.  Single-site inlining is unaffected (the
/// expansion term is zero).
///
/// Deliberately hard-coded rather than promoted to `Config`: the
/// value is a function of the post-mangle call-site length (~ 2-3
/// chars including `(` `)`), not a profile knob.  The pipeline runs
/// inlining ONCE per sweep, sandwiched between const_fold and
/// dead_branch on both sides (see `passes/mod.rs`); multi-sweep
/// convergence catches single-site cases that mature only after later
/// simplification rather than re-running inlining mid-sweep.
const MAX_MULTI_SITE_EXPANSION: usize = 6;

/// Inlining pass parameterised on the per-run node and call-site
/// budgets.
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
    /// Construct a pass with explicit budgets; the defaults live in
    /// `DEFAULT_*` and `MAX_PROFILE_*` constants above and are
    /// selected by [`super::build_ir_passes`] based on the profile.
    pub fn new(max_node_count: usize, max_call_sites: usize) -> Self {
        Self {
            max_node_count,
            max_call_sites,
        }
    }
}

/// Snapshot of an inlinable function: the expression arena alongside
/// the handle that delivers the return value.  Cloned into each
/// caller's arena when the call is replaced.  `argument_types` carries the
/// callee's declared parameter types so the pre-clone OOB gate can size an
/// `Access` base type-derivedly, independent of the caller argument's
/// expression shape.
#[derive(Clone)]
struct InlineTemplate {
    argument_types: Vec<naga::Handle<naga::Type>>,
    return_expr: naga::Handle<naga::Expression>,
    expressions: naga::Arena<naga::Expression>,
}

impl Pass for InliningPass {
    fn name(&self) -> &'static str {
        "function_inlining"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        // A call to a function with an empty body is a no-op statement: the
        // callee cannot observe or affect anything, and argument expressions
        // are pure evaluations that die with the call (compact culls them).
        // The expression-template machinery below excludes ALL void
        // functions, so without this, `fn e(){} ... e();` survives every
        // pass and the empty callee is pinned alive by its own call sites.
        // Calls to preserve-listed functions are kept: their declarations
        // are an external contract and an intact call site is the cheapest
        // proof the declaration stays live.
        let mut changed = delete_calls_to_empty_functions(module, &ctx.config.preserve_symbols);

        let templates = collect_inline_templates(
            module,
            self.max_node_count,
            self.max_call_sites,
            &ctx.config.preserve_symbols,
        );
        if templates.is_empty() {
            return Ok(changed);
        }

        let mut inlined = 0usize;
        for (_, function) in module.functions.iter_mut() {
            inlined += inline_in_function(function, &templates, &module.types);
        }
        for entry in module.entry_points.iter_mut() {
            inlined += inline_in_function(&mut entry.function, &templates, &module.types);
        }
        changed |= inlined > 0;

        Ok(changed)
    }
}

/// Remove every `Call` to a function whose body is empty (or a lone bare
/// `return;`).  Such calls have no observable effect; deleting them lets the
/// next compaction drop the callee itself.  Returns whether anything changed.
fn delete_calls_to_empty_functions(module: &mut naga::Module, preserve: &[String]) -> bool {
    let empty: std::collections::HashSet<naga::Handle<naga::Function>> = module
        .functions
        .iter()
        .filter(|(_, f)| {
            let body_is_empty = matches!(
                f.body.iter().collect::<Vec<_>>().as_slice(),
                [] | [naga::Statement::Return { value: None }]
            );
            body_is_empty
                && f.result.is_none()
                && !f
                    .name
                    .as_deref()
                    .is_some_and(|n| preserve.iter().any(|p| p == n))
        })
        .map(|(h, _)| h)
        .collect();
    if empty.is_empty() {
        return false;
    }

    let mut changed = false;
    for (_, function) in module.functions.iter_mut() {
        changed |= drop_empty_calls_in_block(&mut function.body, &empty);
    }
    for entry in module.entry_points.iter_mut() {
        changed |= drop_empty_calls_in_block(&mut entry.function.body, &empty);
    }
    changed
}

/// Recursively drop `Call`s to `empty` functions from `block`.
fn drop_empty_calls_in_block(
    block: &mut naga::Block,
    empty: &std::collections::HashSet<naga::Handle<naga::Function>>,
) -> bool {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = false;
    for (mut stmt, span) in original.span_into_iter() {
        if let naga::Statement::Call {
            function,
            result: None,
            ..
        } = &stmt
            && empty.contains(function)
        {
            changed = true;
            continue;
        }
        for nested in nested_blocks_mut(&mut stmt) {
            changed |= drop_empty_calls_in_block(nested, empty);
        }
        rebuilt.push(stmt, span);
    }
    *block = rebuilt;
    changed
}

// MARK: Template collection

/// Collect every function eligible for inlining and build its
/// [`InlineTemplate`].
///
/// Safety invariant: only expression-only functions with no side-effecting
/// STATEMENTS are eligible.  The body must be `[Emit*, Return { value }]`
/// with no locals, no stores, no calls, and no other side-effecting
/// statements, which guarantees the template's expression DAG can be cloned
/// into the caller's arena without invalidating any assumptions about stores
/// or control flow between statements.  (Global/texture READS in the body are
/// fine - re-emitting at the call site preserves their call-time timing.)
fn collect_inline_templates(
    module: &naga::Module,
    max_node_count: usize,
    max_call_sites: usize,
    preserve: &[String],
) -> HashMap<naga::Handle<naga::Function>, InlineTemplate> {
    let call_counts = collect_call_counts(module);
    let mut templates = HashMap::new();

    for (function_handle, function) in module.functions.iter() {
        // Preserved functions are an external contract, never templates: a
        // `--preamble` input carries only a STUB body whose real definition
        // arrives when the consumer concatenates the preamble, so baking the
        // stub's expression tree into callers silently bypasses that
        // definition.  Plain `--preserve-symbol` declarations also survive
        // only through intact call sites (`naga::compact` culls call-less
        // functions whenever entry points exist).
        if function
            .name
            .as_deref()
            .is_some_and(|n| preserve.iter().any(|p| p == n))
        {
            continue;
        }
        let call_sites = call_counts.get(&function_handle).copied().unwrap_or(0);
        if call_sites == 0 || call_sites > max_call_sites {
            continue;
        }
        if !function.local_variables.is_empty() || function.result.is_none() {
            continue;
        }

        let Some(return_expr) = extract_inline_return_expression(&function.body) else {
            continue;
        };

        // `visited` is dense over the callee's expression arena and
        // re-checked at every node in the recursive walk; back it
        // with a `Vec<bool>` indexed by `handle.index()` instead of
        // a hashed `HashSet`.  Pre-sized to `function.expressions.len()`
        // so no growth occurs during the analysis.
        let mut visited = vec![false; function.expressions.len()];
        let Some(node_count) = analyze_inline_expression(
            return_expr,
            &function.expressions,
            function.arguments.len(),
            &mut visited,
        ) else {
            continue;
        };

        if node_count == 0 || node_count > max_node_count {
            continue;
        }

        // For multi-site functions, bound the extra nodes introduced by
        // body duplication.  Each additional call site beyond the first
        // copies the entire expression tree, while removing only the
        // single function declaration.
        if call_sites > 1 {
            let expansion = node_count * (call_sites - 1);
            if expansion > MAX_MULTI_SITE_EXPANSION {
                continue;
            }
            // Node counts undercount TEXT: a `Math` node renders its full
            // builtin name (`faceForward(` is 12 characters counted as one
            // node), so duplicating a Math-bearing body across sites grows
            // bytes even inside the node budget - the corpus faceForward /
            // reflect class regressed ~+10 B per extra site this way.
            // Image accessors share the long-spelling problem.  Keeping the
            // helper is the better equilibrium; single-site inlining is
            // unaffected.
            let has_char_heavy_node = function.expressions.iter().any(|(h, e)| {
                visited[h.index()]
                    && matches!(
                        e,
                        naga::Expression::Math { .. }
                            | naga::Expression::ImageSample { .. }
                            | naga::Expression::ImageLoad { .. }
                            | naga::Expression::ImageQuery { .. }
                    )
            });
            if has_char_heavy_node {
                continue;
            }
        }

        templates.insert(
            function_handle,
            InlineTemplate {
                argument_types: function.arguments.iter().map(|a| a.ty).collect(),
                return_expr,
                expressions: function.expressions.clone(),
            },
        );
    }

    templates
}

/// Count call-site references to each function across every function
/// body and entry point; used to gate the `max_call_sites` budget.
fn collect_call_counts(module: &naga::Module) -> HashMap<naga::Handle<naga::Function>, usize> {
    let mut counts = HashMap::new();

    for (_, function) in module.functions.iter() {
        collect_call_counts_in_block(&function.body, &mut counts);
    }
    for entry in module.entry_points.iter() {
        collect_call_counts_in_block(&entry.function.body, &mut counts);
    }

    counts
}

fn collect_call_counts_in_block(
    block: &naga::Block,
    counts: &mut HashMap<naga::Handle<naga::Function>, usize>,
) {
    for statement in block {
        if let naga::Statement::Call { function, .. } = statement {
            *counts.entry(*function).or_insert(0) += 1;
        }
        for nested in nested_blocks(statement) {
            collect_call_counts_in_block(nested, counts);
        }
    }
}

/// Return the expression handle delivered by a body of the form
/// `[Emit*, Return { value }]`, or `None` if the block contains
/// anything else (loops, stores, multi-return, etc.).  Single gate
/// for the pass's purity requirement.
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

/// Count expression nodes reachable from `handle` and ensure every
/// `FunctionArgument` index fits inside the advertised argument list.
/// Returns `None` when the DAG contains an inline-disallowed
/// expression (statement-attached results, local variables, and so
/// on) or an out-of-range argument index.
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

    // Sum the node counts of every child sub-expression.  Read-only walk
    // (no clone of the node just to traverse it); `ok` propagates a child's
    // `None` (disallowed / out-of-range argument) and stops descending
    // siblings, so the per-template `visited` marks match an early-return
    // walk exactly.
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

// MARK: Call-site rewriting

/// Walk `function`'s body, replacing every eligible call with the
/// template's cloned return expression.  Returns the number of call
/// sites rewritten.  The entry-point driver calls this with the same
/// `templates` map so every caller sees the same inlinable set.
fn inline_in_function(
    function: &mut naga::Function,
    templates: &HashMap<naga::Handle<naga::Function>, InlineTemplate>,
    types: &naga::UniqueArena<naga::Type>,
) -> usize {
    let (changed, _) = inline_in_block(
        &mut function.body,
        &mut function.expressions,
        templates,
        types,
        &HashMap::new(),
    );

    if changed > 0 {
        rebuild_function_expressions(function);
        function.named_expressions.clear();
    }

    changed
}

/// Inline eligible calls in `block`, returning the change count AND the
/// expression-replacement map this scope accumulated (inherited entries plus
/// every `CallResult -> inlined-root` mapping recorded here).  The returned
/// map lets a loop thread its `body`'s replacements into its `continuing`
/// block and `break_if`, which naga permits to reference body-defined
/// expressions; without it, inlining a body call leaves the continuing /
/// break_if references pointing at an orphaned `CallResult` (invalid IR that
/// forces a whole-module rollback).
fn inline_in_block(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    templates: &HashMap<naga::Handle<naga::Function>, InlineTemplate>,
    types: &naga::UniqueArena<naga::Type>,
    inherited_replacements: &HashMap<
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    >,
) -> (
    usize,
    HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let mut changed = 0usize;
    let mut replacements = inherited_replacements.clone();

    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        apply_replacements_to_statement(&mut statement, expressions, &replacements);

        match statement {
            naga::Statement::Call {
                function,
                arguments,
                result: Some(result_handle),
            } => {
                if let Some(template) = templates.get(&function)
                    && template.argument_types.len() == arguments.len()
                {
                    let old_len = expressions.len();
                    // Memo backed by `Vec<Option<Handle>>` indexed by
                    // template-arena `handle.index()` instead of a
                    // HashMap.  The template expression arena is the
                    // densest possible address space (one slot per
                    // expression) and `clone_inline_expression` may
                    // re-visit handles many times when the same
                    // sub-expression is shared - the memo lookup is
                    // the hot spot.  Direct indexing removes the
                    // SipHash cost from the inner traversal.
                    let mut memo: Vec<Option<naga::Handle<naga::Expression>>> =
                        vec![None; template.expressions.len()];
                    if let Some(root_handle) = clone_inline_expression(
                        template.return_expr,
                        template,
                        &arguments,
                        expressions,
                        types,
                        &mut memo,
                    ) {
                        push_emit_ranges_for_new_expressions(
                            &mut rebuilt,
                            expressions,
                            old_len,
                            span,
                        );

                        replacements.insert(result_handle, root_handle);
                        changed += 1;
                        continue;
                    }
                }

                rebuilt.push(
                    naga::Statement::Call {
                        function,
                        arguments,
                        result: Some(result_handle),
                    },
                    span,
                );
            }
            naga::Statement::Loop {
                mut body,
                mut continuing,
                mut break_if,
            } => {
                // naga lets the `continuing` block and `break_if` reference
                // expressions defined in `body`, so the continuing recursion
                // (and the break_if handle) must see the replacements `body`
                // produced - otherwise a body-inlined call leaves them
                // referencing an orphaned `CallResult`.
                let (cb, body_replacements) =
                    inline_in_block(&mut body, expressions, templates, types, &replacements);
                changed += cb;
                // `break_if` may reference a `CallResult` defined in the
                // `continuing` block itself (naga allows it), so resolve it
                // against the map the continuing recursion returned - which is
                // seeded from `body_replacements`, hence a superset.  Using
                // only `body_replacements` would leave a continuing-inlined
                // call's result orphaned in `break_if` -> invalid IR -> rollback.
                let (cc, continuing_replacements) = inline_in_block(
                    &mut continuing,
                    expressions,
                    templates,
                    types,
                    &body_replacements,
                );
                changed += cc;
                if let Some(handle) = break_if {
                    break_if = Some(resolve_replacement(handle, &continuing_replacements));
                }
                rebuilt.push(
                    naga::Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            // If / Switch / Block sub-blocks recurse against the CURRENT
            // scope map and their returned maps are dropped: unlike a
            // loop's continuing block, nothing after them may reference
            // their interior expressions.
            mut other => {
                for nested in nested_blocks_mut(&mut other) {
                    changed +=
                        inline_in_block(nested, expressions, templates, types, &replacements).0;
                }
                rebuilt.push(other, span);
            }
        }
    }

    *block = rebuilt;
    (changed, replacements)
}

/// Emit `Emit` statements covering every expression appended to
/// `expressions` after `old_len` that requires an emit range, split
/// around declarative expressions so the ranges remain contiguous.
fn push_emit_ranges_for_new_expressions(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    old_len: usize,
    span: naga::Span,
) {
    if expressions.len() <= old_len {
        return;
    }

    let mut start = None;
    let mut end = None;

    for handle in expressions.range_from(old_len) {
        if expression_needs_emit(&expressions[handle]) {
            if start.is_none() {
                start = Some(handle);
            }
            end = Some(handle);
        } else if let (Some(first), Some(last)) = (start.take(), end.take()) {
            block.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(first, last)),
                span,
            );
        }
    }

    if let (Some(first), Some(last)) = (start, end) {
        block.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(first, last)),
            span,
        );
    }
}

/// Recursively clone `handle` from `template.expressions` into
/// `caller_expressions`, replacing `FunctionArgument` references with
/// the caller-supplied `arguments`.  `memo` guarantees each template
/// handle produces exactly one caller handle so shared sub-DAGs stay
/// shared after cloning.
///
/// `memo` is pre-sized to `template.expressions.len()` by the caller
/// (see `inline_in_block`).  Every `handle` reached through the
/// recursion below comes from `template.expressions` (the recursion
/// only traverses children of the just-cloned expression, which is
/// itself a clone of an entry in the template's arena), so
/// `handle.index() < memo.len()` is an invariant; both the read and
/// write use direct indexing in lockstep, so an invariant violation
/// panics loudly at the offending site rather than silently
/// returning `None` on the read and then panicking on the write
/// (inconsistent diagnostics).
fn clone_inline_expression(
    handle: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller_expressions: &mut naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
    memo: &mut [Option<naga::Handle<naga::Expression>>],
) -> Option<naga::Handle<naga::Expression>> {
    if let Some(mapped) = memo[handle.index()] {
        return Some(mapped);
    }

    let expr = &template.expressions[handle];
    if is_disallowed_inline_expression(expr) {
        return None;
    }

    let mapped = match expr {
        naga::Expression::FunctionArgument(index) => arguments.get(*index as usize).copied()?,
        _ => {
            // Substitution can MANUFACTURE a statically out-of-bounds index
            // the template never had: the template's `v[i]` is a runtime
            // access, but a call site like `f(vec4(...), 6)` maps `i` to a
            // literal, and naga's validator rejects a known-OOB constant
            // index into a fixed-size composite - and a NEGATIVE constant
            // index regardless of base type - either of which used to roll
            // the whole pass back.  Checked BEFORE cloning children so the
            // decline strands nothing: the index resolves through the
            // argument list, the base length through the callee's declared
            // parameter type (covering every caller argument shape) or a
            // structural template composite.
            if let naga::Expression::Access { base, index } = expr
                && let Some(i) =
                    substituted_index_value(*index, template, arguments, caller_expressions)
                && (i < 0
                    || substituted_base_len(*base, template, types)
                        .is_some_and(|len| i as u64 >= len))
            {
                return None;
            }
            let mut cloned = expr.clone();
            try_map_expression_handles_in_place(&mut cloned, &mut |child| {
                clone_inline_expression(child, template, arguments, caller_expressions, types, memo)
            })?;
            // Backstop for composed shapes the pre-clone gate cannot size
            // (base or index produced by nested template expressions).
            // Declining here strands the already-cloned children -
            // unreferenced and individually valid; compact culls them.
            if let naga::Expression::Access { base, index } = cloned
                && let Some(i) = const_index_value(index, caller_expressions)
                && (i < 0
                    || static_composite_len(base, caller_expressions, types)
                        .is_some_and(|len| i as u64 >= len))
            {
                return None;
            }
            caller_expressions.append(cloned, Default::default())
        }
    };

    memo[handle.index()] = Some(mapped);
    Some(mapped)
}

/// Statically-known element count of `ty`: vector size, matrix column
/// count, or fixed array length.  `None` = not indexable with a static
/// bound (the OOB gates then stay out of the way).
fn type_element_count(
    ty: naga::Handle<naga::Type>,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &types[ty].inner {
        naga::TypeInner::Vector { size, .. } => Some(*size as u64),
        naga::TypeInner::Matrix { columns, .. } => Some(*columns as u64),
        naga::TypeInner::Array {
            size: naga::ArraySize::Constant(n),
            ..
        } => Some(n.get() as u64),
        _ => None,
    }
}

/// Statically-known element count of the composite VALUE produced by
/// `handle`, derived structurally (no typifier): `Compose`/`ZeroValue`
/// carry their type, `Splat`/`Swizzle` carry their size directly.
fn static_composite_len(
    handle: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &arena[handle] {
        naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => {
            type_element_count(*ty, types)
        }
        naga::Expression::Splat { size, .. } => Some(*size as u64),
        naga::Expression::Swizzle { size, .. } => Some(*size as u64),
        _ => None,
    }
}

/// Post-substitution constant value of a template `Access` index, resolved
/// WITHOUT cloning: a template-arena literal directly, or the caller
/// argument a `FunctionArgument` maps to.  (`Expression::Constant` indices
/// stay unresolvable here - no module access - and fall to the rollback.)
fn substituted_index_value(
    index: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    arguments: &[naga::Handle<naga::Expression>],
    caller_expressions: &naga::Arena<naga::Expression>,
) -> Option<i64> {
    match &template.expressions[index] {
        naga::Expression::FunctionArgument(k) => {
            const_index_value(*arguments.get(*k as usize)?, caller_expressions)
        }
        naga::Expression::Literal(_) => const_index_value(index, &template.expressions),
        _ => None,
    }
}

/// Static element count of a template `Access` base: the callee's declared
/// parameter type when the base is an argument (independent of the caller
/// argument's expression shape), else a structural template composite.
fn substituted_base_len(
    base: naga::Handle<naga::Expression>,
    template: &InlineTemplate,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<u64> {
    match &template.expressions[base] {
        naga::Expression::FunctionArgument(m) => {
            type_element_count(*template.argument_types.get(*m as usize)?, types)
        }
        _ => static_composite_len(base, &template.expressions, types),
    }
}

/// The compile-time value of an integer-literal index expression, or
/// `None` when the index is not a literal (a genuine runtime access).
fn const_index_value(
    handle: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<i64> {
    match arena[handle] {
        naga::Expression::Literal(naga::Literal::I32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::U32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::I64(v)) => Some(v),
        // Saturate: a u64 index beyond i64::MAX is OOB for any composite.
        naga::Expression::Literal(naga::Literal::U64(v)) => {
            Some(i64::try_from(v).unwrap_or(i64::MAX))
        }
        naga::Expression::Literal(naga::Literal::AbstractInt(v)) => Some(v),
        _ => None,
    }
}

/// Remap every expression handle referenced by `statement` through
/// the inlining `replacements` map.  Covers every naga statement
/// variant exhaustively so inlined call results are consistently
/// substituted across control flow.
fn apply_replacements_to_statement(
    statement: &mut naga::Statement,
    expressions: &mut naga::Arena<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let mut remap =
        |handle: naga::Handle<naga::Expression>| resolve_replacement(handle, replacements);

    match statement {
        naga::Statement::Emit(range) => {
            for handle in range.clone() {
                let expression = expressions.get_mut(handle);
                let _ = try_map_expression_handles_in_place(expression, &mut |h| Some(remap(h)));
            }
        }
        naga::Statement::Block(_) => {}
        naga::Statement::If { condition, .. } => {
            *condition = remap(*condition);
        }
        naga::Statement::Switch { selector, .. } => {
            *selector = remap(*selector);
        }
        naga::Statement::Loop { break_if, .. } => {
            if let Some(handle) = break_if {
                *handle = remap(*handle);
            }
        }
        naga::Statement::Break | naga::Statement::Continue | naga::Statement::Kill => {}
        naga::Statement::Return { value } => {
            if let Some(handle) = value {
                *handle = remap(*handle);
            }
        }
        naga::Statement::ControlBarrier(_) | naga::Statement::MemoryBarrier(_) => {}
        naga::Statement::Store { pointer, value } => {
            *pointer = remap(*pointer);
            *value = remap(*value);
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            *value = remap(*value);
        }
        naga::Statement::Atomic {
            pointer,
            fun,
            value,
            ..
        } => {
            *pointer = remap(*pointer);
            map_atomic_function_handles(fun, &mut remap);
            *value = remap(*value);
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            fun,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            map_atomic_function_handles(fun, &mut remap);
            *value = remap(*value);
        }
        naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            *pointer = remap(*pointer);
        }
        naga::Statement::Call { arguments, .. } => {
            for argument in arguments {
                *argument = remap(*argument);
            }
        }
        naga::Statement::RayQuery { query, fun } => {
            *query = remap(*query);
            map_ray_query_function_handles(fun, &mut remap);
        }
        naga::Statement::RayPipelineFunction(fun) => {
            map_ray_pipeline_function_handles(fun, &mut remap);
        }
        naga::Statement::SubgroupBallot { predicate, .. } => {
            if let Some(handle) = predicate {
                *handle = remap(*handle);
            }
        }
        naga::Statement::SubgroupGather { mode, argument, .. } => {
            map_gather_mode_handles(mode, &mut remap);
            *argument = remap(*argument);
        }
        naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
            *argument = remap(*argument);
        }
        naga::Statement::CooperativeStore { target, data } => {
            *target = remap(*target);
            map_cooperative_data_handles(data, &mut remap);
        }
    }
}

/// Follow `replacements` transitively and return the terminal target.
/// Self-loops terminate early; the callers guarantee the map is
/// acyclic by construction, so a well-formed map always halts.
fn resolve_replacement(
    mut handle: naga::Handle<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    while let Some(next) = replacements.get(&handle).copied() {
        if next == handle {
            break;
        }
        handle = next;
    }
    handle
}

// MARK: Arena compaction

/// Rebuild `function.expressions` after inlining so only reachable
/// expressions survive, remapping every handle referenced by the body,
/// locals, named expressions, and result.  The walk mirrors naga's
/// `compact` pass shape but is scoped to a single function.
fn rebuild_function_expressions(function: &mut naga::Function) {
    let old_expressions = std::mem::take(&mut function.expressions);
    let mut new_expressions = naga::Arena::new();
    let mut handle_map = HashMap::new();

    rebuild_block_expressions(
        &mut function.body,
        &old_expressions,
        &mut new_expressions,
        &mut handle_map,
    );

    // Remap local variable init handles into the new expression arena.
    for (_, local) in function.local_variables.iter_mut() {
        if let Some(init) = &mut local.init {
            *init = clone_expression_handle(
                *init,
                &old_expressions,
                &mut new_expressions,
                &mut handle_map,
            );
        }
    }

    function.expressions = new_expressions;
}

fn rebuild_block_expressions(
    block: &mut naga::Block,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        // Handle Emit specially: clone expressions and split into contiguous
        // sub-ranges to avoid including interleaved non-emittable dependencies.
        if let naga::Statement::Emit(ref range) = statement {
            let mut mapped_handles = Vec::new();
            for handle in range.clone() {
                let mut expression = old_expressions[handle].clone();
                let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
                    Some(clone_expression_handle(
                        child,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    ))
                });
                let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
                handle_map.insert(handle, mapped);
                mapped_handles.push(mapped);
            }

            if !mapped_handles.is_empty() {
                let mut start = mapped_handles[0];
                let mut end = mapped_handles[0];
                for &h in &mapped_handles[1..] {
                    if h.index() == end.index() + 1 {
                        end = h;
                    } else {
                        rebuilt.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                            span,
                        );
                        start = h;
                        end = h;
                    }
                }
                rebuilt.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                    span,
                );
            }
            continue;
        }

        // Handle `Loop` BEFORE the generic remap.  A loop's `break_if`
        // expression is emitted inside `body`/`continuing`, so those blocks
        // must be rebuilt first; then `clone_expression_handle(break_if)`
        // memo-hits the copy its owning Emit just produced.  If we instead let
        // the generic remap below clone break_if first, it appends an
        // un-emitted duplicate and leaves break_if pointing at an expression
        // that is in no Emit range - invalid IR.
        if matches!(statement, naga::Statement::Loop { .. }) {
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = &mut statement
            {
                rebuild_block_expressions(body, old_expressions, new_expressions, handle_map);
                rebuild_block_expressions(continuing, old_expressions, new_expressions, handle_map);
                if let Some(handle) = break_if {
                    *handle = clone_expression_handle(
                        *handle,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    );
                }
            }
            rebuilt.push(statement, span);
            continue;
        }

        remap_statement_handles(&mut statement, &mut |h| {
            clone_expression_handle(h, old_expressions, new_expressions, handle_map)
        });

        // The early Loop path above must have consumed every Loop: reaching
        // the generic remap with one would clone `break_if` BEFORE its owning
        // block is rebuilt, appending an un-emitted duplicate (invalid IR).
        debug_assert!(!matches!(statement, naga::Statement::Loop { .. }));
        for nested in nested_blocks_mut(&mut statement) {
            rebuild_block_expressions(nested, old_expressions, new_expressions, handle_map);
        }

        rebuilt.push(statement, span);
    }

    *block = rebuilt;
}

fn clone_expression_handle(
    handle: naga::Handle<naga::Expression>,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    if let Some(mapped) = handle_map.get(&handle).copied() {
        return mapped;
    }

    let mut expression = old_expressions[handle].clone();
    let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
        Some(clone_expression_handle(
            child,
            old_expressions,
            new_expressions,
            handle_map,
        ))
    });

    let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
    handle_map.insert(handle, mapped);
    mapped
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
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
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

    fn count_calls_to_function(block: &naga::Block, target: naga::Handle<naga::Function>) -> usize {
        let mut count = 0usize;
        for statement in block {
            if let naga::Statement::Call { function, .. } = statement
                && *function == target
            {
                count += 1;
            }
            for nested in nested_blocks(statement) {
                count += count_calls_to_function(nested, target);
            }
        }
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

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 0,
            "helper call should be removed from entry point"
        );
    }

    /// Substituting a call-site literal into a template's runtime `v[i]`
    /// manufactures a statically out-of-bounds constant index that naga's
    /// validator rejects - the clone must decline (call kept) instead of
    /// poisoning the pass into a module-wide rollback.  `run_pass`
    /// validates the post-pass module, so this test fails if the gate is
    /// removed.  The in-bounds call proves the gate is value-sensitive,
    /// not a blanket refusal.
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
        assert!(changed, "the in-bounds call site must still inline");
        let pick = find_function_handle_by_name(&after, "pick");
        assert_eq!(
            count_calls_to_function(&after.entry_points[0].function.body, pick),
            1,
            "exactly the OOB call site must survive as a call"
        );
    }

    /// The pre-clone gate must also size the base from the callee's
    /// DECLARED parameter type (the caller forwards its own argument, so no
    /// structural composite exists in the caller arena) and must reject a
    /// negative index outright - naga flags negative constant indices
    /// regardless of base-length knowledge.  `run_pass` validates, so this
    /// test fails with the gate removed.
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
        let mut surviving_calls = 0;
        for (_, func) in after.functions.iter() {
            surviving_calls += count_calls_to_function(&func.body, pick);
        }
        surviving_calls += count_calls_to_function(&after.entry_points[0].function.body, pick);
        assert_eq!(
            surviving_calls, 2,
            "both statically-invalid call sites must survive as calls"
        );
    }

    #[test]
    fn skips_function_with_local_variables() {
        let source = r#"
fn helper(x: f32) -> f32 {
    var t: f32;
    t = x + 1.0;
    return t;
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
        assert!(!changed, "helper with locals should not be inlined");

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 1,
            "helper call should remain when inlining is skipped"
        );
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
        // Construct a module where the caller has a local with init: Some(...)
        // and inline a simple helper into it.  After inlining, the local's init
        // handle must still be valid in the rebuilt expression arena.

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

        // Manually set init on the local variable to exercise the code path.
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

        // Run inlining pass
        let mut pass = InliningPass::default();
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("inlining should run");
        assert!(changed, "helper should be inlined");

        // After inlining, the local's init handle must be valid.
        let ep_fn = &module.entry_points[0].function;
        let local = &ep_fn.local_variables[local_handle];
        assert!(local.init.is_some(), "local init should still be present");
        let init_handle = local.init.unwrap();
        // The init handle must be within range of the (rebuilt) expression arena.
        assert!(
            init_handle.index() < ep_fn.expressions.len(),
            "local init handle ({}) should be within rebuilt expression arena (len={})",
            init_handle.index(),
            ep_fn.expressions.len(),
        );
        // Verify it's still a literal 0.0
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

    #[test]
    fn expansion_budget_rejects_large_multi_site_function() {
        // A helper with many expression nodes called from 2 sites should be
        // rejected: node_count=9, call_sites=2, expansion=9*(2-1)=9 > 6.
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

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 2, "expected two helper calls before inlining");

        let (changed, after) = run_pass(source);
        assert!(
            !changed,
            "helper should NOT be inlined: expansion budget exceeded"
        );

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(after_calls, 2, "both helper calls should remain");
    }

    #[test]
    fn expansion_budget_allows_small_multi_site_function() {
        // A tiny helper (node_count=3) called from 2 sites should be allowed:
        // expansion=3*(2-1)=3 <= 6.
        let source = r#"
fn helper(x: f32) -> f32 {
    return x + 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let a = helper(1.0);
    let b = helper(2.0);
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;

        let before = naga::front::wgsl::parse_str(source).expect("source should parse");
        let helper_before = find_function_handle_by_name(&before, "helper");
        let before_calls =
            count_calls_to_function(&before.entry_points[0].function.body, helper_before);
        assert_eq!(before_calls, 2, "expected two helper calls before inlining");

        let (changed, after) = run_pass(source);
        assert!(changed, "helper should be inlined: expansion within budget");

        let helper_after = find_function_handle_by_name(&after, "helper");
        let after_calls =
            count_calls_to_function(&after.entry_points[0].function.body, helper_after);
        assert_eq!(
            after_calls, 0,
            "both helper calls should be replaced by inlined expressions"
        );
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_continuing() {
        // A call in the loop BODY whose result is consumed in the CONTINUING
        // block.  naga lets `continuing` reference body-defined expressions,
        // so the continuing recursion must receive the body's inlining
        // replacements; otherwise it references an orphaned `CallResult`
        // (invalid IR), which `run_pass`'s post-pass validation would reject.
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
        let helper = find_function_handle_by_name(&module, "helper");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_calls_to_function(body, helper),
            0,
            "the body call to helper must be inlined (result threads into continuing)"
        );
    }

    #[test]
    fn inlines_call_in_loop_body_used_in_break_if() {
        // Same hazard via `break_if`, which can also reference body-defined
        // expressions.
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
        let limit = find_function_handle_by_name(&module, "limit");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_calls_to_function(body, limit),
            0,
            "the body call to limit must be inlined (result threads into break_if)"
        );
    }
}
