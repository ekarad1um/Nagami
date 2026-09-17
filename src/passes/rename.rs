//! Identifier rename: every user-chosen identifier the pipeline may touch
//! gets a short generated name, deterministic and collision-free against
//! WGSL reserved words and preserved symbols.  Struct type and member names
//! are renamed by the generator (naga's `UniqueArena<Type>` is immutable
//! mid-pipeline); this pass covers the mutable arenas: globals, constants,
//! overrides, functions, arguments, and locals.
//!
//! Name length grows with the number of identifiers (52 single-character
//! names, then two-character, ...), so identifiers are ranked by occurrence
//! weight (declaration plus in-body references) and named heaviest-first,
//! which minimises total identifier bytes by the rearrangement inequality.
//! Every identifier still gets a globally distinct name, so the SET of names
//! drawn is independent of the order and no downstream generator decision
//! (struct types, aliases, and extracted literals avoiding the pool) shifts.

use std::collections::HashSet;

use crate::error::Error;
use crate::handle_set::HandleMap;
use crate::ir::visit::{all_functions, for_each_statement};
use crate::name_gen;
use crate::passes::expr_util::live_expression_ref_counts;
use crate::pipeline::{Pass, PassContext};

/// `preserve` lists names kept verbatim; `mangle` extends renaming to
/// constants and overrides.  `by_render` weighs the names by the text:
/// the pass renders the module under a provisional plan and ranks by the
/// occurrences the emitter produced, where the IR census counts a value
/// the emitter inlines at several uses once.  The tail's instance, whose
/// names ship, weighs so; a re-parse of its text references each inlined
/// copy separately and so counts what it rendered, and names it the same.
#[derive(Debug)]
pub struct RenamePass {
    preserve: HashSet<String>,
    mangle: bool,
    by_render: bool,
}

impl RenamePass {
    /// A pass from the user-facing `preserve_symbols` and the resolved
    /// `mangle` flag, weighing by the IR census.
    pub fn new(preserve_symbols: Vec<String>, mangle: bool) -> Self {
        Self {
            preserve: preserve_symbols.into_iter().collect(),
            mangle,
            by_render: false,
        }
    }

    /// [`Self::new`], weighing by the text (`by_render`).
    pub fn by_render(preserve_symbols: Vec<String>, mangle: bool) -> Self {
        Self {
            by_render: true,
            ..Self::new(preserve_symbols, mangle)
        }
    }
}

impl Pass for RenamePass {
    fn name(&self) -> &'static str {
        "rename_identifiers"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut plan = plan_names(module, &self.preserve, self.mangle);
        if self.by_render {
            // The provisional names only move a decision at a name-length
            // boundary; a module the emitter declines keeps them.  The
            // preamble's declarations are left out as the shipped render
            // leaves them out, so the type spellings counted are its.
            let options = crate::generator::GenerateOptions {
                preamble_names: ctx
                    .tail
                    .map(|tail| tail.preamble_names.clone())
                    .unwrap_or_default(),
                ..crate::generator::GenerateOptions::from_config(ctx.config)
            };
            if let Ok(emission) =
                crate::generator::generate(&plan.applied(module), ctx.info, options)
            {
                plan =
                    plan_names_weighed(module, &self.preserve, self.mangle, emission.name_weights);
                if let Some(tail) = ctx.tail {
                    *tail.type_uses.borrow_mut() = Some(emission.type_uses);
                }
            }
        }
        Ok(plan.apply(module, ctx.name_log))
    }
}

/// The names one sweep assigns, in rank order, before any is applied: a
/// pass that prices a rewrite by the text rename will leave reads them
/// here, and [`RenamePass::run`] applies them.
#[derive(Clone)]
pub(crate) struct NamePlan {
    /// Heaviest first: `(target, weight, declaration sequence)`.
    targets: Vec<(Target, usize, usize)>,
    /// Parallel to `targets`.
    names: Vec<String>,
    /// Every name the draw may not mint, the drawn ones included, and the
    /// counter after the last draw: together they continue the sequence.
    used_names: HashSet<String>,
    counter: usize,
}

/// Rank every renameable identifier of `module` by the IR census and
/// draw its name.
pub(crate) fn plan_names(
    module: &naga::Module,
    preserve: &HashSet<String>,
    mangle: bool,
) -> NamePlan {
    // Weights are structural (handle-based, name-independent), so the
    // assignment is deterministic and idempotent at the convergence
    // fixed point.
    plan_names_weighed(module, preserve, mangle, compute_weights(module))
}

/// [`plan_names`] under the given occurrence weights.
pub(crate) fn plan_names_weighed(
    module: &naga::Module,
    preserve: &HashSet<String>,
    mangle: bool,
    weights: Weights,
) -> NamePlan {
    let mut used_names = collect_reserved_names(module, preserve, mangle);

    // `seq` is the declaration-order tie-break, so equal weights get a
    // stable assignment.
    let mut targets: Vec<(Target, usize, usize)> = Vec::new();
    let mut seq = 0usize;

    if mangle {
        for (h, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref()
                && !preserve.contains(name)
            {
                push_target(&mut targets, &mut seq, Target::Constant(h), &weights);
            }
        }
        for (h, ov) in module.overrides.iter() {
            // An `@id`-less override is identified to the host ONLY by
            // its declaration name (the pipeline `constants` key), so
            // renaming it breaks host specialization; it is reserved
            // instead.
            if let Some(name) = ov.name.as_deref()
                && ov.id.is_some()
                && !preserve.contains(name)
            {
                push_target(&mut targets, &mut seq, Target::Override(h), &weights);
            }
        }
    }

    for (h, global) in module.global_variables.iter() {
        if let Some(name) = global.name.as_deref()
            && preserve.contains(name)
        {
            continue;
        }
        push_target(&mut targets, &mut seq, Target::Global(h), &weights);
    }

    let module_scope: HashSet<&str> = name_gen::module_scope_names(module)
        .chain(name_gen::type_names(module))
        .collect();
    for (fh, function) in module.functions.iter() {
        if !matches!(function.name.as_deref(), Some(n) if preserve.contains(n)) {
            push_target(&mut targets, &mut seq, Target::Function(fh), &weights);
        }
        enumerate_locals(
            function,
            fh.index(),
            preserve,
            &module_scope,
            &weights,
            &mut targets,
            &mut seq,
        );
    }

    for (ei, entry) in module.entry_points.iter().enumerate() {
        // Entry-point names are pipeline-bound and never renamed.
        enumerate_locals(
            &entry.function,
            module.functions.len() + ei,
            preserve,
            &module_scope,
            &weights,
            &mut targets,
            &mut seq,
        );
    }

    // Heaviest first; `seq` is unique per target, so the order is total
    // and independent of sort stability.
    targets.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

    let mut counter = 0usize;
    let names = targets
        .iter()
        .map(|_| next_available_name(&mut counter, &mut used_names))
        .collect();
    NamePlan {
        targets,
        names,
        used_names,
        counter,
    }
}

impl NamePlan {
    /// Where a further target of `weight` ranks: after every target of at
    /// least that weight, its sequence number being the largest.
    fn rank_of_weight(&self, weight: usize) -> usize {
        self.targets.iter().filter(|t| t.1 >= weight).count()
    }

    /// Length of the draw at `rank`; the draw is fixed by the reserved set
    /// alone, whoever takes it.
    fn draw_len(&self, rank: usize) -> usize {
        match self.names.get(rank) {
            Some(name) => name.len(),
            None => name_gen::next_name_unique(&mut self.counter.clone(), &self.used_names).len(),
        }
    }

    /// Length of the name a further target of `weight` would draw.
    pub(crate) fn name_len_at_weight(&self, weight: usize) -> usize {
        self.draw_len(self.rank_of_weight(weight))
    }

    /// Bytes the existing targets pay when a further target of `weight`
    /// joins: every target ranked below it moves one draw down the
    /// sequence, which costs an occurrence per byte the draws lengthen
    /// (the single letters run out at the 52nd).
    pub(crate) fn insertion_cost(&self, weight: usize) -> usize {
        let rank = self.rank_of_weight(weight);
        self.targets[rank..]
            .iter()
            .enumerate()
            .map(|(i, t)| t.1 * (self.draw_len(rank + i + 1) - self.draw_len(rank + i)))
            .sum()
    }

    /// A copy of `module` carrying the planned names.
    pub(crate) fn applied(&self, module: &naga::Module) -> naga::Module {
        let mut renamed = module.clone();
        self.clone().apply(&mut renamed, None);
        renamed
    }

    /// Write the planned names into `module`; `true` when any slot or
    /// `named_expressions` changed.  Module-scope renames are logged
    /// (locals are neither unique nor host-visible), one batch per sweep
    /// so `record_batch` can resolve swaps.
    pub(crate) fn apply(
        self,
        module: &mut naga::Module,
        log: Option<&std::cell::RefCell<crate::name_map::NameLog>>,
    ) -> bool {
        let mut assigned = AssignedNames::new(module);
        for ((target, _, _), name) in self.targets.into_iter().zip(self.names) {
            assigned.insert(target, name);
        }

        let mut module_renames: Vec<(String, String)> = Vec::new();
        let mut changed = false;
        for (h, c) in module.constants.iter_mut() {
            apply_module_name(
                &mut c.name,
                assigned.constant.remove(h),
                &mut changed,
                &mut module_renames,
            );
        }
        for (h, ov) in module.overrides.iter_mut() {
            apply_module_name(
                &mut ov.name,
                assigned.over.remove(h),
                &mut changed,
                &mut module_renames,
            );
        }
        for (h, global) in module.global_variables.iter_mut() {
            apply_module_name(
                &mut global.name,
                assigned.global.remove(h),
                &mut changed,
                &mut module_renames,
            );
        }
        for (fh, function) in module.functions.iter_mut() {
            apply_module_name(
                &mut function.name,
                assigned.function.remove(fh),
                &mut changed,
                &mut module_renames,
            );
            apply_locals(function, fh.index(), &mut assigned, &mut changed);
            changed |= clear_named_expressions(function);
        }
        for (ei, entry) in module.entry_points.iter_mut().enumerate() {
            apply_locals(
                &mut entry.function,
                module.functions.len() + ei,
                &mut assigned,
                &mut changed,
            );
            changed |= clear_named_expressions(&mut entry.function);
        }
        if let Some(log) = log
            && !module_renames.is_empty()
        {
            log.borrow_mut().record_batch(&module_renames);
        }
        changed
    }
}

/// A renameable identifier by arena slot; args and locals carry their
/// [`Body`] so the same handle in two functions never collides.
#[derive(Clone, Copy)]
enum Target {
    Constant(naga::Handle<naga::Constant>),
    Override(naga::Handle<naga::Override>),
    Global(naga::Handle<naga::GlobalVariable>),
    Function(naga::Handle<naga::Function>),
    Arg(Body, usize),
    Local(Body, naga::Handle<naga::LocalVariable>),
}

/// A function body by position: free functions first, then entry points,
/// the order of [`all_functions`] and of the generator's per-function
/// caches; the per-body tables here are indexed by it.
pub(crate) type Body = usize;

/// Names assigned this sweep, bucketed by arena so application is one
/// `iter_mut` per arena; per body, by argument position and by local.
struct AssignedNames {
    constant: HandleMap<naga::Constant, String>,
    over: HandleMap<naga::Override, String>,
    global: HandleMap<naga::GlobalVariable, String>,
    function: HandleMap<naga::Function, String>,
    arg: Vec<Vec<Option<String>>>,
    local: Vec<HandleMap<naga::LocalVariable, String>>,
}

impl AssignedNames {
    fn new(module: &naga::Module) -> Self {
        Self {
            constant: Default::default(),
            over: Default::default(),
            global: Default::default(),
            function: Default::default(),
            arg: all_functions(module)
                .map(|f| vec![None; f.arguments.len()])
                .collect(),
            local: all_functions(module).map(|_| Default::default()).collect(),
        }
    }

    fn insert(&mut self, target: Target, name: String) {
        match target {
            Target::Constant(h) => {
                self.constant.insert(h, name);
            }
            Target::Override(h) => {
                self.over.insert(h, name);
            }
            Target::Global(h) => {
                self.global.insert(h, name);
            }
            Target::Function(h) => {
                self.function.insert(h, name);
            }
            Target::Arg(b, i) => self.arg[b][i] = Some(name),
            Target::Local(b, h) => {
                self.local[b].insert(h, name);
            }
        }
    }
}

/// Occurrence weights; a missing key is a non-renameable handle and counts
/// as 1.  Per body, by argument position and by local.
#[derive(Default, Debug)]
pub(crate) struct Weights {
    global: HandleMap<naga::GlobalVariable, usize>,
    constant: HandleMap<naga::Constant, usize>,
    over: HandleMap<naga::Override, usize>,
    function: HandleMap<naga::Function, usize>,
    arg: Vec<Vec<usize>>,
    local: Vec<HandleMap<naga::LocalVariable, usize>>,
}

impl Weights {
    /// The declaration of every target and every call of a function: what
    /// the text spells once per statement, whichever way the values in
    /// the bodies render.
    pub(crate) fn declared(module: &naga::Module) -> Self {
        let mut w = Weights::default();
        for (h, _) in module.global_variables.iter() {
            *w.global.entry(h).or_insert(0) += 1;
        }
        for (h, _) in module.constants.iter() {
            *w.constant.entry(h).or_insert(0) += 1;
        }
        for (h, _) in module.overrides.iter() {
            *w.over.entry(h).or_insert(0) += 1;
        }
        for (h, _) in module.functions.iter() {
            *w.function.entry(h).or_insert(0) += 1;
        }
        for function in all_functions(module) {
            w.arg.push(vec![1; function.arguments.len()]);
            w.local.push(
                function
                    .local_variables
                    .iter()
                    .map(|(lh, _)| (lh, 1))
                    .collect(),
            );
            count_calls(&function.body, &mut w.function);
        }
        w
    }

    /// `count` renderings of `expr` in `body`, for the target it names, if
    /// any.
    pub(crate) fn reference(&mut self, body: Body, expr: &naga::Expression, count: usize) {
        if count == 0 {
            return;
        }
        match expr {
            naga::Expression::GlobalVariable(g) => *self.global.entry(*g).or_insert(0) += count,
            naga::Expression::LocalVariable(l) => *self.local[body].entry(*l).or_insert(0) += count,
            naga::Expression::FunctionArgument(i) => self.arg[body][*i as usize] += count,
            naga::Expression::Constant(cst) => *self.constant.entry(*cst).or_insert(0) += count,
            naga::Expression::Override(o) => *self.over.entry(*o).or_insert(0) += count,
            _ => {}
        }
    }

    fn of(&self, target: Target) -> usize {
        match target {
            Target::Constant(h) => self.constant.get(h).copied(),
            Target::Override(h) => self.over.get(h).copied(),
            Target::Global(h) => self.global.get(h).copied(),
            Target::Function(h) => self.function.get(h).copied(),
            Target::Arg(b, i) => self.arg.get(b).and_then(|arg| arg.get(i)).copied(),
            Target::Local(b, h) => self.local.get(b).and_then(|local| local.get(h)).copied(),
        }
        .unwrap_or(1)
    }
}

fn push_target(
    targets: &mut Vec<(Target, usize, usize)>,
    seq: &mut usize,
    target: Target,
    weights: &Weights,
) {
    targets.push((target, weights.of(target), *seq));
    *seq += 1;
}

fn enumerate_locals(
    function: &naga::Function,
    body: Body,
    preserve: &HashSet<String>,
    module_scope: &HashSet<&str>,
    weights: &Weights,
    targets: &mut Vec<(Target, usize, usize)>,
    seq: &mut usize,
) {
    // naga gives each block-scoped shadowing `var` its own handle under the
    // shared source name, so a preserved name is kept by its FIRST
    // declaration only: two `var t` in one scope are invalid, and once
    // `defer_vars` sinks one, reads silently re-bind.  A name a module-scope
    // declaration or a struct type carries is never kept on a local: the IR
    // has no declaration position, so a global read the source made before
    // `var t` shadowed it (`let a = t; var t = 0u;`) re-binds to the kept
    // local, and a constructor `S(..)` inlined into a kept `var S`'s scope
    // calls the local.  Locals are not host-visible, so renaming one is
    // always safe.
    fn keeps_name<'n>(
        kept: &mut HashSet<&'n str>,
        preserve: &HashSet<String>,
        module_scope: &HashSet<&str>,
        name: Option<&'n str>,
    ) -> bool {
        match name {
            Some(n) if preserve.contains(n) && !module_scope.contains(n) => kept.insert(n),
            _ => false,
        }
    }
    let mut kept = HashSet::new();
    for (i, argument) in function.arguments.iter().enumerate() {
        if keeps_name(&mut kept, preserve, module_scope, argument.name.as_deref()) {
            continue;
        }
        push_target(targets, seq, Target::Arg(body, i), weights);
    }
    for (lh, local) in function.local_variables.iter() {
        if keeps_name(&mut kept, preserve, module_scope, local.name.as_deref()) {
            continue;
        }
        push_target(targets, seq, Target::Local(body, lh), weights);
    }
}

fn apply_name(slot: &mut Option<String>, name: Option<String>, changed: &mut bool) {
    if let Some(name) = name {
        *changed |= slot.as_deref() != Some(name.as_str());
        *slot = Some(name);
    }
}

/// [`apply_name`] that logs (old, new) for a NAMED slot; an unnamed slot
/// gaining a name is synthetic and unlogged.
fn apply_module_name(
    slot: &mut Option<String>,
    name: Option<String>,
    changed: &mut bool,
    renames: &mut Vec<(String, String)>,
) {
    if let Some(name) = name {
        if let Some(old) = slot.as_deref()
            && old != name
        {
            renames.push((old.to_string(), name.clone()));
        }
        apply_name(slot, Some(name), changed);
    }
}

fn apply_locals(
    function: &mut naga::Function,
    body: Body,
    assigned: &mut AssignedNames,
    changed: &mut bool,
) {
    for (i, argument) in function.arguments.iter_mut().enumerate() {
        apply_name(&mut argument.name, assigned.arg[body][i].take(), changed);
    }
    for (lh, local) in function.local_variables.iter_mut() {
        apply_name(&mut local.name, assigned.local[body].remove(lh), changed);
    }
}

/// Clear `named_expressions` whenever the function has any and report it as
/// a change even when nothing was renamed.  The extra convergence sweep
/// looks like a perf wart but is load-bearing: it lets downstream passes
/// observe IR that settled only in this sweep's earlier passes (DCE catching
/// a global orphaned by a phony-assignment load elimination).
fn clear_named_expressions(function: &mut naga::Function) -> bool {
    if function.named_expressions.is_empty() {
        return false;
    }
    function.named_expressions.clear();
    true
}

// MARK: Occurrence weights

/// Occurrence weights: the declaration plus every live reference in a
/// function or entry body, mirroring the generator's per-function expression
/// ref counts (for an inlined `GlobalVariable(g)` that count is the textual
/// occurrences of `g`).  A size heuristic, not an exact count (module-scope
/// initializers are not counted, and a value the emitter inlines at several
/// uses spells its operands once per use): an imperfect weight only yields
/// a longer name, never a collision, which the all-distinct draw alone
/// guarantees.
fn compute_weights(module: &naga::Module) -> Weights {
    let mut w = Weights::declared(module);
    for (body, function) in all_functions(module).enumerate() {
        let (counts, _live) = live_expression_ref_counts(function);
        for (h, expr) in function.expressions.iter() {
            w.reference(body, expr, counts[h.index()]);
        }
    }
    w
}

/// Call counts, so a frequently-called function earns a shorter name.
fn count_calls(block: &naga::Block, calls: &mut HandleMap<naga::Function, usize>) {
    for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Call { function, .. } = stmt {
            *calls.entry(*function).or_insert(0) += 1;
        }
    });
}

/// The names one rename sweep may not mint.  With `mangle = false`,
/// constant, override, struct type and member names stay verbatim and are
/// reserved.  With `mangle = true` the sweep rewrites constants and
/// overrides, and reserving the previous sweep's assignments would shift
/// every later assignment one slot into a two-sweep oscillation that never
/// converges; the generator mints struct type and member names clear of
/// this pass's assignments, and reserving the source's would make a
/// re-minify skip the letters the previous run minted for them, shifting
/// every later assignment by one.  Every preserve-listed name is reserved
/// unconditionally, including names absent from every arena (a preamble
/// binding a prior pass pruned): the scans see only surviving names, and a
/// re-minted preamble name would make the generator suppress that body
/// declaration as preamble-owned and rebind every reference to the host's
/// binding.
fn collect_reserved_names(
    module: &naga::Module,
    preserve: &HashSet<String>,
    mangle: bool,
) -> HashSet<String> {
    let mut reserved = HashSet::new();

    reserved.extend(preserve.iter().cloned());

    if !mangle {
        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref() {
                reserved.insert(name.to_string());
            }
        }

        for (_, ov) in module.overrides.iter() {
            if let Some(name) = ov.name.as_deref() {
                reserved.insert(name.to_string());
            }
        }
    } else {
        // `@id`-less overrides keep their name (the host's pipeline-constant
        // key).
        for (_, ov) in module.overrides.iter() {
            if let Some(name) = ov.name.as_deref()
                && ov.id.is_none()
            {
                reserved.insert(name.to_string());
            }
        }
    }

    // Entry-point names are host-referenced.
    for entry in module.entry_points.iter() {
        reserved.insert(entry.name.clone());
    }

    // Source struct type and member names, without mangling only: the
    // generator then emits them verbatim, so minting one for a global /
    // function / local would put two same-named symbols in the output
    // (round-trip validation catches it, but compaction silently halves).
    if !mangle {
        for (_, ty) in module.types.iter() {
            if let Some(name) = ty.name.as_deref() {
                reserved.insert(name.to_string());
            }
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                for m in members {
                    if let Some(name) = m.name.as_deref() {
                        reserved.insert(name.to_string());
                    }
                }
            }
        }
    }

    // `A` / `B` / `C` are predeclared cooperative-matrix role names in the
    // `coop_mat<T, role>` type argument.  The generator cannot emit those
    // types, so such modules fall back to naga's wgsl-out, which spells the
    // role literally, and a declaration minted onto one makes naga re-read
    // the role position as that declaration and reject the fallback.  Gated
    // on actual usage since these are the cheapest names.
    if module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::CooperativeMatrix { .. }))
    {
        reserved.insert("A".to_string());
        reserved.insert("B".to_string());
        reserved.insert("C".to_string());
    }

    reserved
}

fn next_available_name(counter: &mut usize, used_names: &mut HashSet<String>) -> String {
    name_gen::next_name_insert(counter, used_names)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashSet;

    fn follows_expected_name_pattern(name: &str) -> bool {
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return false;
        };

        if !first.is_ascii_alphabetic() {
            return false;
        }

        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    fn run_pass(source: &str, preserve: &[&str]) -> (bool, naga::Module) {
        run_pass_with_mangle(source, preserve, false)
    }

    fn run_pass_with_mangle(source: &str, preserve: &[&str], mangle: bool) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = RenamePass::new(preserve.iter().map(|s| s.to_string()).collect(), mangle);
        let config = Config::default();

        let changed =
            PassContext::run_pass(&mut pass, &mut module, &config).expect("rename pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn collect_declaration_names(module: &naga::Module) -> Vec<String> {
        let mut out = Vec::new();

        for (_, global) in module.global_variables.iter() {
            if let Some(name) = global.name.as_deref() {
                out.push(name.to_string());
            }
        }

        for (_, function) in module.functions.iter() {
            if let Some(name) = function.name.as_deref() {
                out.push(name.to_string());
            }
            for argument in function.arguments.iter() {
                if let Some(name) = argument.name.as_deref() {
                    out.push(name.to_string());
                }
            }
            for (_, local) in function.local_variables.iter() {
                if let Some(name) = local.name.as_deref() {
                    out.push(name.to_string());
                }
            }
        }

        for entry in module.entry_points.iter() {
            for argument in entry.function.arguments.iter() {
                if let Some(name) = argument.name.as_deref() {
                    out.push(name.to_string());
                }
            }
            for (_, local) in entry.function.local_variables.iter() {
                if let Some(name) = local.name.as_deref() {
                    out.push(name.to_string());
                }
            }
        }

        out
    }

    fn count_declaration_name(module: &naga::Module, target: &str) -> usize {
        collect_declaration_names(module)
            .into_iter()
            .filter(|name| name == target)
            .count()
    }

    #[test]
    fn renames_non_preserved_identifiers_and_keeps_names_unique() {
        let source = r#"
var<private> global_long_name: f32 = 2.0;

fn helper(input_value: f32) -> f32 {
    var local_value: f32;
    local_value = input_value + global_long_name;
    return local_value;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source, &[]);
        assert!(changed, "rename pass should report changes");

        let decl_names = collect_declaration_names(&module);
        assert!(
            !decl_names.iter().any(|n| n == "global_long_name"),
            "global should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "helper"),
            "helper function should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "input_value"),
            "helper argument should be renamed"
        );
        assert!(
            !decl_names.iter().any(|n| n == "local_value"),
            "helper local should be renamed"
        );

        let unique_count = decl_names.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            unique_count,
            decl_names.len(),
            "renamed declarations should be unique"
        );

        assert!(
            decl_names.iter().all(|n| follows_expected_name_pattern(n)),
            "all generated declaration names should match expected character pattern"
        );

        assert_eq!(
            module.entry_points[0].name, "fs_main",
            "entry point name should not change"
        );
    }

    #[test]
    fn cooperative_matrix_reserves_role_enumerants() {
        // The naga wgsl-out fallback spells the role literally, so a
        // declaration minted onto `A` / `B` / `C` collides with it.
        let source = "enable wgpu_cooperative_matrix;\n\
            var<private> a: coop_mat8x8<f32, A>;\n\
            var<private> bb: coop_mat8x8<f32, B>;\n\
            @group(0) @binding(0) var<storage, read_write> ext: array<f32>;\n\
            @compute @workgroup_size(8, 8, 1) fn main() {\n\
                var c = coopLoad<coop_mat8x8<f32, C>>(&ext[4]);\n\
                var d = coopMultiplyAdd(a, bb, c);\n\
                coopStore(d, &ext[0]);\n\
            }";
        let module = naga::front::wgsl::parse_str(source).expect("coop source should parse");
        let reserved = collect_reserved_names(&module, &HashSet::new(), true);
        for role in ["A", "B", "C"] {
            assert!(
                reserved.contains(role),
                "coop role `{role}` must be reserved"
            );
        }

        let (_, renamed) = run_pass(source, &[]);
        let decl_names = collect_declaration_names(&renamed);
        for role in ["A", "B", "C"] {
            assert!(
                !decl_names.iter().any(|n| n == role),
                "no declaration may be renamed onto coop role `{role}`: {decl_names:?}"
            );
        }
    }

    #[test]
    fn non_coop_module_leaves_role_enumerants_free() {
        // Reserved ONLY for coop modules, so every other module keeps the
        // cheapest names.
        let source = "var<private> some_long_global: f32 = 1.0;\n\
            @compute @workgroup_size(1) fn main() { some_long_global = some_long_global + 1.0; }";
        let module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let reserved = collect_reserved_names(&module, &HashSet::new(), true);
        assert!(
            !reserved.contains("A") && !reserved.contains("B") && !reserved.contains("C"),
            "non-coop module must not auto-reserve A/B/C: {reserved:?}"
        );
    }

    /// The source read the global before the local shadowed it; a kept
    /// `var t` re-binds that read to itself (`for(;t<t;)`).
    #[test]
    fn preserved_module_name_is_never_kept_on_a_local() {
        let source = r#"
var<private> t: u32 = 11u;
fn f() -> u32 {
    let a = t;
    var t: u32 = 0u;
    loop { if (t >= a) { break; } t = t + 3u; }
    return t;
}
@compute @workgroup_size(1) fn main() { t = 15u; _ = f(); }
"#;
        let (_, renamed) = run_pass(source, &["t"]);
        let global_names: Vec<_> = renamed
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.name.clone())
            .collect();
        assert_eq!(global_names, ["t"], "the preserved global keeps its name");
        let (_, f) = renamed.functions.iter().next().expect("f survives");
        let local_names: Vec<_> = f
            .local_variables
            .iter()
            .filter_map(|(_, l)| l.name.clone())
            .collect();
        assert!(
            !local_names.iter().any(|n| n == "t"),
            "the local must not capture the module-scope name: {local_names:?}"
        );
    }

    /// A local sharing a preserved STRUCT TYPE name shadows the type; a
    /// constructor inlining moves into its scope would call the local.
    #[test]
    fn preserved_type_name_is_never_kept_on_a_local() {
        let source = r#"
struct S { v: f32 }
fn mk(x: f32) -> S { return S(x * 2.0); }
fn f() -> f32 { var S: S = S(1.0); let q = mk(S.v); S.v = q.v; return S.v; }
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(f()); }
"#;
        let (_, renamed) = run_pass(source, &["S"]);
        let (_, f) = renamed
            .functions
            .iter()
            .find(|(_, f)| f.local_variables.len() == 1)
            .expect("f has one local");
        let local = f.local_variables.iter().next().unwrap().1.name.clone();
        assert_ne!(
            local.as_deref(),
            Some("S"),
            "the local must not shadow the type"
        );
    }

    #[test]
    fn preserves_requested_symbols_without_reusing_them() {
        let source = r#"
var<private> keep_global: f32 = 1.0;
var<private> rename_global: f32 = 2.0;

fn helper(keep_arg: f32, rename_arg: f32) -> f32 {
    var rename_local: f32;
    rename_local = keep_arg + rename_arg + keep_global + rename_global;
    return rename_local;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0, 2.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source, &["keep_global", "keep_arg"]);
        assert!(
            changed,
            "rename pass should still rename non-preserved symbols"
        );

        assert_eq!(
            count_declaration_name(&module, "keep_global"),
            1,
            "preserved global name should remain and not be reused"
        );
        assert_eq!(
            count_declaration_name(&module, "keep_arg"),
            1,
            "preserved argument name should remain and not be reused"
        );

        let decl_names = collect_declaration_names(&module);
        assert!(decl_names.iter().any(|n| n == "keep_global"));
        assert!(decl_names.iter().any(|n| n == "keep_arg"));
        assert!(!decl_names.iter().any(|n| n == "rename_global"));
        assert!(!decl_names.iter().any(|n| n == "rename_arg"));
        assert!(!decl_names.iter().any(|n| n == "rename_local"));
    }

    #[test]
    fn assigns_shortest_name_to_most_referenced_identifier() {
        // `hot` is declared LAST yet read five times; only weight-sorted
        // assignment gives it "A".
        let source = r#"
var<private> cold_a: f32 = 1.0;
var<private> cold_b: f32 = 2.0;
var<private> cold_c: f32 = 3.0;
var<private> hot: f32 = 4.0;

@fragment
fn fs_main() -> @location(0) vec4f {
    let s = hot + hot + hot + hot + hot + cold_a + cold_b + cold_c;
    return vec4f(s, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass_with_mangle(source, &[], true);
        let weights = compute_weights(&module);

        let heaviest = module
            .global_variables
            .iter()
            .max_by_key(|(h, _)| weights.global.get(h).copied().unwrap_or(0))
            .and_then(|(_, g)| g.name.clone())
            .expect("a global should exist");
        assert_eq!(
            heaviest, "A",
            "the most-referenced identifier must receive the first (shortest) name"
        );
    }

    #[test]
    fn name_generator_sequence_and_pattern_are_expected() {
        let mut counter = 0usize;
        let generated = (0..120)
            .map(|_| name_gen::next_name(&mut counter))
            .collect::<Vec<_>>();

        assert_eq!(generated[0], "A");
        assert_eq!(generated[1], "a");
        assert_eq!(generated[2], "B");

        assert!(
            generated.iter().all(|n| follows_expected_name_pattern(n)),
            "all generated names should follow FIRST/NEXT character-table pattern"
        );
    }

    #[test]
    fn mangle_renames_constants() {
        let source = r#"
const MY_CONSTANT: f32 = 3.14;

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(MY_CONSTANT, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass_with_mangle(source, &[], true);
        assert!(changed, "mangle should rename constants");

        let has_original = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("MY_CONSTANT"));
        assert!(!has_original, "original constant name should be replaced");

        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref() {
                assert!(
                    follows_expected_name_pattern(name),
                    "mangled constant name '{}' should follow pattern",
                    name
                );
            }
        }
    }

    #[test]
    fn mangle_preserves_specified_constants() {
        let source = r#"
const KEEP_ME: f32 = 1.0;
const RENAME_ME: f32 = 2.0;

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(KEEP_ME, RENAME_ME, 0.0, 1.0);
}
"#;

        let (_, module) = run_pass_with_mangle(source, &["KEEP_ME"], true);
        let has_keep = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("KEEP_ME"));
        assert!(has_keep, "preserved constant should keep its name");

        let has_rename = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("RENAME_ME"));
        assert!(!has_rename, "non-preserved constant should be renamed");
    }

    #[test]
    fn no_collision_with_unrenamed_constant_names() {
        // The constant `A` is exactly the first generated name, so the sweep
        // must skip past it.
        let source = r#"
const A: f32 = 1.0;
var<private> long_global_name: f32 = 2.0;

fn helper(x: f32) -> f32 {
    return x + A + long_global_name;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (_, module) = run_pass(source, &[]);

        let has_const_a = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("A"));
        assert!(
            has_const_a,
            "constant A should keep its name when mangle is off"
        );

        let decl_names = collect_declaration_names(&module);
        let a_count = decl_names.iter().filter(|n| n.as_str() == "A").count();
        assert_eq!(
            a_count, 0,
            "no generated declaration name should collide with unrenamed constant 'A'"
        );
    }

    #[test]
    fn mangle_rename_is_idempotent() {
        // Otherwise the convergence loop oscillates between two assignments.
        let source = r#"
const LONG_CONST_A: f32 = 1.0;
const LONG_CONST_B: f32 = 2.0;
var<private> long_global: f32 = 3.0;

fn helper(x: f32) -> f32 {
    var tmp: f32;
    tmp = x + LONG_CONST_A + LONG_CONST_B + long_global;
    return tmp;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;

        let (changed1, module1) = run_pass_with_mangle(source, &[], true);
        assert!(changed1, "first rename should change names");

        let mut module2 = module1.clone();
        let mut pass = RenamePass::new(Vec::new(), true);
        let config = Config::default();
        let changed2 = PassContext::run_pass(&mut pass, &mut module2, &config)
            .expect("second rename should work");
        assert!(
            !changed2,
            "second rename with mangle should be idempotent (no changes)"
        );

        let names1 = collect_declaration_names(&module1);
        let names2 = collect_declaration_names(&module2);
        assert_eq!(names1, names2, "names must be identical across runs");
    }

    /// A deferred `var t = init;` spells `t` once where the IR holds the
    /// declaration and the initialising store: by the census `t` (5)
    /// outranks a global read three times (4), by the text they tie and
    /// the global, declared first, takes the first name - as the re-parse
    /// of the text will rank them.
    #[test]
    fn the_tail_ranks_names_by_what_the_text_spells() {
        let source = "var<private> g: f32;\
            @group(0) @binding(0) var<storage, read_write> o: array<f32>;\
            @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {\
              var t = f32(i) * 2.0; g = t + t * t; o[0] = g; o[1] = g + 1.0; }";
        let name_of_g = |pass: RenamePass| {
            let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
            let mut pass = pass;
            PassContext::run_pass(&mut pass, &mut module, &Config::default()).expect("rename");
            module
                .global_variables
                .iter()
                .next()
                .unwrap()
                .1
                .name
                .clone()
        };
        assert_eq!(
            name_of_g(RenamePass::new(Vec::new(), true)).as_deref(),
            Some("a"),
            "by the census the local outranks the global"
        );
        assert_eq!(
            name_of_g(RenamePass::by_render(Vec::new(), true)).as_deref(),
            Some("A"),
            "by the text the global takes the first name"
        );
    }

    #[test]
    fn clears_named_expressions_and_reports_change_even_when_nothing_renamed() {
        // Gating the clear on `changed` makes a preserve-all pass exit the
        // convergence loop one sweep early, before downstream DCE can remove
        // orphaned globals.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    let y = 1.0;
    return vec4f(y, 0.0, 0.0, 1.0);
}
"#;
        let preserve = ["fs_main", "y"];
        let (changed, module) = run_pass_with_mangle(source, &preserve, true);
        assert!(
            changed,
            "rename must report `changed = true` when it clears `named_expressions`, \
             even if no identifier was renamed - downstream convergence depends on it"
        );
        let entry = module
            .entry_points
            .first()
            .expect("entry point should exist");
        assert!(
            entry.function.named_expressions.is_empty(),
            "named_expressions must be cleared so the WGSL emitter does not pick up \
             stale bindings on the next sweep"
        );
    }

    // MARK: Struct-name reservation regression

    /// Without mangling the generator keeps source struct names verbatim, so
    /// a minted name colliding with one puts two same-named symbols in the
    /// WGSL.
    #[test]
    fn reserves_source_struct_type_names_without_mangle() {
        let src = r#"
struct A { x: f32, y: f32 }
@group(0) @binding(0) var<uniform> g: A;
fn h(p: f32, q: f32, r: f32) -> A {
    var out: A;
    out.x = p + q;
    out.y = r + g.x;
    return out;
}
@fragment fn m() -> @location(0) vec4f {
    let v = h(1.0, 2.0, 3.0);
    return vec4f(v.x, v.y, 0.0, 1.0);
}
"#;
        let module = naga::front::wgsl::parse_str(src).expect("parses");
        let preserve = HashSet::new();
        let reserved = collect_reserved_names(&module, &preserve, /*mangle=*/ false);
        assert!(
            reserved.contains("A"),
            "source struct type name must be in the reserved set so rename does not \
             collide with it"
        );
        assert!(
            reserved.contains("x") && reserved.contains("y"),
            "source struct member names must be in the reserved set"
        );
    }

    /// Under `mangle = true` the generator mints struct type and member
    /// names clear of the pass's assignments, so reserving the source's
    /// would cost letters and, on a re-minify, idempotence
    /// ([`collect_reserved_names`]).
    #[test]
    fn does_not_reserve_source_struct_type_names_with_mangle() {
        let src = r#"
struct A { x: f32 }
@group(0) @binding(0) var<uniform> g: A;
@fragment fn m() -> @location(0) vec4f {
    return vec4f(g.x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).expect("parses");
        let preserve = HashSet::new();
        let reserved = collect_reserved_names(&module, &preserve, /*mangle=*/ true);
        assert!(
            !reserved.contains("A"),
            "a source struct type name is free under mangle"
        );
        assert!(
            !reserved.contains("x"),
            "a source struct member name is free under mangle"
        );
    }

    /// A further target draws the name at its weight's rank; past the 52nd
    /// single letter the draws lengthen, and every target ranked below the
    /// newcomer pays an occurrence per byte its own draw grows.
    #[test]
    fn a_planned_name_and_what_the_others_pay_for_it() {
        let mut src = String::new();
        for i in 0..52 {
            src.push_str(&format!("var<private> g{i}: f32;\n"));
        }
        src.push_str("@compute @workgroup_size(1) fn main() {}");
        let module = naga::front::wgsl::parse_str(&src).expect("source should parse");
        let plan = plan_names(&module, &HashSet::new(), true);
        assert_eq!(plan.names.len(), 52);
        assert!(plan.names.iter().all(|n| n.len() == 1));
        // Weight 1 ties every global and ranks last: the 53rd draw.
        assert_eq!(plan.name_len_at_weight(1), 2);
        assert_eq!(plan.insertion_cost(1), 0);
        // Weight 2 ranks first; the last single-letter holder moves to two
        // letters at its one occurrence.
        assert_eq!(plan.name_len_at_weight(2), 1);
        assert_eq!(plan.insertion_cost(2), 1);
    }

    /// naga gives each block-scoped shadowing `var` its own handle under the
    /// shared source name; a preserved name is kept by the first declaration
    /// only, or both keep it and one scope ends up declaring it twice.
    #[test]
    fn a_preserved_name_is_kept_by_one_declaration_only() {
        let source = r#"
fn f(x: f32) -> f32 {
    var t: f32 = x;
    if (x > 1.0) {
        var t: f32 = 5.0;
        t = t + 1.0;
    }
    return t;
}
"#;
        let (_, module) = run_pass(source, &["t"]);
        assert_eq!(
            count_declaration_name(&module, "t"),
            1,
            "exactly one local keeps the preserved name: {:?}",
            collect_declaration_names(&module)
        );
    }
}
