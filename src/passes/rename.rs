//! Identifier rename pass.  Replaces every user-chosen identifier the
//! pipeline is free to touch with a short generated name, keeping the
//! output deterministic and collision-free against both WGSL reserved
//! words and any symbols the caller asked to preserve.
//!
//! Struct type and member names are renamed at the generator layer
//! (naga's `UniqueArena<Type>` is immutable mid-pipeline), so this
//! pass only covers the mutable arenas: globals, constants, overrides,
//! functions, arguments, and locals.
//!
//! Frequency ordering: name length grows with how many identifiers are in
//! play (52 single-character names, then two-character, ...), so the pass
//! assigns the shortest names to the identifiers that appear most often.  It
//! ranks every renameable identifier by an occurrence weight (its declaration
//! plus its in-body references - the generator's inline/bind ref-count signal)
//! and draws names heaviest-first.  Because every identifier still
//! receives a globally distinct name, the SET of names drawn is independent of
//! the order: ordering only permutes which identifier holds which name, leaving
//! the pool of names the generator must avoid (for struct types, aliases, and
//! extracted literals) unchanged.  So no downstream generator decision shifts,
//! and total identifier bytes are minimised by the rearrangement inequality.

use std::collections::{HashMap, HashSet};

use crate::error::Error;
use crate::name_gen;
use crate::passes::expr_util::{visit_block_expression_handles, visit_expression_children};
use crate::pipeline::{Pass, PassContext};

/// Rename pass state.  `preserve` lists names that must survive
/// verbatim; `mangle` extends the rename scope to include constants
/// and overrides (whose names otherwise leak into the final output).
#[derive(Debug)]
pub struct RenamePass {
    preserve: HashSet<String>,
    mangle: bool,
}

impl RenamePass {
    /// Construct a new pass from the user-facing `preserve_symbols`
    /// vector and the resolved `mangle` flag.
    pub fn new(preserve_symbols: Vec<String>, mangle: bool) -> Self {
        Self {
            preserve: preserve_symbols.into_iter().collect(),
            mangle,
        }
    }
}

impl Pass for RenamePass {
    fn name(&self) -> &'static str {
        "rename_identifiers"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut used_names = collect_reserved_names(module, &self.preserve, self.mangle);

        // Occurrence weights approximate how often each identifier's name
        // appears in the output (its declaration plus its live in-body
        // references).  Assigning the shortest names to the heaviest
        // identifiers minimises total identifier bytes.  Weights are structural
        // (handle/arena based, name independent), so the assignment is
        // deterministic and the pass stays idempotent at the convergence fixed
        // point.
        let weights = compute_weights(module);

        // Enumerate every renameable identifier as a `(target, weight, seq)`
        // triple.  `seq` is the declaration-order index, a deterministic
        // tie-break giving equal-weight identifiers a stable assignment.  The
        // candidate set is exactly the renameable identifiers (the
        // preserve / mangle / `@id` gating below), so the multiset of names
        // drawn depends only on their count, not the order: ordering only
        // repermutes which identifier receives which name.
        let mut targets: Vec<(Target, usize, usize)> = Vec::new();
        let mut seq = 0usize;

        // Mangling extends renaming to module-scope constants and overrides.
        // Struct types and members are handled at the generator layer because
        // naga's `UniqueArena<Type>` is immutable after lowering.
        if self.mangle {
            for (h, c) in module.constants.iter() {
                if let Some(name) = c.name.as_deref()
                    && !self.preserve.contains(name)
                {
                    push_target(&mut targets, &mut seq, Target::Constant(h), &weights);
                }
            }
            for (h, ov) in module.overrides.iter() {
                // An override with an explicit `@id(N)` is identified to the
                // host by its numeric id, so its name is free to mangle.  An
                // `@id`-less override is identified ONLY by its declaration
                // name - the key in the pipeline `constants` record - so
                // renaming it silently breaks host pipeline-constant
                // specialization.  Treat `@id`-less overrides like
                // preserve-listed names; they are reserved in
                // `collect_reserved_names`.
                if let Some(name) = ov.name.as_deref()
                    && ov.id.is_some()
                    && !self.preserve.contains(name)
                {
                    push_target(&mut targets, &mut seq, Target::Override(h), &weights);
                }
            }
        }

        for (h, global) in module.global_variables.iter() {
            if let Some(name) = global.name.as_deref()
                && self.preserve.contains(name)
            {
                continue;
            }
            push_target(&mut targets, &mut seq, Target::Global(h), &weights);
        }

        for (fh, function) in module.functions.iter() {
            // Regular function names are module scope: rename unless preserved.
            if !matches!(function.name.as_deref(), Some(n) if self.preserve.contains(n)) {
                push_target(&mut targets, &mut seq, Target::Function(fh), &weights);
            }
            enumerate_locals(
                function,
                FuncRef::Function(fh),
                &self.preserve,
                &weights,
                &mut targets,
                &mut seq,
            );
        }

        for (ei, entry) in module.entry_points.iter().enumerate() {
            // Entry-point names are pipeline-bound and never renamed.
            enumerate_locals(
                &entry.function,
                FuncRef::Entry(ei),
                &self.preserve,
                &weights,
                &mut targets,
                &mut seq,
            );
        }

        // Heaviest weight first, declaration order breaking ties.  `seq` is
        // unique per target, so the ordering is total and independent of sort
        // stability.
        targets.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

        // Draw names shortest-first into per-arena side tables, then apply.
        // Each draw takes the smallest unused name, so the consumed set is
        // exactly the first N non-reserved names regardless of order - only
        // the identifier-to-name pairing changes.
        let mut counter = 0usize;
        let mut assigned = AssignedNames::default();
        for (target, _, _) in &targets {
            let name = next_available_name(&mut counter, &mut used_names);
            assigned.insert(*target, name);
        }

        let mut changed = false;
        if self.mangle {
            for (h, c) in module.constants.iter_mut() {
                apply_name(&mut c.name, assigned.constant.remove(&h), &mut changed);
            }
            for (h, ov) in module.overrides.iter_mut() {
                apply_name(&mut ov.name, assigned.over.remove(&h), &mut changed);
            }
        }
        for (h, global) in module.global_variables.iter_mut() {
            apply_name(&mut global.name, assigned.global.remove(&h), &mut changed);
        }
        for (fh, function) in module.functions.iter_mut() {
            apply_name(
                &mut function.name,
                assigned.function.remove(&fh),
                &mut changed,
            );
            apply_locals(function, FuncRef::Function(fh), &mut assigned, &mut changed);
            changed |= clear_named_expressions(function);
        }
        for (ei, entry) in module.entry_points.iter_mut().enumerate() {
            apply_locals(
                &mut entry.function,
                FuncRef::Entry(ei),
                &mut assigned,
                &mut changed,
            );
            changed |= clear_named_expressions(&mut entry.function);
        }

        Ok(changed)
    }
}

/// A renameable identifier, identified by the arena slot its name lives in.
/// Args and locals are scoped to a [`FuncRef`] so the same arena handle in
/// two functions never collides in the side tables.
#[derive(Clone, Copy)]
enum Target {
    Constant(naga::Handle<naga::Constant>),
    Override(naga::Handle<naga::Override>),
    Global(naga::Handle<naga::GlobalVariable>),
    Function(naga::Handle<naga::Function>),
    Arg(FuncRef, usize),
    Local(FuncRef, naga::Handle<naga::LocalVariable>),
}

/// Identifies a function body for per-function (argument / local) scoping.
/// Regular functions are keyed by arena handle, entry points by index.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FuncRef {
    Function(naga::Handle<naga::Function>),
    Entry(usize),
}

/// Names assigned this sweep, bucketed by arena so application is a single
/// `iter_mut` per arena rather than a per-handle random-access lookup.
#[derive(Default)]
struct AssignedNames {
    constant: HashMap<naga::Handle<naga::Constant>, String>,
    over: HashMap<naga::Handle<naga::Override>, String>,
    global: HashMap<naga::Handle<naga::GlobalVariable>, String>,
    function: HashMap<naga::Handle<naga::Function>, String>,
    arg: HashMap<(FuncRef, usize), String>,
    local: HashMap<(FuncRef, naga::Handle<naga::LocalVariable>), String>,
}

impl AssignedNames {
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
            Target::Arg(f, i) => {
                self.arg.insert((f, i), name);
            }
            Target::Local(f, h) => {
                self.local.insert((f, h), name);
            }
        }
    }
}

/// Per-identifier occurrence weights used to rank name length.  Every
/// renameable identifier gets an entry (at least its declaration), so a
/// missing key means a non-renameable handle and is treated as weight 1.
#[derive(Default)]
struct Weights {
    global: HashMap<naga::Handle<naga::GlobalVariable>, usize>,
    constant: HashMap<naga::Handle<naga::Constant>, usize>,
    over: HashMap<naga::Handle<naga::Override>, usize>,
    function: HashMap<naga::Handle<naga::Function>, usize>,
    arg: HashMap<(FuncRef, usize), usize>,
    local: HashMap<(FuncRef, naga::Handle<naga::LocalVariable>), usize>,
}

impl Weights {
    fn of(&self, target: Target) -> usize {
        match target {
            Target::Constant(h) => self.constant.get(&h).copied(),
            Target::Override(h) => self.over.get(&h).copied(),
            Target::Global(h) => self.global.get(&h).copied(),
            Target::Function(h) => self.function.get(&h).copied(),
            Target::Arg(f, i) => self.arg.get(&(f, i)).copied(),
            Target::Local(f, h) => self.local.get(&(f, h)).copied(),
        }
        .unwrap_or(1)
    }
}

/// Push one renameable identifier onto the candidate list with its weight and
/// the next declaration-order tie-break index.
fn push_target(
    targets: &mut Vec<(Target, usize, usize)>,
    seq: &mut usize,
    target: Target,
    weights: &Weights,
) {
    targets.push((target, weights.of(target), *seq));
    *seq += 1;
}

/// Enumerate the renameable arguments and locals of one function body,
/// shared by regular functions and entry points.
fn enumerate_locals(
    function: &naga::Function,
    fref: FuncRef,
    preserve: &HashSet<String>,
    weights: &Weights,
    targets: &mut Vec<(Target, usize, usize)>,
    seq: &mut usize,
) {
    for (i, argument) in function.arguments.iter().enumerate() {
        if matches!(argument.name.as_deref(), Some(n) if preserve.contains(n)) {
            continue;
        }
        push_target(targets, seq, Target::Arg(fref, i), weights);
    }
    for (lh, local) in function.local_variables.iter() {
        if matches!(local.name.as_deref(), Some(n) if preserve.contains(n)) {
            continue;
        }
        push_target(targets, seq, Target::Local(fref, lh), weights);
    }
}

/// Write `name` into `slot`, recording whether it actually changed.
fn apply_name(slot: &mut Option<String>, name: Option<String>, changed: &mut bool) {
    if let Some(name) = name {
        *changed |= slot.as_deref() != Some(name.as_str());
        *slot = Some(name);
    }
}

/// Apply this sweep's argument and local names to one function body.
fn apply_locals(
    function: &mut naga::Function,
    fref: FuncRef,
    assigned: &mut AssignedNames,
    changed: &mut bool,
) {
    for (i, argument) in function.arguments.iter_mut().enumerate() {
        apply_name(&mut argument.name, assigned.arg.remove(&(fref, i)), changed);
    }
    for (lh, local) in function.local_variables.iter_mut() {
        apply_name(&mut local.name, assigned.local.remove(&(fref, lh)), changed);
    }
}

/// Clear `named_expressions` whenever the function carries any - even if no
/// identifier was renamed - and report that as a change.  The "report as
/// change" piece looks like a perf wart (it costs one extra convergence sweep
/// on shaders that name expressions but rename nothing else), but it is
/// load-bearing: the extra sweep gives downstream passes a chance to observe
/// IR that settled only in this sweep's earlier passes (e.g. DCE catches an
/// orphaned global that became unreachable once an upstream phony-assignment
/// load was eliminated).
fn clear_named_expressions(function: &mut naga::Function) -> bool {
    if function.named_expressions.is_empty() {
        return false;
    }
    function.named_expressions.clear();
    true
}

// MARK: Occurrence weights

/// Compute occurrence weights ranking identifiers by how often their name is
/// emitted: the declaration plus every live reference in a function or entry
/// body, mirroring the per-function `compute_expression_ref_counts` (for a
/// trivially inlined node like `GlobalVariable(g)`, its ref count equals the
/// textual occurrences of `g`'s name in that body).
///
/// This is a size-ranking heuristic, not an exact emission count: references
/// in module-scope initializers are not counted, and the body walkers cover
/// only the block-bearing statements naga has today.  An imperfect weight can
/// only yield a longer-than-optimal name - never affecting correctness or
/// collision-freedom, which the all-distinct draw alone guarantees.
fn compute_weights(module: &naga::Module) -> Weights {
    let mut w = Weights::default();

    // Declaration occurrences: every emitted declaration prints its name once.
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

    for (fh, function) in module.functions.iter() {
        accumulate_function_weights(&mut w, FuncRef::Function(fh), function);
        count_calls(&function.body, &mut w.function);
    }
    for (ei, entry) in module.entry_points.iter().enumerate() {
        accumulate_function_weights(&mut w, FuncRef::Entry(ei), &entry.function);
        count_calls(&entry.function.body, &mut w.function);
    }

    w
}

/// Fold one function's reference counts into the module-wide weight tables.
fn accumulate_function_weights(w: &mut Weights, fref: FuncRef, function: &naga::Function) {
    // Declaration occurrences for parameters and locals.
    for i in 0..function.arguments.len() {
        *w.arg.entry((fref, i)).or_insert(0) += 1;
    }
    for (lh, _) in function.local_variables.iter() {
        *w.local.entry((fref, lh)).or_insert(0) += 1;
    }

    let counts = function_ref_counts(function);
    for (h, expr) in function.expressions.iter() {
        let c = counts[h.index()];
        if c == 0 {
            continue;
        }
        match expr {
            naga::Expression::GlobalVariable(g) => *w.global.entry(*g).or_insert(0) += c,
            naga::Expression::LocalVariable(l) => *w.local.entry((fref, *l)).or_insert(0) += c,
            naga::Expression::FunctionArgument(i) => {
                *w.arg.entry((fref, *i as usize)).or_insert(0) += c
            }
            naga::Expression::Constant(cst) => *w.constant.entry(*cst).or_insert(0) += c,
            naga::Expression::Override(o) => *w.over.entry(*o).or_insert(0) += c,
            _ => {}
        }
    }
}

/// Per-handle reference counts for one function, mirroring the generator's
/// `compute_expression_ref_counts`: count children of every live (in an
/// `Emit` range) expression plus every statement-level operand.  Dead
/// expressions are excluded so identifiers used only by dead code score 0 and
/// sort last, never claiming a short name.
fn function_ref_counts(function: &naga::Function) -> Vec<usize> {
    let len = function.expressions.len();
    let mut live = vec![false; len];
    mark_emit_live(&function.body, &mut live);

    let mut counts = vec![0usize; len];
    for (h, expr) in function.expressions.iter() {
        if live[h.index()] {
            visit_expression_children(expr, |child| counts[child.index()] += 1);
        }
    }
    // `false` suppresses Emit handles: emission sequencing is not a use.
    visit_block_expression_handles(&function.body, false, &mut |h| counts[h.index()] += 1);
    counts
}

/// Mark every expression handle that appears inside an `Emit` range of
/// `block`, recursing through control flow.  Emission-range membership is the
/// liveness signal `function_ref_counts` filters on.
fn mark_emit_live(block: &naga::Block, live: &mut [bool]) {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    live[h.index()] = true;
                }
            }
            naga::Statement::Block(inner) => mark_emit_live(inner, live),
            naga::Statement::If { accept, reject, .. } => {
                mark_emit_live(accept, live);
                mark_emit_live(reject, live);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    mark_emit_live(&case.body, live);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                mark_emit_live(body, live);
                mark_emit_live(continuing, live);
            }
            _ => {}
        }
    }
}

/// Count `Statement::Call` targets in `block` (recursing through control
/// flow) so a frequently-called function earns a shorter name.
fn count_calls(block: &naga::Block, calls: &mut HashMap<naga::Handle<naga::Function>, usize>) {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Call { function, .. } => {
                *calls.entry(*function).or_insert(0) += 1;
            }
            naga::Statement::Block(inner) => count_calls(inner, calls),
            naga::Statement::If { accept, reject, .. } => {
                count_calls(accept, calls);
                count_calls(reject, calls);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    count_calls(&case.body, calls);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                count_calls(body, calls);
                count_calls(continuing, calls);
            }
            _ => {}
        }
    }
}

/// Build the starting `used_names` set for one rename sweep.
///
/// The policy differs by `mangle` because idempotence hinges on it:
///
/// - With `mangle = false`, constant and override names are kept
///   verbatim and must be reserved so generated names do not collide
///   with them.
/// - With `mangle = true`, those names are themselves rewritten, so
///   reserving the previous sweep's assignments would pollute the used
///   set and shift subsequent assignments one slot, producing a
///   two-sweep oscillation that keeps the pipeline from converging.
///
/// Preserve-listed entries in every arena are always reserved
/// regardless of `mangle` so the user-visible names survive.
fn collect_reserved_names(
    module: &naga::Module,
    preserve: &HashSet<String>,
    mangle: bool,
) -> HashSet<String> {
    let mut reserved = HashSet::new();

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
        // Even when mangling, preserve-listed constants and overrides
        // retain their names and must be reserved.
        for (_, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref()
                && preserve.contains(name)
            {
                reserved.insert(name.to_string());
            }
        }
        for (_, ov) in module.overrides.iter() {
            // Reserve preserve-listed overrides AND every `@id`-less override:
            // the latter keeps its name (the host's pipeline-constant key) and
            // is not renamed, so no mangled identifier may collide with it.
            if let Some(name) = ov.name.as_deref()
                && (preserve.contains(name) || ov.id.is_none())
            {
                reserved.insert(name.to_string());
            }
        }
    }

    for (_, global) in module.global_variables.iter() {
        if let Some(name) = global.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }

    for (_, function) in module.functions.iter() {
        if let Some(name) = function.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
        collect_preserved_function_names(function, preserve, &mut reserved);
    }

    for entry in module.entry_points.iter() {
        reserved.insert(entry.name.clone());
        collect_preserved_function_names(&entry.function, preserve, &mut reserved);
    }

    // Always reserve every source struct type name and struct member
    // name, regardless of `mangle`.  Two reasons:
    //
    // * Under `mangle = false` the generator emits source struct type
    //   names verbatim (see `core.rs::type_names`).  If the rename
    //   counter ever mints one of those names for a global / function
    //   / local, the WGSL output contains two same-named symbols and
    //   the user's struct becomes unreachable.  Round-trip validation
    //   catches the bad output (see lib.rs fallback), but the bug
    //   silently halves compaction quality.
    //
    // * Under `mangle = true` the generator independently re-mangles
    //   struct type / member names via its own `used_names` set in
    //   `core.rs`, so the source names are *not* the final WGSL names.
    //   Reserving them is still harmless: the rename counter just
    //   skips a few short names, and the generator's mangling decides
    //   what each struct is ultimately called.  The cost of reserving
    //   here is bounded by the number of source-named structs.
    //
    // Preserve-listed names in the type / member arenas are an extra
    // case the generator never mangles regardless of `mangle`, so
    // reserving unconditionally also covers them.
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

    // Cooperative-matrix role enumerants `A`/`B`/`C` are predeclared names in
    // the `coop_mat<T, role>` type-argument position.  nagami's generator
    // cannot emit cooperative-matrix types, so such modules fall back to naga's
    // own wgsl-out, which renders the role literally as `A`/`B`/`C`.  If the
    // rename sweep mints one of those names for a global / local / function,
    // naga re-reads the role position as that declaration ("identifier `B`
    // resolves to a declaration" / "declaration of `B` is recursive") and
    // rejects the fallback.  Reserve the three role names so no renamed
    // identifier can collide.  Guarded on actual coop-matrix usage, so every
    // non-coop module is unaffected (`A`/`B`/`C` are the cheapest names the
    // counter would otherwise hand out).
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

/// Reserve every preserve-listed argument and local inside `function`
/// so the rename sweep cannot repurpose those names for unrelated
/// declarations elsewhere in the module.
fn collect_preserved_function_names(
    function: &naga::Function,
    preserve: &HashSet<String>,
    reserved: &mut HashSet<String>,
) {
    for argument in function.arguments.iter() {
        if let Some(name) = argument.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }

    for (_, local) in function.local_variables.iter() {
        if let Some(name) = local.name.as_deref()
            && preserve.contains(name)
        {
            reserved.insert(name.to_string());
        }
    }
}

/// Thin wrapper over [`name_gen::next_name_insert`] so the pass body
/// is readable without the longer function name spelled out.
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
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let changed = pass.run(&mut module, &ctx).expect("rename pass should run");
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
        // The generator cannot emit coop-matrix types, so such modules fall
        // back to naga's wgsl-out, which renders the role literally as
        // `A`/`B`/`C`.  Renaming a declaration onto one of those names collides
        // with the role position, so they must be reserved.
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

        // End-to-end: the rename sweep must not mint a role name for any decl.
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
        // `A`/`B`/`C` are the cheapest names; they are reserved ONLY for coop
        // modules so the whole non-coop corpus keeps them in the rename pool.
        let source = "var<private> some_long_global: f32 = 1.0;\n\
            @compute @workgroup_size(1) fn main() { some_long_global = some_long_global + 1.0; }";
        let module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let reserved = collect_reserved_names(&module, &HashSet::new(), true);
        assert!(
            !reserved.contains("A") && !reserved.contains("B") && !reserved.contains("C"),
            "non-coop module must not auto-reserve A/B/C: {reserved:?}"
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
        // Frequency ordering: the shortest name goes to the most-referenced
        // identifier, not the first-declared one.  `hot` is declared LAST yet
        // read five times; only weight-sorted assignment gives it the first
        // name "A" - plain declaration order would hand it a later name - so
        // this assertion fails if the heaviest-first sort ever regresses to
        // declaration order.
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

        // The heaviest renameable identifier is the global read five times.
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

        // All constant names should follow the short-name pattern.
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
        // With mangle off, constants keep their original names.  The
        // generator must not reassign those same names to other
        // declarations.  Here the constant `A` is exactly the first
        // generated name, so the sweep must skip past it.
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

        // Constant "A" should still exist.
        let has_const_a = module
            .constants
            .iter()
            .any(|(_, c)| c.name.as_deref() == Some("A"));
        assert!(
            has_const_a,
            "constant A should keep its name when mangle is off"
        );

        // No renamed declaration should collide with "A".
        let decl_names = collect_declaration_names(&module);
        let a_count = decl_names.iter().filter(|n| n.as_str() == "A").count();
        assert_eq!(
            a_count, 0,
            "no generated declaration name should collide with unrenamed constant 'A'"
        );
    }

    #[test]
    fn mangle_rename_is_idempotent() {
        // Running the rename pass twice with `mangle = true` on the
        // same module must produce identical results; the second run
        // has to report `changed = false`.  Without this the pipeline's
        // convergence loop oscillates between two name assignments.
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

        // Run a second rename on the already-renamed module.
        let mut module2 = module1.clone();
        let mut pass = RenamePass::new(Vec::new(), true);
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed2 = pass
            .run(&mut module2, &ctx)
            .expect("second rename should work");
        assert!(
            !changed2,
            "second rename with mangle should be idempotent (no changes)"
        );

        // Names should be identical.
        let names1 = collect_declaration_names(&module1);
        let names2 = collect_declaration_names(&module2);
        assert_eq!(names1, names2, "names must be identical across runs");
    }

    #[test]
    fn clears_named_expressions_and_reports_change_even_when_nothing_renamed() {
        // INVERSE regression: a tempting "perf" fix gated this clear
        // on `changed > 0` so a preserve-all pass would not report a
        // change.  Google Tint test corpus showed the convergence
        // loop exits one sweep too early in that mode and downstream
        // DCE never gets a chance to remove orphaned globals.  This
        // test pins the "always clear and report changed" behaviour
        // so a future maintainer who notices the apparent redundancy
        // does not silently re-introduce the regression.
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

    /// Source struct type names must be reserved as the rename pass
    /// mints fresh short identifiers - otherwise (with `mangle = false`,
    /// where the generator keeps source struct names verbatim) the
    /// pass can mint a local / global / function name that collides
    /// with an existing struct name and the emitted WGSL ends up with
    /// two same-named symbols.
    #[test]
    fn reserves_source_struct_type_names_without_mangle() {
        // The struct is named single-character `A`; with enough
        // unrenamed globals and a multi-arg function the rename
        // counter would normally reach `A` quickly and clobber the
        // type name.  We assert `A` appears in the reserved set.
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

    /// Same reservation must apply under `mangle = true`: even though
    /// the generator independently re-mangles type / member names,
    /// reserving them in the rename pass is safe and prevents short-
    /// lived collisions that a future refactor could expose.
    #[test]
    fn reserves_source_struct_type_names_with_mangle() {
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
            reserved.contains("A"),
            "source struct type name must be reserved even under mangle"
        );
        assert!(
            reserved.contains("x"),
            "source struct member name must be reserved even under mangle"
        );
    }
}
