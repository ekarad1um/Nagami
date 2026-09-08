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

use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::error::Error;
use crate::handle_set::HandleMap;
use crate::name_gen;
use crate::passes::expr_util::{for_each_statement, live_expression_ref_counts};
use crate::pipeline::{Pass, PassContext};

/// `preserve` lists names kept verbatim; `mangle` extends renaming to
/// constants and overrides.
#[derive(Debug)]
pub struct RenamePass {
    preserve: HashSet<String>,
    mangle: bool,
}

impl RenamePass {
    /// A pass from the user-facing `preserve_symbols` and the resolved
    /// `mangle` flag.
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

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut used_names = collect_reserved_names(module, &self.preserve, self.mangle);

        // Weights are structural (handle-based, name-independent), so the
        // assignment is deterministic and idempotent at the convergence
        // fixed point.
        let weights = compute_weights(module);

        // `seq` is the declaration-order tie-break, so equal weights get a
        // stable assignment.
        let mut targets: Vec<(Target, usize, usize)> = Vec::new();
        let mut seq = 0usize;

        if self.mangle {
            for (h, c) in module.constants.iter() {
                if let Some(name) = c.name.as_deref()
                    && !self.preserve.contains(name)
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

        // Heaviest first; `seq` is unique per target, so the order is total
        // and independent of sort stability.
        targets.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

        let mut counter = 0usize;
        let mut assigned = AssignedNames::default();
        for (target, _, _) in &targets {
            let name = next_available_name(&mut counter, &mut used_names);
            assigned.insert(*target, name);
        }

        // Module-scope only (locals are neither unique nor host-visible),
        // one batch per sweep so `record_batch` can resolve swaps.
        let mut module_renames: Vec<(String, String)> = Vec::new();
        let mut changed = false;
        if self.mangle {
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
        if let Some(log) = ctx.name_log
            && !module_renames.is_empty()
        {
            log.borrow_mut().record_batch(&module_renames);
        }

        Ok(changed)
    }
}

/// A renameable identifier by arena slot; args and locals carry a
/// [`FuncRef`] so the same handle in two functions never collides.
#[derive(Clone, Copy)]
enum Target {
    Constant(naga::Handle<naga::Constant>),
    Override(naga::Handle<naga::Override>),
    Global(naga::Handle<naga::GlobalVariable>),
    Function(naga::Handle<naga::Function>),
    Arg(FuncRef, usize),
    Local(FuncRef, naga::Handle<naga::LocalVariable>),
}

/// A function body, for argument / local scoping.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum FuncRef {
    Function(naga::Handle<naga::Function>),
    Entry(usize),
}

/// Names assigned this sweep, bucketed by arena so application is one
/// `iter_mut` per arena.
#[derive(Default)]
struct AssignedNames {
    constant: HandleMap<naga::Constant, String>,
    over: HandleMap<naga::Override, String>,
    global: HandleMap<naga::GlobalVariable, String>,
    function: HandleMap<naga::Function, String>,
    arg: FxHashMap<(FuncRef, usize), String>,
    local: FxHashMap<(FuncRef, naga::Handle<naga::LocalVariable>), String>,
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

/// Occurrence weights; a missing key is a non-renameable handle and counts
/// as 1.
#[derive(Default)]
struct Weights {
    global: HandleMap<naga::GlobalVariable, usize>,
    constant: HandleMap<naga::Constant, usize>,
    over: HandleMap<naga::Override, usize>,
    function: HandleMap<naga::Function, usize>,
    arg: FxHashMap<(FuncRef, usize), usize>,
    local: FxHashMap<(FuncRef, naga::Handle<naga::LocalVariable>), usize>,
}

impl Weights {
    fn of(&self, target: Target) -> usize {
        match target {
            Target::Constant(h) => self.constant.get(h).copied(),
            Target::Override(h) => self.over.get(h).copied(),
            Target::Global(h) => self.global.get(h).copied(),
            Target::Function(h) => self.function.get(h).copied(),
            Target::Arg(f, i) => self.arg.get(&(f, i)).copied(),
            Target::Local(f, h) => self.local.get(&(f, h)).copied(),
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
/// initializers are not counted): an imperfect weight only yields a longer
/// name, never a collision, which the all-distinct draw alone guarantees.
fn compute_weights(module: &naga::Module) -> Weights {
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

fn accumulate_function_weights(w: &mut Weights, fref: FuncRef, function: &naga::Function) {
    for i in 0..function.arguments.len() {
        *w.arg.entry((fref, i)).or_insert(0) += 1;
    }
    for (lh, _) in function.local_variables.iter() {
        *w.local.entry((fref, lh)).or_insert(0) += 1;
    }

    let (counts, _live) = live_expression_ref_counts(function);
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

/// Call counts, so a frequently-called function earns a shorter name.
fn count_calls(block: &naga::Block, calls: &mut HandleMap<naga::Function, usize>) {
    for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Call { function, .. } = stmt {
            *calls.entry(*function).or_insert(0) += 1;
        }
    });
}

/// The names one rename sweep may not mint.  With `mangle = false`,
/// constant and override names stay verbatim and are reserved; with
/// `mangle = true` they are rewritten, and reserving the previous sweep's
/// assignments would shift every later assignment one slot into a two-sweep
/// oscillation that never converges.  Every preserve-listed name is
/// reserved unconditionally, including names absent from every arena (a
/// preamble binding a prior pass pruned): the scans see only surviving
/// names, and a re-minted preamble name would make the generator suppress
/// that body declaration as preamble-owned and rebind every reference to the
/// host's binding.
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

    // Source struct type and member names, regardless of `mangle`: without
    // mangling the generator emits them verbatim, so minting one for a
    // global / function / local puts two same-named symbols in the output
    // (round-trip validation catches it, but compaction silently halves);
    // with mangling the generator re-mangles them itself, and reserving
    // costs only a few short names.
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
        let ctx = PassContext {
            config: &config,
            name_log: None,
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
        // Reserved ONLY for coop modules, so the rest of the corpus keeps the
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
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };
        let changed2 = pass
            .run(&mut module2, &ctx)
            .expect("second rename should work");
        assert!(
            !changed2,
            "second rename with mangle should be idempotent (no changes)"
        );

        let names1 = collect_declaration_names(&module1);
        let names2 = collect_declaration_names(&module2);
        assert_eq!(names1, names2, "names must be identical across runs");
    }

    #[test]
    fn clears_named_expressions_and_reports_change_even_when_nothing_renamed() {
        // Gating the clear on `changed` makes a preserve-all pass exit the
        // convergence loop one sweep early, before downstream DCE can remove
        // orphaned globals (seen on the Tint corpus).
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

    /// Under `mangle = true` the generator re-mangles type / member names
    /// itself, but reserving them here is safe and forecloses collisions a
    /// future refactor could expose.
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
