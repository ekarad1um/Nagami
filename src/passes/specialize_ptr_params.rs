//! Pre-validation repair for pointer parameters naga rejects: WGSL core
//! (unrestricted_pointer_parameters) allows parameters pointing into
//! workgroup / storage / uniform space and tint accepts them; naga's
//! validator limits parameters to function / private.
//!
//! Every call site passing a WHOLE global (`Expression::GlobalVariable`)
//! is retargeted to a per-root clone of the callee with the parameter
//! deleted and its uses reading the global directly - tint's own
//! direct-variable-access lowering.  Forwarded pointers resolve
//! recursively at clone-build time.  Element-rooted pointers (`f(&arr[i])`)
//! carry a call-site-dependent access chain a clone cannot express; those
//! call sites stay on the original.
//!
//! naga's uniformity analyzer PANICS (index out of bounds) on a `Call` to
//! a higher-indexed function and clones append at the arena end, so clones
//! are built bottom-up and `restore_call_order` re-sorts the arena behind
//! surviving callers.
//!
//! Runs once from [`crate::run`] ahead of validation.  What it cannot
//! remove - a library helper with no caller, a member- or element-rooted
//! argument - stays a pointer parameter and validates through
//! [`validation_stand_in`] instead.

use super::expr_util::for_each_function_mut;
use crate::handle_set::{HandleMap, HandleSet};
use rustc_hash::FxHashMap;

/// Parameter positions naga's validator rejects; mirrors `valid::function`
/// so trigger and rejection cannot drift.
fn banned_positions(func: &naga::Function, types: &naga::UniqueArena<naga::Type>) -> Vec<u32> {
    func.arguments
        .iter()
        .enumerate()
        .filter_map(|(i, arg)| match types[arg.ty].inner.pointer_space() {
            Some(naga::AddressSpace::Private | naga::AddressSpace::Function) | None => None,
            Some(_) => Some(i as u32),
        })
        .collect()
}

/// Every function with banned positions; empty for a naga-valid module.
fn banned_functions(module: &naga::Module) -> HandleMap<naga::Function, Vec<u32>> {
    module
        .functions
        .iter()
        .filter_map(|(h, f)| {
            let positions = banned_positions(f, &module.types);
            (!positions.is_empty()).then_some((h, positions))
        })
        .collect()
}

/// Rewrite `func` so the parameters at `roots`' ascending positions read
/// the given globals instead: `FunctionArgument` uses become
/// `GlobalVariable`, later argument indices shift down, the arguments
/// disappear.  Both slots are non-emitted pointer expressions, so no
/// Emit-range surgery; a stale `named_expressions` entry would label the
/// global with the dead parameter's name.
fn retarget_params_to_globals(
    func: &mut naga::Function,
    roots: &[(u32, naga::Handle<naga::GlobalVariable>)],
) {
    let mut substituted = Vec::new();
    for (h, expr) in func.expressions.iter_mut() {
        if let naga::Expression::FunctionArgument(arg_idx) = expr {
            let old = *arg_idx;
            if let Some(&(_, gv)) = roots.iter().find(|&&(p, _)| p == old) {
                *expr = naga::Expression::GlobalVariable(gv);
                substituted.push(h);
            } else {
                *arg_idx = old - roots.iter().filter(|&&(p, _)| p < old).count() as u32;
            }
        }
    }
    for h in substituted {
        func.named_expressions.shift_remove(&h);
    }
    for &(p, _) in roots.iter().rev() {
        func.arguments.remove(p as usize);
    }
}

/// A banned callee plus, per banned position, its whole-variable root;
/// positions ascend, so equal call shapes share one clone.
type SpecKey = (
    naga::Handle<naga::Function>,
    Vec<(u32, naga::Handle<naga::GlobalVariable>)>,
);

/// `None` unless EVERY banned argument is a `GlobalVariable` expression:
/// specializing only some positions would leave a still-invalid signature.
fn call_spec_key(
    callee: naga::Handle<naga::Function>,
    arguments: &[naga::Handle<naga::Expression>],
    caller_exprs: &naga::Arena<naga::Expression>,
    banned: &HandleMap<naga::Function, Vec<u32>>,
) -> Option<SpecKey> {
    let positions = banned.get(callee)?;
    let mut roots = Vec::with_capacity(positions.len());
    for &p in positions {
        let arg = *arguments.get(p as usize)?;
        match caller_exprs[arg] {
            naga::Expression::GlobalVariable(gv) => roots.push((p, gv)),
            _ => return None,
        }
    }
    Some((callee, roots))
}

/// Collects the clones `block` needs, and in `declined` the banned callees it
/// still calls directly: those keep a live caller, so the original survives
/// compaction and must not lend its name to a clone.
fn collect_keys_in_block(
    block: &naga::Block,
    caller_exprs: &naga::Arena<naga::Expression>,
    banned: &HandleMap<naga::Function, Vec<u32>>,
    out: &mut Vec<SpecKey>,
    declined: &mut HandleSet<naga::Function>,
) {
    super::expr_util::for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Call {
            function,
            arguments,
            ..
        } = stmt
            && banned.contains_key(function)
        {
            match call_spec_key(*function, arguments, caller_exprs, banned) {
                Some(key) => out.push(key),
                None => {
                    declined.insert(*function);
                }
            }
        }
    });
}

/// Retarget matching calls to their clones; `CallResult` expressions embed
/// the callee handle and are queued for rewriting once the arena is
/// mutable again.
fn rewrite_calls_in_block(
    block: &mut naga::Block,
    caller_exprs: &naga::Arena<naga::Expression>,
    banned: &HandleMap<naga::Function, Vec<u32>>,
    clones: &FxHashMap<SpecKey, naga::Handle<naga::Function>>,
    result_retargets: &mut Vec<(naga::Handle<naga::Expression>, naga::Handle<naga::Function>)>,
) -> bool {
    let mut changed = false;
    for stmt in block.iter_mut() {
        if let naga::Statement::Call {
            function,
            arguments,
            result,
        } = stmt
            && let Some(key) = call_spec_key(*function, arguments, caller_exprs, banned)
            && let Some(&clone) = clones.get(&key)
        {
            *function = clone;
            for &(p, _) in key.1.iter().rev() {
                arguments.remove(p as usize);
            }
            if let Some(r) = result {
                result_retargets.push((*r, clone));
            }
            changed = true;
        }
        for nested in super::expr_util::nested_blocks_mut(stmt) {
            changed |=
                rewrite_calls_in_block(nested, caller_exprs, banned, clones, result_retargets);
        }
    }
    changed
}

/// Create or reuse the clone for `key`, building the clones IT needs first
/// so clone-to-clone calls point backwards.  `None` only past the depth cap
/// (unreachable without recursion); a declined call stays on the original,
/// whose failure lands in the bailout.
fn ensure_clone(
    module: &mut naga::Module,
    key: &SpecKey,
    banned: &HandleMap<naga::Function, Vec<u32>>,
    declined: &HandleSet<naga::Function>,
    clones: &mut FxHashMap<SpecKey, naga::Handle<naga::Function>>,
    used_names: &mut std::collections::HashSet<String>,
    depth: usize,
) -> Option<naga::Handle<naga::Function>> {
    if let Some(&h) = clones.get(key) {
        return Some(h);
    }
    if depth > 64 {
        return None;
    }

    let source = &module.functions[key.0];
    let mut clone = source.clone();
    // The first clone takes the callee's own name so `--preserve-symbol` and
    // the name map keep the user's, but only when every call site is being
    // specialized: a declining one (member / element root, or a forward from
    // a function that keeps its own banned parameter) keeps the original
    // alive past compaction, and two functions of one name reach the
    // generator.  Later clones get `_sp<n>` checked against every
    // module-scope name: a user declaration named like a suffix survives,
    // and under preserve-symbols the rename pass would not erase the
    // collision.
    let base = source.name.as_deref().unwrap_or("");
    let first_of_callee =
        !base.is_empty() && !declined.contains(key.0) && !clones.keys().any(|k| k.0 == key.0);
    let name = if first_of_callee {
        base.to_string()
    } else {
        let mut n = clones.len() + 1;
        loop {
            let candidate = format!("{base}_sp{n}");
            if !used_names.contains(&candidate) {
                break candidate;
            }
            n += 1;
        }
    };
    used_names.insert(name.clone());
    clone.name = Some(name);
    retarget_params_to_globals(&mut clone, &key.1);

    // Substitution can turn the clone's own forwarded calls into whole-var
    // matches; resolve them first so sub-clones append before this clone.
    let mut sub_keys: Vec<SpecKey> = Vec::new();
    // A clone's body is the original's with pointer parameters resolved, so
    // its declines are a subset of the pre-scan's; this one is discarded.
    let mut sub_declined = HandleSet::default();
    collect_keys_in_block(
        &clone.body,
        &clone.expressions,
        banned,
        &mut sub_keys,
        &mut sub_declined,
    );
    sub_keys.sort();
    sub_keys.dedup();
    for sub in &sub_keys {
        ensure_clone(module, sub, banned, declined, clones, used_names, depth + 1)?;
    }
    let naga::Function {
        body, expressions, ..
    } = &mut clone;
    let mut retargets = Vec::new();
    rewrite_calls_in_block(body, expressions, banned, clones, &mut retargets);
    for (h, f) in retargets {
        expressions[h] = naga::Expression::CallResult(f);
    }

    let handle = module.functions.append(clone, naga::Span::UNDEFINED);
    clones.insert(key.clone(), handle);
    Some(handle)
}

/// Rebuild the arena callee-before-caller (stable otherwise) and remap
/// every `Call` / `CallResult`, entry points included: a surviving caller
/// retargeted onto an appended clone is exactly the forward reference the
/// analyzer panics on.  The index lookups are total: every function is
/// numbered before any call is remapped.
fn restore_call_order(module: &mut naga::Module) {
    fn callees_of(func: &naga::Function, out: &mut Vec<naga::Handle<naga::Function>>) {
        crate::passes::expr_util::for_each_statement(&func.body, &mut |stmt| {
            if let naga::Statement::Call { function, .. } = stmt {
                out.push(*function);
            }
        });
    }

    fn emit(
        h: naga::Handle<naga::Function>,
        old: &naga::Arena<naga::Function>,
        map: &mut HandleMap<naga::Function, naga::Handle<naga::Function>>,
        rebuilt: &mut naga::Arena<naga::Function>,
    ) {
        if map.contains_key(h) {
            return;
        }
        let mut callees = Vec::new();
        callees_of(&old[h], &mut callees);
        for callee in callees {
            emit(callee, old, map, rebuilt);
        }
        let mut func = old[h].clone();
        remap_calls(&mut func, map);
        let new_handle = rebuilt.append(func, old.get_span(h));
        map.insert(h, new_handle);
    }

    fn remap_calls(
        func: &mut naga::Function,
        map: &HandleMap<naga::Function, naga::Handle<naga::Function>>,
    ) {
        fn walk(
            block: &mut naga::Block,
            map: &HandleMap<naga::Function, naga::Handle<naga::Function>>,
        ) {
            for stmt in block.iter_mut() {
                if let naga::Statement::Call { function, .. } = stmt {
                    *function = map[*function];
                }
                for nested in crate::passes::expr_util::nested_blocks_mut(stmt) {
                    walk(nested, map);
                }
            }
        }
        walk(&mut func.body, map);
        for (_, expr) in func.expressions.iter_mut() {
            if let naga::Expression::CallResult(f) = expr {
                *f = map[*f];
            }
        }
    }

    let old = std::mem::replace(&mut module.functions, naga::Arena::new());
    let mut map = Default::default();
    let mut rebuilt = naga::Arena::new();
    for (h, _) in old.iter() {
        emit(h, &old, &mut map, &mut rebuilt);
    }
    module.functions = rebuilt;
    for entry in module.entry_points.iter_mut() {
        remap_calls(&mut entry.function, &map);
    }
}

/// `true` when the module changed; the caller validates afterwards.
/// Declines for a module without entry points (compaction is
/// entry-point-rooted, so offending library functions would survive) and
/// for a banned function named in `frozen_fn_names`: a preserved signature
/// is an external contract, and preamble-owned text re-ships verbatim, so
/// dropping the parameter would break the caller that passes it.
pub fn specialize_ptr_params(
    module: &mut naga::Module,
    frozen_fn_names: &std::collections::HashSet<String>,
) -> bool {
    if module.entry_points.is_empty() {
        return false;
    }
    let banned = banned_functions(module);
    if banned.is_empty() {
        return false;
    }
    if banned.keys().any(|&h| {
        module.functions[h]
            .name
            .as_deref()
            .is_some_and(|n| frozen_fn_names.contains(n))
    }) {
        return false;
    }

    // Pre-existing callers only: forwarding chains are still
    // `FunctionArgument`s here and surface after substitution.
    let mut needed: Vec<SpecKey> = Vec::new();
    let mut declined: HandleSet<naga::Function> = Default::default();
    for (_, func) in module.functions.iter() {
        collect_keys_in_block(
            &func.body,
            &func.expressions,
            &banned,
            &mut needed,
            &mut declined,
        );
    }
    for entry in module.entry_points.iter() {
        collect_keys_in_block(
            &entry.function.body,
            &entry.function.expressions,
            &banned,
            &mut needed,
            &mut declined,
        );
    }
    needed.sort();
    needed.dedup();

    // The collision set holds every module-scope name.
    let mut used_names: std::collections::HashSet<String> =
        crate::name_gen::module_scope_names(module)
            .chain(crate::name_gen::type_names(module))
            .map(str::to_owned)
            .collect();
    let mut clones: FxHashMap<SpecKey, naga::Handle<naga::Function>> = Default::default();
    for key in &needed {
        ensure_clone(
            module,
            key,
            &banned,
            &declined,
            &mut clones,
            &mut used_names,
            0,
        );
    }

    // Clones never match `banned` (keyed by pre-existing handles).  Body and
    // arena are disjoint fields, so the walk reads the arena while
    // `CallResult` rewrites wait.
    let mut rewrote = false;
    for (_, func) in module.functions.iter_mut() {
        let naga::Function {
            body, expressions, ..
        } = func;
        let mut retargets = Vec::new();
        rewrote |= rewrite_calls_in_block(body, expressions, &banned, &clones, &mut retargets);
        for (h, f) in retargets {
            expressions[h] = naga::Expression::CallResult(f);
        }
    }
    for entry in module.entry_points.iter_mut() {
        let naga::Function {
            body, expressions, ..
        } = &mut entry.function;
        let mut retargets = Vec::new();
        rewrote |= rewrite_calls_in_block(body, expressions, &banned, &clones, &mut retargets);
        for (h, f) in retargets {
            expressions[h] = naga::Expression::CallResult(f);
        }
    }

    // Retargeting strands the originals and an uncalled banned function is
    // equally fatal, so always compact (entry-point reachability, the sweep
    // DCE's rule); order restoration must precede validation.
    if rewrote {
        restore_call_order(module);
    }
    let functions_before = module.functions.len();
    naga::compact::compact(module, naga::compact::KeepUnused::No);
    rewrote || module.functions.len() != functions_before
}

/// `true` when `arg` is pointer-shaped - a global, a pointer parameter, or
/// an access chain rooted in one.  Anything else at a banned position is a
/// value naga's call check must still see.
fn is_pointer_argument(
    mut arg: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    own_arguments: &[naga::FunctionArgument],
    types: &naga::UniqueArena<naga::Type>,
) -> bool {
    loop {
        match expressions[arg] {
            naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                arg = base;
            }
            naga::Expression::GlobalVariable(_) => return true,
            naga::Expression::FunctionArgument(i) => {
                return own_arguments
                    .get(i as usize)
                    .is_some_and(|a| types[a.ty].inner.pointer_space().is_some());
            }
            _ => return false,
        }
    }
}

/// Drop the pointer arguments every call to a banned callee no longer
/// declares.
fn drop_pointer_arguments(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    own_arguments: &[naga::FunctionArgument],
    types: &naga::UniqueArena<naga::Type>,
    banned: &HandleMap<naga::Function, Vec<u32>>,
) {
    for stmt in block.iter_mut() {
        if let naga::Statement::Call {
            function,
            arguments,
            ..
        } = stmt
            && let Some(positions) = banned.get(function)
        {
            for &p in positions.iter().rev() {
                if arguments
                    .get(p as usize)
                    .is_some_and(|&a| is_pointer_argument(a, expressions, own_arguments, types))
                {
                    arguments.remove(p as usize);
                }
            }
        }
        for nested in super::expr_util::nested_blocks_mut(stmt) {
            drop_pointer_arguments(nested, expressions, own_arguments, types, banned);
        }
    }
}

/// `Some(clone)` when a function keeps a parameter [`specialize_ptr_params`]
/// could not remove, so `crate::io::validate_module` has something naga
/// accepts to validate instead.  Each banned parameter becomes a synthetic
/// module-scope variable of the pointee type in the pointer's own address
/// space (with a resource binding no real variable holds where the space
/// needs one); the body then validates under exactly the load / store /
/// atomic / `workgroupUniformLoad` rules the pointer carried, and calls
/// drop the arguments the callee no longer declares.  Every handle of
/// `module` survives, so the `ModuleInfo` naga returns indexes the real
/// module, the rewritten argument resolving to the same pointer type by
/// value.  The passes then run on the real module: sound, because WGSL
/// forbids a written pointer argument from aliasing another argument or any
/// variable the callee reaches, so a parameter is a root of its own.
/// Costs one argument scan per validation, plus a module clone only for
/// this shape.  Residual: an entry point that reaches a `ptr<immediate>`
/// helper beside a real immediate variable trips naga's one-immediate rule
/// and bails out.
pub fn validation_stand_in(module: &naga::Module) -> Option<naga::Module> {
    let banned = banned_functions(module);
    if banned.is_empty() {
        return None;
    }
    let mut clone = module.clone();
    let mut bindings = 0u32;
    for (fh, func) in module.functions.iter() {
        let Some(positions) = banned.get(fh) else {
            continue;
        };
        let mut roots = Vec::with_capacity(positions.len());
        for &p in positions {
            let arg = &func.arguments[p as usize];
            // `ValuePointer` is the only other pointer shape and no WGSL
            // front end declares one for an argument; naga's own rejection
            // stands if that ever changes.
            let naga::TypeInner::Pointer { base, space } = module.types[arg.ty].inner else {
                return None;
            };
            let binding = matches!(
                space,
                naga::AddressSpace::Storage { .. }
                    | naga::AddressSpace::Uniform
                    | naga::AddressSpace::Handle
            )
            .then(|| {
                bindings += 1;
                naga::ResourceBinding {
                    group: u32::MAX,
                    binding: bindings,
                }
            });
            let gv = clone.global_variables.append(
                naga::GlobalVariable {
                    name: arg.name.clone(),
                    space,
                    binding,
                    ty: base,
                    init: None,
                    memory_decorations: Default::default(),
                },
                naga::Span::UNDEFINED,
            );
            roots.push((p, gv));
        }
        retarget_params_to_globals(&mut clone.functions[fh], &roots);
    }
    for_each_function_mut(
        &mut clone.functions,
        &mut clone.entry_points,
        &mut |caller| {
            let naga::Function {
                body,
                expressions,
                arguments,
                ..
            } = caller;
            drop_pointer_arguments(body, expressions, arguments, &module.types, &banned);
        },
    );
    // Retargeting removed every banned position, so validation sees no
    // pointer parameter and the caller never needs a second stand-in.
    debug_assert!(banned_functions(&clone).is_empty());
    Some(clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> naga::Module {
        crate::io::parse_wgsl(source).expect("test WGSL parses")
    }

    /// naga's own verdict, bypassing the stand-in `validate_module` applies.
    fn naga_rejects(module: &naga::Module) -> bool {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(module)
        .is_err()
    }

    fn assert_recovered(source: &str) -> naga::Module {
        let mut module = parse(source);
        assert!(
            naga_rejects(&module),
            "fixture must start invalid or the pass is untested"
        );
        assert!(specialize_ptr_params(&mut module, &Default::default()));
        assert!(
            !naga_rejects(&module),
            "specialized module validates without the stand-in"
        );
        module
    }

    #[test]
    fn whole_var_root_is_recovered() {
        let module = assert_recovered(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               touch(&sh, l.x);\n\
             }",
        );
        // One clone, original compacted away, no pointer parameters left.
        assert_eq!(module.functions.len(), 1);
        for (_, f) in module.functions.iter() {
            assert!(banned_positions(f, &module.types).is_empty());
        }
    }

    #[test]
    fn two_roots_share_nothing_but_get_one_clone_each() {
        let module = assert_recovered(
            "var<workgroup> a: array<f32, 8>;\n\
             var<workgroup> b: array<f32, 8>;\n\
             fn zero(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 0.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               zero(&a, l.x);\n\
               zero(&b, l.x);\n\
               zero(&a, l.x);\n\
             }",
        );
        // Two roots -> two clones; the third call reuses the `a` clone.
        assert_eq!(module.functions.len(), 2);
    }

    #[test]
    fn forwarded_pointer_resolves_transitively() {
        let module = assert_recovered(
            "var<workgroup> acc: array<f32, 8>;\n\
             fn inner(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             fn outer(p: ptr<workgroup, array<f32, 8>>, i: u32) { inner(p, i); (*p)[i] += 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               outer(&acc, l.x);\n\
             }",
        );
        assert_eq!(module.functions.len(), 2);
        for (_, f) in module.functions.iter() {
            assert!(banned_positions(f, &module.types).is_empty());
        }
    }

    #[test]
    fn value_returning_callee_retargets_call_result() {
        let module = assert_recovered(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn read(p: ptr<workgroup, array<f32, 8>>, i: u32) -> f32 { return (*p)[i]; }\n\
             var<workgroup> out_v: f32;\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               out_v = read(&sh, l.x);\n\
             }",
        );
        // The validator already checks CallResult callees; assert anyway to
        // keep the failure local.
        for (_, f) in module.functions.iter() {
            for (_, e) in f.expressions.iter() {
                if let naga::Expression::CallResult(target) = e {
                    assert!(banned_positions(&module.functions[*target], &module.types).is_empty());
                }
            }
        }
    }

    #[test]
    fn mixed_argument_order_renumbers_survivors() {
        let module = assert_recovered(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn mixd(scale: f32, p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = scale; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               mixd(2.0, &sh, l.x);\n\
             }",
        );
        let (_, clone) = module.functions.iter().next().expect("one clone");
        assert_eq!(clone.arguments.len(), 2, "only the pointer slot is gone");
    }

    #[test]
    fn surviving_caller_function_keeps_valid_call_order() {
        // The helper has no banned parameters and survives in place while
        // its callee becomes an appended clone: the forward-reference shape.
        let module = assert_recovered(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             fn helper(i: u32) { touch(&sh, i); }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               helper(l.x);\n\
             }",
        );
        assert_eq!(module.functions.len(), 2, "helper plus the touch clone");
    }

    #[test]
    fn element_root_declines_specialization_and_validates_through_the_stand_in() {
        let mut module = parse(
            "var<workgroup> arr: array<f32, 8>;\n\
             fn setf(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               setf(&arr[l.x]);\n\
             }",
        );
        assert!(!specialize_ptr_params(&mut module, &Default::default()));
        assert!(
            naga_rejects(&module),
            "an element-rooted pointer has no whole-variable clone"
        );
        crate::io::validate_module(&module)
            .expect("the stand-in validates what specialization leaves");
    }

    #[test]
    fn a_declining_call_site_keeps_the_original_name_off_the_clone() {
        // The element-rooted call keeps `touch` alive, so the whole-variable
        // clone must not be named `touch` as well.
        let mut module = parse(
            "var<workgroup> a: array<u32, 4>;\n\
             var<workgroup> g: array<array<u32, 4>, 2>;\n\
             fn touch(p: ptr<workgroup, array<u32, 4>>, i: u32) { (*p)[i] = 1u; }\n\
             @compute @workgroup_size(4) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               touch(&a, l.x);\n\
               touch(&g[l.x & 1u], l.x);\n\
             }",
        );
        assert!(specialize_ptr_params(&mut module, &Default::default()));
        let names: Vec<_> = module
            .functions
            .iter()
            .filter_map(|(_, f)| f.name.clone())
            .collect();
        assert_eq!(names.len(), 2, "original and clone: {names:?}");
        assert!(
            names[0] != names[1],
            "a surviving original must keep its name to itself: {names:?}"
        );
        crate::io::validate_module(&module).expect("the stand-in covers the survivor");
    }

    #[test]
    fn a_preserved_signature_declines_specialization() {
        let mut module = parse(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               touch(&sh, l.x);\n\
             }",
        );
        let preserved = ["touch".to_string()].into_iter().collect();
        assert!(!specialize_ptr_params(&mut module, &preserved));
        let (_, touch) = module.functions.iter().next().expect("touch survives");
        assert_eq!(
            touch.arguments.len(),
            2,
            "the advertised arity is a contract"
        );
    }

    #[test]
    fn stand_in_is_absent_without_banned_parameters() {
        let module = parse(
            "fn touch(p: ptr<function, f32>) { *p = 1.0; }\n\
             @compute @workgroup_size(1) fn m() { var x = 0.0; touch(&x); }",
        );
        assert!(validation_stand_in(&module).is_none());
    }

    #[test]
    fn stand_in_keeps_every_handle_and_the_argument_pointer_type() {
        // A library helper: no caller anywhere.
        let module = parse(
            "fn scale(p: ptr<workgroup, array<vec2f, 4>>, i: u32) { (*p)[i] = (*p)[i] * 2.0; }",
        );
        assert!(naga_rejects(&module));
        let stand_in = validation_stand_in(&module).expect("banned parameter");
        let (fh, func) = module.functions.iter().next().unwrap();
        assert_eq!(stand_in.functions[fh].arguments.len(), 1);
        assert_eq!(
            stand_in.functions[fh].expressions.len(),
            func.expressions.len()
        );
        assert_eq!(
            stand_in.global_variables.len(),
            module.global_variables.len() + 1
        );
        let info = crate::io::validate_module(&module).expect("validates through the stand-in");
        let pointer_uses = func
            .expressions
            .iter()
            .filter(|(_, e)| matches!(e, naga::Expression::FunctionArgument(0)))
            .filter(|(h, _)| {
                matches!(
                    info[fh][*h].ty.inner_with(&module.types),
                    naga::TypeInner::Pointer {
                        space: naga::AddressSpace::WorkGroup,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            pointer_uses, 1,
            "the real module's argument resolves as its pointer type"
        );
    }

    #[test]
    fn stand_in_keeps_the_address_space_rules() {
        // A runtime-sized pointee exists in storage space only; arrayLength needs it.
        crate::io::validate_module(&parse(
            "fn bump(p: ptr<storage, array<f32>, read_write>, i: u32) {\n\
               if (i < arrayLength(p)) { (*p)[i] += 1.0; }\n\
             }",
        ))
        .expect("read_write storage pointer");
        assert!(
            crate::io::validate_module(&parse(
                "fn bump(p: ptr<storage, array<f32>>, i: u32) { (*p)[i] = 1.0; }",
            ))
            .is_err(),
            "a store through a read-only storage pointer stays an error"
        );
        crate::io::validate_module(&parse(
            "fn inc(p: ptr<workgroup, atomic<u32>>) -> u32 { atomicAdd(p, 1u); return atomicLoad(p); }",
        ))
        .expect("workgroup atomic pointer");
        crate::io::validate_module(&parse(
            "fn fetch(p: ptr<workgroup, u32>) -> u32 { return workgroupUniformLoad(p); }",
        ))
        .expect("workgroupUniformLoad through a pointer");
        crate::io::validate_module(&parse(
            "struct U { v: vec4f }\n\
             fn fetch(p: ptr<uniform, U>) -> vec4f { return (*p).v; }",
        ))
        .expect("uniform pointer");
    }

    #[test]
    fn stand_in_drops_pointer_arguments_at_every_call_shape() {
        let module = parse(
            "struct S { a: array<f32, 4>, b: f32 }\n\
             var<workgroup> s: S;\n\
             fn inner(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
             fn outer(q: ptr<workgroup, f32>, k: f32) { inner(q); *q += k; }\n\
             @compute @workgroup_size(4) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               outer(&s.a[l.x], 2.0);\n\
               inner(&s.b);\n\
             }",
        );
        assert!(naga_rejects(&module));
        crate::io::validate_module(&module).expect("member, element and forwarded roots");
        let stand_in = validation_stand_in(&module).unwrap();
        let mut arities = Vec::new();
        crate::passes::expr_util::for_each_statement(
            &stand_in.entry_points[0].function.body,
            &mut |stmt| {
                if let naga::Statement::Call { arguments, .. } = stmt {
                    arities.push(arguments.len());
                }
            },
        );
        assert_eq!(arities, [1, 0]);
    }

    #[test]
    fn stand_in_leaves_a_value_argument_for_naga_to_reject() {
        // A pass bug that puts a value where the pointer was must stay visible.
        let mut module = parse(
            "var<workgroup> s: array<f32, 4>;\n\
             fn inner(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
             @compute @workgroup_size(4) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               inner(&s[l.x]);\n\
             }",
        );
        let entry = &mut module.entry_points[0].function;
        let index_value = entry
            .expressions
            .iter()
            .find(|(_, e)| {
                matches!(e, naga::Expression::AccessIndex { base, index: 0 }
                    if matches!(entry.expressions[*base], naga::Expression::FunctionArgument(0)))
            })
            .map(|(h, _)| h)
            .expect("l.x");
        for stmt in entry.body.iter_mut() {
            if let naga::Statement::Call { arguments, .. } = stmt {
                arguments[0] = index_value;
            }
        }
        assert!(crate::io::validate_module(&module).is_err());
    }

    #[test]
    fn valid_and_library_modules_are_untouched() {
        // Pointer params in the FUNCTION space are naga-valid: no trigger.
        let mut valid = parse(
            "fn touch(p: ptr<function, f32>) { *p = 1.0; }\n\
             @compute @workgroup_size(1) fn m() { var x = 0.0; touch(&x); }",
        );
        assert!(!specialize_ptr_params(&mut valid, &Default::default()));
        // Library module: cleanup cannot root, so the pass declines.
        let mut library = parse("fn touch(p: ptr<workgroup, f32>) { *p = 1.0; }");
        assert!(!specialize_ptr_params(&mut library, &Default::default()));
        crate::io::validate_module(&library).expect("the stand-in carries the library helper");
    }
}

#[cfg(test)]
mod review_regressions {
    use super::*;

    #[test]
    fn clone_names_keep_the_original_and_avoid_existing_declarations() {
        // Two roots: the first clone inherits `touch`, the second needs a
        // suffix that must not shadow the user's `touch_sp2`.
        let mut module = crate::io::parse_wgsl(
            "var<workgroup> sh: array<f32, 8>;\n\
             var<workgroup> other: array<f32, 8>;\n\
             fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             fn touch_sp2(i: u32) -> f32 { return f32(i); }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               touch(&sh, l.x);\n\
               touch(&other, l.x);\n\
               sh[l.x] += touch_sp2(l.x);\n\
             }",
        )
        .expect("parses");
        assert!(specialize_ptr_params(&mut module, &Default::default()));
        crate::io::validate_module(&module).expect("recovered module validates");
        let mut names: Vec<&str> = module
            .functions
            .iter()
            .filter_map(|(_, f)| f.name.as_deref())
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["touch", "touch_sp2", "touch_sp3"],
            "first clone keeps the original name, the user's touch_sp2 survives untouched"
        );
    }

    #[test]
    fn frozen_function_names_decline_the_repair() {
        // Preamble-owned text re-ships verbatim, so its signature must not
        // change.
        let mut module = crate::io::parse_wgsl(
            "var<workgroup> sh: array<f32, 8>;\n\
             fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               touch(&sh, l.x);\n\
             }",
        )
        .expect("parses");
        let frozen: std::collections::HashSet<String> = ["touch".to_string()].into();
        let functions_before = module.functions.len();
        assert!(!specialize_ptr_params(&mut module, &frozen));
        assert_eq!(module.functions.len(), functions_before);
    }
}
