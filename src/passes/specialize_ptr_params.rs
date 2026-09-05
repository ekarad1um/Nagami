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
//! call sites stay and re-validation falls into the bailout.
//!
//! naga's uniformity analyzer PANICS (index out of bounds) on a `Call` to
//! a higher-indexed function and clones append at the arena end, so clones
//! are built bottom-up and `restore_call_order` re-sorts the arena behind
//! surviving callers.
//!
//! Runs once from [`crate::run`], only after validation failed.

use std::collections::HashMap;

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
    banned: &HashMap<naga::Handle<naga::Function>, Vec<u32>>,
) -> Option<SpecKey> {
    let positions = banned.get(&callee)?;
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

fn collect_keys_in_block(
    block: &naga::Block,
    caller_exprs: &naga::Arena<naga::Expression>,
    banned: &HashMap<naga::Handle<naga::Function>, Vec<u32>>,
    out: &mut Vec<SpecKey>,
) {
    for stmt in block.iter() {
        if let naga::Statement::Call {
            function,
            arguments,
            ..
        } = stmt
            && let Some(key) = call_spec_key(*function, arguments, caller_exprs, banned)
        {
            out.push(key);
        }
        for nested in super::expr_util::nested_blocks(stmt) {
            collect_keys_in_block(nested, caller_exprs, banned, out);
        }
    }
}

/// Retarget matching calls to their clones; `CallResult` expressions embed
/// the callee handle and are queued for rewriting once the arena is
/// mutable again.
fn rewrite_calls_in_block(
    block: &mut naga::Block,
    caller_exprs: &naga::Arena<naga::Expression>,
    banned: &HashMap<naga::Handle<naga::Function>, Vec<u32>>,
    clones: &HashMap<SpecKey, naga::Handle<naga::Function>>,
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
    banned: &HashMap<naga::Handle<naga::Function>, Vec<u32>>,
    clones: &mut HashMap<SpecKey, naga::Handle<naga::Function>>,
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
    // The first clone takes the callee's own name: by the end of the pass
    // the original is compacted away or the whole repair is discarded, so
    // the name never ships twice and `--preserve-symbol` / the name map
    // keep the user's name.  Later clones get `_sp<n>` checked against
    // every module-scope name: a user declaration named like a suffix
    // survives, and under preserve-symbols the rename pass would not erase
    // the collision.
    let base = source.name.as_deref().unwrap_or("");
    let first_of_callee = !base.is_empty() && !clones.keys().any(|k| k.0 == key.0);
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
    let positions: Vec<usize> = key.1.iter().map(|&(p, _)| p as usize).collect();
    let root_of: HashMap<usize, naga::Handle<naga::GlobalVariable>> =
        key.1.iter().map(|&(p, gv)| (p as usize, gv)).collect();

    // Both slots are non-emitted pointer expressions, so no Emit-range
    // surgery; surviving argument indices shift down as in dead_param.
    let mut substituted: Vec<naga::Handle<naga::Expression>> = Vec::new();
    for (h, expr) in clone.expressions.iter_mut() {
        if let naga::Expression::FunctionArgument(arg_idx) = expr {
            let old = *arg_idx as usize;
            if let Some(&gv) = root_of.get(&old) {
                *expr = naga::Expression::GlobalVariable(gv);
                substituted.push(h);
            } else {
                let shift = positions.iter().filter(|&&i| i < old).count();
                *arg_idx = (old - shift) as u32;
            }
        }
    }
    // A stale `named_expressions` entry would label the global with the
    // dead parameter's name.
    for h in substituted {
        clone.named_expressions.shift_remove(&h);
    }
    for &p in positions.iter().rev() {
        clone.arguments.remove(p);
    }

    // Substitution can turn the clone's own forwarded calls into whole-var
    // matches; resolve them first so sub-clones append before this clone.
    let mut sub_keys: Vec<SpecKey> = Vec::new();
    collect_keys_in_block(&clone.body, &clone.expressions, banned, &mut sub_keys);
    sub_keys.sort();
    sub_keys.dedup();
    for sub in &sub_keys {
        ensure_clone(module, sub, banned, clones, used_names, depth + 1)?;
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
/// analyzer panics on.
fn restore_call_order(module: &mut naga::Module) {
    fn callees_of(func: &naga::Function, out: &mut Vec<naga::Handle<naga::Function>>) {
        fn walk(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Function>>) {
            for stmt in block.iter() {
                if let naga::Statement::Call { function, .. } = stmt {
                    out.push(*function);
                }
                for nested in crate::passes::expr_util::nested_blocks(stmt) {
                    walk(nested, out);
                }
            }
        }
        walk(&func.body, out);
    }

    fn emit(
        h: naga::Handle<naga::Function>,
        old: &naga::Arena<naga::Function>,
        map: &mut HashMap<naga::Handle<naga::Function>, naga::Handle<naga::Function>>,
        rebuilt: &mut naga::Arena<naga::Function>,
    ) {
        if map.contains_key(&h) {
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
        map: &HashMap<naga::Handle<naga::Function>, naga::Handle<naga::Function>>,
    ) {
        fn walk(
            block: &mut naga::Block,
            map: &HashMap<naga::Handle<naga::Function>, naga::Handle<naga::Function>>,
        ) {
            for stmt in block.iter_mut() {
                if let naga::Statement::Call { function, .. } = stmt {
                    *function = map[function];
                }
                for nested in crate::passes::expr_util::nested_blocks_mut(stmt) {
                    walk(nested, map);
                }
            }
        }
        walk(&mut func.body, map);
        for (_, expr) in func.expressions.iter_mut() {
            if let naga::Expression::CallResult(f) = expr {
                *f = map[f];
            }
        }
    }

    let old = std::mem::replace(&mut module.functions, naga::Arena::new());
    let mut map = HashMap::new();
    let mut rebuilt = naga::Arena::new();
    for (h, _) in old.iter() {
        emit(h, &old, &mut map, &mut rebuilt);
    }
    module.functions = rebuilt;
    for entry in module.entry_points.iter_mut() {
        remap_calls(&mut entry.function, &map);
    }
}

/// `true` when the module changed; the caller re-validates.  Declines for
/// a module without entry points (compaction is entry-point-rooted, so
/// offending library functions would survive) and for a banned function
/// named in `frozen_fn_names` (preamble-owned text re-ships verbatim, and
/// the preamble emit path hard-errors where a bailout is due).
pub fn specialize_ptr_params(
    module: &mut naga::Module,
    frozen_fn_names: &std::collections::HashSet<String>,
) -> bool {
    if module.entry_points.is_empty() {
        return false;
    }
    let banned: HashMap<naga::Handle<naga::Function>, Vec<u32>> = module
        .functions
        .iter()
        .filter_map(|(h, f)| {
            let positions = banned_positions(f, &module.types);
            (!positions.is_empty()).then_some((h, positions))
        })
        .collect();
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

    // Phase A: scan pre-existing callers.  Forwarding chains are invisible
    // here (still `FunctionArgument`s); `ensure_clone` finds them after
    // substitution.
    let mut needed: Vec<SpecKey> = Vec::new();
    for (_, func) in module.functions.iter() {
        collect_keys_in_block(&func.body, &func.expressions, &banned, &mut needed);
    }
    for entry in module.entry_points.iter() {
        collect_keys_in_block(
            &entry.function.body,
            &entry.function.expressions,
            &banned,
            &mut needed,
        );
    }
    needed.sort();
    needed.dedup();

    // Phase B: clones, bottom-up.  The collision set holds every
    // module-scope name.
    let mut used_names: std::collections::HashSet<String> = module
        .functions
        .iter()
        .filter_map(|(_, f)| f.name.clone())
        .chain(
            module
                .global_variables
                .iter()
                .filter_map(|(_, g)| g.name.clone()),
        )
        .chain(module.constants.iter().filter_map(|(_, c)| c.name.clone()))
        .chain(module.overrides.iter().filter_map(|(_, o)| o.name.clone()))
        .chain(module.types.iter().filter_map(|(_, t)| t.name.clone()))
        .chain(module.entry_points.iter().map(|e| e.name.clone()))
        .collect();
    let mut clones: HashMap<SpecKey, naga::Handle<naga::Function>> = HashMap::new();
    for key in &needed {
        ensure_clone(module, key, &banned, &mut clones, &mut used_names, 0);
    }

    // Phase C: retarget pre-existing callers; clones never match `banned`
    // (keyed by pre-existing handles).  Body and arena are disjoint fields,
    // so the walk reads the arena while `CallResult` rewrites wait.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> naga::Module {
        crate::io::parse_wgsl(source).expect("test WGSL parses")
    }

    fn assert_recovered(source: &str) -> naga::Module {
        let mut module = parse(source);
        assert!(
            crate::io::validate_module(&module).is_err(),
            "fixture must start invalid or the pass is untested"
        );
        assert!(specialize_ptr_params(&mut module, &Default::default()));
        crate::io::validate_module(&module).expect("specialized module validates");
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
    fn element_root_declines_and_leaves_module_invalid() {
        let mut module = parse(
            "var<workgroup> arr: array<f32, 8>;\n\
             fn setf(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
             @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
               setf(&arr[l.x]);\n\
             }",
        );
        specialize_ptr_params(&mut module, &Default::default());
        assert!(
            crate::io::validate_module(&module).is_err(),
            "an element-rooted pointer cannot be whole-var specialized; the \
             caller must fall through to the bailout"
        );
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
        // Preamble-owned text re-ships verbatim, so no module repair helps
        // and the emit path would hard-error; the pass must decline.
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
