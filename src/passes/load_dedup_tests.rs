use super::*;
use crate::config::Config;
use crate::handle_set::{HandleMap, HandleSet};

fn run_pass(source: &str) -> (bool, naga::Module) {
    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let mut pass = LoadDedupPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        name_log: None,
    };

    let changed = pass
        .run(&mut module, &ctx)
        .expect("load dedup pass should run");
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
    (changed, module)
}

fn count_loads_from_local(function: &naga::Function) -> usize {
    function
        .expressions
        .iter()
        .filter(|(handle, expr)| {
            if let naga::Expression::Load { pointer } = expr
                && let naga::Expression::LocalVariable(_) = &function.expressions[*pointer]
            {
                return is_handle_in_any_emit(&function.body, *handle);
            }
            false
        })
        .count()
}

fn is_handle_in_any_emit(block: &naga::Block, target: naga::Handle<naga::Expression>) -> bool {
    block.iter().any(|statement| {
        if let naga::Statement::Emit(range) = statement {
            return range.clone().any(|h| h == target);
        }
        nested_blocks(statement).any(|b| is_handle_in_any_emit(b, target))
    })
}

#[test]
fn nested_load_keeps_backing_store_alive() {
    // The liveness scan must count a depth>=2 nested load (`e.a.x`) that
    // `get_pointer_key` cannot forward, or `e` is marked dead, `e = s`
    // removed, and the load reads the zero-default.
    let source = r#"
struct Inner { x: f32, y: f32 }
struct Outer { a: Inner, b: f32 }
@fragment
fn fs(@location(0) k: f32) -> @location(0) vec4f {
    var e: Outer;
    var s: Outer;
    s.a.x = k;
    s.a.y = k;
    s.b = k;
    e = s;
    let whole = e;
    let nested = e.a.x;
    return vec4f(whole.b, nested, 0.0, 1.0);
}
"#;
    let (_changed, module) = run_pass(source);
    let func = &module.entry_points[0].function;
    let e = func
        .local_variables
        .iter()
        .find(|(_, lv)| lv.name.as_deref() == Some("e"))
        .map(|(h, _)| h)
        .expect("local `e` must exist");
    assert!(
        count_stores_to_local(func, e) >= 1,
        "the `e = s` store must survive: `e` is still read via the nested `e.a.x`"
    );
}

#[test]
fn deduplicates_consecutive_loads_from_same_local() {
    let source = r#"
fn hash(p: vec2<f32>) -> f32 {
    var v: vec3<f32>;
    v = fract(vec3<f32>(p.x, p.y, p.x) * 0.1313);
    let a = v;
    let b = v;
    let c = v;
    v = a + vec3(dot(b, c.yzx + vec3(3.333)));
    let x = v.x;
    let y = v.y;
    let z = v.z;
    return fract((x + y) * z);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let h = hash(vec2(1.0, 2.0));
    return vec4f(h);
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "should detect redundant loads");

    let hash_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("hash"))
        .map(|(_, f)| f)
        .expect("hash function should exist");

    let active_loads = count_loads_from_local(hash_fn);
    assert!(
        active_loads < 3,
        "expected fewer than 3 active whole-var loads, got {}",
        active_loads
    );
}

#[test]
fn forwards_multi_store_variable_in_straight_line() {
    // Each store seeds the cache.
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    a = x + 1.0;
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "pass should forward multi-store loads");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "both loads should be forwarded (multi-store variable in straight-line code)"
    );
}

#[test]
fn forwards_single_store_copy_variable() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var tmp: f32;
    tmp = x * 2.0;
    let y = tmp + 1.0;
    return y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "pass should report changes");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "single-store variable loads should be forwarded"
    );
}

#[test]
fn handles_no_local_variables() {
    let source = r#"
fn pure_fn(x: f32) -> f32 {
    return x * 2.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(pure_fn(1.0));
}
"#;

    let (changed, _) = run_pass(source);
    assert!(!changed, "no locals means nothing to deduplicate");
}

#[test]
fn dead_store_removed_for_forwarded_variable() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var tmp: f32;
    tmp = x;
    return tmp;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "pass should report changes");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    fn count_stores(block: &naga::Block) -> usize {
        let mut count = 0;
        for stmt in block {
            if matches!(stmt, naga::Statement::Store { .. }) {
                count += 1;
            }
            for nested in nested_blocks(stmt) {
                count += count_stores(nested);
            }
        }
        count
    }
    assert_eq!(
        count_stores(&test_fn.body),
        0,
        "dead store to forwarded variable should be removed"
    );
}

#[test]
fn var_with_init_not_forwarded() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 0.0;
    a = x;
    let v = a;
    return v;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (_, module) = run_pass(source);
    // Only validity is asserted.
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
}

#[test]
fn cache_invalidated_after_if_branches() {
    let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    if c {
        a = x + 1.0;
    } else {
        a = x + 2.0;
    }
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, true));
}
"#;

    let (_, module) = run_pass(source);
    // Only validity is asserted.
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
}

#[test]
fn chained_copy_variables_forwarded() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x * 2.0;
    var b: f32;
    b = a;
    return b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "pass should report changes");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "chained copy variables should all be forwarded"
    );
}

#[test]
fn dynamic_index_store_invalidates_cache() {
    // Whole-variable loads so `count_loads_from_local` sees them.
    let source = r#"
fn test_fn(idx: i32) -> vec3<f32> {
    var v: vec3<f32>;
    v = vec3(1.0, 2.0, 3.0);
    let a = v;
    v[idx] = 99.0;
    let b = v;
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let r = test_fn(0);
    return vec4f(r, 1.0);
}
"#;

    let (_, module) = run_pass(source);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");

    // `a`'s forward is undone (`v` stays live via `b` and the Compose is
    // complex); `b` is never forwarded.
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 2,
        "both loads preserved: store forwarding reverted for non-dead variable with complex expression, got {} active loads",
        active_loads
    );
}

#[test]
fn trace_ray_payload_invalidates_cache_and_marks_escaped() {
    let mut module = naga::Module::default();

    let f32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::F32),
        },
        naga::Span::UNDEFINED,
    );

    let accel_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::AccelerationStructure {
                vertex_return: false,
            },
        },
        naga::Span::UNDEFINED,
    );

    let mut function = naga::Function::default();

    let local_payload = function.local_variables.append(
        naga::LocalVariable {
            name: Some("payload_var".into()),
            ty: f32_ty,
            init: None,
        },
        naga::Span::UNDEFINED,
    );

    let ptr_payload = function.expressions.append(
        naga::Expression::LocalVariable(local_payload),
        naga::Span::UNDEFINED,
    );
    let load1 = function.expressions.append(
        naga::Expression::Load {
            pointer: ptr_payload,
        },
        naga::Span::UNDEFINED,
    );
    let load2 = function.expressions.append(
        naga::Expression::Load {
            pointer: ptr_payload,
        },
        naga::Span::UNDEFINED,
    );
    let lit_val = function.expressions.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        naga::Span::UNDEFINED,
    );

    let accel_global = module.global_variables.append(
        naga::GlobalVariable {
            name: Some("accel".into()),
            space: naga::AddressSpace::Handle,
            binding: None,
            ty: accel_ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        naga::Span::UNDEFINED,
    );
    let accel_expr = function.expressions.append(
        naga::Expression::GlobalVariable(accel_global),
        naga::Span::UNDEFINED,
    );

    let desc_global = module.global_variables.append(
        naga::GlobalVariable {
            name: Some("desc".into()),
            space: naga::AddressSpace::Private,
            binding: None,
            ty: f32_ty,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        naga::Span::UNDEFINED,
    );
    let desc_expr = function.expressions.append(
        naga::Expression::GlobalVariable(desc_global),
        naga::Span::UNDEFINED,
    );

    // Store; Emit(load1); TraceRay(payload=&payload_var); Emit(load2): the
    // TraceRay may write the payload, so load2 must not dedup to load1.
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Store {
            pointer: ptr_payload,
            value: lit_val,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            acceleration_structure: accel_expr,
            descriptor: desc_expr,
            payload: ptr_payload,
        }),
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
        naga::Span::UNDEFINED,
    );
    function.body = body;

    let mut replacements = HandleMap::default();
    let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
    let mut all_loads = HandleMap::default();
    let mut seeded_by_store = HandleMap::default();
    let scope_idx = ExpressionScopeIndex::build(&function.body, &function.expressions);
    collect_redundant_loads(
        &function.body,
        &function.expressions,
        &scope_idx,
        &mut cache,
        &mut replacements,
        &mut all_loads,
        &mut seeded_by_store,
        false,
        &mut HandleSet::default(),
    );

    assert!(
        !replacements.contains_key(load2),
        "load after TraceRay must not be deduplicated (cache should be invalidated)"
    );

    let escaped = locals_passed_by_pointer(&function.body, &function.expressions);
    assert!(
        escaped.contains(local_payload),
        "local passed as TraceRay payload should be marked as escaped"
    );
}

#[test]
fn cooperative_store_through_data_pointer_invalidates_cache() {
    // Mirrors naga's lowering of `coopStore(matrix_value, &mat_var, stride)`:
    // `target` is the matrix VALUE and `data.pointer` the destination, so the
    // invalidation must key on `data.pointer`'s root (a CooperativeMatrix
    // `target` can never resolve to a LocalVariable).
    let mut module = naga::Module::default();

    let f32_scalar = naga::Scalar::F32;
    let coop_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::CooperativeMatrix {
                columns: naga::CooperativeSize::Sixteen,
                rows: naga::CooperativeSize::Sixteen,
                scalar: f32_scalar,
                role: naga::CooperativeRole::C,
            },
        },
        naga::Span::UNDEFINED,
    );

    let mut function = naga::Function::default();
    function.arguments.push(naga::FunctionArgument {
        name: Some("mat_in".into()),
        ty: coop_ty,
        binding: None,
    });

    let local_mat = function.local_variables.append(
        naga::LocalVariable {
            name: Some("mat_var".into()),
            ty: coop_ty,
            init: None,
        },
        naga::Span::UNDEFINED,
    );

    let ptr_mat = function.expressions.append(
        naga::Expression::LocalVariable(local_mat),
        naga::Span::UNDEFINED,
    );
    let matrix_arg = function
        .expressions
        .append(naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED);
    let load1 = function.expressions.append(
        naga::Expression::Load { pointer: ptr_mat },
        naga::Span::UNDEFINED,
    );
    let load2 = function.expressions.append(
        naga::Expression::Load { pointer: ptr_mat },
        naga::Span::UNDEFINED,
    );
    let stride = function.expressions.append(
        naga::Expression::Literal(naga::Literal::U32(16)),
        naga::Span::UNDEFINED,
    );

    // Store(&mat_var, mat_in); Emit(load1); coopStore(mat_in, &mat_var, 16);
    // Emit(load2): load2 must not dedup to load1.
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Store {
            pointer: ptr_mat,
            value: matrix_arg,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::CooperativeStore {
            target: matrix_arg,
            data: naga::CooperativeData {
                pointer: ptr_mat,
                stride,
                row_major: false,
            },
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
        naga::Span::UNDEFINED,
    );
    function.body = body;

    let mut replacements = HandleMap::default();
    let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
    let mut all_loads = HandleMap::default();
    let mut seeded_by_store = HandleMap::default();
    let scope_idx = ExpressionScopeIndex::build(&function.body, &function.expressions);
    collect_redundant_loads(
        &function.body,
        &function.expressions,
        &scope_idx,
        &mut cache,
        &mut replacements,
        &mut all_loads,
        &mut seeded_by_store,
        false,
        &mut HandleSet::default(),
    );

    assert!(
        !replacements.contains_key(load2),
        "load after CooperativeStore must not be deduplicated \
             (cache must be invalidated via data.pointer's root local)"
    );

    let mut escaped = HandleSet::default();
    let mut partially_stored = HandleSet::default();
    collect_escaped_and_partially_stored(
        &function.body,
        &function.expressions,
        &mut escaped,
        &mut partially_stored,
    );
    assert!(
        partially_stored.contains(local_mat),
        "local written by CooperativeStore must be flagged as partially_stored \
             (matrix write through data.pointer may not cover the full local)"
    );
}

#[test]
fn atomic_invalidates_cache_and_counts_as_store() {
    let mut module = naga::Module::default();

    let u32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Atomic(naga::Scalar::U32),
        },
        naga::Span::UNDEFINED,
    );

    let mut function = naga::Function::default();

    let local_var = function.local_variables.append(
        naga::LocalVariable {
            name: Some("atom_var".into()),
            ty: u32_ty,
            init: None,
        },
        naga::Span::UNDEFINED,
    );

    let ptr_var = function.expressions.append(
        naga::Expression::LocalVariable(local_var),
        naga::Span::UNDEFINED,
    );
    let load1 = function.expressions.append(
        naga::Expression::Load { pointer: ptr_var },
        naga::Span::UNDEFINED,
    );
    let load2 = function.expressions.append(
        naga::Expression::Load { pointer: ptr_var },
        naga::Span::UNDEFINED,
    );
    let lit_val = function.expressions.append(
        naga::Expression::Literal(naga::Literal::U32(1)),
        naga::Span::UNDEFINED,
    );
    let atomic_result = function.expressions.append(
        naga::Expression::AtomicResult {
            ty: u32_ty,
            comparison: false,
        },
        naga::Span::UNDEFINED,
    );

    // Store; Emit(load1); Atomic(Add); Emit(load2): load2 must not dedup to
    // load1.
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Store {
            pointer: ptr_var,
            value: lit_val,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Atomic {
            pointer: ptr_var,
            fun: naga::AtomicFunction::Add,
            value: lit_val,
            result: Some(atomic_result),
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
        naga::Span::UNDEFINED,
    );
    function.body = body;

    let store_counts = count_local_stores(&function.body, &function.expressions);
    assert_eq!(
        store_counts.get(local_var).copied().unwrap_or(0),
        2,
        "Store + Atomic should count as 2 stores"
    );

    let mut replacements = HandleMap::default();
    let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
    let mut all_loads = HandleMap::default();
    let mut seeded_by_store = HandleMap::default();
    let scope_idx = ExpressionScopeIndex::build(&function.body, &function.expressions);
    collect_redundant_loads(
        &function.body,
        &function.expressions,
        &scope_idx,
        &mut cache,
        &mut replacements,
        &mut all_loads,
        &mut seeded_by_store,
        false,
        &mut HandleSet::default(),
    );

    assert!(
        !replacements.contains_key(load2),
        "load after Atomic must not be deduplicated (cache should be invalidated)"
    );
}

#[test]
fn deduplicates_loads_through_same_dynamic_index() {
    let source = r#"
fn test_fn(idx: i32) -> f32 {
    var v: vec3<f32>;
    v = vec3(1.0, 2.0, 3.0);
    let a = v[idx];
    let b = v[idx];
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let r = test_fn(0);
    return vec4f(r, 0.0, 0.0, 1.0);
}
"#;

    let (changed, module) = run_pass(source);
    assert!(
        changed,
        "should detect redundant loads through same dynamic index"
    );

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let dynamic_load_count = test_fn
        .expressions
        .iter()
        .filter(|(handle, expr)| {
            if let naga::Expression::Load { pointer } = expr
                && let naga::Expression::Access { base, .. } = &test_fn.expressions[*pointer]
                && matches!(
                    test_fn.expressions[*base],
                    naga::Expression::LocalVariable(_)
                )
            {
                return is_handle_in_any_emit(&test_fn.body, *handle);
            }
            false
        })
        .count();

    assert!(
        dynamic_load_count < 2,
        "expected at most 1 active dynamic-indexed load, got {}",
        dynamic_load_count
    );
}

#[test]
fn init_seeded_load_forwarded_and_dead_local_removed() {
    let source = r#"
fn test_fn() -> f32 {
    var a: f32 = 0.0;
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn());
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "init-seeded load should be forwarded");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "load of init-seeded variable should be forwarded"
    );
}

#[test]
fn init_seeded_then_overwritten_uses_store_value() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 0.0;
    a = x;
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(5.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "load should be forwarded to store value");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "load after overwrite should forward to store value, not init"
    );
}

#[test]
fn forwarding_survives_loop_for_unmodified_local() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = x;
    var sum: f32 = 0.0;
    var i: i32 = 0;
    loop {
        if i >= 4 { break; }
        sum += 1.0;
        continuing {
            i += 1;
        }
    }
    return a + sum;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "load of a should be forwarded across the loop");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let a_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("a"))
        .map(|(h, _)| h)
        .expect("variable a should exist");
    assert_eq!(
        store_count.get(a_handle).copied().unwrap_or(0),
        0,
        "stores to a should be eliminated (dead local after forwarding)"
    );
}

#[test]
fn forwarding_survives_if_for_unmodified_local() {
    let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = x;
    var b: f32 = 0.0;
    if c {
        b = 1.0;
    } else {
        b = 2.0;
    }
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, true));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "load of a should be forwarded across the if");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let a_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("a"))
        .map(|(h, _)| h)
        .expect("variable a should exist");
    assert_eq!(
        store_count.get(a_handle).copied().unwrap_or(0),
        0,
        "stores to a should be eliminated (dead local after forwarding across if)"
    );
}

#[test]
fn no_forwarding_across_loop_when_modified() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = x;
    var i: i32 = 0;
    loop {
        if i >= 4 { break; }
        a += 1.0;
        continuing {
            i += 1;
        }
    }
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (_, module) = run_pass(source);

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let a_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("a"))
        .map(|(h, _)| h)
        .expect("variable a should exist");
    assert!(
        store_count.get(a_handle).copied().unwrap_or(0) > 0,
        "stores to a should NOT be eliminated when loop modifies it"
    );
}

#[test]
fn dead_store_eliminated_when_overwritten_before_load() {
    let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var p: vec3f;
    p = x;
    p = x + vec3f(1.0);
    return p.x + p.y + p.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "should detect dead store");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let p_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("p"))
        .map(|(h, _)| h)
        .expect("variable p should exist");
    assert_eq!(
        store_count.get(p_handle).copied().unwrap_or(0),
        1,
        "first dead store to p should be removed, leaving only one"
    );
}

#[test]
fn dead_store_not_eliminated_when_loaded_between_stores() {
    let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var p: vec3f;
    p = x;
    let a = p.x;
    p = x + vec3f(1.0);
    return a + p.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

    let (_, module) = run_pass(source);

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let p_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("p"))
        .map(|(h, _)| h)
        .expect("variable p should exist");
    assert_eq!(
        store_count.get(p_handle).copied().unwrap_or(0),
        2,
        "both stores to p should remain when first is loaded before second"
    );
}

#[test]
fn dead_store_chain_eliminates_all_but_last() {
    // vec3f + field access keep load-dedup from forwarding the whole-variable
    // Store, so exactly one Store stays.
    let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var a: vec3f;
    a = x;
    a = x + vec3f(1.0);
    a = x + vec3f(2.0);
    return a.x + a.y + a.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "should detect dead stores in chain");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
    let a_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("a"))
        .map(|(h, _)| h)
        .expect("variable a should exist");
    assert_eq!(
        store_count.get(a_handle).copied().unwrap_or(0),
        1,
        "only the last store in a chain should survive"
    );
}

#[test]
fn undo_phase_reverts_complex_forwarding_for_non_dead_variable() {
    // The partial store `v.x = 0` keeps `v` live via `b`, so the complex
    // (Binary) forward of `a` must be undone.
    let source = r#"
fn test_fn(x: vec3<f32>) -> f32 {
    var v: vec3<f32>;
    v = x + vec3<f32>(1.0, 2.0, 3.0);
    let a = v;
    v.x = 0.0;
    let b = v;
    return a.x + b.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(1.0)));
}
"#;

    let (_, module) = run_pass(source);
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 2,
        "undo phase should revert complex forwarding for non-dead variable, got {} active loads",
        active_loads
    );
}

#[test]
fn undo_phase_keeps_simple_forwarding_for_non_dead_variable() {
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    a = x + 1.0;
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "pass should forward loads");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    // v1 -> x is simple; v2 -> x+1 is kept because `a` ends up dead.
    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 0,
        "all loads should be forwarded (simple and dead-local cases)"
    );
}

#[test]
fn compose_init_not_seeded_for_multi_load_variable() {
    // A Compose init is not seeded, so loads dedup Load-to-Load instead of
    // duplicating the constructor.
    let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    let a = v;
    let b = v;
    return a.x + b.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn());
}
"#;

    let (changed, module) = run_pass(source);
    assert!(changed, "should deduplicate consecutive loads");
    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 1,
        "Compose init should not be forwarded; Load-to-Load dedup keeps 1 active load, got {}",
        active_loads
    );

    let v_handle = test_fn
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("v"))
        .map(|(h, _)| h)
        .expect("variable v should exist");
    assert!(
        test_fn.local_variables[v_handle].init.is_some(),
        "variable v should retain its init (not marked dead)"
    );
}

#[test]
fn chain_resolution_flattens_load_chains_after_undo() {
    // load1 -> expr is undone, so load2 -> load1 (Load-to-Load) must still
    // resolve.
    let source = r#"
fn test_fn(x: vec3<f32>) -> f32 {
    var a: vec3<f32>;
    a = x + vec3<f32>(1.0, 2.0, 3.0);
    let v1 = a;
    let v2 = a;
    a.x = 0.0;
    let v3 = a;
    return v1.x + v2.y + v3.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(1.0)));
}
"#;

    let (_, module) = run_pass(source);
    let _ = crate::io::validate_module(&module)
        .expect("module should remain valid after chain resolution");

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn function should exist");

    // v1 canonical, v2 deduplicated to v1, v3 fresh after the invalidation.
    let active_loads = count_loads_from_local(test_fn);
    assert_eq!(
        active_loads, 2,
        "chain resolution should preserve Load-to-Load dedup after undo, got {} active loads",
        active_loads
    );
}

// Dead init removal (Phase 1: zero inits, Phase 2: dead non-zero inits)

fn local_has_init(function: &naga::Function, name: &str) -> bool {
    function
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some(name))
        .map(|(_, v)| v.init.is_some())
        .unwrap_or(false)
}

#[test]
fn zero_init_f32_removed() {
    let source = r#"
fn test_fn() -> f32 {
    var a: f32 = 0.0;
    a = 1.0;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let f = module.functions.iter().next().unwrap().1;
    assert!(!local_has_init(f, "a"), "zero f32 init should be removed");
}

#[test]
fn zero_init_i32_removed() {
    let source = r#"
fn test_fn() -> i32 {
    var a: i32 = 0i;
    a = 1i;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(f32(test_fn())); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let f = module.functions.iter().next().unwrap().1;
    assert!(!local_has_init(f, "a"), "zero i32 init should be removed");
}

#[test]
fn zero_init_bool_removed() {
    let source = r#"
fn test_fn() -> bool {
    var a: bool = false;
    a = true;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(f32(test_fn())); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let f = module.functions.iter().next().unwrap().1;
    assert!(!local_has_init(f, "a"), "false bool init should be removed");
}

#[test]
fn non_zero_init_preserved_when_loaded() {
    // Compose inits are not seeded, so the first Load reads the init and
    // keeps `v` alive.
    let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    return v.x + v.y + v.z;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
    let (_, module) = run_pass(source);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
    let f = module.functions.iter().next().unwrap().1;
    let v_init = f
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("v"))
        .map(|(_, v)| v.init.is_some());
    assert_eq!(
        v_init,
        Some(true),
        "non-zero Compose init should be preserved"
    );
}

#[test]
fn zero_init_vec3_compose_removed() {
    let source = r#"
fn test_fn() -> vec3<f32> {
    var v: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    v = vec3<f32>(1.0, 2.0, 3.0);
    return v;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(), 1.0); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let f = module.functions.iter().next().unwrap().1;
    assert!(
        !local_has_init(f, "v"),
        "all-zero Compose init should be removed"
    );
}

#[test]
fn partially_non_zero_compose_preserved() {
    let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    return v.y;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
    let (_, module) = run_pass(source);
    let f = module.functions.iter().next().unwrap().1;
    // Forwarding may eliminate `v`; if it survives, the init must stay.
    let v_init = f
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("v"))
        .map(|(_, v)| v.init.is_some());
    if let Some(has_init) = v_init {
        assert!(
            has_init,
            "partially non-zero Compose init should be preserved"
        );
    }
}

// Phase 2: dead non-zero init removal

#[test]
fn dead_non_zero_scalar_init_removed() {
    // After the load forwards to `x` its Emit is gone, so `find_dead_inits`
    // sees `Store(a, x)` first.
    let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 5.0;
    a = x;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0)); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
}

#[test]
fn dead_non_zero_compose_init_removed() {
    let source = r#"
fn test_fn(x: vec3<f32>) -> vec3<f32> {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    v = x;
    return v;
}
@fragment fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(4.0, 5.0, 6.0)), 1.0);
}
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let f = module.functions.iter().next().unwrap().1;
    assert!(
        !local_has_init(f, "v"),
        "dead Compose init should be removed"
    );
}

#[test]
fn dead_init_across_unrelated_control_flow() {
    let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = 5.0;
    var b: f32;
    if c { b = 1.0; } else { b = 2.0; }
    a = x;
    return a + b;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0, true)); }
"#;
    let (changed, module) = run_pass(source);
    assert!(changed);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
}

#[test]
fn init_preserved_when_read_through_control_flow() {
    // On the else path the init reaches `return a`.
    let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = 5.0;
    if c { a = x; }
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0, true)); }
"#;
    let (_, module) = run_pass(source);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
    let f = module.functions.iter().next().unwrap().1;
    let a_init = f
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("a"))
        .map(|(_, v)| v.init.is_some());
    assert_eq!(
        a_init,
        Some(true),
        "init must be preserved when the variable is read through control flow"
    );
}

#[test]
fn dead_init_partial_store_prevents_removal() {
    // A partial (field) store reads the old value, so the init is used.
    let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    v.x = 99.0;
    return v.x + v.y + v.z;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
    let (_, module) = run_pass(source);
    let _ = crate::io::validate_module(&module).expect("module should remain valid");
    let f = module.functions.iter().next().unwrap().1;
    let v_init = f
        .local_variables
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("v"))
        .map(|(_, v)| v.init.is_some());
    if let Some(has_init) = v_init {
        assert!(
            has_init,
            "partial store reads old value - init must be preserved"
        );
    }
}

fn run_dead_branch_then_load_dedup(source: &str) -> (bool, naga::Module) {
    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        name_log: None,
    };

    let mut db = crate::passes::dead_branch::DeadBranchPass;
    let _ = db.run(&mut module, &ctx).expect("dead_branch should run");
    let _ = crate::io::validate_module(&module).expect("valid after dead_branch");

    let mut ld = LoadDedupPass;
    let changed = ld.run(&mut module, &ctx).expect("load_dedup should run");
    let _ = crate::io::validate_module(&module).expect("valid after load_dedup");

    (changed, module)
}

fn assert_wgsl_round_trips(module: &naga::Module) {
    let info = crate::io::validate_module(module).expect("module should validate");
    let wgsl =
        naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
            .expect("WGSL emission should succeed");
    let reparsed = naga::front::wgsl::parse_str(&wgsl);
    assert!(
        reparsed.is_ok(),
        "emitted WGSL should re-parse, got error: {:?}\nemitted WGSL:\n{}",
        reparsed.err(),
        wgsl
    );
}

#[test]
fn forward_ref_replacement_preserves_short_circuit_local() {
    // naga lowers `a && b` to `if (a) { local = b; } else { local = false; }`
    // and the re-sugar rebuilds it as a `Binary(LogicalAnd)` appended at the
    // arena's end, so the store-forward `Load(local) -> Binary` is a forward
    // reference the apply loop refuses; `local` must then not be marked dead
    // or the Store goes while the Load persists.
    let source = r#"
fn test_fn(a: f32, b: f32, c: f32) -> f32 {
    let hit = a > 0.0 && b > 0.0 && c > 0.0;
    return select(0.0, 1.0, hit);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, 2.0, 3.0));
}
"#;

    let (_, module) = run_dead_branch_then_load_dedup(source);
    assert_wgsl_round_trips(&module);

    let test_fn = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
        .map(|(_, f)| f)
        .expect("test_fn should exist");

    for (lh, lvar) in test_fn.local_variables.iter() {
        let ptr_h = test_fn
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, naga::Expression::LocalVariable(l) if *l == lh))
            .map(|(h, _)| h);
        let Some(ptr_h) = ptr_h else { continue };

        let has_load = test_fn
            .expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::Load { pointer } if *pointer == ptr_h));
        if !has_load {
            continue;
        }

        let has_store = has_store_to(&test_fn.body, ptr_h);
        let has_init = lvar.init.is_some();
        assert!(
            has_store || has_init,
            "local {:?} has loads but no store/init - forward-ref replacement bug",
            lvar.name
        );
    }
}

fn has_store_to(block: &naga::Block, ptr_h: naga::Handle<naga::Expression>) -> bool {
    block.iter().any(|stmt| {
        if let naga::Statement::Store { pointer, .. } = stmt {
            return *pointer == ptr_h;
        }
        nested_blocks(stmt).any(|b| has_store_to(b, ptr_h))
    })
}

#[test]
fn forward_ref_chained_short_circuit_preserves_locals() {
    // Each link of the chain carries a forward-reference Binary replacement.
    let source = r#"
fn test_fn(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let hit = a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0;
    return select(0.0, 1.0, hit);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, 2.0, 3.0, 4.0));
}
"#;

    let (_, module) = run_dead_branch_then_load_dedup(source);
    assert_wgsl_round_trips(&module);
}

// `remove_dead_stores_in_block` pending-store precision: a non-aliasing
// statement (Barrier, Atomic on a global, ImageStore) must not save a dead
// Store, and a Return/Kill terminator marks every pending Store dead.

fn count_stores_to_local(
    function: &naga::Function,
    local: naga::Handle<naga::LocalVariable>,
) -> usize {
    fn walk(
        block: &naga::Block,
        expressions: &naga::Arena<naga::Expression>,
        local: naga::Handle<naga::LocalVariable>,
        count: &mut usize,
    ) {
        for stmt in block {
            if let naga::Statement::Store { pointer, .. } = stmt
                && let naga::Expression::LocalVariable(lh) = expressions[*pointer]
                && lh == local
            {
                *count += 1;
            }
            for nested in nested_blocks(stmt) {
                walk(nested, expressions, local, count);
            }
        }
    }
    let mut n = 0;
    walk(&function.body, &function.expressions, local, &mut n);
    n
}

fn local_handle_by_name(
    function: &naga::Function,
    name: &str,
) -> naga::Handle<naga::LocalVariable> {
    function
        .local_variables
        .iter()
        .find(|(_, lv)| lv.name.as_deref() == Some(name))
        .map(|(h, _)| h)
        .unwrap_or_else(|| panic!("local {name} not found"))
}

#[test]
fn dead_store_removed_across_atomic_on_global() {
    // The atomic targets a storage atomic that cannot alias `x`, so `x = 1`
    // is dead; the trailing If keeps the load live so only the per-local
    // invalidation can remove it.
    let source = r#"
@group(0) @binding(0) var<storage, read_write> g: atomic<i32>;
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    atomicAdd(&g, 1);
    x = 2;
    if (c) { x = 3; }
    return x;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
    let (_, module) = run_pass(source);
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .map(|(_, f)| f)
        .expect("f exists");
    let x = local_handle_by_name(f, "x");
    assert_eq!(
        count_stores_to_local(f, x),
        2,
        "dead Store before atomic should be removed; remaining = x=2 + x=3"
    );
}

#[test]
fn dead_store_removed_across_barrier() {
    // Barriers reference no function-local pointers.
    let source = r#"
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    workgroupBarrier();
    x = 2;
    if (c) { x = 3; }
    return x;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
    let (_, module) = run_pass(source);
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .map(|(_, f)| f)
        .expect("f exists");
    let x = local_handle_by_name(f, "x");
    assert_eq!(
        count_stores_to_local(f, x),
        2,
        "dead Store before barrier should be removed"
    );
}

#[test]
fn dead_trailing_store_removed_before_return_unit() {
    // `remove_dead_stores_in_function` alone: the Return drains
    // `pending_store`.  The full pass reaches the same result via
    // `dead_store_ids`, so this pins the standalone semantic.
    let source = r#"
fn f() -> i32 {
    var x: i32;
    var y: i32;
    y = 7;
    x = 1;
    return y;
}
@compute @workgroup_size(1) fn main() { _ = f(); }
"#;
    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let f_handle = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .map(|(h, _)| h)
        .expect("f exists");
    let f = module.functions.get_mut(f_handle);
    let x = local_handle_by_name(f, "x");
    let y = local_handle_by_name(f, "y");
    let changed = remove_dead_stores_in_function(f);
    assert!(changed, "should mark trailing Store dead");
    assert_eq!(
        count_stores_to_local(f, x),
        0,
        "trailing Store of `x` before Return should be removed by terminator drain"
    );
    assert_eq!(
        count_stores_to_local(f, y),
        1,
        "Store of `y` is read by `return y` and must be preserved"
    );
}

#[test]
fn dead_store_kept_across_call_with_pointer_arg() {
    // The callee may read the pending Store through the `ptr<function>`
    // argument, so per-arg invalidation must keep it.
    let source = r#"
fn g(p: ptr<function, i32>) -> i32 {
    return *p;
}
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    let y = g(&x);
    x = 2;
    if (c) { x = 3; }
    return x + y;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
    let (_, module) = run_pass(source);
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .map(|(_, f)| f)
        .expect("f exists");
    let x = local_handle_by_name(f, "x");
    assert_eq!(
        count_stores_to_local(f, x),
        3,
        "Store observable through ptr arg must NOT be removed"
    );
}

#[test]
fn dead_store_removed_across_call_without_pointer_arg() {
    // The call's pointer argument targets `y`, so `x = 1` is dead; the
    // trailing If keeps dedup_loads from trimming `x = 2` / `x = 3`.
    let source = r#"
fn g(p: ptr<function, i32>) -> i32 {
    return *p;
}
fn f(c: bool) -> i32 {
    var x: i32;
    var y: i32;
    x = 1;
    let r = g(&y);
    x = 2;
    if (c) { x = 3; }
    return x + r;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
    let (_, module) = run_pass(source);
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("f"))
        .map(|(_, f)| f)
        .expect("f exists");
    let x = local_handle_by_name(f, "x");
    assert_eq!(
        count_stores_to_local(f, x),
        2,
        "dead Store across call with non-aliasing ptr arg should be removed"
    );
}

// MARK: Block / If / Switch scope-leak regressions
//
// A value bound by an `Emit` inside a nested Block/If/Switch is read through
// a variable after it; forwarding the post-block load to the in-block handle
// is IR the validator rejects ("expression used outside its scope"), which
// `run_pass`'s validation catches.

#[test]
fn block_scope_leak_regression_statement_block() {
    let source = r#"
fn f(a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    {
        let temp = vec3<f32>(a, b, c);
        x = temp;
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(1.0, 2.0, 3.0); }
"#;
    let (_, _module) = run_pass(source);
}

#[test]
fn block_scope_leak_regression_if_branch() {
    let source = r#"
fn f(c: bool, a: f32, b: f32, d: f32) -> vec3<f32> {
    var x: vec3<f32>;
    if c {
        let temp = vec3<f32>(a, b, d);
        x = temp;
    } else {
        x = vec3<f32>(0.0);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(true, 1.0, 2.0, 3.0); }
"#;
    let (_, _module) = run_pass(source);
}

// MARK: ExpressionScopeIndex regressions

#[test]
fn scope_index_assigns_monotonic_positions_in_emit_order() {
    // Emit(e0, e1) at position 0, Store(p, v) at 1, Emit(e2) at 2.
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    // Placeholder literals: only handle identity matters.
    let p = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        naga::Span::UNDEFINED,
    );
    let v = arena.append(
        naga::Expression::Literal(naga::Literal::U32(1)),
        naga::Span::UNDEFINED,
    );
    let e0 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(2)),
        naga::Span::UNDEFINED,
    );
    let e1 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(3)),
        naga::Span::UNDEFINED,
    );
    let e2 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(4)),
        naga::Span::UNDEFINED,
    );
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(e0, e1)),
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Store {
            pointer: p,
            value: v,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(e2, e2)),
        naga::Span::UNDEFINED,
    );

    let idx = ExpressionScopeIndex::build(&body, &arena);

    assert_eq!(idx.handle_position(e0), Some(0));
    assert_eq!(idx.handle_position(e1), Some(0));
    assert_eq!(idx.store_position((p, v)), Some(1));
    assert_eq!(idx.handle_position(e2), Some(2));
    assert!(idx.is_in_subtree(&body, e0));
    assert!(idx.is_in_subtree(&body, e2));
}

#[test]
fn scope_index_recurses_into_control_flow() {
    // Store(p,v) at 0; If at 1 with accept Emit(a) at 2 and reject Emit(b)
    // at 3; Store(p2,v2) at 4.
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let p = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        naga::Span::UNDEFINED,
    );
    let v = arena.append(
        naga::Expression::Literal(naga::Literal::U32(1)),
        naga::Span::UNDEFINED,
    );
    let cond = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        naga::Span::UNDEFINED,
    );
    let a = arena.append(
        naga::Expression::Literal(naga::Literal::U32(2)),
        naga::Span::UNDEFINED,
    );
    let b = arena.append(
        naga::Expression::Literal(naga::Literal::U32(3)),
        naga::Span::UNDEFINED,
    );
    let p2 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(4)),
        naga::Span::UNDEFINED,
    );
    let v2 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(5)),
        naga::Span::UNDEFINED,
    );

    let mut accept = naga::Block::new();
    accept.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(a, a)),
        naga::Span::UNDEFINED,
    );
    let mut reject = naga::Block::new();
    reject.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(b, b)),
        naga::Span::UNDEFINED,
    );

    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Store {
            pointer: p,
            value: v,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::If {
            condition: cond,
            accept,
            reject,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Store {
            pointer: p2,
            value: v2,
        },
        naga::Span::UNDEFINED,
    );

    let idx = ExpressionScopeIndex::build(&body, &arena);

    assert_eq!(idx.store_position((p, v)), Some(0));
    assert_eq!(idx.handle_position(a), Some(2));
    assert_eq!(idx.handle_position(b), Some(3));
    assert_eq!(idx.store_position((p2, v2)), Some(4));
    assert!(idx.store_position((p2, v2)).unwrap() > idx.handle_position(a).unwrap());
    assert!(idx.store_position((p2, v2)).unwrap() > idx.handle_position(b).unwrap());

    // Re-borrow through the body so the block addresses match production
    // callers'.
    let (accept_ref, reject_ref) = match &body[1] {
        naga::Statement::If { accept, reject, .. } => (accept, reject),
        _ => unreachable!(),
    };
    assert!(idx.is_in_subtree(accept_ref, a));
    assert!(!idx.is_in_subtree(accept_ref, b));
    assert!(idx.is_in_subtree(reject_ref, b));
    assert!(!idx.is_in_subtree(reject_ref, a));
}

#[test]
fn scope_index_collides_stores_to_minimum() {
    // Two Stores with one (ptr, val) identity record the minimum position,
    // so a Load between them counts as after the earliest (conservative).
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let p = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        naga::Span::UNDEFINED,
    );
    let v = arena.append(
        naga::Expression::Literal(naga::Literal::U32(1)),
        naga::Span::UNDEFINED,
    );
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Store {
            pointer: p,
            value: v,
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::Store {
            pointer: p,
            value: v,
        },
        naga::Span::UNDEFINED,
    );

    let idx = ExpressionScopeIndex::build(&body, &arena);

    assert_eq!(idx.store_position((p, v)), Some(0));
}

/// Statement-bound results (Call, Atomic, WorkGroupUniformLoad, Subgroup*,
/// RayQueryFunction::Proceed) sit in no Emit range; without a recorded
/// position a has_later_live_load check against one fails and the producing
/// Store is misclassified.  Atomic covers the `Option<Handle>` arm; the
/// walker shares the codepath for the other variants.
#[test]
fn scope_index_records_statement_bound_result_handles() {
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let ptr = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        naga::Span::UNDEFINED,
    );
    let val = arena.append(
        naga::Expression::Literal(naga::Literal::U32(1)),
        naga::Span::UNDEFINED,
    );
    // Literal placeholders: the index checks handle identity only.
    let atomic_result = arena.append(
        naga::Expression::Literal(naga::Literal::U32(2)),
        naga::Span::UNDEFINED,
    );
    let wg_result = arena.append(
        naga::Expression::Literal(naga::Literal::U32(3)),
        naga::Span::UNDEFINED,
    );

    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Atomic {
            pointer: ptr,
            fun: naga::AtomicFunction::Add,
            value: val,
            result: Some(atomic_result),
        },
        naga::Span::UNDEFINED,
    );
    body.push(
        naga::Statement::WorkGroupUniformLoad {
            pointer: ptr,
            result: wg_result,
        },
        naga::Span::UNDEFINED,
    );

    let idx = ExpressionScopeIndex::build(&body, &arena);

    assert_eq!(idx.handle_position(atomic_result), Some(0));
    assert_eq!(idx.handle_position(wg_result), Some(1));

    assert!(idx.is_in_subtree(&body, atomic_result));
    assert!(idx.is_in_subtree(&body, wg_result));
}

/// The empty-block interval `[enter, enter)` must reject every handle; loop
/// continuing blocks are routinely empty.
#[test]
fn scope_index_empty_block_subtree_membership() {
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let e0 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        naga::Span::UNDEFINED,
    );
    // Emit(e0) at 0, an empty Block at 1.
    let inner = naga::Block::new();
    let mut body = naga::Block::new();
    body.push(
        naga::Statement::Emit(naga::Range::new_from_bounds(e0, e0)),
        naga::Span::UNDEFINED,
    );
    body.push(naga::Statement::Block(inner), naga::Span::UNDEFINED);

    let idx = ExpressionScopeIndex::build(&body, &arena);

    let inner_ref = match &body[1] {
        naga::Statement::Block(b) => b,
        _ => unreachable!(),
    };
    assert!(idx.is_in_subtree(&body, e0));
    assert!(!idx.is_in_subtree(inner_ref, e0));
}

/// Pre-emit expressions (Literal, Constant, LocalVariable) are bound by no
/// statement and in scope everywhere: `handle_position` is `None` and
/// subtree membership false (the filter sites `||` with `needs_pre_emit`).
#[test]
fn scope_index_pre_emit_handles_have_no_position() {
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let unused_literal = arena.append(
        naga::Expression::Literal(naga::Literal::U32(42)),
        naga::Span::UNDEFINED,
    );
    let body = naga::Block::new();
    let idx = ExpressionScopeIndex::build(&body, &arena);
    assert_eq!(idx.handle_position(unused_literal), None);
    assert!(!idx.is_in_subtree(&body, unused_literal));
}

#[test]
fn block_scope_leak_regression_switch_case() {
    // `has_default` and no fall-through make the switch eligible for
    // meet-over-branches, which would carry the in-case value forward.
    let source = r#"
fn f(sel: u32, a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    switch sel {
        case 0u: {
            let temp = vec3<f32>(a, b, c);
            x = temp;
        }
        default: {
            x = vec3<f32>(0.0);
        }
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(0u, 1.0, 2.0, 3.0); }
"#;
    let (_, _module) = run_pass(source);
}

/// Statement-bound results (CallResult, AtomicResult,
/// WorkGroupUniformLoadResult, Subgroup*Result, RayQueryProceedResult) are
/// let-bound at the statement's block, not in an Emit range, so a cache
/// entry `Local(x) -> CallResult` must be filtered at the closing brace too.
#[test]
fn block_scope_leak_regression_call_result_in_block() {
    let source = r#"
fn helper(a: f32, b: f32, c: f32) -> vec3<f32> { return vec3<f32>(a, b, c); }
fn f(a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    {
        x = helper(a, b, c);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(1.0, 2.0, 3.0); }
"#;
    let (_, _module) = run_pass(source);
}

/// The CallResult-seeded Store inside an `If` branch.
#[test]
fn block_scope_leak_regression_call_result_in_if_branch() {
    let source = r#"
fn helper(a: f32, b: f32, c: f32) -> vec3<f32> { return vec3<f32>(a, b, c); }
fn f(cond: bool, a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    if cond {
        x = helper(a, b, c);
    } else {
        x = vec3<f32>(0.0);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(true, 1.0, 2.0, 3.0); }
"#;
    let (_, _module) = run_pass(source);
}

/// The statement-bound `Atomic` result.
#[test]
fn block_scope_leak_regression_atomic_result_in_block() {
    let source = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
fn f() -> u32 {
    var x: u32;
    {
        x = atomicAdd(&counter, 1u);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(); }
"#;
    let (_, _module) = run_pass(source);
}

/// On exit from a nested `Statement::Block` that wrote `failed`, the cache
/// rollback restores the pre-block init entry (`Local(failed) ->
/// Literal(false)`); a later `Load(failed)` in another block then forwards
/// to `false`, emitting `failed = false | X`, structurally valid WGSL that
/// drops every earlier `failed |= ...`.  Rollback must drop entries for
/// locals written inside the block, as the If/Switch/Loop arms do.  Signal:
/// at most one init-forwarded `failed=false|` (the first write is correct).
#[test]
fn block_arm_does_not_forward_stale_init_after_inner_block_writes() {
    let source = r#"
fn test_fn(a: bool, b: bool, c: bool) -> bool {
    var failed = false;
    {
        failed |= a;
    }
    {
        failed |= b;
    }
    {
        failed |= c;
    }
    return failed;
}

@compute @workgroup_size(1) fn main() {
    _ = test_fn(true, false, false);
}
"#;
    let (_, module) = run_pass(source);

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("module should validate");
    let mut out = String::new();
    let mut writer =
        naga::back::wgsl::Writer::new(&mut out, naga::back::wgsl::WriterFlags::empty());
    writer.write(&module, &info).expect("should emit WGSL");

    // Whitespace-insensitive match.
    let stripped: String = out.chars().filter(|c| !c.is_whitespace()).collect();

    let bad_pattern_count = stripped.matches("failed=false|").count();
    assert!(
        bad_pattern_count <= 1,
        "expected at most 1 init-forwarded `failed=false|` (the first \
             write), found {} occurrences.  Pre-fix bug forwards \
             Load(failed) inside nested blocks to the stale init, \
             producing additional `failed=false|X` writes that drop \
             accumulator state.  Output:\n{}",
        bad_pattern_count,
        out,
    );
}

/// True when a `Divide`/`Modulo` right operand is an integer-zero literal,
/// directly or through the `Splat`/`Compose` naga wraps a scalar divisor in
/// for `vecN / scalar`: the shape `decline_static_error_forwards` prevents.
fn has_const_zero_divisor(function: &naga::Function) -> bool {
    fn operand_is_zero(
        exprs: &naga::Arena<naga::Expression>,
        handle: naga::Handle<naga::Expression>,
    ) -> bool {
        match &exprs[handle] {
            naga::Expression::Literal(lit) => is_integer_zero_literal(lit),
            naga::Expression::Splat { value, .. } => operand_is_zero(exprs, *value),
            naga::Expression::Compose { components, .. } => {
                components.iter().any(|&c| operand_is_zero(exprs, c))
            }
            _ => false,
        }
    }
    function.expressions.iter().any(|(_, e)| {
        matches!(
            e,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Divide | naga::BinaryOperator::Modulo,
                right,
                ..
            } if operand_is_zero(&function.expressions, *right)
        )
    })
}

#[test]
fn declines_forwarding_zero_into_scalar_divisor() {
    // Forwarding `b -> 0u` makes `a / 0u` a const divide-by-zero
    // (shader-creation error): naga rejects it and the whole pass rolls back
    // with a spurious warning; `run_pass` validates, so the guard's absence
    // panics.
    let src = "\
@compute @workgroup_size(1)
fn f() {
    var a = 10u;
    var b = 0u;
    let r = a / b;
}";
    let (_changed, module) = run_pass(src);
    assert!(
        !has_const_zero_divisor(&module.entry_points[0].function),
        "guard must decline forwarding zero into a scalar divisor"
    );
}

#[test]
fn declines_forwarding_zero_into_vector_divisor() {
    // `vecN / scalar` splats the scalar divisor, so the forwarded zero hides
    // under a `Splat` - the guard must recurse into it.
    let src = "\
@compute @workgroup_size(1)
fn f() {
    var a = vec3<u32>(1u, 2u, 3u);
    var b = 0u;
    let r = a / b;
}";
    let (_changed, module) = run_pass(src);
    assert!(
        !has_const_zero_divisor(&module.entry_points[0].function),
        "guard must decline forwarding zero into a splatted vector divisor"
    );
}

#[test]
fn declines_forwarding_overflow_into_shift_amount() {
    // Forwarding `s -> 40u` makes `1i << 40u` a const shift past the 32-bit
    // width; `run_pass`'s validation is the assertion.
    let src = "\
@compute @workgroup_size(1)
fn f() {
    var a = 1i;
    var s = 40u;
    let r = a << s;
}";
    let (_changed, _module) = run_pass(src);
}

#[test]
fn still_forwards_legal_shift_and_divisor() {
    // The guard is surgical, not a blanket ban on shift/div RHS.
    let src = "\
@compute @workgroup_size(1)
fn f() {
    var a = 256u;
    var sh = 2u;
    var d = 4u;
    let r = (a >> sh) / d;
}";
    let (changed, module) = run_pass(src);
    assert!(changed, "legal shift/divide forwards must still fire");
    assert!(
        !has_const_zero_divisor(&module.entry_points[0].function),
        "a non-zero divisor is never flagged"
    );
}

/// Dawn on Metal flushes a negative-zero LITERAL to +0 while a runtime
/// negation keeps the sign, so `-0.0` never forwards into a load: the
/// initializer form and the store form both keep their reads.
#[test]
fn dead_stores_clear_at_a_discard_and_the_scan_continues() {
    // The terminator drains the pending-store map, and the block keeps
    // storing afterwards: the scan must survive the reset.
    let (changed, module) = run_pass(
        "@fragment fn m(@location(0) x: f32) -> @location(0) vec4f {\n\
           var a: f32;\n\
           a = x;\n\
           discard;\n\
           a = 2.0;\n\
           return vec4f(a);\n\
         }",
    );
    assert!(changed, "the overwritten store is dead");
    let f = &module.entry_points[0].function;
    assert_eq!(
        f.body
            .iter()
            .filter(|s| matches!(s, naga::Statement::Store { .. }))
            .count(),
        0,
        "the first store is overwritten and the second forwards into the return"
    );
}

#[test]
fn module_scope_values_are_not_forwarded_into_a_runtime_read() {
    // An override has no value until pipeline creation and a constant's tree
    // lives in an arena the pass does not hold, so neither can be shown free
    // of a `-0.0` that Dawn's Metal backend would flush.
    for (decl, init, forwarded) in [
        ("override OV: f32 = -0.0;", "OV", false),
        ("const CN: f32 = -0.0;", "CN", false),
        ("", "0.5", true),
    ] {
        let source = format!(
            "{decl}\n\
             @group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n\
               var z = {init};\n\
               out[0] = bitcast<u32>(-(z * z));\n\
             }}"
        );
        let (_, module) = run_pass(&source);
        let loads = count_loads_from_local(&module.entry_points[0].function);
        assert_eq!(loads == 0, forwarded, "init {init}: {loads} loads");
    }
}

#[test]
fn negative_zero_literals_are_not_forwarded() {
    for (init, store, forwarded) in [
        ("-(0.0)", "", false),
        ("0.5", "", true),
        ("0.0", "z = -(0.0);", false),
        ("0.0", "z = 0.5;", true),
    ] {
        let source = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n\
               var z = {init};\n\
               {store}\n\
               out[0] = bitcast<u32>(-z);\n\
             }}"
        );
        let (_, module) = run_pass(&source);
        let loads = count_loads_from_local(&module.entry_points[0].function);
        assert_eq!(
            loads == 0,
            forwarded,
            "init {init} store {store:?}: {loads} loads"
        );
    }
}
