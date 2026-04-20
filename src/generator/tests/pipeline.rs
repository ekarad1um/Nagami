//! End-to-end IR-pass integration tests.  Each case runs the full
//! `parse -> validate -> IR passes -> generate` chain through
//! [`super::helpers::compact_with_passes`] and asserts on properties
//! of the emitted WGSL that would regress if a pass's invariants
//! drifted out of sync with the emitter.

use super::helpers::*;

// MARK: IR pass integration tests

#[test]
fn const_fold_negate_i32_min_no_panic() {
    // Negating i32::MIN must not panic (overflow).  The fold should be skipped,
    // leaving the runtime negation intact.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<i32, 1>;
        @compute @workgroup_size(1)
        fn main() {
            let x = i32(-2147483648);
            out[0] = -x;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("const_fold_negate_i32_min: {out}");
    // The output must NOT have folded away the negation - it should still
    // contain a negation operator or the original expression, not a bare literal.
    assert!(
        out.contains("-i32(") || out.contains("= -"),
        "negate of i32::MIN should not be folded: {out}"
    );
}

#[test]
fn const_fold_negate_normal_i32_folds() {
    // Normal i32 negation should still fold fine.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<i32, 1>;
        @compute @workgroup_size(1)
        fn main() {
            let x = 42i;
            out[0] = -x;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("const_fold_negate_normal_i32: {out}");
    // Should fold to -42
    assert!(
        out.contains("-42"),
        "normal i32 negation should fold: {out}"
    );
}

#[test]
fn load_dedup_invalidates_cache_on_call_with_pointer() {
    // A function call that takes a pointer to a local must invalidate the
    // load cache for that local - subsequent loads must NOT reuse stale values.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 2>;

        fn modify(p: ptr<function, f32>) {
            *p += 1.0;
        }

        @compute @workgroup_size(1)
        fn main() {
            var x: f32 = 10.0;
            out[0] = x;
            modify(&x);
            out[1] = x;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("load_dedup_call_invalidation: {out}");
    // After modify(&x), the second store must load x fresh.
    // It must NOT be the literal 10.0 (stale cached value).
    assert!(
        !out.contains("out[1] = 10") && !out.contains("out[1]=10"),
        "load after call through pointer must not use stale cached value: {out}"
    );
}

#[test]
fn load_dedup_no_pointer_arg_still_deduplicates() {
    // When no pointer to the local is passed, loads should still be deduplicated.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 2>;

        @compute @workgroup_size(1)
        fn main() {
            var x: f32 = 10.0;
            out[0] = x;
            out[1] = x;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("load_dedup_no_pointer_still_dedup: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn dead_branch_strips_code_after_return_from_folded_if_true() {
    // `if (true) { return 1.0; }` folds to `return 1.0;`, making the
    // subsequent store dead.  The dead-branch pass must strip it.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 2>;
        fn test_fn() -> f32 {
            if (true) { return 1.0; }
            out[0] = 2.0;
            return 0.0;
        }
        @compute @workgroup_size(1)
        fn main() {
            out[1] = test_fn();
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("dead_branch_strip_after_return: {out}");
    // The folded function should just return 1.0, with no trace of dead code.
    assert!(
        !out.contains("2.0") && !out.contains("0.0"),
        "dead code after folded-if return must be stripped: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn dead_branch_strips_code_after_both_branches_terminate() {
    // When both branches of a non-constant if terminate, subsequent code is dead.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 2>;
        fn test_fn(c: bool) -> f32 {
            if (c) { return 1.0; } else { return 2.0; }
            out[0] = 99.0;
            return 0.0;
        }
        @compute @workgroup_size(1)
        fn main() {
            out[1] = test_fn(true);
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("dead_branch_both_branches_terminate: {out}");
    assert!(
        !out.contains("99"),
        "dead code after if where both branches return must be stripped: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn or_chain_redundant_true_stores_eliminated() {
    // `a || b || c` lowers to two ifs with separate locals.  After
    // coalescing merges them, the reject branches `else { d = true; }`
    // become redundant and should be cleared by the dead-branch pass.
    let src = r#"
        fn or3(a: bool, b: bool, c: bool) -> bool {
            return a || b || c;
        }
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(f32(or3(true, false, true)));
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    println!("or_chain: {out}");
    // After optimization the redundant `else { d = true; }` branches
    // should be gone.  Count occurrences of "=true" or "= true" stores
    // inside if-else - at most one should remain (the first if's else).
    let true_stores = out.matches("=true").count() + out.matches("= true").count();
    assert!(
        true_stores <= 1,
        "chained || should have at most 1 true-store after optimization, got {true_stores}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn derivative_argument_keeps_float_type() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: f32;
        @fragment fn main() {
            out = dpdx(1.0);
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    assert!(
        out.contains("dpdx(1.") || out.contains("dpdx(1f") || out.contains("dpdx(1.0"),
        "dpdx argument must remain float-typed: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn matrix_cast_emits_and_validates() {
    let src = r#"
        enable f16;
        @group(0) @binding(0) var<storage, read_write> out: mat2x2<f32>;

        @compute @workgroup_size(1)
        fn main() {
            let m16 = mat2x2<f16>(1.0h, 0.0h, 0.0h, 1.0h);
            out = mat2x2<f32>(m16);
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    assert!(
        out.contains("mat2x2<f32>(") || out.contains("mat2x2f("),
        "matrix cast should be emitted via matrix constructor: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn deferred_loop_init_not_absorbed_when_var_used_later() {
    let src = r#"
        @compute @workgroup_size(1)
        fn main() {
            var c: i32;
            c = 0;
            loop {
                if c < 4 {
                } else {
                    break;
                }
                continuing {
                    c = c + 1;
                }
            }
            loop {
                if c < 8 {
                } else {
                    break;
                }
                continuing {
                    c = c + 1;
                }
            }
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn image_store_can_be_for_loop_update() {
    let src = r#"
        @group(0) @binding(0) var img: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(1)
        fn main() {
            var i: i32 = 0;
            loop {
                if i < 4 {
                } else {
                    break;
                }
                i = i + 1;
                continuing {
                    textureStore(img, vec2<i32>(i, 0), vec4<f32>(1.0, 1.0, 1.0, 1.0));
                }
            }
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "loop should be reconstructed as for: {out}"
    );
    assert!(
        out.contains("textureStore("),
        "textureStore must be preserved in update clause: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_uniform_load_can_appear_in_for_header() {
    let src = r#"
        var<workgroup> a: i32;
        var<workgroup> b: i32;

        @compute @workgroup_size(1)
        fn main() {
            for (var i = 0; i < workgroupUniformLoad(&a); i += workgroupUniformLoad(&b)) {
            }
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "expected for-loop reconstruction: {out}"
    );
    assert!(
        out.contains("workgroupUniformLoad(&"),
        "workgroupUniformLoad should remain in loop header path: {out}"
    );
    let wgul_count = out.matches("workgroupUniformLoad(&").count();
    assert_eq!(
        wgul_count, 2,
        "expected exactly 2 workgroupUniformLoad calls (condition + update), got {wgul_count}: {out}"
    );
    assert_valid_wgsl(&out);
}
