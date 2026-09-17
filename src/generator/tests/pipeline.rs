//! End-to-end IR-pass integration: each case runs the full pipeline through
//! [`super::helpers::compact_with_passes`] and pins emitted-text properties
//! that regress when a pass's invariants drift from the emitter.

use super::helpers::*;

// MARK: IR pass integration tests

#[test]
fn const_fold_negate_i32_min_no_panic() {
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
    assert!(
        out.contains("-i32(") || out.contains("= -"),
        "negate of i32::MIN should not be folded: {out}"
    );
}

#[test]
fn const_fold_negate_normal_i32_folds() {
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
    assert!(
        out.contains("-42"),
        "normal i32 negation should fold: {out}"
    );
}

#[test]
fn load_dedup_invalidates_cache_on_call_with_pointer() {
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
    assert!(
        !out.contains("out[1] = 10") && !out.contains("out[1]=10"),
        "load after call through pointer must not use stale cached value: {out}"
    );
}

#[test]
fn load_dedup_no_pointer_arg_still_deduplicates() {
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
    assert!(
        !out.contains("2.0") && !out.contains("0.0"),
        "dead code after folded-if return must be stripped: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn dead_branch_strips_code_after_both_branches_terminate() {
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
    // `a || b || c` lowers to two ifs with separate locals; once coalescing
    // merges them the `else { d = true; }` arms are redundant.
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
    // At most the first if's else survives.
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

/// An `f16` constant under a builtin folds in the pass, not only at naga's
/// re-parse of the output, so a second run has nothing left to shrink.
#[test]
fn an_f16_constant_under_a_builtin_folds_in_one_pass() {
    let src = r#"
        enable f16;
        @group(0) @binding(0) var<storage, read_write> out: f16;
        const c = 1.5703125h;

        @compute @workgroup_size(1)
        fn main() {
            var arg = vec3<f16>(0.0h);
            out = sin(c) + length(arg);
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(out.contains("= 1h + length(vec3h());"), "{out}");
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

#[test]
fn workgroup_uniform_load_reused_in_condition_not_duplicated() {
    // For-reconstruction re-emits a guard preload's `workgroupUniformLoad(&p)`
    // at every occurrence; a result reused in the condition would run the
    // barrier twice per iteration, which re-validation cannot catch, so
    // emission must fall back to a plain `loop` with one `let`.
    let src = r#"
        var<workgroup> a: u32;
        @group(0) @binding(0) var<storage, read_write> outv: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                let w = workgroupUniformLoad(&a);
                if w * w < 100u {} else { break; }
                outv[i] = i;
                i = i + 1u;
            }
        }
    "#;
    let out = compact(src);
    let wgul_count = out.matches("workgroupUniformLoad(&").count();
    assert_eq!(
        wgul_count, 1,
        "reused WorkGroupUniformLoad must be emitted exactly once (bound to a \
         let via loop fallback), not duplicated into the for-condition: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A preload is materialised only where its result is emitted, so a barrier
/// kept purely for its side effect would vanish from a for-header; the
/// preload-safety predicate requires exactly one emission.
#[test]
fn workgroup_uniform_load_dead_preload_barrier_not_dropped() {
    let src = r#"
        var<workgroup> sh: u32;
        @group(0) @binding(0) var<storage, read_write> data: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                let bar = workgroupUniformLoad(&sh);
                if i >= 10u { break; }
                data[i] = i;
                continuing { i = i + 1u; }
            }
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("workgroupUniformLoad"),
        "dead preload barrier must survive (not be dropped): {out}"
    );
    assert!(
        !out.contains("for("),
        "must not become a for-loop that drops the unused preload: {out}"
    );
    assert_valid_wgsl(&out);
}

/// With no core update statement a continuing preload has no for-update
/// clause to land in, so conversion must be refused.
#[test]
fn workgroup_uniform_load_dead_continuing_preload_not_dropped() {
    let src = r#"
        var<workgroup> sh: u32;
        @group(0) @binding(0) var<storage, read_write> data: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                if i >= 10u { break; }
                data[i] = i;
                i = i + 1u;
                continuing {
                    let bar = workgroupUniformLoad(&sh);
                }
            }
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("workgroupUniformLoad"),
        "dead continuing preload barrier must survive: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_preload_used_via_expression_in_tail_not_dropped() {
    // The tail-use oracle must recurse into expression children; a flat
    // operand check would drop the preload's `let` and emit an undeclared
    // `_e<n>`.
    let src = r#"
        var<workgroup> wg: u32;
        @group(0) @binding(0) var<storage, read_write> outv: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                let w = workgroupUniformLoad(&wg);
                if i >= 4u { break; }
                outv[i] = w + 1u;
                continuing { i = i + 1u; }
            }
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_preload_used_as_atomic_compare_in_tail_not_dropped() {
    // The `compare` operand of `atomicCompareExchangeWeak` is a
    // statement-operand position the tail-use oracle must cover.
    let src = r#"
        var<workgroup> wg: u32;
        @group(0) @binding(0) var<storage, read_write> a: atomic<u32>;
        @group(0) @binding(1) var<storage, read_write> outv: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                let w = workgroupUniformLoad(&wg);
                if i >= 4u { break; }
                let r = atomicCompareExchangeWeak(&a, w, 5u);
                outv[i] = select(0u, 1u, r.exchanged);
                continuing { i = i + 1u; }
            }
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_counter_declared_when_preload_used_after_guard() {
    // `try_emit_for_loop` bails to a plain loop here, so the counter analysis
    // must read the same `emittable_for_loop_shape`.
    let src = r#"
        var<workgroup> wg_limit: u32;
        @group(0) @binding(0) var<storage, read_write> sink: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                let lim = workgroupUniformLoad(&wg_limit);
                if i < lim {} else { break; }
                sink[i] = lim;
                continuing { i = i + 1u; }
            }
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_uniform_load_reused_in_update_not_duplicated() {
    // An update preload reused within its statement (`outv[w] = w`) would
    // materialise the barrier twice.
    let src = r#"
        var<workgroup> b: u32;
        @group(0) @binding(0) var<storage, read_write> outv: array<u32, 64>;
        @compute @workgroup_size(64)
        fn main() {
            var i: u32 = 0u;
            loop {
                if i >= 10u { break; }
                i = i + 1u;
                continuing {
                    let w = workgroupUniformLoad(&b);
                    outv[w] = w;
                }
            }
        }
    "#;
    let out = compact(src);
    let wgul_count = out.matches("workgroupUniformLoad(&").count();
    assert_eq!(
        wgul_count, 1,
        "reused update WorkGroupUniformLoad must be emitted exactly once: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A counter both absorbed into the for-init and still flagged deferred is
/// re-declared by its in-body update; the body `var b` shadows the for-init
/// counter (valid WGSL) and freezes it, an infinite loop re-validation cannot
/// catch.
#[test]
fn nested_loop_counter_not_redeclared_in_for_body() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<u32>;
        @compute @workgroup_size(1) fn main() {
            var acc: u32 = 0u;
            var i: u32 = 0u;
            loop {
                if (i >= 2u) { break; }
                var j: u32 = 0u;
                loop {
                    if (j >= 2u) { break; }
                    acc = acc + 1u;
                    j = j + 1u;
                }
                i = i + 1u;
            }
            buf[0] = acc;
        }
    "#;
    // Beautified output, so match the spaced forms.
    let out = compact_with_passes(src, Profile::Max);
    assert_valid_wgsl(&out);
    let after = out
        .split("for (var ")
        .nth(1)
        .expect("the inner loop should reconstruct to a `for` with an absorbed counter");
    let counter: String = after
        .chars()
        .take_while(|&c| c.is_ascii_alphanumeric() || c == '_')
        .collect();
    assert!(!counter.is_empty(), "for-init counter name expected: {out}");
    assert!(
        !after.contains(&format!("var {counter} =")),
        "for-init counter `{counter}` must not be re-declared in the loop body \
         (shadowing it freezes the counter -> infinite loop): {out}"
    );
}

// MARK: Mutated-load binding

/// After passes the IR is `let _a = A[0]; let _b = A[1]; A[0] = _b; A[1] = _a;`;
/// `_a` is read after `A[0]` was overwritten, so inlining it yields
/// `A[1] = A[0]`.
#[test]
fn global_swap_load_not_inlined_across_store() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> A: array<i32>;
        @compute @workgroup_size(1) fn main() {
            var x = A[0]; var y = A[1];
            let t = x; x = y; y = t;
            A[0] = x; A[1] = y;
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("A[1] = A[0]"),
        "A[0] read after its own store must be let-bound, not re-read post-write: {out}"
    );
    assert!(
        out.contains("let "),
        "the swap snapshot must be let-bound: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The for-update clause is emitted into the header, before the body binding,
/// so a continuing update reading a body snapshot of a body-mutated place
/// would inline the post-write place; conversion must bail to a plain `loop`.
#[test]
fn for_loop_update_must_not_inline_mutated_load() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> A: array<i32>;
        @group(0) @binding(1) var<storage, read_write> OUT: array<i32>;
        @compute @workgroup_size(1) fn main() {
            var i = 0;
            loop {
                if (i >= 8) { break; }
                let snap = A[0];
                A[0] = A[0] + 1;
                OUT[i] = snap;
                continuing { i = i + snap; }
            }
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("+= A[0]"),
        "the counter update must use the bound pre-write snapshot, not re-read A[0]: {out}"
    );
    assert!(
        out.contains("continuing"),
        "for-conversion must bail to a plain loop so the body binding precedes the update: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The back-edge lets a later iteration's write reach the post-loop use, so
/// the pre-loop snapshot must bind.
#[test]
fn load_before_loop_used_after_must_bind() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> A: array<i32>;
        var<private> g: i32;
        @compute @workgroup_size(1) fn main() {
            let snap = g;
            for (var i = 0; i < 4; i = i + 1) { g = g + 1; }
            A[0] = snap; A[1] = g;
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    let let_pos = out
        .find("let ")
        .expect("the pre-loop snapshot must be let-bound");
    let loop_pos = out
        .find("for ")
        .or_else(|| out.find("loop"))
        .expect("a loop must be emitted");
    assert!(
        let_pos < loop_pos,
        "the snapshot of `g` must be bound BEFORE the loop that mutates it: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The must-bind bail must not block ordinary counted loops.
#[test]
fn counted_loop_still_reconstructs_for() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> A: array<i32>;
        @compute @workgroup_size(1) fn main() {
            for (var i = 0; i < 10; i = i + 1) { A[i] = i; }
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        out.contains("for ("),
        "a plain counted loop must still become a for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The preload pointer is relocated into the for-update slot ahead of the
/// body binding, so `&W[snap]` would inline `snap` as its post-write place;
/// conversion must bail.
#[test]
fn for_loop_update_preload_pointer_must_not_inline_mutated_load() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> a: array<i32>;
        var<workgroup> W: array<i32, 16>;
        @compute @workgroup_size(1) fn main() {
            var B = 0;
            loop {
                if (B >= 5) { break; }
                let snap = a[0];
                a[0] = a[1];
                a[1] = snap;
                continuing {
                    let w = workgroupUniformLoad(&W[snap]);
                    B = B + w;
                }
            }
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("+= workgroupUniformLoad"),
        "the preload pointer must use the bound pre-write snapshot, not be relocated \
         into the for-update where it re-reads the swapped element: {out}"
    );
    assert!(
        out.contains("continuing"),
        "for-conversion must bail so the body binding precedes the preload: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A guard preload relocated into the for-condition runs the barrier at
/// iteration top, so a must-bind load defined before it (`A[0]`, read by guard
/// and body) would be re-read post-barrier with a different cross-invocation
/// value; conversion must bail.
#[test]
fn for_loop_guard_preload_must_not_read_must_bind_load_post_barrier() {
    let src = r#"
        var<workgroup> A: array<i32, 4>;
        var<workgroup> X: i32;
        @group(0) @binding(0) var<storage, read_write> OUT: array<i32>;
        @compute @workgroup_size(64) fn main() {
            var i = 0;
            loop {
                let v = A[0];
                let w = workgroupUniformLoad(&X);
                if (v > w) { break; }
                OUT[i] = v;
                continuing { i = i + 1; }
            }
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("for ("),
        "a guard-preload loop reading a must-bind load post-barrier must bail to plain loop: {out}"
    );
    let snap = out
        .find("[0]")
        .expect("the workgroup-array element snapshot must be emitted");
    let barrier = out
        .find("workgroupUniformLoad")
        .expect("the barrier must be preserved");
    assert!(
        snap < barrier,
        "the workgroup-array snapshot must be bound BEFORE the workgroupUniformLoad barrier: {out}"
    );
    assert_valid_wgsl(&out);
}

/// `A[0]` is read only by the body tail, so the bail must key on loads defined
/// in the pre-guard region, not on the condition's operand cone.
#[test]
fn for_loop_guard_preload_body_only_must_bind_load_snapshotted_before_barrier() {
    let src = r#"
        var<workgroup> A: array<i32, 4>;
        var<workgroup> X: i32;
        @group(0) @binding(0) var<storage, read_write> OUT: array<i32>;
        @compute @workgroup_size(64) fn main() {
            var i = 0;
            loop {
                let v = A[0];
                let w = workgroupUniformLoad(&X);
                if (w > 5) { break; }
                OUT[i] = v;
                continuing { i = i + 1; }
            }
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("for ("),
        "a body-only must-bind load defined before a guard-preload barrier must bail to plain loop: {out}"
    );
    let snap = out
        .find("[0]")
        .expect("the workgroup-array element snapshot must be emitted");
    let barrier = out
        .find("workgroupUniformLoad")
        .expect("the barrier must be preserved");
    assert!(
        snap < barrier,
        "the workgroup-array snapshot must precede the barrier even when the condition does not read it: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Call-result inlining (reads_locals)

/// Exercises the `reads_locals` path of `find_inlineable_calls`; counting a
/// by-value read as a pointer-to-local leaves the call `let`-bound.
#[test]
fn by_value_local_arg_call_inlined_when_no_intervening_write() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        fn f(x: i32) -> i32 { var a = x; for (var k = 0; k < 3; k = k + 1) { a = a * 2 + k; } return a; }
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) gi: u32) {
            var arr = array<i32, 4>(1, 2, 3, 4);
            let i = i32(gi) & 3;
            let c = f(arr[i]);
            O[0] = c;
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        !out.contains("let "),
        "a by-value local-arg call with no intervening write should inline at its \
         use site (no `let` binding survives): {out}"
    );
    assert_valid_wgsl(&out);
}

/// Inlining past the store would re-read the post-store element.  The array
/// element and runtime store value keep the store live and block the SSA
/// splitting that would make a scalar case safe to inline.
#[test]
fn by_value_local_arg_call_bound_across_write_to_read_local() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        fn f(x: i32) -> i32 { var a = x; for (var k = 0; k < 3; k = k + 1) { a = a * 2 + k; } return a; }
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) gi: u32) {
            var arr = array<i32, 4>(1, 2, 3, 4);
            let i = i32(gi) & 3;
            let c = f(arr[i]);
            arr[i] = i32(gi) * 7 + 1;
            O[0] = c;
            O[1] = arr[i];
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    // `O` is mangled; recover its name from the declaration.
    let decl_tail = out
        .split("read_write>")
        .nth(1)
        .expect("storage declaration expected");
    let arr_name: String = decl_tail
        .trim_start()
        .chars()
        .take_while(|&c| c.is_ascii_alphanumeric() || c == '_')
        .collect();
    assert!(!arr_name.is_empty(), "storage array name expected: {out}");
    // An unsound inline puts a call `(` in the first store's RHS.
    let after_o0 = out
        .split(&format!("{arr_name}[0]"))
        .nth(1)
        .expect("the O[0] store must be emitted");
    let use_stmt = after_o0.split(';').next().unwrap_or("");
    assert!(
        !use_stmt.contains('('),
        "a call reading a local must be bound before a store to that local, not \
         relocated into the post-store use site: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: de Morgan equality fold (`!(a == b)` -> `a != b`)

#[test]
fn not_equal_folds_to_not_equal_operator() {
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) n: u32) {
            let a = i32(n);
            let p = !(a == 2);
            buf[0] = select(0, 1, p);
        }
        "#,
        Profile::Max,
    );
    assert!(out.contains("!="), "!(a==b) should fold to a!=b: {out}");
    assert!(
        !out.contains("!("),
        "the `!(` wrapper should be gone: {out}"
    );
}

#[test]
fn not_notequal_folds_to_equal_operator() {
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) n: u32) {
            let a = i32(n);
            let p = !(a != 2);
            buf[0] = select(0, 1, p);
        }
        "#,
        Profile::Max,
    );
    assert!(out.contains("=="), "!(a!=b) should fold to a==b: {out}");
    assert!(
        !out.contains("!("),
        "the `!(` wrapper should be gone: {out}"
    );
}

#[test]
fn negated_equality_folds_under_outer_comparison() {
    // The folded `a != b` becomes an operand of an outer comparison, which WGSL
    // forbids bare; an emit-time fold could not parenthesise it, having lost
    // the comparison shape.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3u) {
            let a = i32(g.x); let b = i32(g.y); let d = i32(g.z);
            let r = (!(a == b)) == (d > 0);
            buf[0] = select(0, 1, r);
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("!="),
        "inner equality should fold to !=: {out}"
    );
    assert!(!out.contains("!("), "no `!(` wrapper should survive: {out}");
}

#[test]
fn negated_vector_equality_folds_componentwise() {
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3u) {
            let va = vec2(i32(g.x), i32(g.y));
            let vb = vec2(1, 2);
            let r = !(va == vb);
            buf[0] = select(0, 1, all(r));
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("!="),
        "vector !(va==vb) should fold to va!=vb: {out}"
    );
}

// MARK: short-circuit full collapse + idempotence (forwarding)

#[test]
fn short_circuit_guard_chain_fully_collapses_and_is_idempotent() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3u) {
            let a = i32(g.x);
            let b = i32(g.y);
            let c = i32(g.z);
            if (a < 10 && b < 20 && c < 30) { buf[0] = a + b + c; }
        }
    "#;
    let out1 = compact_with_passes(src, Profile::Max);
    assert!(
        out1.contains("&&"),
        "conditions should fold to `&&`: {out1}"
    );
    assert!(
        !out1.contains("= false") && !out1.contains("=false"),
        "the lowered `else {{ t = false }}` ladder must be gone: {out1}"
    );
    assert_eq!(
        out1.matches("if").count(),
        1,
        "the chain should collapse to a single `if`: {out1}"
    );
    let out2 = compact_with_passes(&out1, Profile::Max);
    assert!(
        out2.len() <= out1.len(),
        "re-minifying must not grow (idempotence): {} -> {} bytes\n  {out1}\n  {out2}",
        out1.len(),
        out2.len()
    );
}

#[test]
fn short_circuit_value_position_left_lowered_and_idempotent() {
    // Folding a value-position `&&` that feeds an expression would create an
    // `&&` store the forwarder cannot collapse, so it stays lowered.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3u) {
            let a = i32(g.x);
            let b = i32(g.y);
            let hit = a < 10 && b < 20;
            buf[0] = select(0, 1, hit) | 2;
        }
    "#;
    let out1 = compact_with_passes(src, Profile::Max);
    let out2 = compact_with_passes(&out1, Profile::Max);
    assert!(
        out2.len() <= out1.len(),
        "value-position && must be idempotent: {} -> {} bytes\n  {out1}\n  {out2}",
        out1.len(),
        out2.len()
    );
}

// MARK: short-circuit forwarding correctness regressions

#[test]
fn forward_copy_chain_preserves_value_not_zero_init() {
    // Redirects must resolve transitively; otherwise the intermediate load is
    // orphaned and reads the removed local's zero-init.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> g: array<i32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            let k = id.x;
            var a = g[k] + 1;
            var b = a;
            g[k] = b;
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("++") || out.contains("+ 1") || out.contains("+1"),
        "forwarded copy chain must preserve g[k]+1 (not zero-init 0): {out}"
    );
}

#[test]
fn short_circuit_value_used_in_if_keeps_body_reachable() {
    // `var ok = a && b; if ok {..}` lowers to an inner `&&` temp plus the `ok`
    // copy; forwarding both must not leave a never-assigned `var ok: bool`
    // that reads false.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> g: i32;
        @compute @workgroup_size(1) fn main() {
            var ok = g > 0 && g < 10;
            if (ok) { g = 1; }
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("&&"),
        "guard must keep its `&&` condition: {out}"
    );
    assert!(
        !out.contains(": bool") && !out.contains(":bool"),
        "no never-assigned bool temp should remain (dead-body bug): {out}"
    );
}

#[test]
fn mixed_and_or_logical_is_parenthesized() {
    // WGSL gives `&&`/`||` no relative precedence; naga accepts the bare form,
    // so check the text.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> g: f32;
        @fragment fn main(@location(0) p: vec4f) -> @location(0) f32 {
            if (p.x > 2.0 || (p.y > 1.0 && p.z > 1.0)) { g = 7.0; }
            return g;
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("||(") || out.contains("|| ("),
        "`&&` inside `||` must be parenthesized: {out}"
    );
}

#[test]
fn bitwise_operand_of_logical_is_parenthesized() {
    // `&&`/`||` operands are `relational_expression`s, which cannot contain a
    // bitwise expression; naga round-trips the bare form but Tint rejects
    // "mixing '|' and '&&' requires parenthesis".  The `fn` operands are bool
    // atoms, so `)&&` / `)||` can only come from the bitwise wrap.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> g: i32;
        fn h(a: bool, b: bool, c: bool) -> bool { return (a | b) && c; }
        fn k(a: bool, b: bool, c: bool) -> bool { return (a & b) || c; }
        @compute @workgroup_size(1) fn main() {
            if (h(g > 0, g > 1, g > 2)) { g = 1; }
            if (k(g > 3, g > 4, g > 5)) { g = 2; }
        }
        "#,
        Profile::Max,
    );
    // Whitespace-normalised so beautify mode does not matter.
    let compact: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        compact.contains(")&&"),
        "`|` operand of `&&` must be parenthesized (no bare `b|c&&d`): {out}"
    );
    assert!(
        compact.contains(")||"),
        "`&` operand of `||` must be parenthesized (no bare `b&c||d`): {out}"
    );
}

#[test]
fn loop_invariant_not_forwarded_into_nested_loop_guard() {
    // `forward_single_store_locals` must only forward within one block:
    // folding the outer-block invariant into the inner guard (`1 < 10` ->
    // `true`) lets dead-branch strip the loop's only exit, leaving a bare
    // `loop {}` Tint rejects ("loop does not exit").
    let out = compact_with_passes(
        r#"
        @fragment fn main() -> @location(0) vec4f {
            var r : i32 = 0;
            var c : i32 = 0;
            loop {
                if (c < 1) {} else { break; }
                if (r >= 0) { break; }
                r = r + 1;
                var n : i32 = 10;
                var d : i32 = 0;
                loop {
                    let x = n;
                    if (1 < x) {} else { break; }
                    d = d + 1;
                }
                c = c + 1;
            }
            return vec4f(f32(r));
        }
        "#,
        Profile::Max,
    );
    let compact: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        compact.contains("1<"),
        "inner loop guard must survive (invariant not forwarded into it): {out}"
    );
}

#[test]
fn read_before_write_local_not_forwarded() {
    // A forward must key on the load's materialisation (`Emit`), not a later
    // consumer: `Load(t)` is emitted before the store, so `snap` holds the
    // zero-init.  The stored literal is dead; reaching the output means `snap`
    // was forwarded.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> out: u32;
        fn f() -> u32 {
            var t: u32;
            let snap = t;
            t = 12345u;
            return snap;
        }
        @compute @workgroup_size(1) fn main() { out = f(); }
        "#,
        Profile::Max,
    );
    assert!(
        !out.contains("12345"),
        "read-before-write `snap` must stay the zero-init value, not the \
         forwarded store: {out}"
    );
}
