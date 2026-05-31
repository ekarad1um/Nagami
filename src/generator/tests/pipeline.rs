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

#[test]
fn workgroup_uniform_load_reused_in_condition_not_duplicated() {
    // The for-loop reconstruction binds a guard WorkGroupUniformLoad to inline
    // `workgroupUniformLoad(&p)` text and re-emits it at every occurrence.  A
    // result reused in the condition (`w*w < n`) would execute the barrier /
    // uniform load twice per iteration - a semantic change re-validation does
    // not catch.  The generator must fall back to plain `loop` emission, which
    // binds the load to a single `let`.
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

/// A `workgroupUniformLoad` preload whose result is NEVER used by the loop
/// condition or update (a barrier kept purely for its side effect) must not be
/// dropped by for-loop reconstruction.  A preload is materialised only where
/// its `result` is emitted; with zero uses in the for-header it would vanish
/// entirely, silently deleting the workgroup barrier.  The preload-safety
/// predicate requires EXACTLY one emission, so this stays a plain loop where
/// the preload keeps its own statement.
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

/// Companion: a `continuing` block holding a `workgroupUniformLoad` whose
/// result is unused must also keep the barrier.  With no core update statement
/// the preload has no for-update clause to be emitted into, so for-conversion
/// must be refused.
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
    // A body WorkGroupUniformLoad result used in the tail INSIDE an expression
    // (`outv[i] = w + 1u`) must block for-reconstruction.  The tail-use oracle
    // must recurse into emitted expression children; a flat operand-equality
    // check would miss it, drop the preload's `let`, and emit an undeclared
    // `_e<n>` - invalid WGSL.
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
    // `w` used as the `compare` operand of atomicCompareExchangeWeak - a
    // statement-operand position the old oracle's `_ => false` / partial Atomic
    // arm missed entirely.
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
    // When a body WorkGroupUniformLoad preload result is used AFTER the guard,
    // `try_emit_for_loop` bails to plain `loop` emission.  The counter-var
    // suppression decision (`is_for_loop_candidate`) must agree, or the
    // counter is left undeclared - invalid WGSL.  Assert the generator output
    // re-validates (it would not if `i` were undeclared).
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
    // The counter `i` must be declared (the generator must not have suppressed
    // its `var` while emitting a plain `loop`).
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_uniform_load_reused_in_update_not_duplicated() {
    // Same hazard for a continuing-block update preload reused across the
    // update statement (`outv[w] = w` materialises `w` twice).
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

/// A counter whose `var` is BOTH absorbed into a reconstructed `for` init AND
/// still flagged deferred must have its deferred flag cleared when the for-init
/// declares it.  Otherwise its in-body update re-declares it -
/// `for(var b=0u;b<2;){...var b=b+1;}` - and the body `var b` SHADOWS the
/// for-init counter, so the counter never advances: an infinite loop.  The
/// output is valid WGSL (shadowing is legal), so re-validation alone cannot
/// catch it; assert the for-init counter is not re-declared inside the loop.
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
    // `compact_with_passes` beautifies (`for (var b = 0u; ...)`), so split on the
    // spaced form and check for a spaced re-declaration `var <counter> =`.
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

/// Global array swap: after passes collapse the temporaries, the IR is
/// `let _a = A[0]; let _b = A[1]; A[0] = _b; A[1] = _a;`.  `_a` is read at
/// `A[1] = _a` AFTER `A[0]` was overwritten, so it must be bound; inlining it
/// produced the miscompile `A[0] = A[1]; A[1] = A[0]` (both = old A[1]).
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

/// A loop whose `continuing` update reads a body-local snapshot of a value the
/// body mutates must NOT be reconstructed as a `for(...)` loop: the update
/// clause is emitted into the for-header BEFORE the body binding, so it would
/// inline the snapshot as the post-write place.  The conversion must bail to a
/// plain `loop { ... continuing { ... } }` where the body binding precedes the
/// continuing use.
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
    // Broken for-conversion emits `for(...; i += A[0]) {...}`, reading the
    // post-increment A[0].  The fix bails to a plain loop with `continuing`.
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

/// A load taken BEFORE a loop and used AFTER it, where the loop body writes the
/// loaded place, must be bound: the back-edge means a later iteration's write
/// reaches the post-loop use, so inlining would read the mutated value.
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
    // The snapshot of `g` must be let-bound before the loop; otherwise both
    // stores read the post-loop `g`.
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

/// Non-regression: an ordinary counted loop whose update does NOT reference a
/// must-bind load still reconstructs as a `for(...)` loop.
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

/// A `continuing` `workgroupUniformLoad` whose POINTER indexes by a body-local
/// must-bind load must also block for-conversion: the preload pointer is
/// relocated into the for-update slot (emitted before the body binding), so
/// `&W[snap]` would inline `snap` as its post-write place.  The fix bails to a
/// plain loop where the body binding precedes the preload.
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
    // Broken for-conversion hoists the preload into the header:
    // `for(...; B += workgroupUniformLoad(&W[a[0]])) {...}`, indexing by the
    // post-swap a[0].  The fix bails to a plain loop with `continuing`.
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

/// A `workgroupUniformLoad` GUARD PRELOAD relocated into the for-condition is a
/// barrier evaluated at iteration top.  A must-bind load defined before it -
/// here `A[0]`, snapshotted pre-barrier in the source and read by both the guard
/// and the body - would, after for-conversion, be re-read POST-barrier (the
/// header runs before the body), yielding a different cross-invocation value.
/// The conversion must bail to a plain loop where the snapshot precedes the
/// barrier.
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
        .find("= A[0]")
        .expect("the A[0] snapshot must be let-bound");
    let barrier = out
        .find("workgroupUniformLoad")
        .expect("the barrier must be preserved");
    assert!(
        snap < barrier,
        "the A[0] snapshot must be bound BEFORE the workgroupUniformLoad barrier: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Companion to [`for_loop_guard_preload_must_not_read_must_bind_load_post_barrier`]:
/// the must-bind load `A[0]` is NOT referenced by the guard condition (only by the
/// body tail), so checking the condition's operand cone alone would miss it.  The
/// hazard is the load being DEFINED in the pre-guard region (ahead of the relocated
/// barrier), so the bail keys on that region, not the condition.
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
        .find("= A[0]")
        .expect("the A[0] snapshot must be let-bound");
    let barrier = out
        .find("workgroupUniformLoad")
        .expect("the barrier must be preserved");
    assert!(
        snap < barrier,
        "the A[0] snapshot must precede the barrier even when the condition does not read it: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Call-result inlining (reads_locals)

/// A single-use `Call` whose argument reads a function-local BY VALUE
/// (`f(arr[i])`) is inlined at its use site when no statement between the call
/// and the use writes a local the argument read.  This exercises the
/// `reads_locals` path in `find_inlineable_calls`, dormant until
/// `arg_has_pointer_to_local` stopped over-counting by-value reads as
/// pointers-to-local; a regression there leaves the call `let`-bound.
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

/// Soundness companion: when a `Store` to a local the argument read sits between
/// the call and its use, the call must NOT be inlined past it - inlining would
/// re-read the post-store value.  `reads_locals` keeps the call bound, so its
/// argument is evaluated before the store and the use reads the snapshot.  (The
/// array element + runtime store value keep the store genuinely live and block
/// the SSA splitting that would make a scalar case safe to inline.)
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
    // `O` is mangled, so recover its emitted name from the storage declaration
    // rather than guessing the mangler's lettering (the previous `A[0]` anchor
    // only worked by coincidence).
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
    // The snapshot use must reference the BOUND result by name, not re-invoke the
    // call: an unsound inline would relocate `f(arr[i])` past the store, so the
    // first storage-array store's RHS (`<arr>[0] = ...`) would contain a call `(`.
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
