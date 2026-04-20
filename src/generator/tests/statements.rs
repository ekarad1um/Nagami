//! Statement-level emission tests: `switch`, `continue`, `discard`,
//! barrier family, `loop` with `continuing` / `break_if`, for-loop
//! reconstruction and init absorption, nested-loop ref-count
//! adjustments, diagnostic and enable directives, and call-result
//! inlining across control flow.  Each MARK block targets one
//! statement shape or inlining scenario.

use super::helpers::*;

// MARK: switch statement

#[test]
fn switch_i32_roundtrip() {
    let src = r#"
            fn f(x: i32) -> i32 {
                var r = 0i;
                switch x {
                    case 0: { r = 10; }
                    case 1: { r = 20; }
                    default: { r = 30; }
                }
                return r;
            }
            @compute @workgroup_size(1) fn main() { _ = f(0); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("switch"),
        "switch keyword should be present: {out}"
    );
    assert!(
        out.contains("case"),
        "case keyword should be present: {out}"
    );
    assert!(
        out.contains("default"),
        "default keyword should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn switch_u32_case_has_suffix() {
    let src = r#"
            fn f(x: u32) -> u32 {
                var r = 0u;
                switch x {
                    case 0u: { r = 1u; }
                    case 1u: { r = 2u; }
                    default: { r = 3u; }
                }
                return r;
            }
            @compute @workgroup_size(1) fn main() { _ = f(0u); }
        "#;
    let out = compact(src);
    // U32 case values must have the `u` suffix in emitted WGSL.
    assert!(
        out.contains("case 0u") || out.contains("case 0u{"),
        "u32 case should have u suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn switch_coalesced_cases_roundtrip() {
    let src = r#"
            fn f(x: i32) -> i32 {
                var r = 0i;
                switch x {
                    case 0, 1, 2: { r = 10; }
                    default: { r = 20; }
                }
                return r;
            }
            @compute @workgroup_size(1) fn main() { _ = f(0); }
        "#;
    let out = compact(src);
    // Coalesced case labels should not emit `fallthrough`.
    assert!(
        !out.contains("fallthrough"),
        "fallthrough should not appear: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: continue statement

#[test]
fn continue_in_loop_roundtrip() {
    let src = r#"
            fn f() -> i32 {
                var sum = 0i;
                var i = 0i;
                loop {
                    if i >= 5 { break; }
                    i += 1;
                    if i == 3 { continue; }
                    sum += i;
                }
                return sum;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("continue"),
        "continue statement should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: discard (Kill) statement

#[test]
fn discard_in_fragment_roundtrip() {
    let src = r#"
            @fragment fn main(@location(0) alpha: f32) -> @location(0) vec4f {
                if alpha < 0.5 { discard; }
                return vec4f(1.0, 0.0, 0.0, alpha);
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("discard"),
        "discard statement should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Barrier statements

#[test]
fn workgroup_barrier_roundtrip() {
    let src = r#"
            var<workgroup> wg_data: array<f32, 64>;
            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_index) lid: u32) {
                wg_data[lid] = f32(lid);
                workgroupBarrier();
                _ = wg_data[0];
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("workgroupBarrier()"),
        "workgroupBarrier should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn storage_barrier_roundtrip() {
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<f32, 64>;
            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_index) lid: u32) {
                buf[lid] = f32(lid);
                storageBarrier();
                _ = buf[0];
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("storageBarrier()"),
        "storageBarrier should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn texture_barrier_roundtrip() {
    let src = r#"
            @group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, read_write>;
            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_index) lid: u32) {
                textureStore(tex, vec2u(lid, 0u), vec4f(1.0, 0.0, 0.0, 1.0));
                textureBarrier();
                let v = textureLoad(tex, vec2u(0u, 0u));
                _ = v;
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("textureBarrier()"),
        "textureBarrier should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn combined_workgroup_storage_barrier_roundtrip() {
    // Tests that both barriers appear and the output is valid.
    let src = r#"
        var<workgroup> wg_data: array<f32, 64>;
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 64>;
        @compute @workgroup_size(64)
        fn main(@builtin(local_invocation_index) lid: u32) {
            wg_data[lid] = f32(lid);
            buf[lid] = f32(lid);
            workgroupBarrier();
            storageBarrier();
            _ = wg_data[0];
            _ = buf[0];
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("workgroupBarrier()"),
        "workgroupBarrier should be present: {out}"
    );
    assert!(
        out.contains("storageBarrier()"),
        "storageBarrier should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Loop with continuing and break_if

#[test]
fn loop_with_continuing_roundtrip() {
    // This loop pattern is converted to a for-loop:
    // `loop { if i >= 10 { break; } ... continuing { i += 1; } }` -> `for(;i<10;i+=1){...}`
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var i = 0i;
                var sum = 0f;
                loop {
                    if i >= 10 { break; }
                    sum += 1.0;
                    continuing {
                        i += 1;
                    }
                }
                return vec4f(sum, 0.0, 0.0, 1.0);
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "loop with continuing should be reconstructed as for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn loop_with_break_if_roundtrip() {
    // `break if` must appear inside the `continuing` block in WGSL.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var i = 0i;
                var sum = 0f;
                loop {
                    sum += 1.0;
                    continuing {
                        i += 1;
                        break if i >= 10;
                    }
                }
                return vec4f(sum, 0.0, 0.0, 1.0);
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("break if"),
        "break if should be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: For-loop reconstruction

#[test]
fn for_loop_from_reject_break_pattern() {
    // `if cond {} else { break; }` - naga-lowered from `for(;cond;)`.
    let src = r#"
            fn f() -> i32 {
                var sum = 0i;
                for (var i = 0i; i < 10; i += 1) {
                    sum += i;
                }
                return sum;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "should reconstruct as for-loop: {out}"
    );
    assert!(
        !out.contains("loop"),
        "should not contain loop keyword: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_from_accept_break_pattern() {
    // `if cond { break; }` - condition is negated for the for-header.
    let src = r#"
            fn f() -> i32 {
                var sum = 0i;
                var i = 0i;
                loop {
                    if i >= 10 { break; }
                    sum += i;
                    continuing {
                        i += 1;
                    }
                }
                return sum;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "should reconstruct as for-loop: {out}"
    );
    assert!(
        out.contains("<10") || out.contains("< 10"),
        "exit condition >= should be negated to <: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_update_clause() {
    // Loop with empty continuing -> `for(;cond;)` with no update.
    let src = r#"
            fn f() -> i32 {
                var sum = 0i;
                var i = 0i;
                loop {
                    if i >= 5 { break; }
                    i += 1;
                    sum += i;
                }
                return sum;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("for("),
        "should reconstruct as for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn break_if_stays_as_loop() {
    // `break if` in the continuing block must remain as `loop`.
    let src = r#"
            fn f() -> i32 {
                var i = 0i;
                loop {
                    i += 1;
                    continuing {
                        break if i >= 10;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("break if"),
        "break if in continuing must stay: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_with_compound_assign_update() {
    // Compound assign `i += 1` should appear in the for-update clause.
    let src = r#"
            fn f() -> f32 {
                var sum = 0f;
                for (var i = 0f; i < 5.0; i += 1.0) {
                    sum += i;
                }
                return sum;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(out.contains("for("), "should be a for-loop: {out}");
    assert!(
        out.contains("+="),
        "compound assign should appear in update: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn loop_body_not_starting_with_if_break_stays_as_loop() {
    // Loop whose body doesn't start with if-break guard stays as `loop`.
    let src = r#"
            fn f() -> i32 {
                var i = 0i;
                loop {
                    i += 1;
                    if i >= 10 { break; }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    // The if-break is NOT the first statement, so no for-loop conversion.
    assert!(!out.contains("for("), "should stay as loop: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Diagnostic directives

#[test]
fn module_level_diagnostic_preserved() {
    let src = r#"
        diagnostic(off, derivative_uniformity);
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            return textureSample(t, s, uv);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("diagnostic(off,derivative_uniformity)"),
        "module-level diagnostic directive must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn per_function_diagnostic_preserved() {
    let src = r#"
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @diagnostic(off, derivative_uniformity)
        @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            return textureSample(t, s, uv);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@diagnostic(off,derivative_uniformity)"),
        "per-function @diagnostic attribute must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn both_module_and_function_diagnostic() {
    let src = r#"
        diagnostic(off, derivative_uniformity);
        @group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @diagnostic(off, derivative_uniformity)
        @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            return textureSample(t, s, uv);
        }
    "#;
    let out = compact(src);
    // Module-level (no @) and function-level (with @) should both be present.
    assert!(
        out.contains("diagnostic(off,derivative_uniformity);"),
        "module-level diagnostic must be present: {out}"
    );
    assert!(
        out.contains("@diagnostic(off,derivative_uniformity)"),
        "per-function @diagnostic must be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Enable directives

#[test]
fn enable_f16_emitted_for_half_precision() {
    let src = r#"
        enable f16;
        @compute @workgroup_size(1)
        fn main() {
            var x: f16 = 1.0h;
            _ = x;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("enable f16;"),
        "enable f16 should be emitted when f16 types are used: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn enable_clip_distances() {
    let src = r#"
        enable clip_distances;
        struct VertexOutput {
            @builtin(position) pos: vec4<f32>,
            @builtin(clip_distances) clip: array<f32, 1>,
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
            var o: VertexOutput;
            o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            o.clip[0] = 1.0;
            return o;
        }
    "#;
    let out = compact(src);
    assert!(
        out.starts_with("enable clip_distances;"),
        "clip_distances enable must be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn enable_draw_index() {
    let src = r#"
        enable draw_index;
        @group(0) @binding(0) var<storage, read_write> buf: array<u32, 64>;
        @vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(draw_index) di: u32) -> @builtin(position) vec4<f32> {
            buf[vi] = di;
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.starts_with("enable draw_index;"),
        "draw_index enable must be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn enable_primitive_index() {
    let src = r#"
        enable primitive_index;
        @group(0) @binding(0) var<storage, read_write> buf: array<u32, 64>;
        @fragment fn fs(@builtin(primitive_index) pi: u32) -> @location(0) vec4<f32> {
            buf[0] = pi;
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.starts_with("enable primitive_index;"),
        "primitive_index enable must be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Call result inlining

#[test]
fn single_use_call_result_inlined() {
    let src = r#"
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @fragment fn fs() -> @location(0) vec4<f32> {
            let a = helper(1.0);
            return vec4<f32>(a, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    // The single-use call result should be inlined directly,
    // so the fs body should NOT have a let binding for the call.
    assert!(
        !out.contains("=helper("),
        "single-use call result should be inlined, not let-bound: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn multi_use_call_result_not_inlined() {
    let src = r#"
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @fragment fn fs() -> @location(0) vec4<f32> {
            let a = helper(1.0);
            return vec4<f32>(a, a, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    // Multi-use call result must be bound to a let variable.
    assert!(
        out.contains("let "),
        "multi-use call result should be let-bound: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn consecutive_single_use_calls_inlined() {
    let src = r#"
        fn f(x: f32) -> f32 { return x + 1.0; }
        @fragment fn fs() -> @location(0) vec4<f32> {
            let a = f(1.0);
            let b = f(2.0);
            let c = f(3.0);
            return vec4<f32>(a, b, c, 1.0);
        }
    "#;
    let out = compact(src);
    // All three single-use call results should be inlined.
    let call_count = out.matches("f(").count();
    // The function definition has "fn f(" and the 3 inline calls have "f("
    assert!(
        call_count >= 4,
        "expected fn def + 3 inline calls, got {call_count} matches: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn call_before_store_not_inlined() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @compute @workgroup_size(1) fn main() {
            let a = helper(1.0);
            buf[0] = 42.0;
            buf[1] = a;
        }
    "#;
    let out = compact(src);
    // The call result is used after a Store - must NOT be inlined.
    assert!(
        out.contains("let "),
        "call before store should remain as let binding: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Call result inlining across control flow / side effects

// Regression tests for `find_inlineable_calls` consumption detection.
// Before the fix, a subsequent side-effecting statement (If, non-local
// Store, etc.) would clear the pending set even when the single-use
// Call result had already been consumed by an earlier Emit/Store/Return.

/// Use is in an `if` condition (direct handle reference) AFTER the Call.
/// The `if` clears `pending`, but consumption must fire first.
#[test]
fn call_result_used_before_if_is_inlined() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @compute @workgroup_size(1) fn main() {
            let a = helper(1.0);
            if a > 0.0 {
                buf[0] = 42.0;
            }
        }
    "#;
    let out = compact(src);
    // Inlined form should produce `if helper(...)>0{...}` with no let.
    assert!(
        out.contains("if helper("),
        "call result consumed by `if` condition should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Use is in a local Store BEFORE a later `if`.  Variant of the shader that
/// motivated this fix: `let D = B(..); M = vec3(M.x, M.yz * D); if (...) {}`.
#[test]
fn call_result_used_in_local_store_before_if_is_inlined() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        fn rot(b: f32) -> mat2x2<f32> {
            let c = cos(b);
            let d = sin(b);
            return mat2x2<f32>(vec2<f32>(c, d), vec2<f32>(-d, c));
        }
        @compute @workgroup_size(1) fn main() {
            var M: vec3<f32> = vec3<f32>(6.0);
            let D = rot(0.5);
            M = vec3<f32>(M.x, M.yz * D);
            // After the Call+Emit+Store, an `if` appears later.
            if M.x > 0.0 {
                buf[0] = M.y;
            }
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("=rot(") && !out.contains("= rot("),
        "call result consumed in local Store before `if` should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Use is in Return BEFORE a later non-local Store (via control flow
/// structure that guarantees Return happens first).  The Return consumes
/// the Call result.
#[test]
fn call_result_used_in_return_is_inlined() {
    let src = r#"
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @fragment fn fs() -> @location(0) vec4<f32> {
            let a = helper(3.0);
            let v = vec4<f32>(a, 0.0, 0.0, 1.0); // consumed by Emit
            return v;
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("=helper("),
        "call result used before Return should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Negative: a non-local Store between the Call and its use must keep
/// the call result let-bound (to preserve call-vs-store ordering even
/// when the callee might be impure).
#[test]
fn call_result_with_nonlocal_store_before_use_not_inlined() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
        fn helper(x: f32) -> f32 { return x * 2.0; }
        @compute @workgroup_size(1) fn main() {
            let a = helper(1.0);
            buf[0] = 42.0;   // non-local Store BEFORE `a` is used
            buf[1] = a;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("let "),
        "call with intervening non-local store must stay let-bound: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Negative: a second Call between the first Call and its use must
/// prevent inlining.  Uses two *impure* functions (both write to `buf`) so
/// the pure-function inliner pass cannot eliminate them and hide the bug.
/// Inlining would reorder `a_` after `b_`, which is observable when both
/// callees write to the same memory location.
#[test]
fn call_result_with_intervening_call_not_inlined() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 8>;
        fn a_(i: i32) -> i32 { buf[0] = 1; return i; }
        fn b_(i: i32) -> i32 { buf[0] = 2; return i; }
        @compute @workgroup_size(1) fn main() {
            let va = a_(1);
            let vb = b_(2);     // intervening impure Call
            buf[5] = va;        // use of `va` happens AFTER b_'s side effect
            buf[6] = vb;
        }
    "#;
    let out = compact(src);
    // Neither call may be inlined: inlining `a_(1)` into `buf[5] = a_(1)`
    // reorders it after `b_(2)`, which is observable via the shared buf[0]
    // write.  Both function names must still appear as `let`-bound calls.
    assert!(
        (out.contains("=a_(") || out.contains("= a_("))
            && (out.contains("=b_(") || out.contains("= b_(")),
        "both calls must stay let-bound when their use crosses another call: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Positive: a Call whose argument *is* the prior Call's single-use result.
/// The argument consumes the pending handle, so nesting the call directly
/// is safe and preferable.
#[test]
fn call_result_used_as_next_call_argument_is_inlined() {
    let src = r#"
        fn f(x: f32) -> f32 { return x * 2.0; }
        fn g(x: f32) -> f32 { return x + 1.0; }
        @fragment fn fs() -> @location(0) vec4<f32> {
            let a = f(1.0);
            let b = g(a);      // `a` is consumed as g's argument
            return vec4<f32>(b, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    // Expect a nested form like `g(f(1))` with no binding for f's result.
    assert!(
        out.contains("g(f(") || out.contains("g(f("),
        "call used as next call's argument should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: For-loop Block unwrapping

#[test]
fn for_loop_no_double_braces() {
    // naga wraps the user's for-body in `Statement::Block`, which would
    // produce `for(...){ { body } }`.  The emitter should unwrap the inner block.
    let src = r#"
        fn f() -> f32 {
            var sum = 0f;
            for (var i = 0u; i < 10u; i += 1u) {
                sum += f32(i);
            }
            return sum;
        }
        @compute @workgroup_size(1) fn main() { _ = f(); }
    "#;
    let out = compact_beautified(src);
    assert!(
        !out.contains("{ {") && !out.contains("{\n    {"),
        "for-loop body should not have double braces: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: For-loop init absorption

#[test]
fn for_loop_absorbs_initialized_local() {
    // `var i = 0i; loop { if i>=10 {break;} ... continuing { i+=1; } }`
    // should become `for(var i=0;i<10;i+=1){...}` with var absorbed.
    let src = r#"
        fn f() -> i32 {
            var sum = 0i;
            var i = 0i;
            loop {
                if i >= 10 { break; }
                sum += i;
                continuing {
                    i += 1;
                }
            }
            return sum;
        }
        @compute @workgroup_size(1) fn main() { _ = f(); }
    "#;
    let out = compact(src);
    assert!(
        out.contains("for(var i=0"),
        "initialized local should be absorbed into for-init: {out}"
    );
    // The standalone `var i` declaration should NOT appear before the for-loop.
    let for_pos = out.find("for(").unwrap();
    let before_for = &out[..for_pos];
    assert!(
        !before_for.contains("var i"),
        "var i should not appear before the for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_absorbs_uninit_local() {
    // After dead-init removal, a loop counter may have no init (WGSL
    // zero-initialises).  The generator should emit `for(var i:i32;...)`.
    let src = r#"
        fn f() -> i32 {
            var sum = 0i;
            var i: i32;
            loop {
                if i >= 10 { break; }
                sum += i;
                continuing {
                    i += 1;
                }
            }
            return sum;
        }
        @compute @workgroup_size(1) fn main() { _ = f(); }
    "#;
    let out = compact(src);
    assert!(
        out.contains("for(var "),
        "uninit local should be absorbed into for-init: {out}"
    );
    // The standalone `var i` declaration should NOT appear before the for-loop.
    if let Some(for_pos) = out.find("for(") {
        let before_for = &out[..for_pos];
        assert!(
            !before_for.contains("var i"),
            "var i should not appear before the for-loop: {out}"
        );
    }
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_when_used_after_loop() {
    // var i is used after the loop, so it must NOT be absorbed.
    let src = r#"
        fn f() -> i32 {
            var i = 0i;
            loop {
                if i >= 10 { break; }
                continuing {
                    i += 1;
                }
            }
            return i;
        }
        @compute @workgroup_size(1) fn main() { _ = f(); }
    "#;
    let out = compact(src);
    // The var should be declared outside the for-loop.
    assert!(
        !out.contains("for(var i"),
        "var used after loop should not be absorbed into for-init: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_from_break_if_loop() {
    // `break if` loops are not for-loop candidates, so the var must stay.
    let src = r#"
        fn f() -> i32 {
            var i = 0i;
            loop {
                i += 1;
                continuing {
                    break if i >= 10;
                }
            }
            return i;
        }
        @compute @workgroup_size(1) fn main() { _ = f(); }
    "#;
    let out = compact(src);
    assert!(
        out.contains("break if"),
        "break if should be preserved: {out}"
    );
    assert!(
        !out.contains("for(var i"),
        "break-if loop should not absorb var into for-init: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_when_continuing_has_multiple_stmts() {
    // Continuing has 2 non-Emit statements (store to i + store to buf).
    // try_emit_for_loop rejects this, so the var must NOT be suppressed.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 64>;
        fn f() {
            var i = 0f;
            loop {
                if i >= 10 { break; }
                continuing {
                    i += 1;
                    buf[0] = 99.0;
                }
            }
        }
        @compute @workgroup_size(1) fn main() { f(); }
    "#;
    let out = compact(src);
    // The loop should NOT be converted to for-loop.
    assert!(
        !out.contains("for("),
        "loop with multi-stmt continuing should not become for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_when_no_if_break_guard() {
    // Body doesn't start with an If-break guard - the Call comes first.
    // try_emit_for_loop rejects this, so the var must NOT be suppressed.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32, 64>;
        fn side_effect() { buf[0] = 1.0; }
        fn f() {
            var i = 0;
            loop {
                side_effect();
                if i >= 10 { break; }
                continuing {
                    i += 1;
                }
            }
        }
        @compute @workgroup_size(1) fn main() { f(); }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_update_call_uses_call_emission_rules() {
    // The update-clause call must use the same emission path as ordinary
    // calls so that mangled function names and pointer arguments stay valid.
    let src = r#"
        fn increment_counter_stepper(p: ptr<function, i32>) {
            *p = *p + 1;
        }

        fn f() -> i32 {
            var i = 0;
            for (; i < 4; increment_counter_stepper(&i)) {}
            return i;
        }

        @compute @workgroup_size(1)
        fn main() {
            _ = f();
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    assert!(
        out.contains("for ("),
        "loop should stay reconstructed as for: {out}"
    );
    assert!(
        !out.contains("increment_counter_stepper"),
        "helper name should be mangled consistently in update clause: {out}"
    );
    assert!(
        out.contains('&'),
        "pointer argument should be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Nested for-loop ref_count adjustment

#[test]
fn nested_for_loops_both_reconstruct() {
    // Two independent nested for-loops should both be reconstructed.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum = 0.0;
            for (var i = 0.0; i < 4.0; i += 1.0) {
                for (var j = 0.0; j < 4.0; j += 1.0) {
                    sum += i * j;
                }
            }
            out[0] = sum;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn nested_for_loops_no_redundant_let_for_loop_var() {
    // The loop variable should be inlined in the body, not bound to a
    // redundant `let` (the bug fixed by the ref_count adjustment).
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum = 0.0;
            for (var i = 0.0; i < 4.0; i += 1.0) {
                let a = (i + 0.5) * 0.1;
                for (var j = 0.0; j < 4.0; j += 1.0) {
                    let b = (j + 0.5) * 0.1;
                    sum += a * b;
                }
            }
            out[0] = sum;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    // Both loops should be for-loops.
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn nested_for_loop_inner_condition_references_outer_var() {
    // Inner loop condition `j < i` references the outer loop variable.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum = 0.0;
            for (var i = 0.0; i < 8.0; i += 1.0) {
                for (var j = 0.0; j < i; j += 1.0) {
                    sum += (i + 0.5) * (j + 0.5);
                }
            }
            out[0] = sum;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn triple_nested_for_loops() {
    // Three levels of nesting - all should reconstruct as for-loops.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum = 0.0;
            for (var i = 0.0; i < 4.0; i += 1.0) {
                for (var j = 0.0; j < 4.0; j += 1.0) {
                    for (var k = 0.0; k < 4.0; k += 1.0) {
                        sum += (i + 0.5) * (j + 0.5) * (k + 0.5);
                    }
                }
            }
            out[0] = sum;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 3,
        "expected 3 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn nested_for_loop_outer_var_multi_use_in_inner() {
    // Outer loop variable used multiple times in the inner body.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum = 0.0;
            for (var i = 0.0; i < 4.0; i += 1.0) {
                for (var j = 0.0; j < 4.0; j += 1.0) {
                    sum += i * i + j;
                }
            }
            out[0] = sum;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn sequential_for_loops_independent() {
    // Two sequential (not nested) for-loops.  Ref_count adjustments must
    // not leak between them.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<f32, 64>;
        @compute @workgroup_size(1)
        fn main() {
            var sum1 = 0.0;
            for (var i = 0.0; i < 4.0; i += 1.0) {
                sum1 += (i + 0.5) * 0.1;
            }
            var sum2 = 0.0;
            for (var j = 0.0; j < 4.0; j += 1.0) {
                sum2 += (j + 0.5) * 0.1;
            }
            out[0] = sum1 + sum2;
        }
    "#;
    let out = compact_with_passes(src, Profile::Aggressive);
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}
