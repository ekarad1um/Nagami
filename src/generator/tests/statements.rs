//! Statement emission: `switch`, `continue`, `discard`, barriers, `loop` with
//! `continuing`/`break_if`, for-loop reconstruction and init absorption,
//! nested-loop ref counts, directives, and call-result inlining across
//! control flow.

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
    assert!(
        out.contains("case 0u") || out.contains("case 0u{"),
        "u32 case should have u suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A literal selector must carry the case labels' suffix: `switch 0 { case 0u:
/// ... }` is a naga selector/case type mismatch.
#[test]
fn switch_with_literal_u32_selector_matches_case_suffix() {
    let src = r#"
            fn f() -> u32 {
                var r = 0u;
                switch 0u {
                    case 0u: { r = 1u; }
                    default: { r = 2u; }
                }
                return r;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("switch 0u"),
        "literal-selector u32 switch must emit selector with `u` suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn switch_with_literal_i32_selector_matches_case_suffix() {
    let src = r#"
            fn f() -> i32 {
                var r = 0i;
                switch 0i {
                    case 0: { r = 1i; }
                    default: { r = 2i; }
                }
                return r;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
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
    // naga lowers `for(;cond;)` to `if cond {} else { break; }`.
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
    let call_count = out.matches("f(").count();
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
    assert!(
        out.contains("let "),
        "call before store should remain as let binding: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Call result inlining across control flow / side effects

// `find_inlineable_calls`: a side-effecting statement must not clear a
// single-use Call result already consumed by an earlier Emit/Store/Return.

/// The `if` clears `pending`; consumption by its condition must fire first.
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
    assert!(
        out.contains("if helper("),
        "call result consumed by `if` condition should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

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

/// The binding preserves call-vs-store order for a possibly impure callee.
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

/// Both callees write `buf[0]`, so the pure-function inliner cannot remove
/// them and inlining `a_` past `b_` would be an observable reorder.
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
    assert!(
        (out.contains("=a_(") || out.contains("= a_("))
            && (out.contains("=b_(") || out.contains("= b_(")),
        "both calls must stay let-bound when their use crosses another call: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The argument consumes the pending handle, so nesting the call is safe.
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
    assert!(
        out.contains("g(f(") || out.contains("g(f("),
        "call used as next call's argument should be inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Inlining would re-read `v` at the use site, after `v = v * 2`; the call
/// text is identical either way, so assert on evaluation order.
#[test]
fn call_arg_reading_local_not_inlined_across_store_to_that_local() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 8>;
        fn impure(x: i32) -> i32 { buf[7] = 99; return x + 100; }
        @compute @workgroup_size(1) fn main() {
            var v: i32 = i32(buf[6]);
            let c = impure(v);   // argument loads local `v`
            v = v * 2;           // store to `v` BEFORE `c` is used
            buf[0] = c;
            buf[1] = v;
        }
    "#;
    let out = compact(src);
    let call_pos = out.find("impure(").expect("call must be emitted");
    let store_pos = out
        .find("v*=2")
        .expect("the `v = v * 2` store must be emitted");
    assert!(
        call_pos < store_pos,
        "call reading `v` must be evaluated BEFORE `v = v*2` (let-bound), not \
         inlined after it: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The callee derefs `&d` at call time, so inlining past `d = d*2` reads the
/// post-store value; only the call's position reveals it.
#[test]
fn call_pointer_arg_not_inlined_across_store_to_pointee() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
        fn g(p: ptr<function, i32>) -> i32 { return *p + 100; }
        @compute @workgroup_size(1) fn main() {
            var d: i32 = i32(buf[3]);
            let c = g(&d);
            d = d * 2;
            buf[0] = c;
            buf[1] = d;
        }
    "#;
    let out = compact(src);
    let call_pos = out.find("g(&").expect("call must be emitted");
    let store_pos = out
        .find("d*=2")
        .expect("the `d = d * 2` store must be emitted");
    assert!(
        call_pos < store_pos,
        "call taking `&d` must be evaluated BEFORE `d = d*2` (let-bound), not \
         inlined after it: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The callee may write the pointee, so inlining past `let t = d;` would move
/// the write after the read; the Store-interference check misses this, so
/// pointer-to-local calls are un-inlineable outright.
#[test]
fn call_pointer_arg_not_inlined_across_later_read_of_pointee() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
        fn g(p: ptr<function, i32>) -> i32 { *p = *p + 100; return *p; }
        @compute @workgroup_size(1) fn main() {
            var d: i32 = i32(buf[3]);
            let c = g(&d);
            let t = d;
            buf[0] = c;
            buf[1] = t;
            buf[2] = t;
        }
    "#;
    let out = compact(src);
    let call_pos = out.find("g(&").expect("call must be emitted");
    let read_pos = out
        .find("=d;")
        .or_else(|| out.find("=d}"))
        .or_else(|| out.rfind("d;"))
        .expect("a read of `d` must be emitted");
    assert!(
        call_pos < read_pos,
        "call taking `&d` (callee writes the pointee) must be evaluated BEFORE \
         the later read of `d`, not inlined past it: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The interference check must stay precise: an argument that does not read
/// the stored local still inlines across the store.
#[test]
fn call_arg_not_reading_stored_local_still_inlines_across_store() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 8>;
        fn pure_add(x: i32) -> i32 { return x + 100; }
        @compute @workgroup_size(1) fn main() {
            var v: i32 = i32(buf[6]);
            let c = pure_add(7);   // argument reads no local
            v = v * 2;             // store to an UNRELATED local
            buf[0] = c;
            buf[1] = v;
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("let ") && !out.contains("var c"),
        "call whose arg does not read `v` should still inline across the \
         `v = v*2` store: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Inlining relocates the call's write to the use site, where left-to-right
/// evaluation reads `g` first (`11 - 1`, not `777 - 1`); only the call's own
/// `let` preserves the order, so assert it is not folded into the store.
#[test]
fn call_writing_global_not_inlined_past_read_of_that_global() {
    let src = r#"
        var<private> g: i32;
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        fn writes_g() -> i32 { g = 777; return 1; }
        @compute @workgroup_size(1) fn main() {
            g = 11;
            let c = writes_g();   // writes g = 777
            let r = g;            // reads the POST-call g (777)
            O[0] = r - c;         // 777 - 1; inlining would give 11 - 1
        }
    "#;
    let out = compact(src);
    let store_rhs = out
        .split("O[0]")
        .nth(1)
        .expect("the O[0] store must be emitted");
    let rhs = store_rhs.split(';').next().unwrap_or("");
    assert!(
        !rhs.contains('('),
        "an impure (global-writing) call must be bound before the read it would \
         reorder against, not inlined into the store RHS: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The purity gate must not over-bind a pure callee.
#[test]
fn pure_call_still_inlines_at_single_use() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        fn dbl(x: i32) -> i32 { return x * 2 + 1; }
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) gi: u32) {
            let c = dbl(i32(gi));
            O[0] = c;
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("let "),
        "a pure single-use call should inline at its use site: {out}"
    );
    assert_valid_wgsl(&out);
}

/// An impure call that is the whole RHS of the immediately-following store to
/// a bare variable neither moves nor gains a sibling, so it inlines despite
/// impurity (the `outv = atomicAdd(...)` idiom).
#[test]
fn impure_call_as_store_rhs_inlines_adjacently() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> ctr: atomic<u32>;
        @group(0) @binding(1) var<storage, read_write> outv: u32;
        fn bump() -> u32 { return atomicAdd(&ctr, 1u); }
        @compute @workgroup_size(1) fn main() {
            outv = bump();
        }
    "#;
    let out = compact(src);
    // The `let` inside `bump` is unrelated, so assert on the store text.
    assert!(
        out.contains("outv=bump()"),
        "an impure call that is the whole RHS of the immediately-following store \
         to a bare variable should inline at that store, not bind: {out}"
    );
    assert_valid_wgsl(&out);
}

/// An unread impure call cannot be DCE'd, so a dead `let x = ...;` would
/// survive forever; the bare call must be emitted instead.
#[test]
fn discarded_impure_call_result_drops_dead_let() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> ctr: atomic<u32>;
        @compute @workgroup_size(1) fn main() {
            atomicAdd(&ctr, 5u);
            atomicAdd(&ctr, 3u);
        }
    "#;
    let out = compact(src);
    assert_eq!(
        out.matches("atomicAdd(&ctr,").count(),
        2,
        "both atomic calls should be emitted: {out}"
    );
    // Only the dead-binding form: unrelated `let`s may exist.
    assert!(
        !out.contains("=atomicAdd"),
        "an unread impure-call result must not bind `let <name>=atomicAdd(...)`: {out}"
    );
    assert_valid_wgsl(&out);
}

/// With every other operand memory-free the call is the statement's sole
/// memory access, so nothing can reorder across its side effect.
#[test]
fn impure_call_inlines_into_memfree_surround_expression() {
    let src = r#"
        var<private> seed: f32;
        @group(0) @binding(0) var<storage, read_write> outv: f32;
        fn prng() -> f32 { seed = seed * 1.1 + 0.3; return seed; }
        @compute @workgroup_size(1) fn main() {
            outv = (prng() - 0.5) * 0.1;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("(prng()-"),
        "the impure call should inline into the literal-only expression: {out}"
    );
    assert!(
        !out.contains("let "),
        "no `let` binding should survive when the call inlines: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The literal sibling leaves the call as the condition's sole memory access.
#[test]
fn impure_call_inlines_into_if_condition_with_literal_sibling() {
    let src = r#"
        var<private> tick: i32;
        @group(0) @binding(0) var<storage, read_write> outv: i32;
        fn step() -> i32 { tick = tick + 1; return tick; }
        @compute @workgroup_size(1) fn main() {
            if (step() == 3) { outv = 1; }
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("if step()=="),
        "the impure call should inline into the if-condition: {out}"
    );
    assert!(
        !out.contains("let "),
        "no `let` binding should survive when the call inlines: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A sibling load (`pre`, of the global `bump` writes) makes the surround
/// memory-reading: inlining `pre + bump()` lets left-to-right evaluation
/// reorder the read against the increment.  Fails the moment the
/// memory-free-surround check admits a load sibling.
#[test]
fn impure_call_not_inlined_when_sibling_reads_memory() {
    let src = r#"
        var<private> g: i32;
        @group(0) @binding(0) var<storage, read_write> outv: i32;
        fn bump() -> i32 { g = g + 1; return g; }
        @compute @workgroup_size(1) fn main() {
            let pre = g;
            outv = pre + bump();
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("=bump()"),
        "an impure call with a memory-reading sibling must stay let-bound: {out}"
    );
    assert!(
        !out.contains("+bump()"),
        "the impure call must NOT inline next to the load sibling: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Writes through a helper's pointer parameter into the caller's own local
/// never escape, so the per-parameter effect analysis must rate the function
/// pure.
#[test]
fn fn_using_param_writing_helper_on_own_local_is_pure() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        fn modf_like(v: f32, ip: ptr<function, f32>) -> f32 { *ip = trunc(v); return v - *ip; }
        fn noise(x: f32) -> f32 { var w: f32; let f = modf_like(x, &w); return f * 2.0 + w; }
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) gi: u32) {
            let c = noise(f32(gi));
            O[0] = i32(c);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("=noise("),
        "a function whose helper writes only its OWN local is pure and should \
         inline at its use: {out}"
    );
    assert!(
        out.contains("noise("),
        "the call must still be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

/// Passing `&global` to a pointer-writing helper is an escaping write; rating
/// it "contained" would inline the call past the read and miscompile.
#[test]
fn fn_writing_global_via_helper_param_not_inlined_past_read() {
    let src = r#"
        var<private> G: i32;
        @group(0) @binding(0) var<storage, read_write> O: array<i32>;
        @group(0) @binding(1) var<uniform> seed: i32;
        fn helper(p: ptr<private, i32>) { *p = 777; }
        fn writes_g() -> i32 { helper(&G); return seed; }   // writes G THROUGH helper's param
        @compute @workgroup_size(1) fn main() {
            G = 11;
            let c = writes_g();   // writes G = 777
            let r = G;            // reads the POST-call G
            O[0] = r - c;         // inlining would read pre-call G (reorder)
        }
    "#;
    let out = compact(src);
    let store_rhs = out
        .split("O[0]")
        .nth(1)
        .expect("the O[0] store must be emitted");
    let rhs = store_rhs.split(';').next().unwrap_or("");
    assert!(
        !rhs.contains('('),
        "a function writing a global through a helper's pointer param is impure \
         and must be bound before the read, not inlined: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: For-loop Block unwrapping

#[test]
fn for_loop_no_double_braces() {
    // naga wraps the for-body in a `Statement::Block`.
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
    // Dead-init removal can leave a counter with no initializer.
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
    assert!(
        !out.contains("for(var i"),
        "var used after loop should not be absorbed into for-init: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_from_break_if_loop() {
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
    // `try_emit_for_loop` rejects a two-statement continuing, so absorption
    // must not suppress the var.
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
    assert!(
        !out.contains("for("),
        "loop with multi-stmt continuing should not become for-loop: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn for_loop_no_absorb_when_no_if_break_guard() {
    // `try_emit_for_loop` rejects a body that opens with a Call, so absorption
    // must not suppress the var.
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
    let for_count = out.matches("for (").count();
    assert!(
        for_count == 2,
        "expected 2 for-loops, found {for_count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn nested_for_loop_inner_condition_references_outer_var() {
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
    // Ref-count adjustments must not leak between sequential loops.
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

// MARK: Mutated-load binding

/// `t` is a `Load` of `x`; inlined as the bare name it reads `x` after `x = y`
/// overwrote it, so both end up with the old `y`.
#[test]
fn swap_via_temp_load_must_bind() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
        @compute @workgroup_size(1) fn main() {
            var x = buf[0]; var y = buf[1];
            let t = x; x = y; y = t;
            buf[0] = x; buf[1] = y;
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("x=y;y=x"),
        "swap snapshot must be let-bound, not inlined to the overwritten var: {out}"
    );
    assert!(
        out.contains("x=y;"),
        "the swap body must still be emitted: {out}"
    );
    assert_valid_wgsl(&out);
}

/// `a = buf[0]` is used after `buf[0] = b` overwrites it, so it must bind; `b`
/// is used before any write to `buf[1]` and stays inlined.
#[test]
fn array_element_swap_load_must_bind() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
        @compute @workgroup_size(1) fn main() {
            let a = buf[0]; let b = buf[1];
            buf[0] = b; buf[1] = a;
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("buf[1]=buf[0]"),
        "buf[0] read after its store must be let-bound (not re-read post-write): {out}"
    );
    assert!(
        out.contains("let "),
        "the snapshot load must be let-bound: {out}"
    );
    assert_valid_wgsl(&out);
}

/// A function-argument pointer's place is unresolved, so the analysis treats
/// the `*p = 5` store as invalidating the `*p` snapshot.
#[test]
fn load_through_pointer_param_must_bind_across_write() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        fn f(p: ptr<function, i32>) -> i32 { let s = *p; *p = 5; return s; }
        @compute @workgroup_size(1) fn main() { var v = buf[0]; buf[1] = f(&v); }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("return *p"),
        "the pre-store snapshot of *p must be let-bound, not re-read after `*p = 5`: {out}"
    );
    assert_valid_wgsl(&out);
}

/// `*p` never reads a named local of this function, so a store to a local
/// must not force it to bind.
#[test]
fn pointer_param_load_inlined_across_unrelated_local_store() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        fn f(p: ptr<function,i32>) -> i32 {
            let s = *p;
            var t = 0; t = 7;
            return s + t;
        }
        @compute @workgroup_size(1) fn main() { var v = buf[0]; buf[1] = f(&v); }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("=*p"),
        "*p must NOT be let-bound across a store to an unrelated local: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The caller may pass one target to both parameters, so a store through `q`
/// may alias `*p`.
#[test]
fn pointer_param_load_bound_across_store_through_other_param() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        fn g(p: ptr<function,i32>, q: ptr<function,i32>) -> i32 {
            let s = *p;
            *q = 99;
            return s + *q;
        }
        @compute @workgroup_size(1) fn main() { var v = buf[0]; var w = buf[1]; buf[2] = g(&v, &w); }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("return *p"),
        "*p (possibly aliased by *q) must be snapshotted before `*q = 99`, not re-read after: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: pointer parameters as pointer operands

/// A pointer parameter is already the pointer the builtins want; `&p` would
/// be a pointer to a pointer.  A writable storage pointer spells its access
/// mode, a read-only one keeps WGSL's default.
#[test]
fn pointer_parameters_are_pointer_operands_already() {
    let out = compact(
        "fn f(p: ptr<workgroup, atomic<u32>>, q: ptr<workgroup, u32>,\n\
              r: ptr<storage, array<f32>, read_write>, s: ptr<storage, f32>) -> u32 {\n\
           atomicAdd(p, 1u);\n\
           atomicStore(p, 2u);\n\
           let a = atomicLoad(p);\n\
           let b = workgroupUniformLoad(q);\n\
           (*r)[0] = *s;\n\
           return a + b + arrayLength(r);\n\
         }",
    );
    for needle in [
        "atomicAdd(p,",
        "atomicStore(p,",
        "atomicLoad(p)",
        "workgroupUniformLoad(q)",
        "arrayLength(r)",
        "ptr<storage,array<f32>,read_write>",
        "ptr<storage,f32>",
    ] {
        assert!(out.contains(needle), "{needle} missing in {out}");
    }
    assert!(!out.contains('&'), "{out}");
    assert_valid_wgsl(&out);
}
