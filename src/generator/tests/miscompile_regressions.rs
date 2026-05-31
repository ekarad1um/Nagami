//! End-to-end regression tests for verified miscompiles and strict-WGSL
//! emission bugs.  Each runs the full `parse -> IR passes -> generate`
//! pipeline via [`crate::run`] at the default (`max`) profile and asserts a
//! semantic invariant on the minified text; a failure means a previously
//! reproduced defect - valid-but-wrong output, or a form strict WGSL parsers
//! reject - is back.

use super::helpers::assert_valid_wgsl;
use crate::config::Config;

/// Run the full pipeline at the default profile and return minified text.
fn minify(src: &str) -> String {
    let out = crate::run(src, &Config::default()).expect("run failed");
    assert_valid_wgsl(&out.source);
    out.source
}

/// An identity fold (`0 + x`) over a single-use storage `Load` must NOT
/// relocate the read past an intervening `Store` to the same place; doing so
/// would make `data[1] = 0u + data[0]` read the post-`Store` value.
#[test]
fn identity_fold_does_not_reorder_load_past_store() {
    let src = "@group(0)@binding(0) var<storage,read_write> data:array<u32>;\
        @compute @workgroup_size(1) fn main(){ let a=data[0]; data[0]=99u; data[1]=0u+a; }";
    let out = minify(src);
    // The load is bound BEFORE the store; the second store uses the bound
    // value, never a fresh `A[0]` read after the overwrite.
    assert!(
        out.contains("let a=A[0];A[0]=99;A[1]=0+a;"),
        "identity fold reordered a load past a store: {out}"
    );
    assert!(
        !out.contains("A[1]=A[0]"),
        "data[1] must not re-read A[0] after it is overwritten: {out}"
    );
}

/// The involution fold (`-(-x)`) shares the load-relocation hazard, and the
/// surviving double-negation must emit `- -a` - a space disambiguates the
/// WGSL-reserved `--` token.
#[test]
fn involution_fold_does_not_reorder_load_and_avoids_double_minus() {
    let src = "@group(0)@binding(0) var<storage,read_write> data:array<i32>;\
        @compute @workgroup_size(1) fn main(){ let a=data[0]; data[0]=99; data[1]=-(-a); }";
    let out = minify(src);
    assert!(
        out.contains("let a=A[0];A[0]=99;A[1]=- -a;"),
        "involution reordered a load past a store or emitted `--`: {out}"
    );
    assert!(
        !out.contains("--"),
        "must not emit the WGSL-reserved `--` token: {out}"
    );
}

/// A single-use `textureLoad` must be bound to a `let` before an intervening
/// `textureStore` to the same storage texture, not inlined past it (it would
/// read the post-store texel).
#[test]
fn image_load_bound_before_intervening_texture_store() {
    let src = "@group(0)@binding(0) var tex: texture_storage_2d<r32float, read_write>;\
        @compute @workgroup_size(1) fn main(){\
          let c = vec2i(0,0);\
          let v = textureLoad(tex, c).x;\
          textureStore(tex, c, vec4f(99.0));\
          textureStore(tex, vec2i(1,1), vec4f(v));\
        }";
    let out = minify(src);
    assert!(
        out.contains("let B=textureLoad(A,a);textureStore(A,a,"),
        "textureLoad must be bound before the intervening textureStore: {out}"
    );
}

/// Loop CSE must not redirect a `continuing`-block expression to a body `let`
/// defined after a `continue` (the continue iteration skips it); the
/// continuing block must recompute the shared expression.
#[test]
fn cse_does_not_share_body_let_into_continuing_past_continue() {
    let src = "@group(0) @binding(0) var<storage, read_write> buf: array<f32, 64>;\
        @group(0) @binding(1) var<uniform> k: f32;\
        @compute @workgroup_size(1)\
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\
            let p = f32(gid.x);\
            var i = 0u; var acc = 0.0;\
            loop {\
                if i >= 10u { break; }\
                let w = acc + p;\
                if i == 5u { continue; }\
                let a = sin(w)*cos(k) + tan(w)*sin(k);\
                buf[i] = a; buf[i + 1u] = a;\
                continuing {\
                    let b = sin(w)*cos(k) + tan(w)*sin(k);\
                    buf[0] = b; acc = acc + 1.0; i = i + 1u;\
                }\
            }\
        }";
    let out = minify(src);
    // The continuing-block store recomputes its own `sin(...)` chain rather
    // than referencing a body binding the `continue` skips.
    assert!(
        out.contains("continuing{A[0]=sin("),
        "continuing must recompute the shared expression, not reuse a body let: {out}"
    );
}

/// Redundant-store elimination must compare float literals bitwise, so a
/// conditional `+0.0` store against a known `-0.0` init is NOT dropped - the
/// sign of zero is observable.
#[test]
fn neg_zero_conditional_store_is_not_dropped() {
    let src = "@fragment fn main(@location(0) c: f32, @location(1) d: f32) -> @location(0) vec4f {\
        var x: f32 = -0.0; if c > 0.5 { x = 0.0; } let r = x + d; return vec4f(r); }";
    let out = minify(src);
    // The `if` and its `x = 0.0` store survive; `x` is not hardcoded to -0.0.
    assert!(
        out.contains("var B=-0f;if A>.5{B=0;}"),
        "the +0.0-vs--0.0 conditional store was wrongly eliminated: {out}"
    );
}

/// Swizzle/member access on a `ptr<function, vecN>` value must emit an
/// explicit `(*p)` deref - strict WGSL parsers reject `p.xyz`.
#[test]
fn pointer_value_swizzle_is_dereferenced() {
    let src = "fn rd(q: ptr<function, vec4<f32>>) -> vec3<f32> {\
          return vec3<f32>((*q).x, (*q).y, (*q).z) * 2.0 + vec3<f32>((*q).z, (*q).y, (*q).x);\
        }\
        @fragment fn main(@location(0) k: f32) -> @location(0) vec4f {\
          var v = vec4f(k, k*2.0, k*3.0, k*4.0);\
          var w = vec4f(k*5.0, k*6.0, k*7.0, k*8.0);\
          let r = rd(&v) + rd(&w);\
          return vec4f(r, 1.0);\
        }";
    let out = minify(src);
    assert!(
        out.contains("(*a).xyz"),
        "pointer-value swizzle must be dereferenced as (*p).xyz: {out}"
    );
    // The bug form is a bare swizzle on the pointer parameter name.
    assert!(
        !out.contains("return a.xyz") && !out.contains("a)->vec3f{return a."),
        "must not emit a bare swizzle on a pointer value: {out}"
    );
}

/// An override WITHOUT an explicit `@id` is keyed by its name in the host
/// pipeline `constants` record, so mangling must preserve it; an `@id`-bearing
/// override is keyed numerically and stays renamable.
#[test]
fn id_less_override_name_is_preserved_under_mangle() {
    let src = "@id(7) override has_id: f32 = 1.0;\
        override no_id: f32 = 2.0;\
        @fragment fn main() -> @location(0) vec4f { return vec4f(has_id * no_id); }";
    let out = minify(src);
    assert!(
        out.contains("override no_id:"),
        "@id-less override name must be preserved: {out}"
    );
    assert!(
        !out.contains("override has_id:"),
        "@id-bearing override should still be mangled: {out}"
    );
}

/// A vector `Compose` built from sub-vectors (`vec4(v2, v2)`) must not be
/// splat-elided to a non-scalar operand; doing so emits `v2 * v4` and forces a
/// fallback to naga's verbose emitter.
#[test]
fn subvector_compose_not_splat_elided() {
    let src = "@group(0)@binding(0) var<uniform> v: vec2f;\
        @group(0)@binding(1) var<uniform> w: vec4f;\
        @fragment fn main() -> @location(0) vec4f { let a = v; return vec4<f32>(a, a) * w; }";
    let out = minify(src);
    // The compose stays a full vec4 constructor; no bare `vec2 * vec4`.
    assert!(
        out.contains("vec4f(A,A)*a") || out.contains("vec4f(A,A) * a"),
        "sub-vector compose was wrongly splat-elided: {out}"
    );
}

/// A storage texture reached through a `binding_array` (`texs[i]`) is still a
/// hazardous-load target: a single-use `textureLoad(texs[i], ..)` must be
/// bound before an intervening `textureStore` to the same element, not inlined
/// past it.  The writability test must peel the `BindingArray` to the element
/// texture type, else the load is dropped from the hazard set and reordered.
#[test]
fn binding_array_texture_load_bound_before_store() {
    let src = "@group(0) @binding(0) var texs: binding_array<texture_storage_2d<r32uint, read_write>, 4>;\
        @compute @workgroup_size(1) fn main(){\
          let c = vec2i(0,0);\
          let v = textureLoad(texs[0], c).x;\
          textureStore(texs[0], c, vec4u(99u));\
          textureStore(texs[0], vec2i(1,1), vec4u(v));\
        }";
    let out = minify(src);
    assert!(
        out.contains("let B=textureLoad(A[0],a);textureStore(A[0],a,"),
        "binding-array textureLoad must be bound before the intervening store: {out}"
    );
}

/// An inlined whole-pointee `Load` of a pointer value (`let e = (*p)`) used as
/// a postfix base must be parenthesised - `(*p).field`, not the bare
/// `*p.field`, which parses as `*(p.field)` and is rejected (forcing a verbose
/// naga fallback).
#[test]
fn inlined_pointee_load_postfix_base_is_parenthesized() {
    let src = "struct M { mx: vec4<f32>, my: vec4<f32> }\
        fn f(d: ptr<function, M>, s: f32) { let e = (*d); (*d).mx = e.mx + vec4f(s); }\
        @fragment fn main() -> @location(0) vec4f { var m: M; m.my = vec4f(1.0); f(&m, 2.0); return m.mx; }";
    let out = minify(src);
    // The inlined `(*d).mx` postfix is parenthesised; the member access binds
    // to the deref, not to the pointer name.
    assert!(
        out.contains("(*a).c=(*a).c+B"),
        "inlined pointee-Load postfix base must be (*p).field: {out}"
    );
    // It must NOT have fallen back to naga's verbose multi-line emitter (which
    // the pre-fix invalid `*a.mx` forced).
    assert!(
        out.matches('\n').count() <= 1,
        "output looks like the verbose naga fallback (lost minification): {out}"
    );
}
