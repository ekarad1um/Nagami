//! End-to-end regression tests for verified miscompiles and strict-WGSL
//! emission bugs.  Each runs the full `parse -> IR passes -> generate`
//! pipeline via [`crate::run`] at the default (`max`) profile and asserts a
//! semantic invariant on the minified text; a failure means a previously
//! reproduced defect - valid-but-wrong output, or a form strict WGSL parsers
//! reject - is back.

use super::helpers::assert_valid_wgsl;
use crate::config::{Config, TraceConfig};

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
    // than referencing a body binding the `continue` skips.  Name-agnostic and
    // emission-shape-agnostic: a recompute leaves a `sin(` inside the
    // continuing block (whether inlined into the store or rebound), whereas a
    // buggy reuse would reference the body binding and emit no `sin(` there.
    let continuing = out
        .split_once("continuing{")
        .map(|(_, rest)| rest)
        .expect("a continuing block must be emitted");
    assert!(
        continuing.contains("sin("),
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
    // Name-agnostic: the `-0f` init keeps its sign (not folded to +0.0), and
    // both the guard (`>.5{`) and the conditional `=0` store survive.
    assert!(
        out.contains("=-0f"),
        "the -0.0 initializer must keep its sign: {out}"
    );
    assert!(
        out.contains(">.5{") && out.contains("=0;}"),
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
    // The compose stays a full TWO-component vec4 constructor (no splat-
    // elision to a scalar `vec2 * vec4`).  The `f` element suffix may be
    // dropped because the `vec2f` components already pin the f32 element
    // type, so both `vec4f(A,A)` and the shorter `vec4(A,A)` are accepted -
    // what matters is the two-component `(A,A)` shape, not a scalar splat.
    assert!(
        out.contains("vec4(A,A)*a")
            || out.contains("vec4(A,A) * a")
            || out.contains("vec4f(A,A)*a")
            || out.contains("vec4f(A,A) * a"),
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
    // Names are mangled, so assert the structural invariant rather than exact
    // ids: the inlined deref is parenthesised as `(*ptr).member`, never the
    // WGSL-invalid `*ptr.member` (which parses as `*(ptr.member)`).  The
    // deref'd pointer is the function's first argument, `a`.
    assert!(
        out.contains("(*a)."),
        "inlined pointee-Load postfix base must be parenthesised `(*p).field`: {out}"
    );
    assert!(
        !out.contains("*a."),
        "must not emit the WGSL-invalid unparenthesised `*ptr.field`: {out}"
    );
    // It must NOT have fallen back to naga's verbose multi-line emitter (which
    // the pre-fix invalid `*a.mx` forced).
    assert!(
        out.matches('\n').count() <= 1,
        "output looks like the verbose naga fallback (lost minification): {out}"
    );
}

/// A `var<immediate>` (push-constant / immediate-data) global must keep its
/// address space.  Emitting `var<private>` instead (a per-invocation,
/// zero-initialised mutable with a different data source) is a silent
/// miscompile: the shader reads default-initialised memory rather than the
/// host-supplied immediate data.  Guards against `address_space` mapping
/// `AddressSpace::Immediate` to the `"private"` fallback.
#[test]
fn immediate_address_space_is_preserved() {
    let src = "var<immediate> pc: vec4<f32>;\
        @fragment fn main() -> @location(0) vec4<f32> { return pc; }";
    let out = minify(src);
    assert!(
        out.contains("var<immediate>"),
        "var<immediate> must be preserved, not rewritten to another space: {out}"
    );
    assert!(
        !out.contains("var<private>"),
        "var<immediate> must not become var<private> (changes the data source): {out}"
    );
}

/// An f16 value that survives only as a bare literal inside a conversion
/// (`f32(1.0h + 2.0h)` folds to `f32(3h)`) still requires `enable f16;` in
/// the output.  The directive scan must catch f16 *literals* / casts, not
/// just registered f16 *types*; otherwise the emitted text is rejected by
/// naga ("the `f16` enable extension is not enabled") and the whole run
/// fails.  `minify` re-parses + validates, so a missing directive panics here.
#[test]
fn enable_f16_emitted_for_surviving_f16_literal_in_conversion() {
    // No explicit `enable f16;`: detection must inject it on input AND the
    // emitter must re-assert it on output for the folded `f32(3h)`.
    let src = "@fragment fn main() -> @location(0) vec4<f32> { \
        let x = 1.0h + 2.0h; return vec4<f32>(f32(x), 0.0, 0.0, 1.0); }";
    let out = minify(src);
    assert!(
        out.contains("enable f16;"),
        "output uses an f16 literal but dropped `enable f16;`: {out}"
    );
}

/// `lhs = rhs * lhs` must NOT fold to the compound `lhs *= rhs` when the
/// product is a non-commutative linear-algebra product (matrix*matrix,
/// matrix*vector, vector*matrix): `*=` desugars to `lhs = lhs * rhs`, the
/// opposite (transposed) product.  A matrix `m = n * m` must stay `m = n * m`.
#[test]
fn matrix_compound_assign_preserves_non_commutative_order() {
    // mat*mat, rhs-self: the swapped fold would be a silent miscompile.
    let mm = minify(
        "@group(0)@binding(0) var<storage,read_write> m: mat3x3<f32>;\
         @group(0)@binding(1) var<storage,read> n: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { m = n * m; }",
    );
    assert!(
        mm.contains("A=a*A;") && !mm.contains("*="),
        "mat*mat `m = n*m` was folded to a swapped `*=` (transposed product): {mm}"
    );
    // mat*vec, rhs-self: `v = M * v` must keep the column-vector product.
    let mv = minify(
        "@group(0)@binding(0) var<storage,read_write> v: vec3<f32>;\
         @group(0)@binding(1) var<storage,read> M: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { v = M * v; }",
    );
    assert!(
        mv.contains("A=a*A;") && !mv.contains("*="),
        "mat*vec `v = M*v` was folded to a swapped `*=`: {mv}"
    );
    // The left-self matrix fold IS order-preserving and must still apply.
    let ls = minify(
        "@group(0)@binding(0) var<storage,read_write> m: mat3x3<f32>;\
         @group(0)@binding(1) var<storage,read> n: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { m = m * n; }",
    );
    assert!(
        ls.contains("A*=a;"),
        "left-self matrix `m = m*n` should still fold to `m *= n`: {ls}"
    );
    // A genuinely commutative matrix*scalar must still fold (no over-restrict).
    let ms = minify(
        "@group(0)@binding(0) var<storage,read_write> m: mat3x3<f32>;\
         @group(0)@binding(1) var<storage,read> k: f32;\
         @compute @workgroup_size(1) fn main() { m = k * m; }",
    );
    assert!(
        ms.contains("A*=a;"),
        "commutative matrix*scalar `m = k*m` should still fold to `m *= k`: {ms}"
    );
}

/// A both-literal F16 multiply by zero (`x * 0.0h`) must NOT be folded by the
/// absorbing rule, because `eval_binary` has no F16 arm to compute the IEEE
/// sign of the product: cloning the matched `0.0h` would take its sign, not
/// the product's (`-2.0h * 0.0h` is `-0.0h`, not `+0.0h`).  The fold is
/// declined so the bare product survives and re-parses to the right signed
/// zero.  Guards the `is_integer_zero` gate on the absorbing-Multiply arm.
#[test]
fn f16_multiply_by_zero_is_not_mis_signed() {
    let out = minify(
        "enable f16;\
         @group(0)@binding(0) var<storage,read_write> out: array<f16, 4>;\
         @compute @workgroup_size(1) fn main() { let a = -2.0h; out[0] = a * 0.0h; }",
    );
    // The signed product must survive as `-2h*0h` (re-parses to -0.0h), never
    // collapse to a positive-zero store `A[0]=0h;`.
    assert!(
        out.contains("-2h*0h"),
        "f16 `-2.0h * 0.0h` must keep the signed product, not fold to +0.0h: {out}"
    );
    assert!(
        !out.contains("A[0]=0h"),
        "f16 `-2.0h * 0.0h` was mis-signed to +0.0h: {out}"
    );
}

/// An override-sized array (`array<i32, O*2>`) must round-trip without
/// aborting the process.  naga 29's own WGSL back-end hits `unreachable!()`
/// on the override size expression; under the release `panic = "abort"`
/// strategy that crashes (SIGABRT).  nagami emits these types with its own
/// generator, so `run` must detect the override-sized array and skip the
/// naga baseline/fallback emit rather than invoke it.
#[test]
fn override_sized_array_does_not_abort() {
    let out = minify(
        "override O = 123;\
         alias A = array<i32, O*2>;\
         var<workgroup> W: A;\
         @compute @workgroup_size(1) fn main() { let p: ptr<workgroup, A> = &W; (*p)[0] = 42; }",
    );
    // `minify` re-parses + validates, so reaching here means no abort and a
    // valid override-sized array survived.
    assert!(
        out.contains("array<i32,") && out.contains("var<workgroup>"),
        "override-sized workgroup array must survive minification: {out}"
    );
}

/// A value computed BEFORE a loop and used in its `break_if` must stay bound
/// there: load_dedup must not forward the load to a read inside the loop,
/// which would observe the loop-mutated place across the back-edge.  Here
/// `cond(n)` is loop-invariant (n=0 at the call), so the break condition must
/// read the pre-loop value, never the `n` mutated each iteration - otherwise a
/// loop that never breaks (`cond(0)==false`) wrongly terminates.
#[test]
fn loop_break_if_does_not_read_loop_mutated_load() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<i32>;\
        fn cond(v: i32) -> bool { return v >= 500; }\
        @compute @workgroup_size(1) fn main() {\
            var n: i32 = 0; var count: i32 = 0; let c = cond(n);\
            loop { count = count + 1; out[0] = count;\
                continuing { n = n + 100; break if c; } } }";
    let out = minify(src);
    // The break value is bound immediately before the loop and the break uses
    // that binding; it never re-reads the var mutated in `continuing`.
    assert!(
        out.contains("let b=A;loop{") && out.contains("break if b>=500"),
        "loop-invariant break value was relocated into the loop: {out}"
    );
}

/// A `switch` whose case contains a reachable bare `break` (not as the case's
/// last statement) falls through to the code after the switch, so that code is
/// live and must not be dropped.  Here `out[0]=x` after the switch is reached
/// when `gid.x==1 && x>0` takes the bare break.
#[test]
fn switch_with_nested_bare_break_keeps_code_after_switch() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<i32>;\
        @group(0)@binding(1) var<storage,read> inp:array<i32>;\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\
            let x = inp[gid.x];\
            switch i32(gid.x) {\
                case 1: { if (x > 0) { break; } out[0] = -1; return; }\
                default: { out[0] = -2; return; }\
            }\
            out[0] = x; }";
    let out = minify(src);
    // The bare break survives AND the post-switch store is not eliminated.
    assert!(
        out.contains("if b>0{break;}"),
        "the bare break inside the case was lost: {out}"
    );
    assert!(
        out.contains("}A[0]=b;}"),
        "post-switch store was wrongly dropped as unreachable: {out}"
    );
}

/// An implicit-derivative expression (`dpdx`) in uniform control flow must not
/// be sunk into a non-uniform branch by single-use inlining: that violates the
/// WGSL uniformity rules and strict parsers (Tint/Dawn/browsers) reject it,
/// even though naga's validator does not.  It must stay bound at its original
/// (uniform) position before the non-uniform `if`.
#[test]
fn implicit_derivative_not_sunk_into_non_uniform_branch() {
    let src = "@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {\
        let d = dpdx(tc.x);\
        if (tc.y > 0.5) { return vec4<f32>(d, 0.0, 0.0, 1.0); }\
        return vec4<f32>(1.0); }";
    let out = minify(src);
    // `dpdx` is bound before the branch; the branch consumes the binding.
    assert!(
        out.contains("let a=dpdx(A.x);if "),
        "dpdx was sunk out of uniform control flow: {out}"
    );
}

/// `dead_param` must not strip a parameter from a `--preserve-symbol` function:
/// its signature is an external contract, so the arity must survive even when a
/// parameter is unused.  (Seven call sites keep the function from being inlined
/// away, isolating the dead-param path.)
#[test]
fn preserve_symbol_keeps_unused_parameter_arity() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<f32>;\
        fn kept_api(a: f32, unused: f32) -> f32 { return a * 2.0 + a; }\
        @compute @workgroup_size(1) fn main() {\
            out[0]=kept_api(out[1],out[2]); out[3]=kept_api(out[4],out[5]);\
            out[6]=kept_api(out[7],out[8]); out[9]=kept_api(out[10],out[11]);\
            out[12]=kept_api(out[13],out[14]); out[15]=kept_api(out[16],out[17]);\
            out[18]=kept_api(out[19],out[20]); }";
    let config = Config {
        preserve_symbols: vec!["kept_api".to_string()],
        ..Default::default()
    };
    let out = crate::run(src, &config).expect("run failed").source;
    assert_valid_wgsl(&out);
    assert!(
        out.contains("fn kept_api(a:f32,B:f32)"),
        "preserved function lost its unused parameter (arity changed): {out}"
    );
    assert!(
        out.contains("kept_api(A[1],A[2])"),
        "call site to a preserved function dropped an argument: {out}"
    );
}

/// `flag_used_loads` stops recursing once it pins a written load, since the
/// binding materialises its whole operand cone at the pin site.  That must not
/// under-bind a NESTED load: here the outer `arr[kk]` is pinned before the
/// store, and `kk` (a snapshot of `k`) is bound independently because `k` is
/// later written - so `out[0]` reads the pre-store `arr[0]` and the separate
/// `out[1] = kk` reads the pre-write `k` (0).
#[test]
fn pinned_load_does_not_under_bind_nested_index_load() {
    let src = "@group(0)@binding(0) var<storage,read_write> arr:array<i32>;\
        @group(0)@binding(1) var<storage,read_write> out:array<i32>;\
        @compute @workgroup_size(1) fn main() {\
            var k: i32 = 0; let kk = k; let av = arr[kk];\
            k = 1; arr[0] = 99; out[0] = av; out[1] = kk; }";
    let out = minify(src);
    assert!(
        out.contains("let B=A[0];A[0]=99;") && out.contains("a[0]=B;"),
        "nested-load pinning under-bound the outer load: {out}"
    );
    assert!(
        out.contains("a[1]=0;"),
        "inner index load read the mutated k instead of its pre-write value: {out}"
    );
}

/// `--validate-each-pass` re-emits WGSL text after every pass; for an
/// override-sized array that path must not invoke naga's panicking backend.
/// This guards the pipeline driver's per-pass emit, distinct from the
/// crate-root baseline emit covered by `override_sized_array_does_not_abort`.
#[test]
fn override_sized_array_survives_per_pass_text_emit() {
    let config = Config {
        trace: TraceConfig {
            validate_each_pass: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let src = "override O = 123;\
         alias A = array<i32, O*2>;\
         var<workgroup> W: A;\
         @compute @workgroup_size(1) fn main() { let p: ptr<workgroup, A> = &W; (*p)[0] = 42; }";
    // Reaching the assert means the per-pass text emit did not abort and the
    // pipeline still converged (validate_each_pass turns any per-pass
    // validation failure or non-convergence into an Err).
    let out = crate::run(src, &config).expect("validate_each_pass must not abort or fail");
    assert!(
        out.source.contains("array<i32,"),
        "override-sized array did not survive: {}",
        out.source
    );
}

/// A scalar `bitcast<T>` whose operand is an inline literal must pin the
/// operand's concrete type (emit the typed/suffixed form), because bitcast
/// reinterprets BITS and a bare token re-types as an abstract literal.
/// `bitcast<u32>(1.0f)` collapsing to `bitcast<u32>(1)` is a silent VALUE
/// miscompile (f32 1.0 is `0x3F800000`, but bare `1` is i32 `0x1`), and
/// `bitcast<f32>(<u32 > i32::MAX>)` losing its `u` overflows the i32 default
/// and is rejected by spec-conformant consumers.
#[test]
fn bitcast_scalar_literal_source_keeps_concrete_type() {
    // Whole-number float source must NOT collapse to a bare int.
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @compute @workgroup_size(1) fn main(){ o[0]=bitcast<u32>(1.0f); o[1]=bitcast<u32>(2.0f); }",
    );
    assert!(
        !out.contains("bitcast<u32>(1)") && !out.contains("bitcast<u32>(2)"),
        "whole-number float bitcast source collapsed to a bare int (value miscompile): {out}"
    );
    assert!(
        out.contains("bitcast<u32>(1f)") && out.contains("bitcast<u32>(2f)"),
        "float bitcast source did not keep its concrete f32 form: {out}"
    );
    // Unsigned source above i32::MAX must keep its `u` suffix.
    let out2 = minify(
        "@group(0)@binding(0) var<storage,read_write> r:array<f32>;\
         @compute @workgroup_size(1) fn main(){ r[0]=bitcast<f32>(3212836864u); }",
    );
    assert!(
        out2.contains("3212836864u"),
        "out-of-i32-range u32 bitcast source lost its `u` suffix (strict-parser reject): {out2}"
    );
}

/// The bitcast type-pinning must hold even when the literal is ALSO repeated
/// enough to be hoisted into a shared `const` by literal extraction.  That
/// const is abstract-typed (the bare form merges across types), so a
/// `bitcast<u32>(C)` reference would reinterpret the abstract-int default
/// instead of the float - the same value miscompile through a const.  The
/// bitcast use must stay inline-typed (`1024f`), while the const still serves
/// its other (type-pinning) uses; and the extraction counter must not leave a
/// dangling const when ALL uses are bitcasts.
#[test]
fn bitcast_scalar_literal_pinned_even_when_extracted() {
    // 1024.0 repeated: one bitcast + eight f32 stores -> extracted to a const
    // for the stores, but the bitcast stays inline-typed.
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @group(0)@binding(1) var<storage,read_write> f:array<f32>;\
         @compute @workgroup_size(1) fn main(){ o[0]=bitcast<u32>(1024.0);\
            f[0]=1024.0; f[1]=1024.0; f[2]=1024.0; f[3]=1024.0;\
            f[4]=1024.0; f[5]=1024.0; f[6]=1024.0; f[7]=1024.0; }",
    );
    assert!(
        out.contains("bitcast<u32>(1024f)"),
        "extracted-literal bitcast source was not pinned to its concrete type: {out}"
    );
    // All-bitcast: every use is a bitcast, so extraction must decline (no
    // substitutable uses) rather than leave a dangling `const`.
    let out2 = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @compute @workgroup_size(1) fn main(){\
            o[0]=bitcast<u32>(1024.0); o[1]=bitcast<u32>(1024.0);\
            o[2]=bitcast<u32>(1024.0); o[3]=bitcast<u32>(1024.0);\
            o[4]=bitcast<u32>(1024.0); o[5]=bitcast<u32>(1024.0); }",
    );
    assert!(
        out2.contains("bitcast<u32>(1024f)") && !out2.contains("const"),
        "all-bitcast extraction left a dangling const or failed to pin: {out2}"
    );
}

/// A relational child of an equality operator (and any comparison child of a
/// comparison parent) must stay parenthesised: WGSL puts all six comparisons
/// at one non-associative grammar level requiring `shift_expression` operands,
/// so `a<b==c<d` is a parse error in spec-conformant consumers (Dawn/Tint:
/// "mixing '<' and '==' requires parenthesis"), though naga round-trips it.
#[test]
fn chained_comparison_keeps_required_parens() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @group(0)@binding(1) var<storage,read> i:array<i32>;\
         @compute @workgroup_size(1) fn main(){ o[0]=u32(i[0]<i[1] == i[2]<i[3]); }",
    );
    assert!(
        out.contains(")==(") || out.contains(") == ("),
        "chained comparison must parenthesise its relational children: {out}"
    );
}

/// naga's WGSL front-end (`proc::ensure_block_returns`) injects an implicit
/// `return;` after a tail `loop`, so a non-void function whose dead-code-
/// stripped trailing return leaves it ending in an always-returning loop
/// round-trips into an invalid module ("return expression None does not match
/// the declared return type").  The generator must synthesise a trailing
/// zero-value return.
#[test]
fn terminal_loop_non_void_fn_synthesises_trailing_return() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:i32;\
         fn f()->i32{ var i:i32; i=0; loop{ if (true) { return 7; } i=i+1; \
            continuing{ break if i>3; } } return 9; }\
         @compute @workgroup_size(1) fn m(){ o=f(); }",
    );
    // `if (true)` makes the loop body always return, so dead-branch strips the
    // unreachable `return 9`; the synthesised `return 0i;` follows the loop.
    assert!(
        out.contains("}return 0i;}"),
        "terminal loop in a non-void fn must gain a trailing zero return: {out}"
    );
}

/// The `ensure_block_returns` injection recurses into `switch` cases, so a
/// non-void function ending in a `switch` whose every case tails in a loop is
/// also rejected.  One trailing top-level return suppresses the whole recursive
/// injection (naga only inspects the block's last statement).
#[test]
fn terminal_loop_in_switch_case_synthesises_trailing_return() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:i32;\
         fn f(p:i32)->i32{ switch p { case 0: { loop { return 1; } } \
            default: { loop { return 2; } } } return 9; }\
         @compute @workgroup_size(1) fn m(){ o=f(0); }",
    );
    assert!(
        out.contains("return 0i;}"),
        "switch-of-terminal-loops in a non-void fn must gain a trailing return: {out}"
    );
}

/// Negative guard for the terminal-loop fix: a non-void function that already
/// ends in real returns (here, an `if`-guarded pair, kept un-inlined by 7 call
/// sites) must NOT gain a synthesised zero return - that would cost bytes and
/// signals an over-broad terminating predicate.
#[test]
fn normal_returning_fn_not_padded_with_zero_return() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<i32>;\
         fn g(x:i32)->i32{ if (x>0) { return x; } return -x; }\
         @compute @workgroup_size(1) fn m(){ o[0]=g(1);o[1]=g(2);o[2]=g(3);\
            o[3]=g(4);o[4]=g(5);o[5]=g(6);o[6]=g(7); }",
    );
    assert!(
        !out.contains("return 0"),
        "normal value-returning fn must not gain a synthesised zero return: {out}"
    );
}

/// naga 29's WGSL back-end hits `unreachable!()` (and, under release
/// `panic = "abort"`, SIGABRTs) when writing a non-const override / global-var
/// initializer such as `override h = 2 * d;`.  Because nagami eagerly emits
/// that naga baseline before its own generator runs, the baseline must be
/// skipped for such modules (the generator emits them correctly).
#[test]
fn override_binary_initializer_minifies_without_abort() {
    let out = minify(
        "override d: f32;\
         override h = 2.0 * d;\
         var<private> g: f32 = d * 10.0;\
         @group(0)@binding(0) var<storage,read_write> o:f32;\
         @compute @workgroup_size(1) fn m(){ o = h + g; }",
    );
    assert!(
        out.contains("override h") && out.contains("*d"),
        "override binary initializer must survive minification: {out}"
    );
}

/// Cooperative-matrix role enumerants `A`/`B`/`C` are predeclared in the
/// `coop_mat<T, role>` type-argument position.  The generator cannot emit coop
/// types, so the module falls back to naga's wgsl-out; if the rename sweep
/// minted `A`/`B`/`C` for a global/local it collided with the role
/// ("identifier `B` resolves to a declaration").  Reserving the role names when
/// a coop type is present keeps the fallback valid.
#[test]
fn cooperative_matrix_minifies_without_role_collision() {
    let out = minify(
        "enable wgpu_cooperative_matrix;\
         var<private> a: coop_mat8x8<f32, A>;\
         var<private> bb: coop_mat8x8<f32, B>;\
         @group(0) @binding(0) var<storage, read_write> ext: array<f32>;\
         @compute @workgroup_size(8, 8, 1) fn main() {\
            var c = coopLoad<coop_mat8x8<f32, C>>(&ext[4]);\
            var d = coopMultiplyAdd(a, bb, c);\
            coopStore(d, &ext[0]); }",
    );
    // `minify` already re-validates; assert the role keywords still render and
    // no declaration was renamed onto them (round-trip would otherwise fail).
    assert!(
        out.contains("coop_mat8x8<f32, A>") || out.contains("coop_mat8x8<f32,A>"),
        "coop role positions must survive: {out}"
    );
}
