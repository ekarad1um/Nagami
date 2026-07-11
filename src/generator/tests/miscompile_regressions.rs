//! End-to-end regression tests over the full `parse -> IR passes -> generate`
//! pipeline ([`crate::run`], default `max` profile), each asserting an
//! invariant on the minified text.  A failure is either a CORRECTNESS
//! regression - a verified miscompile (valid-but-wrong output) or a form
//! strict WGSL parsers reject is back - or a MINIFICATION-QUALITY regression -
//! an optimization stopped firing, or the output grew.

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

/// An override-sized array (`array<i32, O*2>`) must round-trip without aborting.
/// naga's WGSL back-end hits `unreachable!()` on the override size expression,
/// SIGABRTing under the release `panic = "abort"`.  nagami emits these types with
/// its own generator, so `run` must detect the override-sized array and skip the
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

/// A void function's tail if/switch arms need no closing `return;` - falling
/// off the arm ends the function - and naga's `ensure_block_returns`
/// re-synthesises them on every re-parse, so leaving them makes the text
/// gain an `else{return;}` per arm per round trip (a former grow class).
#[test]
fn void_fn_tail_switch_case_returns_are_elided() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() {\
            switch o { case 0u: { o = 1u; return; } default: { o = 2u; return; } }\
        }";
    let out = minify(src);
    assert!(
        !out.contains("return"),
        "tail-position case returns are redundant in a void fn: {out}"
    );
    assert!(
        out.contains("switch"),
        "the switch itself must survive: {out}"
    );
}

/// The elision must not touch a return that still skips code: here the arm
/// `return;` jumps over the trailing store, so it is load-bearing.
#[test]
fn void_fn_non_tail_return_survives_elision() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() {\
            if o > 3u { o = 1u; return; } else { o = 2u; return; }\
        }";
    let out = minify(src);
    // Else-elision hoists the reject arm behind the if, making the accept's
    // return load-bearing (it must skip the hoisted store).
    assert!(
        out.contains("return;}"),
        "the branch-skipping return must survive: {out}"
    );
}

/// Constructor suffix elision via literal pins: a float-form literal pins
/// f32 and an int literal pins i32, so `vec4f`/`vec3i` shrink to the bare
/// inferring form; u32 (whose abstract default is i32) must keep its suffix.
#[test]
fn ctor_suffix_drops_only_under_a_literal_pin() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: vec4f;\
        @group(0)@binding(1) var<storage,read_write> u: vec2u;\
        @compute @workgroup_size(1) fn main() {\
            o = vec4f(0.5, o.x, o.y, o.z);\
            u = vec2u(4u, u.x);\
        }";
    let out = minify(src);
    assert!(
        out.contains("vec4(.5,"),
        "the float-form `.5` pins f32, so the suffix must drop: {out}"
    );
    assert!(
        !out.contains("vec2u(4") || out.contains("vec2("),
        "a u32 component (concrete, non-literal) may pin; bare `4` alone must not: {out}"
    );
}

/// naga's compactor roots `special_types`, so a dead `__frexp_result_f16`
/// (and its f16 member scalar) survives every DCE; the directive scan must
/// not credit those dead types with an `enable f16;` the emitted text never
/// uses - the spurious enable dropped on re-minify (non-idempotent) and
/// retained an f16 device-feature requirement the output no longer has.
#[test]
fn dead_frexp_special_type_does_not_emit_f16_enable() {
    let out = minify("enable f16; @compute @workgroup_size(1) fn m(){ let r = frexp(1.5h); }");
    assert!(
        !out.contains("enable f16;"),
        "no f16 survives DCE, so the enable must not be emitted: {out}"
    );
    // Control: live f16 must keep the enable.
    let live = minify(
        "enable f16;\
         @group(0)@binding(0) var<storage,read_write> o: f32;\
         @compute @workgroup_size(1) fn m(){ o = f32(f16(o) * 2h); }",
    );
    assert!(
        live.contains("enable f16;"),
        "live f16 usage must keep the enable: {live}"
    );
}

/// tint's parser rejects expression nesting deeper than 512 while naga
/// accepts arbitrary depth, so unbounded single-use `let` inlining can
/// flatten a long chain into text only naga accepts.  The emitter must
/// split such chains with forced bindings at the depth cap.
#[test]
fn deep_single_use_chain_is_depth_capped() {
    // 600 distinct private scalars summed in one expression: a ~600-deep
    // left-leaning Binary chain with nothing for const-fold/CSE to collapse.
    let mut src = String::new();
    for i in 0..600 {
        src.push_str(&format!("var<private> v{i}: f32;"));
    }
    src.push_str("@group(0)@binding(0) var<storage,read_write> o: f32;");
    src.push_str("@compute @workgroup_size(1) fn main() { o = v0");
    for i in 1..600 {
        src.push_str(&format!("+v{i}"));
    }
    src.push_str("; }");
    let out = minify(&src);
    // Without the cap the whole chain inlines into the store (zero lets);
    // with it, at least one forced binding splits the chain below tint's
    // 512-deep parser limit.
    assert!(
        out.contains("let "),
        "a >512-deep chain must be split by at least one forced binding"
    );
}

/// tint's loop-exit analysis is syntactic - it credits a loop with an exit
/// for a `break` inside `if false { .. }` without const-evaluating the
/// condition - while naga 30 validates a fully exit-less `loop{}` as legal.
/// Folding the const-false `if` away therefore turns tint-valid input into
/// tint-rejected output ("loop does not exit") that the naga self-check
/// cannot catch.  The fold must keep a dropped block that carries the
/// enclosing loop's only exit.
#[test]
fn loop_only_exit_inside_const_false_if_is_kept() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() {\
            loop { o = o + 1u; if (false) { break; } }\
        }";
    let out = minify(src);
    assert!(
        out.contains("if false{break;}"),
        "the loop's only lexical exit must survive the const-false fold: {out}"
    );
}

/// Same class through the `break_if` spelling: `break if false;` never breaks
/// at runtime, but it is the only lexical exit tint sees.
#[test]
fn loop_only_exit_break_if_false_is_kept() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() {\
            loop { o = o + 1u; continuing { break if false; } }\
        }";
    let out = minify(src);
    assert!(
        out.contains("break if false;"),
        "the loop's only lexical exit must survive the break-if-false fold: {out}"
    );
}

/// Same class through a const-selector switch collapse: the dropped
/// `default` carries the `return` that is the loop's only exit.
#[test]
fn loop_only_exit_in_dropped_switch_case_is_kept() {
    let src = "fn f() -> u32 {\
            var i: u32 = 0u;\
            loop { i = i + 1u; switch (0u) { case 0u: { } default: { return i; } } }\
            return 0u;\
        }\
        @group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() { o = f(); }";
    let out = minify(src);
    assert!(
        out.contains("default") && out.contains("return"),
        "the dropped switch case's return is the loop's only exit and must survive: {out}"
    );
}

/// The guards must not pessimize regular dead-branch folding: outside a
/// loop, a const-false `if` folds away even when it carries a `return`, and
/// inside a loop a const-false `if` WITHOUT an exit still folds when the
/// loop has a real exit of its own.
#[test]
fn loop_exit_guard_does_not_block_ordinary_folds() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        fn g() -> u32 { if (false) { return 1u; } return 2u; }\
        @compute @workgroup_size(1) fn main() {\
            var i: u32 = 0u;\
            loop { i = i + 1u; if (false) { o = 5u; } if (i > 3u) { break; } }\
            o = g() + i;\
        }";
    let out = minify(src);
    assert!(
        !out.contains("if false"),
        "const-false ifs with no load-bearing exit must still fold: {out}"
    );
    assert!(
        !out.contains("=5"),
        "the dead store inside the folded branch must be gone: {out}"
    );
}

/// A ray-query module must minify through nagami's generator instead of
/// SIGABRTing in naga's WGSL writer (`Statement::RayQuery => unreachable!()`),
/// which the mandatory baseline emit used to reach before
/// `module_needs_naga_baseline_skip` gated ray queries.  The output must carry
/// `enable wgpu_ray_query;` - naga's own writer never emits it (it cannot
/// write the statements at all), so nagami's directive scan goes beyond
/// mirroring naga here; without it the self-check re-parse rejects the text.
#[test]
fn ray_query_minifies_without_baseline_abort() {
    let src = "enable wgpu_ray_query;\
        @group(0)@binding(0) var acc: acceleration_structure;\
        @group(0)@binding(1) var<storage,read_write> out: vec4<f32>;\
        @compute @workgroup_size(1) fn main(){\
            var rq: ray_query;\
            rayQueryInitialize(&rq, acc, RayDesc(4u,255u,0.1,100.0,vec3<f32>(0.0),vec3<f32>(0.0,1.0,0.0)));\
            loop { let p = rayQueryProceed(&rq); if !p { break; } }\
            let hit = rayQueryGetCommittedIntersection(&rq);\
            out = vec4<f32>(hit.t, f32(hit.kind), 0.0, 1.0);\
        }";
    let out = minify(src);
    assert!(
        out.contains("enable wgpu_ray_query;"),
        "query-only module must keep the input-faithful ray-query enable: {out}"
    );
    assert!(
        out.contains("rayQueryInitialize(&") && out.contains("rayQueryGetCommittedIntersection(&"),
        "ray-query builtins must render inline, not as `_eN` placeholders: {out}"
    );
    assert!(
        !out.contains("wgpu_ray_tracing_pipeline"),
        "no pipeline signal is present, so the pipeline enable must not appear: {out}"
    );
}

/// `rayQueryGenerateIntersection`'s hit_t argument gets no expected-type
/// propagation from naga's lowerer, so a whole-number f32 literal emitted in
/// bare-int shortest form (`10.0` -> `10`) re-parses as i32 and fails
/// validation ("Hit distance must be an f32").  The emitter must pin the
/// typed spelling in that slot.
#[test]
fn ray_query_generate_intersection_hit_t_stays_f32() {
    let src = "enable wgpu_ray_query;\
        @group(0)@binding(0) var acc: acceleration_structure;\
        @compute @workgroup_size(1) fn main(){\
            var rq: ray_query;\
            rayQueryInitialize(&rq, acc, RayDesc(4u,255u,0.1,100.0,vec3<f32>(0.0),vec3<f32>(0.0,1.0,0.0)));\
            let kind = rayQueryGetCandidateIntersection(&rq).kind;\
            if kind == 3u { rayQueryGenerateIntersection(&rq, 10.0); }\
            rayQueryTerminate(&rq);\
        }";
    let out = minify(src);
    assert!(
        out.contains("rayQueryGenerateIntersection(") && out.contains("10f"),
        "hit_t must keep a typed f32 spelling (bare `10` re-parses as i32): {out}"
    );
}

/// The `vertex_return` flavors: the type flags must survive on both
/// `ray_query` and `acceleration_structure`, `getCommittedHitVertexPositions`
/// must render inline (naga's writer has it as `unreachable!()`), and BOTH
/// enables must be present (the flag needs `wgpu_ray_query_vertex_return`,
/// the base types still need `wgpu_ray_query`).
#[test]
fn ray_query_vertex_return_round_trips() {
    let src = "enable wgpu_ray_query;\
        enable wgpu_ray_query_vertex_return;\
        @group(0)@binding(0) var acc: acceleration_structure<vertex_return>;\
        @group(0)@binding(1) var<storage,read_write> out: vec3<f32>;\
        @compute @workgroup_size(1) fn main(){\
            var rq: ray_query<vertex_return>;\
            rayQueryInitialize(&rq, acc, RayDesc(4u,255u,0.1,100.0,vec3<f32>(0.0),vec3<f32>(0.0,1.0,0.0)));\
            loop { let p = rayQueryProceed(&rq); if !p { break; } }\
            let verts = getCommittedHitVertexPositions(&rq);\
            out = verts[0];\
        }";
    let out = minify(src);
    assert!(
        out.contains("enable wgpu_ray_query;")
            && out.contains("enable wgpu_ray_query_vertex_return;"),
        "both ray-query enables must be emitted: {out}"
    );
    assert!(
        out.contains("ray_query<vertex_return>")
            && out.contains("acceleration_structure<vertex_return>"),
        "vertex_return type flags must survive the round trip: {out}"
    );
    assert!(
        out.contains("getCommittedHitVertexPositions(&"),
        "vertex-position query must render inline: {out}"
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

/// naga's WGSL back-end hits `unreachable!()` (SIGABRTing under release
/// `panic = "abort"`) when writing a non-const override / global-var initializer
/// such as `override h = 2 * d;`.  nagami eagerly emits that naga baseline before
/// its own generator runs, so the baseline must be skipped for such modules (the
/// generator emits them correctly).
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

/// Several single-use results of a pure, non-inlined function feeding one
/// expression must ALL inline in a single pass; collapsing only the last one
/// leaves the rest as `let`s that a *second* minify pass would inline - a
/// non-idempotent size drift.
#[test]
fn pure_call_lets_inline_in_a_single_pass() {
    let src = "fn bx(p:vec2f,b:vec2f)->f32{let d=abs(p)-b;\
        return length(max(d,vec2f(0.0)))+min(max(d.x,d.y),0.0);}\
        @fragment fn fs(@builtin(position) pos:vec4f)->@location(0) f32{\
        let p=pos.xy;let a=bx(p,vec2f(1.0));let b=bx(p,vec2f(2.0));\
        let c=bx(p,vec2f(3.0));return min(min(a,b),c);}";
    let out = minify(src);
    assert!(
        out.contains("return min(min(A("),
        "pure-call lets were not all inlined in one pass: {out}"
    );
    assert!(
        !out.contains("=A("),
        "a pure call result is still let-bound (single-pass-only inline): {out}"
    );
    assert_eq!(
        minify(&out),
        out,
        "minify must be idempotent on the pure-call chain"
    );
}

/// A matrix built from explicit scalar columns emits in the shorter flat
/// all-scalar form `mat2x2f(a,b,c,d)` rather than per-column `vec2f(...)`.
#[test]
fn matrix_scalar_columns_flatten() {
    // Normally-formatted source: a hand-pre-minified input here can end up
    // SHORTER than the emit (CSE let-binds cos/sin at their break-even
    // point), which trips the ship-input-when-not-smaller guard and hides
    // the flattening this test observes.
    let src = "@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {\
        let m = mat2x2f(cos(p.x), sin(p.x), -sin(p.x), cos(p.x));\
        return vec4f(m[0], m[1]);\
    }";
    let out = minify(src);
    assert!(
        out.contains("mat2x2f(a,B,-B,a)"),
        "matrix scalar columns were not flattened to the shorter form: {out}"
    );
    assert!(
        !out.contains("mat2x2f(vec2"),
        "matrix kept the longer per-column constructor form: {out}"
    );
}

/// Flattening must NOT expand a shared / let-bound vector column into N
/// repeated scalars (that grows the output); the short shared name wins via the
/// build-then-compare-shorter gate.
#[test]
fn matrix_shared_column_keeps_name_form() {
    let src = "@fragment fn fs(@builtin(position) p:vec4f)->@location(0) f32{\
        let a=vec3f(p.x);return mat3x3f(a,a,a)[0].x;}";
    let out = minify(src);
    assert!(
        out.contains("mat3x3f(a,a,a)"),
        "shared matrix column was wrongly expanded to scalars: {out}"
    );
}

/// An array constructor whose element type is pinned by a concretely-typed
/// component (here `vec2f(...)`) drops its template parameters to the shorter
/// inferring form `array(...)`.
#[test]
fn array_type_params_elided_when_concretely_pinned() {
    let src = "fn g(i:u32)->vec2f{var a=array(vec2f(-1.5,0.0),vec2f(1.0,2.0),vec2f(3.0,4.0));return a[i];}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(g(u32(p.x)),0,0);}";
    let out = minify(src);
    assert!(
        out.contains("array(c("),
        "concrete-element array constructor was not elided to array(...): {out}"
    );
}

/// Critical soundness: an array of bare abstract literals must NEVER elide
/// its template parameters - `array<u32,2>(7,9)` -> `array(7,9)` would silently
/// re-infer the element type as i32.  The constructor keeps the explicit (or
/// aliased) typed name; it never becomes a bare `array(7,...)`.
#[test]
fn array_type_params_kept_for_abstract_literal_elements() {
    let src = "fn g(i:u32)->u32{var a=array<u32,2>(7u,9u);return a[i];}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(f32(g(u32(p.x))));}";
    let out = minify(src);
    assert!(
        !out.contains("array(7"),
        "abstract-literal array was wrongly elided (would retype u32 -> i32): {out}"
    );
}

/// An all-literal vector constant constructed identically in 3+ live sites
/// is hoisted into a shared module `const` by the pre-rename `const-hoist` IR
/// pass, so the (now frequent) constant gets a short frequency-assigned name -
/// and re-minification is a fixed point (the constant participates in renaming
/// the same way every pass).
#[test]
fn repeated_vector_constant_is_hoisted_to_a_shared_const() {
    let src = "@group(0)@binding(0) var<storage,read_write> o:vec4f;\
        fn f(v:vec4f){o=o+v;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) f32{\
        switch u32(p.x){\
        case 0u{f(vec4f(7.0,2.0,9.0,3.0));}\
        case 1u{f(vec4f(7.0,2.0,9.0,3.0));}\
        default{f(vec4f(7.0,2.0,9.0,3.0));}}return 0.0;}";
    let out = minify(src);
    assert!(
        out.contains("const A=c(7,2,9,3);"),
        "repeated vector constant was not hoisted to a shared const: {out}"
    );
    // The three sites all reference the hoisted name, not the inline vector.
    assert!(
        !out.contains("c(7,2,9,3)") || out.matches("c(7,2,9,3)").count() == 1,
        "a use site still inlines the vector instead of the const: {out}"
    );
    assert_eq!(minify(&out), out, "const-hoist output must be idempotent");
}

/// A write-only local (stored to, never read / escaped / atomic-touched) is
/// eliminated entirely - var, stores, and the dead value expressions all go -
/// while a genuine side effect in the same function survives.
#[test]
fn write_only_local_is_eliminated() {
    let src = "@group(0)@binding(0) var<storage,read_write> o:f32;\
        @compute @workgroup_size(1) fn main(){var d=1.0;d=2.0;o=5.0;}";
    let out = minify(src);
    assert!(
        !out.contains("var "),
        "write-only local was not eliminated: {out}"
    );
    assert!(
        out.contains("=5"),
        "the real store was wrongly removed: {out}"
    );
}

/// Critical soundness: a local whose pointer ESCAPES to a callee must keep
/// its stores - the callee can read the value through the pointer.
#[test]
fn escaped_local_stores_are_not_removed() {
    let src = "fn sink(p:ptr<function,f32>){}\
        @group(0)@binding(0) var<storage,read_write> o:f32;\
        @compute @workgroup_size(1) fn main(){var d=1.0;d=2.0;sink(&d);o=5.0;}";
    let out = minify(src);
    assert!(
        out.contains("var "),
        "an escaped local (passed by pointer) was wrongly eliminated: {out}"
    );
}

/// An INLINE (single-use) all-`+0` vector folds to the shorter zero-value
/// constructor `vec2i()`.
#[test]
fn inline_all_zero_vector_folds_to_zero_value() {
    let src = "@group(0)@binding(0) var t:texture_2d<f32>;\
        @fragment fn fs()->@location(0) vec4f{return textureLoad(t,vec2i(0,0),0);}";
    let out = minify(src);
    assert!(
        out.contains("vec2i()"),
        "inline all-zero vec2i did not fold to vec2i(): {out}"
    );
}

/// Idempotence gate: a
/// MULTI-use zero vector is `let`-bound, and the bound value must stay
/// `vec2f(0)`, NOT fold to `vec2f()`.  naga re-parses `vec2f()` as a
/// non-emittable `ZeroValue` that can never be re-bound, so folding the bound
/// case would force the generator to inline `vec2f()` at every use on
/// re-minification - a non-idempotent size blow-up.  The bound `vec2f(0)`
/// re-parses to a bindable `Splat` and stays bound; the output is a fixed point.
#[test]
fn bound_multi_use_zero_vector_stays_splat_and_is_idempotent() {
    let src = "fn d(a:vec2f,b:vec2f)->f32{return distance(a,b);}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) f32{\
        let h=vec2f(0.0,0.0);return d(h,p.xy)+d(h,p.zw)+d(h,p.yx);}";
    let out = minify(src);
    assert!(
        out.contains("=vec2f(0);"),
        "bound multi-use zero was not kept as the round-trip-stable splat vec2f(0): {out}"
    );
    assert!(
        !out.contains("=vec2f();"),
        "bound zero was folded to vec2f() (would over-inline on re-minify): {out}"
    );
    assert_eq!(
        minify(&out),
        out,
        "A3 bound-zero handling must be idempotent"
    );
}

/// Critical soundness: `-0.0` has a non-zero bit pattern, so it must NEVER
/// fold to a zero-value constructor (that would flip the sign bit:
/// `1.0/-0.0 == -inf` vs `+inf`).
#[test]
fn negative_zero_vector_is_never_folded() {
    let src = "fn g(x:f32)->vec2f{return vec2f(-0.0,-0.0)*x;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(g(p.x),0,0);}";
    let out = minify(src);
    assert!(out.contains("-0."), "the -0.0 sign bit was lost: {out}");
}

/// A struct local built up member-by-member and then read is coalesced into
/// a single struct constructor, with members in DECLARATION order (the pass
/// reorders from source-assignment order - value-safe because every member
/// value is a pre-materialised handle computed before its store).
#[test]
fn struct_field_build_coalesces_to_constructor() {
    let src = "struct T{a:f32,b:f32,c:f32,}\
        fn mk(x:f32,y:f32,z:f32)->T{var t:T;t.a=x*2.0;t.b=y*3.0;t.c=z*4.0;return t;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        let r=mk(p.x,p.y,p.z);return vec4f(r.a,r.b,r.c,1.0);}";
    let out = minify(src);
    assert!(
        out.contains("return a(") || out.contains("=a("),
        "struct field-build was not coalesced into a constructor: {out}"
    );
    assert!(
        !out.contains(".b="),
        "member stores survived coalescing: {out}"
    );
}

/// Critical soundness: a member value that READS a sibling member
/// (`t.b = t.a + 1`) must NOT be coalesced - the constructor would read an
/// unset member.  The field-by-field build is preserved.
#[test]
fn struct_build_with_sibling_read_is_not_coalesced() {
    let src = "struct T{a:f32,b:f32,}\
        fn mk(x:f32)->T{var t:T;t.a=x*2.0;t.b=t.a+1.0;return t;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        let r=mk(p.x);return vec4f(r.a,r.b,0,0);}";
    let out = minify(src);
    // Not coalesced => the struct local survives as a `var` (a coalesced build
    // is `return T(...)` with no local).  `minify` re-validates, so the staged
    // build is also confirmed value-correct.
    assert!(
        out.contains("var "),
        "a sibling-dependent struct build was wrongly coalesced (would read an unset member): {out}"
    );
}

/// An identity swizzle (`.xy` on a vec2, `.xyz` on a vec3, `.xyzw` on a
/// vec4 - selects every lane in order) is a no-op and is elided to the base.
#[test]
fn identity_swizzle_is_elided() {
    let src = "@group(0)@binding(0) var<uniform> s:vec2f;\
        @fragment fn fs()->@location(0) vec4f{let r=s.xy;return vec4f(r,0,0);}";
    let out = minify(src);
    assert!(
        !out.contains(".xy"),
        "identity swizzle `.xy` on a vec2 was not elided: {out}"
    );
}

/// Parenthesisation safety: an identity swizzle whose base needs
/// parentheses as an operand (a `Binary`) must NOT be elided - dropping it
/// would leave the base unparenthesised in the parent expression.
#[test]
fn identity_swizzle_over_binary_base_is_kept() {
    let src = "@group(0)@binding(0) var<uniform> s:vec2f;\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        let r=(s+p.xy).xy*p.zw;return vec4f(r,0,0);}";
    let out = minify(src);
    assert!(
        out.contains(").xy"),
        "identity swizzle over a Binary base was wrongly elided (parenthesisation hazard): {out}"
    );
}

/// A maximal run of >=2 equal scalar components collapses to a sub-vector
/// splat when that is shorter: `vec4f(.333,.333,.333,1)` -> `vec4f(vec3f(.333),1)`.
#[test]
fn vector_subsplat_run_collapses_when_shorter() {
    let src = "@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        return vec4f(0.333,0.333,0.333,1.0)*p.x;}";
    let out = minify(src);
    assert!(
        out.contains("vec3f(.333)"),
        "a long-valued scalar run was not collapsed to a sub-vector splat: {out}"
    );
}

/// No-grow gate: a sub-splat that would be LONGER than the inline form is
/// discarded - `vec4f(0,0,0,2)` with no short `vec3f` alias stays inline, never
/// grows to `vec4f(vec3f(),2)`.
#[test]
fn vector_subsplat_does_not_grow() {
    let src = "@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        return vec4f(0.0,0.0,0.0,2.0)*p.x;}";
    let out = minify(src);
    assert!(
        !out.contains("vec3f()"),
        "an unprofitable zero sub-splat grew the output: {out}"
    );
}

/// When TWO struct locals are built interleaved in one function, collapsing
/// both must not drop or misplace the unrelated statements between the two
/// builds: every plan's store indices are positions in the *same* pre-mutation
/// body, so the rebuild consults all of them in one pass rather than rewriting
/// per local (which would shift later indices and silently drop the live
/// `g = 7777;` store between the builds).
#[test]
fn interleaved_struct_builds_do_not_drop_live_statements() {
    let src = "struct S{a:i32,b:i32,}struct R{x:i32,y:i32,}\
        @group(0)@binding(0) var<storage,read_write> g:i32;\
        @compute @workgroup_size(1) fn main(){\
        var t:S;var u:R;\
        t.a=1;u.x=100;u.y=200;g=7777;t.b=2;\
        g=g+t.a+t.b*10+u.x*1000+u.y*100000;}";
    let out = minify(src);
    // The live store between the two interleaved builds must survive.
    assert!(
        out.contains("7777"),
        "the live `g = 7777` store between two interleaved struct builds was dropped: {out}"
    );
    // Both structs must still collapse to a constructor (no leftover member
    // stores re-introduced by a misplaced splice): the only `=` stores left to
    // the struct locals are their constructor inits, so no `.<member>=` remains.
    assert!(
        !out.contains(".b=2") && !out.contains(".y=200"),
        "an interleaved struct build left orphan member stores: {out}"
    );
}

/// Two single-use pure calls feeding one `if` condition must BOTH inline into
/// the condition, not leave one `let`-bound.  An `Emit` merges sibling pending
/// calls onto one carrier; consuming only the first (a prior bug) needlessly
/// bound the second to a `let`.  Both survived the same clears, so both are
/// safe to inline at the shared use site - a byte-minimality guard.
#[test]
fn sibling_pure_calls_in_if_condition_both_inline() {
    // `f`/`g` are pure (write only locals) and loop-bearing, so neither is
    // fully inlined - each survives as a `Call` whose single-use result feeds
    // the `==`.  Their own bodies emit no `let`, so any `let` in the output can
    // only be a needlessly-bound call result.
    let src = "\
        fn f(x:f32)->f32{ var s=0.0; for(var i=0;i<9;i=i+1){ s=s+x; } return s; }\
        fn g(x:f32)->f32{ var s=1.0; for(var i=0;i<9;i=i+1){ s=s*x; } return s; }\
        @fragment fn main(@location(0) v:f32,@location(1) w:f32)->@location(0) f32{\
            if(f(v)==g(w)){ return 1.0; } return 0.0; }";
    let out = minify(src);
    assert!(
        !out.contains("let "),
        "both pure calls should inline into the `if` condition, none `let`-bound: {out}"
    );
}
