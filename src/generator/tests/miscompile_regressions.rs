//! End-to-end regressions over the full pipeline ([`crate::run`], default
//! profile): each test pins a verified miscompile, a strict-parser rejection,
//! or a minification-quality loss (an optimization that stopped firing, or
//! output that grew).

use super::helpers::assert_valid_wgsl;
use crate::config::{Config, TraceConfig};

/// The CLI minifies on a big-stack worker; the deep-chain tests overflow the
/// default test-thread stack in debug.  A panic inside `f` is re-raised.
fn on_big_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    match std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .expect("spawn big-stack test thread")
        .join()
    {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn minify(src: &str) -> String {
    minify_preserving(src, &[])
}

/// `minify` with `preserve` kept by name: a helper called once is otherwise
/// spliced into its caller, and these fixtures test the generator's
/// handling of the CALL.
fn minify_preserving(src: &str, preserve: &[&str]) -> String {
    let src = src.to_string();
    let preserve: Vec<String> = preserve.iter().map(|s| s.to_string()).collect();
    on_big_stack(move || {
        let config = Config {
            preserve_symbols: preserve,
            ..Config::default()
        };
        let out = crate::run(&src, &config).expect("run failed");
        assert_valid_wgsl(&out.source);
        out.source
    })
}

/// An identity fold (`0 + x`) over a single-use storage `Load` must NOT
/// relocate the read past an intervening `Store` to the same place; doing so
/// would make `data[1] = 0u + data[0]` read the post-`Store` value.
#[test]
fn identity_fold_does_not_reorder_load_past_store() {
    let src = "@group(0)@binding(0) var<storage,read_write> data:array<u32>;\
        @compute @workgroup_size(1) fn main(){ let a=data[0]; data[0]=99u; data[1]=0u+a; }";
    let out = minify(src);
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

/// A single-use `textureLoad` inlined past an intervening `textureStore` to
/// the same storage texture would read the post-store texel.
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
        out.contains("let a=textureLoad(A,vec2i());textureStore(A,vec2i(),"),
        "textureLoad must be bound before the intervening textureStore: {out}"
    );
}

/// No dedup may redirect a `continuing`-block expression to a body `let`
/// defined after a `continue` (the continue iteration skips it); the
/// continuing block must recompute the shared expression.
#[test]
fn body_let_is_not_shared_into_continuing_past_continue() {
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
    // Name-agnostic: a recompute leaves a `sin(` inside the continuing block;
    // a reuse of the body binding leaves none.
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
/// explicit `(*p)` deref - strict WGSL parsers reject `p.xyz`.  `rd` is
/// preserved: its two-site clone renders shorter and would dissolve the
/// pointer parameter under test.
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
    let out = minify_preserving(src, &["rd"]);
    assert!(
        out.contains("(*a).xyz"),
        "pointer-value swizzle must be dereferenced as (*p).xyz: {out}"
    );
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
    // The `f` suffix may drop (the `vec2f` components pin f32); the
    // two-component shape is what matters.
    assert!(
        out.contains("vec4(A,A)*a")
            || out.contains("vec4(A,A) * a")
            || out.contains("vec4f(A,A)*a")
            || out.contains("vec4f(A,A) * a"),
        "sub-vector compose was wrongly splat-elided: {out}"
    );
}

/// The writability test must peel a `BindingArray` to its element texture, or
/// a single-use `textureLoad(texs[i], ..)` leaves the hazard set and inlines
/// past an intervening `textureStore` to the same element.
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
        out.contains("let a=textureLoad(A[0],vec2i());textureStore(A[0],vec2i(),"),
        "binding-array textureLoad must be bound before the intervening store: {out}"
    );
}

/// An inlined whole-pointee `Load` used as a postfix base must render
/// `(*p).field`; bare `*p.field` parses as `*(p.field)` and forces the naga
/// fallback.
#[test]
fn inlined_pointee_load_postfix_base_is_parenthesized() {
    let src = "struct M { mx: vec4<f32>, my: vec4<f32> }\
        fn f(d: ptr<function, M>, s: f32) { let e = (*d); (*d).mx = e.mx + vec4f(s); }\
        @fragment fn main() -> @location(0) vec4f { var m: M; m.my = vec4f(1.0); f(&m, 2.0); return m.mx; }";
    let out = minify_preserving(src, &["f"]);
    // Names are mangled; the deref'd pointer is the first argument `a`.
    assert!(
        out.contains("(*a)."),
        "inlined pointee-Load postfix base must be parenthesised `(*p).field`: {out}"
    );
    assert!(
        !out.contains("*a."),
        "must not emit the WGSL-invalid unparenthesised `*ptr.field`: {out}"
    );
    assert!(
        out.matches('\n').count() <= 1,
        "output looks like the verbose naga fallback (lost minification): {out}"
    );
}

/// `var<immediate>` emitted as `var<private>` silently reads zero-initialised
/// memory instead of the host-supplied immediate data.
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

/// The directive scan must catch f16 literals and casts, not just registered
/// f16 types: `f16(v) + 2.0h` keeps its literal (a runtime operand, so the
/// conversion cannot fold) and still needs `enable f16;` or naga rejects
/// the text.
#[test]
fn enable_f16_emitted_for_surviving_f16_literal_in_conversion() {
    // No source `enable f16;`: detection injects it on input and the emitter
    // must re-assert it on output.
    let src = "@fragment fn main(@location(0) v: f32) -> @location(0) vec4<f32> { \
        let x = f16(v) + 2.0h; return vec4<f32>(f32(x), 0.0, 0.0, 1.0); }";
    let out = minify(src);
    assert!(
        out.contains("enable f16;"),
        "output uses an f16 literal but dropped `enable f16;`: {out}"
    );
}

/// `lhs = rhs * lhs` must not fold to `lhs *= rhs` for a matrix/vector
/// product: `*=` desugars to `lhs = lhs * rhs`, the transposed product.
#[test]
fn matrix_compound_assign_preserves_non_commutative_order() {
    // mat*mat, rhs-self.
    let mm = minify(
        "@group(0)@binding(0) var<storage,read_write> m: mat3x3<f32>;\
         @group(0)@binding(1) var<storage,read> n: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { m = n * m; }",
    );
    assert!(
        mm.contains("A=a*A;") && !mm.contains("*="),
        "mat*mat `m = n*m` was folded to a swapped `*=` (transposed product): {mm}"
    );
    // mat*vec, rhs-self.
    let mv = minify(
        "@group(0)@binding(0) var<storage,read_write> v: vec3<f32>;\
         @group(0)@binding(1) var<storage,read> M: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { v = M * v; }",
    );
    assert!(
        mv.contains("A=a*A;") && !mv.contains("*="),
        "mat*vec `v = M*v` was folded to a swapped `*=`: {mv}"
    );
    // Left-self is order-preserving and must still fold.
    let ls = minify(
        "@group(0)@binding(0) var<storage,read_write> m: mat3x3<f32>;\
         @group(0)@binding(1) var<storage,read> n: mat3x3<f32>;\
         @compute @workgroup_size(1) fn main() { m = m * n; }",
    );
    assert!(
        ls.contains("A*=a;"),
        "left-self matrix `m = m*n` should still fold to `m *= n`: {ls}"
    );
    // matrix*scalar commutes and must still fold.
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

/// A float product's zero carries the sign of both operands (`-2.0h * 0.0h`
/// is `-0.0h`, a literal Dawn on Metal reads as `+0.0`), so the absorber
/// matches integer zeros only (`Class::IntZero`) and the evaluator never
/// manufactures a `-0.0`: `x * 0.0h` stays.
#[test]
fn f16_multiply_by_zero_is_not_mis_signed() {
    let out = minify(
        "enable f16;\
         @group(0)@binding(0) var<storage,read_write> out: array<f16, 4>;\
         @compute @workgroup_size(1) fn main() { let a = -2.0h; out[0] = a * 0.0h; }",
    );
    assert!(
        out.contains("*0h"),
        "f16 `-2.0h * 0.0h` must keep the signed product, not fold to +0.0h: {out}"
    );
    assert!(
        !out.contains("A[0]=0h"),
        "f16 `-2.0h * 0.0h` was mis-signed to +0.0h: {out}"
    );
}

/// naga's WGSL back-end hits `unreachable!()` on an override size expression
/// (SIGABRT under `panic = "abort"`), so `run` must skip the baseline/fallback
/// emit for override-sized arrays.
#[test]
fn override_sized_array_does_not_abort() {
    let out = minify(
        "override O = 123;\
         alias A = array<i32, O*2>;\
         var<workgroup> W: A;\
         @compute @workgroup_size(1) fn main() { let p: ptr<workgroup, A> = &W; (*p)[0] = 42; }",
    );
    assert!(
        out.contains("array<i32,") && out.contains("var<workgroup>"),
        "override-sized workgroup array must survive minification: {out}"
    );
}

/// Tail-arm `return;`s in a void function are redundant, and naga's
/// `ensure_block_returns` re-synthesises them on every re-parse, so keeping
/// them grows the text by an `else{return;}` per arm per round trip.
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

/// `if c { A; return; } else { B; return; }`, the shape naga's front-end
/// gives every void tail, must minify to `if c { A } else { B }`: the pin on
/// `strip_tail_void_returns` running ahead of else-elision.
#[test]
fn void_fn_arm_returns_fold_into_if_else() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() {\
            if o > 3u { o = 1u; return; } else { o = 2u; return; }\
        }";
    let out = minify(src);
    assert!(
        out.contains("{A=1;}else{A=2;}") && !out.contains("return"),
        "each store keeps its own arm and no return survives: {out}"
    );
}

/// The inliner's node budget counts `faceForward(...)` as one node but it
/// renders 12+ characters, so duplicating it per call site costs more than
/// the declaration saves.
#[test]
fn multi_site_math_helper_is_not_duplicated() {
    let src = "fn h(x: vec2f) -> vec2f { return faceForward(x, x, x); }\
        @group(0)@binding(0) var<storage,read_write> o: vec2f;\
        @compute @workgroup_size(1) fn main() { o = h(o) + h(o.yx) + h(o * 2.0); }";
    let out = minify(src);
    assert_eq!(
        out.matches("faceForward").count(),
        1,
        "the Math body must render once, in the kept helper: {out}"
    );
}

/// The expression-template inliner excludes void functions, so without the
/// dedicated empty-callee deletion `fn e(){} ... e();` stays pinned alive by
/// its own call sites.
#[test]
fn calls_to_empty_functions_dissolve() {
    let src = "fn e() {}\
        fn f() {}\
        @group(0)@binding(0) var<storage,read_write> o: u32;\
        @compute @workgroup_size(1) fn main() { e(); f(); f(); o = 1u; }";
    let out = minify(src);
    assert!(
        !out.contains("fn e") && !out.contains("fn f") && !out.contains("();"),
        "empty functions and their call sites must dissolve: {out}"
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
/// survives DCE; crediting it with `enable f16;` retains a device-feature
/// requirement the text no longer has and drops on re-minify (non-idempotent).
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

/// tint rejects expression nesting deeper than 512 while naga accepts any
/// depth, so unbounded single-use `let` inlining can flatten a chain into text
/// only naga accepts; the emitter splits it with forced bindings.
#[test]
fn deep_single_use_chain_is_depth_capped() {
    // 600 distinct scalars: nothing for const-fold/CSE to collapse.
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
    assert!(
        out.contains("let "),
        "a >512-deep chain must be split by at least one forced binding"
    );
}

/// tint's loop-exit analysis is syntactic (a `break` inside `if false {}`
/// counts) while naga validates an exit-less `loop{}`, so folding the
/// const-false `if` turns tint-valid input into "loop does not exit" the naga
/// self-check cannot catch; a dropped block carrying the loop's only exit
/// must be kept.
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

/// `break if false;` never breaks at runtime but is the only lexical exit
/// tint sees.
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

/// The `default` dropped by a const-selector switch collapse carries the
/// `return` that is the loop's only exit.
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

/// The exit guards must not pessimise ordinary folds: outside a loop a
/// const-false `if` folds even with a `return`; inside a loop one without an
/// exit folds when the loop has a real exit of its own.
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

/// Splicing `if (true) { continue; }` makes the trailing `break` unreachable
/// under tint's behavior sequencing ("loop does not exit"); the kept `if`
/// preserves `Next` via its empty dead arm.
#[test]
fn loop_exit_shading_const_if_continue_is_kept() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: array<u32, 8>;\
        @compute @workgroup_size(1) fn main() {\
            var i: u32 = 0u;\
            loop { o[i % 8u] = i; i = i + 1u; if (true) { continue; } break; }\
        }";
    let out = minify(src);
    assert!(
        out.contains("if true{continue;}") && out.contains("break;"),
        "splicing the all-continue arm shades the trailing break from tint: {out}"
    );
}

/// A const-selector switch whose matched chain is `{ continue; }`: splicing
/// it would shade the trailing `break`.
#[test]
fn loop_exit_shading_const_switch_chain_is_kept() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: array<u32, 8>;\
        @compute @workgroup_size(1) fn main() {\
            var i: u32 = 0u;\
            loop {\
                o[i % 8u] = i; i = i + 1u;\
                switch (1i) { case 1i: { continue; } case 2i: { o[0] = 9u; } default: { } }\
                break;\
            }\
        }";
    let out = minify(src);
    assert!(
        out.contains("switch") && out.contains("break;"),
        "splicing the all-continue matched chain shades the trailing break: {out}"
    );
}

/// A collapse inside a dynamic switch's sole case: the spliced `continue`
/// makes the whole switch never-falls-through and shades the loop-level
/// `break` after it.
#[test]
fn loop_exit_shading_nested_case_collapse_keeps_outer_break() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: array<u32, 8>;\
        @compute @workgroup_size(1) fn main() {\
            var i: u32 = 0u;\
            loop {\
                o[i % 8u] = i; i = i + 1u;\
                switch (i32(i)) { default: { if (true) { continue; } } }\
                break;\
            }\
        }";
    let out = minify(src);
    assert!(
        out.contains("break;"),
        "the loop-level break must stay reachable after the nested collapse: {out}"
    );
}

/// An all-paths-`break` arm is the loop's exit, so `if (true) { break; }`
/// still folds to a bare `break`.
#[test]
fn loop_exit_shading_guard_still_folds_break_splice() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: array<u32, 8>;\
        @compute @workgroup_size(1) fn main() {\
            var i: u32 = 0u;\
            loop { o[i % 8u] = i; i = i + 1u; if (true) { break; } }\
        }";
    let out = minify(src);
    assert!(
        !out.contains("if true") && out.contains("break;"),
        "an exit-carrying splice is safe and must still fold: {out}"
    );
}

/// `rayQueryGetCandidateIntersection` reads the query's current traversal
/// state, so a read held across `rayQueryProceed` must bind at its Emit point
/// rather than re-evaluate after the second Proceed (a value miscompile the
/// self-check cannot see).
#[test]
fn ray_query_candidate_read_stays_between_proceeds() {
    let src = "enable wgpu_ray_query;\
        @group(0) @binding(0) var acc: acceleration_structure;\
        @group(0) @binding(1) var<storage, read_write> out: vec4<f32>;\
        @compute @workgroup_size(1) fn main() {\
            var q: ray_query;\
            rayQueryInitialize(&q, acc, RayDesc(0u, 0xFFu, 0.1, 100.0, vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(0.0, 1.0, 0.0)));\
            let p1 = rayQueryProceed(&q);\
            let i = rayQueryGetCandidateIntersection(&q);\
            let p2 = rayQueryProceed(&q);\
            out = vec4<f32>(i.t, f32(i.kind), f32(u32(p1)), f32(u32(p2)));\
        }";
    let out = minify(src);
    assert_eq!(
        out.matches("rayQueryGetCandidateIntersection").count(),
        1,
        "the stateful read must not be duplicated per use: {out}"
    );
    let first_proceed = out.find("rayQueryProceed").expect("first proceed");
    let get = out
        .find("rayQueryGetCandidateIntersection")
        .expect("intersection read");
    let second_proceed = out[first_proceed + 1..]
        .find("rayQueryProceed")
        .map(|p| p + first_proceed + 1)
        .expect("second proceed");
    assert!(
        first_proceed < get && get < second_proceed,
        "the read must stay between the two Proceeds: {out}"
    );
}

/// naga's expression validator (unlike its statement validator) admits a
/// `ptr<function, ray_query>` argument as the query operand; the emitter must
/// pass the pointer through verbatim, not take `&` of it.
#[test]
fn ray_query_read_through_pointer_param_minifies() {
    let src = "enable wgpu_ray_query;\
        @group(0) @binding(0) var acc: acceleration_structure;\
        @group(0) @binding(1) var<storage, read_write> out: f32;\
        fn get_t(p: ptr<function, ray_query>) -> f32 {\
            return rayQueryGetCommittedIntersection(p).t;\
        }\
        @compute @workgroup_size(1) fn main() {\
            var q: ray_query;\
            rayQueryInitialize(&q, acc, RayDesc(0u, 0xFFu, 0.1, 100.0, vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(0.0, 1.0, 0.0)));\
            let p1 = rayQueryProceed(&q);\
            out = get_t(&q);\
        }";
    let out = minify(src);
    assert!(
        out.contains("rayQueryGetCommittedIntersection("),
        "pointer-parameter query reads must emit without a spurious `&`: {out}"
    );
}

/// Tail-return elision pops the void function's tail `if` off the emission
/// slice; the for-init absorption's later-use scan must still see it, or a
/// counter read only there is absorbed into a for-scoped `var` and the
/// self-check ships the input verbatim.
#[test]
fn void_tail_if_use_blocks_for_init_absorption() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: f32;\
        @group(0)@binding(1) var<storage> s: f32;\
        @compute @workgroup_size(1) fn main() {\
            var i: i32;\
            o = s + 1.0;\
            i = i32(o);\
            for (; i < 10; i++) { o = o + 1.0; }\
            if s > 0.5 { o = f32(i); }\
        }";
    let out = minify(src);
    assert!(
        out.len() < src.len() - 40,
        "the scope bug ships the input verbatim (0% minification): {out}"
    );
}

/// Right-nested subtraction costs tint's recursive parser ~4 frames per
/// parenthesised level (vs ~1 flat) and naga's frontend stops near 197
/// levels, so the depth cap must count frame-weighted units, not raw nodes.
#[test]
fn deep_paren_chain_is_bound_within_parser_budgets() {
    let mut expr = String::from("s");
    for _ in 0..150 {
        expr = format!("(s - {expr})");
    }
    let src = format!(
        "@group(0)@binding(0) var<storage,read_write> o: f32;\
         @group(0)@binding(1) var<storage> s: f32;\
         @compute @workgroup_size(1) fn main() {{ o = {expr}; }}"
    );
    let out = minify(&src);
    assert!(
        out.matches("let ").count() >= 2,
        "a 150-level paren chain must be split by the frame-weighted cap: {out}"
    );
}

/// A for-loop guard renders inline into the header before any Emit-range
/// processing, bypassing the depth gate; an over-deep guard chain must fall
/// back to plain `loop` emission where the gate binds it.
#[test]
fn deep_for_guard_chain_falls_back_to_loop_emission() {
    let mut cond = String::from("f32(i)");
    for _ in 0..80 {
        cond = format!("abs({cond} + 0.5)");
    }
    let src = format!(
        "@group(0)@binding(0) var<storage,read_write> o: f32;\
         @compute @workgroup_size(1) fn main() {{\
             var i: i32 = 0;\
             loop {{\
                 if ({cond} > 1e30) {{ break; }}\
                 o = o + 1.0; i = i + 1;\
             }}\
         }}"
    );
    let out = minify(&src);
    assert!(
        !out.contains("for("),
        "an over-deep guard must not inline into a for-header: {out}"
    );
    assert!(
        out.matches("let ").count() >= 2,
        "the guard chain must be depth-bound on the loop path: {out}"
    );
}

/// A guard preload's `result` renders as a childless leaf that hides the
/// pointer's depth from the for-header cap, so a deep preload pointer must
/// also force the plain-`loop` fallback.
#[test]
fn deep_for_preload_pointer_falls_back_to_loop_emission() {
    let mut idx = String::from("r");
    for _ in 0..90 {
        idx = format!("{idx}+r");
    }
    let src = format!(
        "@group(0)@binding(0) var<storage,read_write> o: u32;\
         @group(0)@binding(1) var<uniform> r: u32;\
         var<workgroup> wg: array<u32,8>;\
         @compute @workgroup_size(1) fn main() {{\
             for (var i: u32 = 0u; i < workgroupUniformLoad(&wg[({idx})%8u]); i = i + 1u) {{ o = o + 1u; }}\
         }}"
    );
    let out = minify(&src);
    assert!(
        !out.contains("for("),
        "an over-deep preload pointer must not inline into a for-header: {out}"
    );
    // Control: a shallow preload pointer still converts to `for(`.
    let shallow = "@group(0)@binding(0) var<storage,read_write> o: u32;\
         var<workgroup> wg: array<u32,8>;\
         @compute @workgroup_size(1) fn main() {\
             for (var i: u32 = 0u; i < workgroupUniformLoad(&wg[2]); i = i + 1u) { o = o + 1u; }\
         }";
    assert!(
        minify(shallow).contains("for("),
        "a shallow preload pointer must still convert to a for-header"
    );
}

/// naga's WGSL writer has `Statement::RayQuery => unreachable!()`, so the
/// baseline emit must be skipped for ray-query modules; and since that writer
/// never emits `enable wgpu_ray_query;`, the directive scan must add it itself
/// or the self-check re-parse rejects the text.
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

/// naga's lowerer propagates no expected type into
/// `rayQueryGenerateIntersection`'s hit_t, so a bare-int shortest form
/// (`10.0` -> `10`) re-parses as i32 ("Hit distance must be an f32"); the slot
/// must keep a typed spelling.
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

/// `vertex_return`: the type flags must survive on both `ray_query` and
/// `acceleration_structure`, `getCommittedHitVertexPositions` renders inline
/// (naga's writer has it as `unreachable!()`), and both enables are needed
/// (the flag needs `wgpu_ray_query_vertex_return`, the base types
/// `wgpu_ray_query`).
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

/// A constant tree neither evaluator computes (an `unpack*` root, a subnormal
/// `bitcast`) must not reach a failable operator inline: the input kept the
/// shift/division at runtime behind a local, and forwarding it makes tint
/// const-evaluate at shader creation (`1 << 31` changes sign, `1/subnormal`
/// overflows).  A literal integer bitcast folds instead, taking the tree
/// above it along; a tree the folder merely left alone (a matrix product) was
/// a const-expression in the input and stays inline.
#[test]
fn opaque_constant_tree_binds_before_a_failable_operator() {
    let head = "@group(0)@binding(0) var<storage,read_write> out:array<u32>;";
    let shift = format!(
        "{head}@compute @workgroup_size(1) fn main() {{\
            var v = i32(unpack2x16float(0x3c003c00u).x); out[0] = u32(v << 31u); }}"
    );
    let out = minify(&shift);
    assert!(
        out.contains("let ") && out.contains("<<31") && !out.contains(".x)<<31"),
        "unpack-rooted shift operand must be let-bound: {out}"
    );
    let folded = format!(
        "{head}@compute @workgroup_size(1) fn main() {{\
            var v = clamp(sign(bitcast<i32>(2u)), -100i, 100i); out[0] = u32(v << 31u); }}"
    );
    let out = minify(&folded);
    assert!(
        out.contains("=2147483648;") && !out.contains("bitcast") && !out.contains("<<"),
        "a literal bitcast folds, and the shift and conversion above it: {out}"
    );
    // A `let`: a `var` is kept by the forwarders' static-error decline (a
    // compound const target in a divisor slot) and never reaches the
    // generator's binding.
    let div = format!(
        "{head}@compute @workgroup_size(1) fn main() {{\
            let b = bitcast<f32>(1u); out[0] = u32(1.0 / b); }}"
    );
    let out = minify(&div);
    assert!(
        out.contains("=1f;") && out.contains("/bitcast<f32>(1u)"),
        "division by a bitcast-rooted tree must bind an operand: {out}"
    );
    let matrix = format!(
        "{head}@fragment fn fs() -> @location(0) vec2f {{\
            return clamp(mat3x2f(-0.1146, 0.5, -0.3854, -0.4542, 0.5, -0.0458) * vec3f(1.0) + 0.5,\
                vec2f(0.0), vec2f(1.0)); }}"
    );
    let out = minify(&matrix);
    assert!(
        !out.contains("let "),
        "a non-opaque constant tree must stay inline: {out}"
    );
}

/// load_dedup must not forward a pre-loop load into a `break_if` read inside
/// the loop, which would observe the loop-mutated place across the back-edge:
/// `cond(n)` is loop-invariant, so a never-breaking loop would wrongly
/// terminate.  The init is a builtin so the pre-loop read cannot fold.  The
/// comparison over the load is loop-invariant work, which the loop pin binds
/// ahead of the loop as well.
#[test]
fn loop_break_if_does_not_read_loop_mutated_load() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<i32>;\
        fn cond(v: i32) -> bool { return v >= 500; }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
            var n: i32 = i32(id.x); var count: i32 = 0; let c = cond(n);\
            loop { count = count + 1; out[0] = count;\
                continuing { n = n + 100; break if c; } } }";
    let out = minify(src);
    let loop_at = out.find("loop{").expect("loop emitted: {out}");
    assert!(
        out[..loop_at].contains("let C=A;") && out[..loop_at].contains("=C>=500;"),
        "loop-invariant break value was relocated into the loop: {out}"
    );
}

/// A reachable bare `break` inside a case falls through to the code after the
/// switch, so that code is live: `out[0]=x` is reached when
/// `gid.x==1 && x>0`.
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
    assert!(
        out.contains("if B[a.x]>0{break;}"),
        "the bare break inside the case was lost: {out}"
    );
    assert!(
        out.contains("}A[0]=B[a.x];}"),
        "post-switch store was wrongly dropped as unreachable: {out}"
    );
}

/// Single-use inlining must not sink an implicit derivative (`dpdx`) into a
/// non-uniform branch: WGSL uniformity rules forbid it and Tint/Dawn reject
/// it, though naga's validator does not.  The arm's store keeps the guard
/// out of the return merge.
#[test]
fn implicit_derivative_not_sunk_into_non_uniform_branch() {
    let src = "var<private> p: f32;\
        @fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {\
        let d = dpdx(tc.x);\
        if (tc.y > 0.5) { p = 1.0; return vec4<f32>(d, 0.0, 0.0, 1.0); }\
        return vec4<f32>(1.0); }";
    let out = minify(src);
    assert!(
        out.contains("=dpdx(A.x);if "),
        "dpdx was sunk out of uniform control flow: {out}"
    );
}

/// A `--preserve-symbol` function's signature is an external contract, so
/// `dead_param` must keep its unused parameter.  Seven call sites keep the
/// function from being inlined away.
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

/// `flag_used_loads` stops recursing once it pins a written load (the binding
/// materialises the whole operand cone), which must not under-bind a nested
/// load: `arr[kk]` is pinned before the store and `kk` binds independently
/// because `k` is later written.
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

/// `--validate-each-pass` re-emits text after every pass; the driver's
/// per-pass emit must not invoke naga's panicking backend on an
/// override-sized array.
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
    // `validate_each_pass` turns a per-pass validation failure or
    // non-convergence into an Err.
    let out = crate::run(src, &config).expect("validate_each_pass must not abort or fail");
    assert!(
        out.source.contains("array<i32,"),
        "override-sized array did not survive: {}",
        out.source
    );
}

/// A scalar `bitcast` of a literal folds to the reinterpreted bits so the
/// value cannot be lost to re-typing (`bitcast<u32>(1.0f)` as
/// `bitcast<u32>(1)` is i32 `0x1`, not `0x3F800000`); where the fold declines
/// (a non-finite result) the operand keeps its `u`, or a strict parser
/// overflows the i32 default.
#[test]
fn bitcast_scalar_literal_folds_or_keeps_concrete_type() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @compute @workgroup_size(1) fn main(){ o[0]=bitcast<u32>(1.0f); o[1]=bitcast<u32>(2.0f); }",
    );
    assert!(
        out.contains("1065353216") && out.contains("1073741824") && !out.contains("bitcast"),
        "float bitcast sources must fold to their bits: {out}"
    );
    let out2 = minify(
        "@group(0)@binding(0) var<storage,read_write> r:array<f32>;\
         @compute @workgroup_size(1) fn main(){ let bits = 4286578688u; r[0]=bitcast<f32>(bits); }",
    );
    assert!(
        out2.contains("4286578688u") && out2.contains("bitcast<f32>("),
        "a -inf bitcast must survive with its `u`-suffixed source: {out2}"
    );
}

/// An extracted literal `const` is abstract-typed, so `bitcast<f32>(C)` would
/// concretize to i32 and overflow above i32::MAX; the bitcast use must keep a
/// concrete u32 spelling while the const serves its other uses, and
/// all-bitcast uses must not leave a dangling const.  The -inf bits keep the
/// fold out.
#[test]
fn bitcast_scalar_literal_pinned_even_when_extracted() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:array<u32>;\
         @group(0)@binding(1) var<storage,read_write> f:array<f32>;\
         @compute @workgroup_size(1) fn main(){ let bits = 4286578688u; f[0]=bitcast<f32>(bits);\
            o[0]=4286578688u; o[1]=4286578688u; o[2]=4286578688u; o[3]=4286578688u;\
            o[4]=4286578688u; o[5]=4286578688u; o[6]=4286578688u; o[7]=4286578688u; }",
    );
    assert!(
        out.contains("bitcast<f32>(") && (out.contains(":u32=") || out.contains("(4286578688u)")),
        "extracted-literal bitcast source was not pinned to its concrete type: {out}"
    );
    let out2 = minify(
        "@group(0)@binding(0) var<storage,read_write> f:array<f32>;\
         @compute @workgroup_size(1) fn main(){ let bits = 4286578688u;\
            f[0]=bitcast<f32>(bits); f[1]=bitcast<f32>(bits); f[2]=bitcast<f32>(bits);\
            f[3]=bitcast<f32>(bits); f[4]=bitcast<f32>(bits); f[5]=bitcast<f32>(bits); }",
    );
    assert!(
        out2.contains("4286578688u") && !out2.contains("const"),
        "all-bitcast extraction left a dangling const or failed to pin: {out2}"
    );
}

/// All six comparisons share one non-associative grammar level with
/// `shift_expression` operands, so `a<b==c<d` is a Dawn/Tint parse error
/// ("mixing '<' and '==' requires parenthesis") though naga round-trips it.
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

/// naga's `ensure_block_returns` injects `return;` after a tail `loop`, so a
/// non-void function whose stripped trailing return leaves it ending in an
/// always-returning loop round-trips invalid ("return expression None does
/// not match"); the generator synthesises a zero-value return.
#[test]
fn terminal_loop_non_void_fn_synthesises_trailing_return() {
    let out = minify(
        "@group(0)@binding(0) var<storage,read_write> o:i32;\
         fn f()->i32{ var i:i32; i=0; loop{ if (true) { return 7; } i=i+1; \
            continuing{ break if i>3; } } return 9; }\
         @compute @workgroup_size(1) fn m(){ o=f(); }",
    );
    // `if (true)` makes the body always return, so `return 9` is stripped.
    assert!(
        out.contains("}return 0i;}"),
        "terminal loop in a non-void fn must gain a trailing zero return: {out}"
    );
}

/// The `ensure_block_returns` injection recurses into `switch` cases; one
/// trailing top-level return suppresses the whole recursion (naga inspects
/// only the block's last statement).
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

/// A function already ending in real returns (7 call sites keep it
/// un-inlined) must not gain a synthesised zero return: an over-broad
/// terminating predicate costs bytes.
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

/// naga's WGSL back-end hits `unreachable!()` writing a non-const
/// override/global initializer such as `override h = 2 * d;`, so the eager
/// baseline emit must be skipped for such modules.
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

/// Coop-matrix role enumerants `A`/`B`/`C` are predeclared and the generator
/// cannot emit coop types, so the module uses naga's wgsl-out, where a rename
/// onto `A`/`B`/`C` collides with the role ("identifier `B` resolves to a
/// declaration"); the names are reserved when a coop type is present.
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
    assert!(
        out.contains("coop_mat8x8<f32, A>") || out.contains("coop_mat8x8<f32,A>"),
        "coop role positions must survive: {out}"
    );
}

/// Several single-use results of a pure, non-inlined function feeding one
/// expression must all inline in one pass; collapsing only the last leaves
/// `let`s a second pass would inline (non-idempotent size drift).
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

/// A matrix built from explicit scalar columns emits in the shorter flat form
/// `mat2x2f(a,b,c,d)` rather than per-column `vec2f(...)`.
#[test]
fn matrix_scalar_columns_flatten() {
    // Formatted source: a pre-minified input could come out shorter than the
    // emit and trip the ship-input guard, hiding the flattening.
    let src = "@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {\
        let m = mat2x2f(cos(p.x), sin(p.x), -sin(p.x), cos(p.x));\
        return vec4f(m[0], m[1]);\
    }";
    let out = minify(src);
    assert!(
        out.contains("mat2x2f(cos(A.x),sin(A.x),-sin(A.x),cos(A.x))"),
        "matrix scalar columns were not flattened to the shorter form: {out}"
    );
    assert!(
        !out.contains("mat2x2f(vec2"),
        "matrix kept the longer per-column constructor form: {out}"
    );
}

/// Flattening must not expand a shared, let-bound column into N repeated
/// scalars; the build-then-compare-shorter gate keeps the short name.
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

/// An array constructor whose element type a concretely-typed component
/// (`vec2f(...)`) pins drops its template parameters to `array(...)`.
#[test]
fn array_type_params_elided_when_concretely_pinned() {
    // Two call sites: called once, `g` would be spliced into `fs`.
    let src = "fn g(i:u32)->vec2f{var a=array(vec2f(-1.5,0.0),vec2f(1.0,2.0),vec2f(3.0,4.0));return a[i];}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(g(u32(p.x)),g(u32(p.y)));}";
    let out = minify(src);
    // Name-agnostic: the element constructor is a one-letter alias.
    assert!(
        out.contains("array(") && !out.contains("array<"),
        "concrete-element array constructor was not elided to array(...): {out}"
    );
}

/// An array of bare abstract literals must never elide its template
/// parameters: `array<u32,2>(7,9)` -> `array(7,9)` re-infers i32.
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

/// An all-literal vector built identically at 3+ live sites is hoisted into a
/// module `const` by the pre-rename `const-hoist` pass, so it gets a short
/// frequency-assigned name and re-minification is a fixed point.
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
    assert!(
        !out.contains("c(7,2,9,3)") || out.matches("c(7,2,9,3)").count() == 1,
        "a use site still inlines the vector instead of the const: {out}"
    );
    assert_eq!(minify(&out), out, "const-hoist output must be idempotent");
}

/// A write-only local (never read, escaped, or atomic-touched) goes entirely,
/// var and stores alike, while a real side effect beside it survives.
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

/// A local whose pointer escapes to a callee keeps its stores.  The sink
/// needs a real body: a call to an empty callee is deleted outright, which
/// would leave this exercising nothing.
#[test]
fn escaped_local_stores_are_not_removed() {
    let src = "@group(0)@binding(1) var<storage,read_write> g:f32;\
        fn sink(p:ptr<function,f32>){ g = *p; }\
        @group(0)@binding(0) var<storage,read_write> o:f32;\
        @compute @workgroup_size(1) fn main(){var d=1.0;d=2.0;sink(&d);o=5.0;}";
    let out = minify_preserving(src, &["sink"]);
    assert!(
        out.contains("var "),
        "an escaped local (passed by pointer) was wrongly eliminated: {out}"
    );
}

/// An inline (single-use) all-`+0` vector folds to the zero-value `vec2i()`.
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

/// A `let`-bound multi-use zero vector is one `ZeroValue` after the fold -
/// `vec2f()`, the spelling a re-parse yields - which the emitter never
/// `let`-binds (no `Emit`); const_hoist shares it through a module `const`
/// where that pays, so every pass lands on the same text.
#[test]
fn bound_multi_use_zero_vector_is_one_zero_value_shared_by_a_constant() {
    let src = "fn d(a:vec2f,b:vec2f)->f32{return distance(a,b);}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) f32{\
        let h=vec2f(0.0,0.0);return d(h,p.xy)+d(h,p.zw)+d(h,p.yx);}";
    let out = minify(src);
    assert!(
        out.contains("const ") && out.matches("vec2f()").count() == 1,
        "one zero value declared once for three uses: {out}"
    );
    assert!(
        !out.contains("vec2f(0)"),
        "the zero vector has no splat spelling left: {out}"
    );
    assert_eq!(minify(&out), out, "bound-zero handling must be idempotent");
}

/// `-0.0` has a non-zero bit pattern, so it must never fold to a zero-value
/// constructor (`1.0/-0.0` is `-inf`, not `+inf`).
#[test]
fn negative_zero_vector_is_never_folded() {
    let src = "fn g(x:f32)->vec2f{return vec2f(-0.0,-0.0)*x;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{return vec4f(g(p.x),0,0);}";
    let out = minify(src);
    assert!(out.contains("-0."), "the -0.0 sign bit was lost: {out}");
}

/// A struct local built member-by-member then read coalesces into one
/// constructor in declaration order; reordering from assignment order is
/// value-safe because every member value is a handle materialised before its
/// store.
#[test]
fn struct_field_build_coalesces_to_constructor() {
    let src = "struct T{a:f32,b:f32,c:f32,}\
        fn mk(x:f32,y:f32,z:f32)->T{var t:T;t.a=x*2.0;t.b=y*3.0;t.c=z*4.0;return t;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        let r=mk(p.x,p.y,p.z);return vec4f(r.a,r.b,r.c,1.0);}";
    let out = minify(src);
    let name = out
        .split("struct ")
        .nth(1)
        .and_then(|rest| rest.split('{').next())
        .expect("the struct survives");
    assert!(
        out.contains(&format!("={name}(")) || out.contains(&format!("return {name}(")),
        "struct field-build was not coalesced into a constructor: {out}"
    );
    assert!(
        !out.contains(".b=") && !out.contains(".a="),
        "member stores survived coalescing: {out}"
    );
}

/// A member value that reads a sibling (`t.b = t.a + 1`) must not coalesce:
/// the constructor would read an unset member.
#[test]
fn struct_build_with_sibling_read_is_not_coalesced() {
    let src = "struct T{a:f32,b:f32,}\
        fn mk(x:f32)->T{var t:T;t.a=x*2.0;t.b=t.a+1.0;return t;}\
        @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
        let r=mk(p.x);return vec4f(r.a,r.b,0,0);}";
    let out = minify(src);
    // A coalesced build is `return T(...)` with no local.
    assert!(
        out.contains("var "),
        "a sibling-dependent struct build was wrongly coalesced (would read an unset member): {out}"
    );
}

/// An identity swizzle (every lane in order) is a no-op and elides to the
/// base.
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

/// An identity swizzle over a `Binary` base must not elide: dropping it would
/// leave the base unparenthesised in the parent expression.
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

/// A sub-splat longer than the inline form is discarded: `vec4f(0,0,0,2)`
/// with no short `vec3f` alias must not grow to `vec4f(vec3f(),2)`.
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

/// Every struct-build plan's store indices are positions in the same
/// pre-mutation body, so the rebuild must consult all plans in one pass;
/// rewriting per local shifts later indices and drops the live `g = 7777;`
/// between two interleaved builds.
#[test]
fn interleaved_struct_builds_do_not_drop_live_statements() {
    let src = "struct S{a:i32,b:i32,}struct R{x:i32,y:i32,}\
        @group(0)@binding(0) var<storage,read_write> g:i32;\
        @compute @workgroup_size(1) fn main(){\
        var t:S;var u:R;\
        t.a=1;u.x=100;u.y=200;g=7777;t.b=2;\
        g=g+t.a+t.b*10+u.x*1000+u.y*100000;}";
    let out = minify(src);
    assert!(
        out.contains("7777"),
        "the live `g = 7777` store between two interleaved struct builds was dropped: {out}"
    );
    // Both builds must still collapse: no `.<member>=` store may remain.
    assert!(
        !out.contains(".b=2") && !out.contains(".y=200"),
        "an interleaved struct build left orphan member stores: {out}"
    );
}

/// An `Emit` merges sibling pending calls onto one carrier; consuming only
/// the first needlessly binds the second, though both survived the same
/// clears and are safe at the shared use site.
#[test]
fn sibling_pure_calls_in_if_condition_both_inline() {
    // `f`/`g` are pure but loop-bearing, so they survive as calls; their
    // bodies emit no `let`, so any `let` is a needlessly bound result.
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

/// A drained tail shell (`if f() { return; }` -> `if f() {}`) over a stashed
/// single-use call must keep the call (`tail_cone_has_stashed_call`); with
/// the IR stripping its tail returns first, the shell is gone before
/// emission and the call stands as a statement of its own.
#[test]
fn vacuous_tail_if_keeps_stashed_impure_call() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<u32,4>;\
        fn f()->bool{ out[0]=out[0]+1u; return out[0]>10u; }\
        @compute @workgroup_size(1) fn main(){ out[1]=7u; if f(){ return; } }";
    let out = minify_preserving(src, &["f"]);
    assert!(
        out.contains("f();") && !out.contains("if f()"),
        "the call must survive its drained consumer as a statement: {out}"
    );
}

/// Every case body drains to empty and the selector holds the stashed call.
#[test]
fn vacuous_tail_switch_keeps_stashed_selector_call() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<u32,4>;\
        fn g()->u32{ out[0]=out[0]+1u; return out[0]; }\
        @compute @workgroup_size(1) fn main(){ out[1]=7u;\
            switch g() { case 0u: { return; } default: { return; } } }";
    let out = minify_preserving(src, &["g"]);
    assert!(
        out.contains("g();") && !out.contains("switch"),
        "the selector call must survive its drained switch as a statement: {out}"
    );
}

/// A single-use pure call in a loop body consumed only in `continuing` must
/// stay bound in the body: inlining relocates its text past the continuing
/// block's stores, so the argument re-reads post-increment values (`acc`
/// sums f(2..4), not f(1..3)).
#[test]
fn body_call_consumed_in_continuing_stays_bound() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: vec2f;\
        fn f(p: f32) -> f32 { var t: f32 = p * 2.0; if (p > 1000.0) { t = 0.0; } return t; }\
        @compute @workgroup_size(1) fn main() {\
            var x: f32 = 1.0; var acc: f32 = 0.0; var i: i32 = 0;\
            loop { let c = f(x);\
                continuing { x = x + 1.0; acc = acc + c; i = i + 1; break if i >= 3; } }\
            o.x = acc; }\
        @compute @workgroup_size(1) fn main2() { o.y = f(4.0); }";
    let out = minify(src);
    assert!(
        out.contains("let B=a(C);continuing"),
        "the body call must bind before the continuing block that consumes it: {out}"
    );
    assert!(
        !out.contains("+=a(C)"),
        "the call text must not relocate into the continuing block: {out}"
    );
}

/// The `break if` condition evaluates after the continuing block, so a body
/// call consumed there would re-read post-increment arguments (2 iterations
/// instead of 3).
#[test]
fn body_call_consumed_in_break_if_stays_bound() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: vec2f;\
        fn f(p: f32) -> f32 { var t: f32 = p * 2.0; if (p > 1000.0) { t = 0.0; } return t; }\
        @compute @workgroup_size(1) fn main() {\
            var x: f32 = 1.0;\
            loop { o.x = o.x + 1.0; let c = f(x);\
                continuing { x = x + 1.0; break if c > 4.0; } } }\
        @compute @workgroup_size(1) fn main2() { o.y = f(4.0); }";
    let out = minify(src);
    assert!(
        out.contains("let B=a(C);"),
        "the body call feeding `break if` must stay bound in the body: {out}"
    );
    assert!(
        out.contains("break if B>4"),
        "`break if` must read the pre-increment binding: {out}"
    );
}

/// A pure call consumed as another pure call's argument bakes the inner text
/// into the outer's stash, so the outer must inherit the inner's
/// argument-local reads or `f(g(y))` floats past `y = ...` and re-reads the
/// post-store value.
#[test]
fn composed_call_inherits_inner_argument_reads() {
    let src = "@group(0)@binding(0) var<storage,read_write> o: vec2f;\
        fn g(p: f32) -> f32 { var t: f32 = p * 2.0; if (p > 1000.0) { t = 0.0; } return t; }\
        fn f(q: f32) -> f32 { var t: f32 = q + 10.0; if (q > 1000.0) { t = 0.0; } return t; }\
        @compute @workgroup_size(1) fn main() {\
            var y: f32 = o.y; var acc: f32 = 0.0; var i: i32 = 0;\
            loop { let a = g(y); let c = f(a); y = f32(i) * 5.0; acc = acc + c; i = i + 1;\
                if (i >= 3) { break; } }\
            o.x = acc; }\
        @compute @workgroup_size(1) fn main2() { o.y = g(3.0) + f(4.0); }";
    let out = minify(src);
    assert!(
        out.contains("let b=c(B(E));E="),
        "the composed call must bind BEFORE the store to the inner argument's local: {out}"
    );
    assert!(
        !out.contains("+=c(B(E))"),
        "the composed call text must not relocate past the store to `b`: {out}"
    );
}

/// Inlining literal args manufactures pairs whose checked folds would decline
/// (`MIN / -1`, `MIN % -1`, sign-overflowing `<<`); declined text fails naga's
/// re-parse const-eval in both generator output and fallback (exit 2 on valid
/// input).  WGSL defines the runtime values (e1, 0, the bit-pattern shift), so
/// the folds must produce them.
#[test]
fn manufactured_min_div_rem_shl_fold_to_defined_values() {
    let src = "@group(0)@binding(0) var<storage,read_write> out:array<i32,4>;\
        fn f(x:i32)->i32{ return x / -1; }\
        fn g(x:i32)->i32{ return x % -1; }\
        fn h(x:i32)->i32{ return x << 1u; }\
        @compute @workgroup_size(1) fn main(){\
            out[0]=f(-2147483648); out[1]=g(-2147483648); out[2]=h(1073741824); }";
    let out = minify(src);
    assert!(
        out.contains("i32(-2147483648)"),
        "MIN / -1 and 1073741824 << 1u must fold to the defined value MIN: {out}"
    );
    assert!(
        out.contains("A[1]=0;"),
        "MIN % -1 must fold to the defined value 0: {out}"
    );
}

/// A float division by literal zero has no valid const-expression spelling
/// (inf), so it must ship as a RUNTIME division: neither the compacted-input
/// bailout nor a hard error on valid input.  The DIVISOR is what stays
/// runtime - the sign-sensitive guard declines forwarding a float zero into a
/// `/` slot, `5f / +0.0` and `5f / -0.0` differing.
#[test]
fn float_div_zero_ships_as_runtime_division() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: f32;\n\
        @compute @workgroup_size(1)\n\
        fn main() {\n  var a: f32 = 5.0;\n  var b: f32 = 0.0;\n  out = a / b;\n}";
    let result = crate::run(src, &crate::config::Config::default())
        .expect("float division by zero must minify, not error");
    assert!(
        result.report.bailout.is_none(),
        "division by zero must not bail out: {}",
        result.source
    );
    assert!(
        result.source.contains("var a=0f;A=5/a;"),
        "the divisor must stay a variable so the division stays runtime: {}",
        result.source
    );
    assert_valid_wgsl(&result.source);
}

/// A chain of single-use pure calls stashes each call inside the next
/// (`A(A(A(...)))`) with no `let`; `render_depth` pricing the stash as a leaf
/// lets 127+ links ship past tint's parser recursion limit while naga accepts
/// it, so the recorded stash depth must force a binding.
#[test]
fn deep_pure_call_chain_binds_past_depth_budget() {
    let mut src = String::from(
        "@group(0)@binding(0) var<storage,read_write> o: f32;\
         fn f(x: f32) -> f32 { return x * 0.5 + 0.25; }\
         @compute @workgroup_size(1) fn main(){ var a: f32 = 1.0; let b0 = f(a);",
    );
    for i in 1..150 {
        src.push_str(&format!("let b{i} = f(b{});", i - 1));
    }
    src.push_str("o = b149; }");
    let out = minify(&src);
    let mut cur = 0i32;
    let mut max_depth = 0i32;
    for c in out.chars() {
        match c {
            '(' => {
                cur += 1;
                max_depth = max_depth.max(cur);
            }
            ')' => cur -= 1,
            _ => {}
        }
    }
    assert!(
        max_depth <= 80,
        "stashed-call chain must bind past the depth budget, got paren depth {max_depth}: {}",
        &out[..out.len().min(400)]
    );
}

/// load_dedup's switch meet must not apply when any case breaks: a bare
/// `break` reaches post-switch code carrying the pre-break value (the
/// SPIR-V-structurizer phi idiom), so meeting the fall-off-the-end states
/// forwards a value that makes the local dead and deletes its stores.
#[test]
fn switch_meet_keeps_stores_when_case_breaks() {
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<f32>;\
        @group(0)@binding(1) var<uniform> u: vec4u;\
        @compute @workgroup_size(1) fn main() {\
            let s = i32(u.x); let c = u.y == 1u; let v = f32(u.z);\
            var x: f32 = 3.0;\
            switch (s) {\
                case 0 { if (c) { break; } x = v; }\
                default { x = v; }\
            }\
            out[0] = x; }";
    let out = minify(src);
    assert!(
        out.contains("var a=3f;"),
        "the init must survive - the break path reads it: {out}"
    );
    assert!(
        out.contains("a=f32(A.z);}default{a=f32(A.z);}"),
        "case stores must survive - only the fall-through paths write v: {out}"
    );
    assert!(
        out.contains("B[0]=a;"),
        "the post-switch read must consult the local, not a forwarded value: {out}"
    );
}

/// Every argument of a shared-type float builtin rendering as a bare
/// whole-number literal leaves the call typed AbstractInt (`mix(1,2,1)`),
/// which naga rejects even though tint accepts it - and the rejection is the
/// self-check's, so ONE such call dropped the whole module to naga's emitter.
#[test]
fn all_literal_float_builtin_args_keep_a_typed_pin() {
    for call in ["mix(1.f, 2.f, 1.f)", "smoothstep(2.f, 4.f, 3.f)"] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> out: f32;\n\
             fn helper() -> f32 {{ return {call}; }}\n\
             @compute @workgroup_size(1) fn main() {{ out = helper(); }}"
        );
        let output = crate::run(&src, &Config::default()).expect("must minify");
        let gen_report = output
            .report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "generator_emit")
            .expect("generator_emit pass must exist");
        assert!(
            !gen_report.rolled_back,
            "{call} must not drop the module to naga's emitter: {}",
            output.source
        );
        assert_valid_wgsl(&output.source);
    }
}

/// A declaration that survives renaming may take a predeclared alias
/// spelling (`fn vec4f()` is legal WGSL and entry-point names are always
/// preserved).  Emitting the short form then names the user's symbol, so the
/// long form must stand.
#[test]
fn a_shadowed_predeclared_alias_falls_back_to_the_long_form() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: vec4<f32>;\n\
        @compute @workgroup_size(1) fn vec4f() { out = vec4<f32>(1.0, 2.0, 3.0, 4.0); }";
    let output = crate::run(src, &Config::default()).expect("must minify");
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(
        !gen_report.rolled_back,
        "an entry point named `vec4f` must not cost the custom generator: {}",
        output.source
    );
    assert!(
        output.source.contains("vec4<f32>"),
        "the shadowed short alias must not spell the type: {}",
        output.source
    );
    assert_valid_wgsl(&output.source);
}

/// The same collision through a preamble used to be a hard error: the
/// preamble path has no fallback emitter, so the self-check failure escaped
/// as `emit error` (exit 2) on valid input.
#[test]
fn a_preamble_shadowing_an_alias_still_minifies() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: vec4<f32>;\n\
        @compute @workgroup_size(1) fn main() {\n\
          var s: vec4f;\n  s.i = 3;\n  out = vec4<f32>(f32(s.i), 1.0, 1.0, 1.0);\n}";
    let config = Config {
        preamble: Some("struct vec4f { i : i32, }".to_string()),
        ..Default::default()
    };
    let output = crate::run(src, &config).expect("a shadowed alias must not hard-error");
    assert!(
        output.source.contains("vec4<f32>"),
        "the preamble's `vec4f` must not spell the builtin type: {}",
        output.source
    );
}

/// Lossy float rounding is applied when the literal is PRINTED, so a rule
/// judged on the IR's values can be violated only in the output: rounding
/// `smoothstep(0.01, 0.02, x)` to one decimal collapses both edges to `0`,
/// which tint rejects and naga - the emit self-check - does not.
#[test]
fn rounded_literals_are_judged_by_the_printed_value() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: f32;\n\
        @compute @workgroup_size(1) fn main() { out = smoothstep(0.01, 0.02, out); }";
    let config = Config {
        float_precision: crate::config::FloatPrecision::all(
            crate::config::PrecisionMode::DecimalPlaces(1),
        ),
        ..Default::default()
    };
    let output = crate::run(src, &config).expect("must minify");
    assert!(
        !output.source.contains("smoothstep(0,0,"),
        "both edges rounded together; one must be bound to keep the call runtime: {}",
        output.source
    );
    assert_valid_wgsl(&output.source);
}

/// Inlining can make a divisor or shift amount a const-expression the input's
/// was not: `5u % (d + 1u - 1u)` called as `f(0u)`.  naga's validator passes
/// the manufactured shape (`arena_static_error_slot`); the driver's slot check
/// in `io::validate_module` reports it, so the run rolls the inlining back
/// instead of shipping lexically compacted with every optimization lost.
#[test]
fn inlined_argument_never_manufactures_a_static_error_slot() {
    // (source, a token proving the IR pipeline ran)
    let cases = [
        (
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             fn f(d: u32) -> u32 { return 5u / (d + 1u - 1u); }\n\
             @compute @workgroup_size(1) fn main() { o[0] = f(0u); }",
            "5/(B+1-1)",
        ),
        (
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             fn f(s: u32) -> u32 { return 5u << (s + 1u); }\n\
             @compute @workgroup_size(1) fn main() { o[0] = f(63u); }",
            "5u<<(B+1)",
        ),
        // The failable operator on the CALLER's side, fed by the inlined
        // return value: a decline inside the callee's clone cannot see it.
        (
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             fn f(x: u32) -> u32 { return x - 1u; }\n\
             @compute @workgroup_size(1) fn main() { o[0] = 5u / f(1u); }",
            "5/a(1)",
        ),
        // The zero arrives through a module `const`, not a literal argument.
        (
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             const Z: u32 = 1u;\n\
             fn f(d: u32) -> u32 { return 5u / (d - Z); }\n\
             @compute @workgroup_size(1) fn main() { o[0] = f(1u); }",
            "5/(B-1)",
        ),
    ];
    for (src, marker) in cases {
        let result = crate::run(src, &Config::default()).expect("must minify");
        assert!(
            result.report.bailout.is_none(),
            "must not degrade to the lexical bailout: {:?}\n{}",
            result.report.bailout,
            result.source
        );
        assert!(
            result.source.contains(marker),
            "expected the un-inlined, renamed form containing {marker}: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// `-x`, `x * y` and `x / y` must keep the const-ness the input gave their
/// operands.  `const_fold` used to narrow `select(0f, v, false)` to `0f`,
/// turning `-select(...)` const: the input stored `-0.0` and the output
/// `+0.0`, found by the GPU differential fuzzer.  `load_dedup` reaches the
/// same shape by forwarding the variable's initializer.
#[test]
fn a_sign_sensitive_float_slot_keeps_the_input_const_ness() {
    let cases = [
        // const_fold's narrowing: the other operand is a storage read, so
        // nothing can be forwarded and only the `select` fold is at fault.
        "let rt = bitcast<f32>(o[1]);\n  let res = -(select(0.0f, rt, false));",
        // load_dedup's forwarding: `v`'s initializer completes the slot.
        "var v = 1.0f;\n  v = -(select(0.0f, v, false));\n  let res = v;",
        "var v = 1.0f;\n  v = select(-0.25f, v, false) * select(0.0f, v, false);\n  let res = v;",
    ];
    for body in cases {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n  {body}\n  o[0] = bitcast<u32>(res);\n}}"
        );
        let result = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            result.source.contains("select("),
            "the select must survive so the slot stays runtime: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// `select(x, x, c) -> x` drops a RUNTIME condition, so the slot turns const
/// with no literal folded anywhere - the crossing the value guard on the
/// literal walk cannot see.  `-select(z, z, c)` shipped as `-0.`, device-read
/// as `+0.0` where the input gave `-0.0`.  `clone_over` carries the guard.
#[test]
fn narrowing_a_select_keeps_a_sign_sensitive_slot_runtime() {
    let cases = [
        // `accept == reject` needs ONE handle, which a `let` supplies; two
        // spellings of `0.0` are two handles and never reach the arm.
        "let z = 0.0;\n  let res = -(select(z, z, c));",
        // A `ZeroValue` source, so the guard cannot key on the literal.
        "let zv = vec2f();\n  let res = (-(select(zv, zv, c))).y;",
        // The `*` slot, not just the negation.
        "let z = 0.0;\n  let res = select(z, z, c) * bitcast<f32>(o[3]);",
    ];
    for body in cases {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n  let c = o[2] > 0u;\n  {body}\n  \
             o[0] = bitcast<u32>(res);\n}}"
        );
        let result = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            result.source.contains("select("),
            "the select must survive so the slot stays runtime: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// The SIBLING half: a sign-sensitive operator re-signs a zero, so the operand
/// that completes the crossing need not be the zero itself.  In
/// `(v - 1.) * 0.` the zero is const from the start and `v - 1.` crosses when
/// the never-written local reads as zero, shipping the const-expression
/// `-1. * 0.`; device-read `+0.0` against the input's `-0.0`.  Pinning only a
/// folded ZERO left this open.
#[test]
fn a_sibling_crossing_keeps_a_sign_sensitive_slot_runtime() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
               @compute @workgroup_size(1) fn main() {\n  var v: f32;\n  \
               o[0] = bitcast<u32>((v - 1.0) * 0.0);\n}";
    let result = crate::run(src, &Config::default()).expect("must minify");
    // The local must survive as a runtime read; every all-literal spelling of
    // the product is the miscompile.
    assert!(
        result.source.contains("var "),
        "the zero-init local must survive: {}",
        result.source
    );
    assert!(
        !result.source.contains("bitcast<u32>(-1"),
        "`v - 1.` must not fold into the product: {}",
        result.source
    );
    assert_valid_wgsl(&result.source);
}

/// The three passes that substitute a value for a runtime BINDING - register
/// promotion, load forwarding, inlining - each cross a sign-sensitive slot the
/// same way, and each guard used to key on the value being a literal `-0.0`
/// rather than on the crossing.  So `t = -1.` forwarded into `t * 0.` shipped
/// the const `-1. * 0.`, device-read `+0.0` against the input's `-0.0`; a
/// parameter read is runtime by construction, which is why `g(-1.)` does it
/// too.  Rooting the test at the OPERATOR is what sees the zero sibling.
#[test]
fn substituting_a_binding_keeps_a_sign_sensitive_slot_runtime() {
    // (body, the text that proves the binding survived)
    let cases = [
        ("var t: f32;\n  t = -1.0;\n  let res = t * 0.0;", "var "),
        ("var t: f32;\n  t = 0.0;\n  let res = -t;", "var "),
        ("var v = -1.0;\n  let res = v * 0.0;", "var "),
        ("var t: f32;\n  t = -1.0;\n  let res = 0.0 / t;", "var "),
        // A `ZeroValue` is a zero the operator re-signs, and it is not a
        // `Literal`, so a zero test that only reads literals misses it.
        (
            "var t: f32;\n  t = -1.0;\n  let res = (vec2f(t, 1.0) * vec2f()).x;",
            "var ",
        ),
    ];
    for (body, keeps) in cases {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             @compute @workgroup_size(1) fn main() {{\n  {body}\n  \
             o[0] = bitcast<u32>(res);\n}}"
        );
        let result = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            result.source.contains(keeps),
            "the binding must survive so the slot stays runtime: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// The inlining half: a callee's parameter read is runtime, so substituting a
/// const argument is itself the crossing.  `g(-1.)` into `a * 0.` needs the
/// operator-rooted test - the zero is in the CALLEE and the crossing operand
/// is the argument.  The no-zero and runtime-argument calls must still inline,
/// or the guard is just a veto on floats.
#[test]
fn inlining_a_const_argument_keeps_a_sign_sensitive_slot_runtime() {
    // (callee, call, must the call survive?)
    let cases = [
        ("fn c(a: f32) -> f32 { return -a; }", "c(0.0)", true),
        ("fn c(a: f32) -> f32 { return a * 0.0; }", "c(-1.0)", true),
        (
            "fn c(a: f32, b: f32) -> f32 { return a * b; }",
            "c(-1.0, 0.0)",
            true,
        ),
        ("fn c(a: f32) -> f32 { return a / -1.0; }", "c(0.0)", true),
        // No zero anywhere: inlining must still happen.
        ("fn c(a: f32) -> f32 { return a * 2.0; }", "c(3.0)", false),
        // A runtime argument leaves the slot runtime either way.
        (
            "fn c(a: f32) -> f32 { return -a; }",
            "c(bitcast<f32>(o[1]))",
            false,
        ),
        // A float `ZeroValue` counts as the zero...
        (
            "fn c(a: vec2f) -> vec2f { return a * vec2f(); }",
            "c(vec2f(-1.0, 1.0)).x",
            true,
        ),
        // ...but an INTEGER one has no signed zero, and the untyped roots
        // stop being inert if it is counted: this must still inline.
        (
            "fn c(a: vec4i) -> vec4i { return -a; }",
            "c(vec4i()).x",
            false,
        ),
    ];
    for (callee, call, keeps_call) in cases {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             {callee}\n\
             @compute @workgroup_size(1) fn main() {{\n  \
             o[0] = bitcast<u32>({call});\n}}"
        );
        let result = crate::run(&src, &Config::default()).expect("must minify");
        // Only `fn main` is left once the callee is inlined: nothing else
        // keeps a single-use function alive.
        let inlined = result.source.matches("fn ").count() == 1;
        assert_eq!(
            inlined, !keeps_call,
            "call {call} into {callee}: inlined={inlined}: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// A `Binary`'s operands pin each other's type, but a LITERAL pair pins
/// nothing: two f32 whose bare spellings are integer-shaped re-parse as
/// AbstractInt, so `-2. * 0.` shipped as `-2*0` - the integer 0, in a
/// `bitcast` the wrong type family, not just the wrong zero.  Reached
/// wherever a wholly const float product survives const_fold's `-0.0`
/// decline, which the input's own elided `let` still allows.
#[test]
fn a_float_literal_pair_keeps_its_type() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
               @compute @workgroup_size(1) fn main() {\n  let a = -2.0;\n  \
               let b = 0.0;\n  o[0] = bitcast<u32>(a * b);\n}";
    let result = crate::run(src, &Config::default()).expect("must minify");
    assert!(
        result.source.contains("-2f*0"),
        "one operand must carry its type so the pair stays f32: {}",
        result.source
    );
    assert_valid_wgsl(&result.source);
}

/// naga's `ensure_block_returns` appends its dead `return;` inside the LAST
/// statement's nested blocks, so a diverging loop in an `if` / `else` /
/// `switch` arm carries one where nothing at the function tail does.  The
/// validator then rejects the module on its own INPUT (`InvalidReturnType`)
/// and it bails out to lexical compaction.
#[test]
fn a_diverging_loop_inside_a_branch_still_minifies() {
    let cases = [
        "fn h(a: u32, i: u32) -> u32 {\n  var v = a;\n\
         if ((v ^ i) > 9u) {\n    loop { v = v + 7u; if (v > 25u) { return v; } }\n\
         } else {\n    return v * 3u;\n  }\n}",
        "fn h(a: u32, i: u32) -> u32 {\n  var v = a;\n\
         switch (v & 1u) {\n\
         case 0u: { loop { v = v + 7u; if (v > 25u) { return v; } } }\n\
         default: { return v + i; }\n  }\n}",
        // A `Block` tail, the third arm of naga's descent.
        "fn h(a: u32, i: u32) -> u32 {\n  var v = a;\n\
         { loop { v = v + 7u; if (v > 25u) { return v + i; } } }\n}",
    ];
    for helper in cases {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
             {helper}\n\
             @compute @workgroup_size(1) fn main() {{ o[0] = h(o[1], 2u); }}"
        );
        let result = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            result.report.bailout.is_none(),
            "the input must validate, not bail out: {:?}",
            result.report.bailout
        );
        assert!(
            !result.source.contains("var v = a"),
            "the IR pipeline must have run, not just lexical compaction: {}",
            result.source
        );
        assert_valid_wgsl(&result.source);
    }
}

/// `-0` re-parses as the ABSTRACT INTEGER zero, which has no sign bit, so a
/// float zero under a `Negate` must keep its float marker.  The literal's own
/// `-0.0` always did; a `+0.0` whose minus comes from an enclosing `Negate`
/// reaches the emitter only now that `const_fold` declines to fold `-(0.0)`
/// into a literal, and rendered as `-0` it turned `-0.0 * x` into `+0.0 * x`.
#[test]
fn a_negated_float_zero_keeps_its_float_marker() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
        @group(0) @binding(1) var<storage, read> inp: array<u32>;\n\
        @compute @workgroup_size(1) fn main() {\n\
          let z = select(0.0, 0.0, true) * 7.216;\n\
          out[0] = bitcast<u32>(-(z) * bitcast<f32>(inp[1]));\n\
        }";
    let out = minify(src);
    assert!(
        out.contains("-0.*") || out.contains("-0f*"),
        "the negated zero must stay float-typed, not the abstract int `-0`: {out}"
    );
    assert_valid_wgsl(&out);
}

/// The text a short-circuit `op` evaluates conditionally: from the operator
/// to the brace that opens the consuming `if`.
fn short_circuit_rhs_text<'a>(out: &'a str, op: &str) -> &'a str {
    let start = out.find(op).unwrap_or_else(|| panic!("no `{op}` in {out}")) + op.len();
    let rest = &out[start..];
    &rest[..rest.find('{').expect("the condition opens a block")]
}

/// A single-use IMPURE call consumed by the right operand of `&&` / `||`
/// must not be sunk there: the input ran it unconditionally, and the
/// short-circuit skips its write on every lane where the left side decides
/// (GPU-verified: the input reads back 1 through `out[0]`, the sunk form 0).
/// The hole needs a memory-free left operand; a storage read there was
/// already declined by the reorder test.
#[test]
fn an_impure_call_is_not_sunk_into_a_short_circuit_operand() {
    for (cond, op) in [("g.x == 7u && t", "&&"), ("g.x == 7u || t", "||")] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             var<private> c: u32;\n\
             fn f() -> bool {{ c = 1u; return true; }}\n\
             @compute @workgroup_size(1)\n\
             fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
               let t = f();\n  if ({cond}) {{ out[1] = 1u; }}\n  out[0] = c;\n}}"
        );
        let out = minify_preserving(&src, &["f"]);
        assert!(
            !short_circuit_rhs_text(&out, op).contains('('),
            "the call must stay bound before the `if`: {out}"
        );
    }
}

/// The pure half of the same hole: a callee carrying a derivative or an
/// implicit-LOD sample relocated under a non-uniform `&&` operand is a
/// uniformity error Dawn rejects, so the carrier never bubbles through a
/// short-circuit wrapper.  Baseline profile, so the IR inliner does not
/// dissolve the call first; the arm's store keeps the guard out of the
/// return merge.
#[test]
fn a_pure_call_with_a_derivative_is_not_sunk_into_a_short_circuit_operand() {
    let src = "@group(0) @binding(0) var t: texture_2d<f32>;\n\
               @group(0) @binding(1) var s: sampler;\n\
               var<private> p: f32;\n\
               fn f(uv: vec2f) -> f32 { return textureSample(t, s, uv).x; }\n\
               @fragment fn main(@location(0) uv: vec2f, @location(1) k: f32) -> @location(0) vec4f {\n\
                 let v = f(uv);\n  if (k > 0.5 && v > 0.5) { p = 1.0; return vec4f(1.0); }\n  return vec4f(0.0);\n}";
    let config = Config {
        profile: crate::config::Profile::Baseline,
        ..Config::default()
    };
    let out = crate::run(src, &config).expect("run failed").source;
    assert_valid_wgsl(&out);
    assert!(
        !short_circuit_rhs_text(&out, "&&").contains('('),
        "the sampling call must stay bound before the `if`: {out}"
    );
}

/// `discard` is not a terminator: execution continues as a helper
/// invocation, so a store pending at it is still read - here by the quad
/// neighbours through `dpdx`.  The store was deleted and the derivative
/// const-folded to `dpdx(0f)`, a wrong colour on a real render target.
#[test]
fn a_store_pending_at_a_discard_survives() {
    let src = "@fragment\n\
               fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {\n\
                 var x: f32 = 0.0;\n\
                 if (p.y > 100.0) { x = p.x * 3.0; discard; }\n\
                 let d = dpdx(x);\n  return vec4<f32>(d, 0.0, 0.0, 1.0);\n}";
    let out = minify(src);
    assert!(out.contains("*3;discard;"), "the store must survive: {out}");
    assert!(
        !out.contains("dpdx(0"),
        "the derivative must read the local: {out}"
    );
}

/// Float `%` reads differently as a const-expression (`0x1p25 % 3` is 2
/// const-evaluated, 0 on the GPU), so every pass that can make the slot const
/// must decline for ANY operand - forwarding, the zero-init fold, inlining;
/// the `*` twin has no rule without a zero and must still fold.
#[test]
fn a_float_modulo_keeps_the_input_const_ness() {
    let forwarded = |body: &str| {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size(1) fn main() {{\n  {body}\n}}"
        );
        minify(&src)
    };
    let out = forwarded("var a = 33554432.0;\n  var b = 3.0;\n  o[0] = a % b;");
    assert!(
        out.contains("var ") && !out.contains("0x1p25%3"),
        "load_dedup must not forward both operands: {out}"
    );
    let out = forwarded("var a: f32;\n  o[0] = 33554432.0 % (a + 3.0);");
    assert!(
        out.contains("var ") && !out.contains("0x1p25%3"),
        "const_fold must not fold the zero-init local into the divisor: {out}"
    );
    let out = forwarded("var a = 33554432.0;\n  var b = 3.0;\n  o[0] = a * b;");
    assert!(
        !out.contains("var "),
        "`*` without a zero still folds: {out}"
    );

    let src = "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
               fn g(a: f32) -> f32 { return a % 3.0; }\n\
               @compute @workgroup_size(1) fn main() { o[0] = g(33554432.0); }";
    let out = minify(src);
    assert!(
        out.contains("fn ") && !out.contains("0x1p25%"),
        "inlining must not substitute the const argument: {out}"
    );
}

/// A `workgroupUniformLoad` preload carries a barrier and must run exactly
/// once per iteration; inlined into a `for` guard's `&&` right operand it
/// runs on the lanes the left side lets through.  The DAG path count said
/// "one" and admitted it.
#[test]
fn a_preload_under_a_short_circuit_operand_keeps_the_plain_loop() {
    let src = "var<workgroup> flag: u32;\n\
               @group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
               @compute @workgroup_size(4) fn main(@builtin(local_invocation_index) li: u32) {\n\
                 var i: u32 = 0u;\n\
                 loop {\n    let r = workgroupUniformLoad(&flag);\n\
                   if (!(i < 4u && r == 0u)) { break; }\n    out[li] = i;\n\
                   continuing { i = i + 1u; }\n  }\n}";
    let out = minify(src);
    assert!(
        out.contains("loop{") && !out.contains("for("),
        "the for-conversion must be declined: {out}"
    );
}

/// A `continuing` clause referencing a result the BODY binds (an atomic here)
/// cannot move into a `for` header, which renders before the body: it
/// printed naga's `_e<n>` placeholder, and the counter's `var` suppression
/// had to stay in lockstep with the decline or the loop lost its declaration.
#[test]
fn a_continuing_clause_reading_a_body_bound_result_keeps_the_plain_loop() {
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<atomic<u32>>;\n\
               @compute @workgroup_size(1) fn main() {\n\
                 var i: u32 = 0u;\n\
                 loop {\n    if (i >= 10u) { break; }\n\
                   let old = atomicAdd(&out[0], 1u);\n\
                   continuing { i = i + old; }\n  }\n}";
    let out = minify(src);
    assert!(
        out.contains("var ") && out.contains("loop{") && !out.contains("_e"),
        "{out}"
    );
}

/// A helper with its own `@diagnostic(...)` filter is checked under that
/// filter; cloned into a caller governed by the module default, its
/// non-uniform sample is an error Dawn rejects, so it is never a template.
#[test]
fn a_helper_with_a_diagnostic_filter_is_not_inlined() {
    let src = "@group(0) @binding(0) var t: texture_2d<f32>;\n\
               @group(0) @binding(1) var s: sampler;\n\
               @diagnostic(off, derivative_uniformity)\n\
               fn f(uv: vec2f) -> f32 { return textureSample(t, s, uv).x; }\n\
               @fragment fn main(@location(0) uv: vec2f, @location(1) k: f32) -> @location(0) vec4f {\n\
                 if (k > 0.5) { return vec4f(f(uv)); }\n  return vec4f(0.0);\n}";
    let out = minify(src);
    assert!(
        out.contains("@diagnostic(off,derivative_uniformity)fn "),
        "the filtered helper must survive with its attribute: {out}"
    );
}

/// A single-use call stashed for inline emission is priced as a short NAME by
/// the `let` threshold, so a multi-use `-a` over it needed ten uses to bind
/// and was re-rendered per use instead - `-f()+-f()`, the callee's write
/// repeated (a pure callee's texture sample multiplied the same way).  The
/// wrapper must bind at two uses like any sub-expression.
#[test]
fn a_cheap_wrapper_over_a_stashed_call_binds_before_reuse() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n\
               var<private> p: u32;\n\
               fn f() -> f32 { p = p + 1u; return 1.0; }\n\
               @compute @workgroup_size(1) fn main() {\n\
                 let a = f();\n  let c = -a;\n  out[0] = c + c;\n  out[1] = f32(p);\n}";
    for profile in [
        crate::config::Profile::Baseline,
        crate::config::Profile::Max,
    ] {
        let config = Config {
            profile,
            preserve_symbols: vec!["f".to_string()],
            ..Config::default()
        };
        let out = crate::run(src, &config).expect("run failed").source;
        assert_valid_wgsl(&out);
        // `fn f()`, `fn main()`, and exactly one call.
        assert_eq!(
            out.matches("()").count(),
            3,
            "the call must render once: {out}"
        );
    }
}

/// A const float `%` pair is exact `fmod` (2.0) where the GPU computes the
/// stepwise form (0.0 on Metal); the input's slot was runtime, so one
/// operand stays `let`-bound.
#[test]
fn const_float_modulo_pair_stays_runtime() {
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\
        @compute @workgroup_size(1) fn main(){ let a = 33554432.0f; out[0] = bitcast<u32>(a % 3.0f); }";
    let out = minify(src);
    assert!(
        !out.contains("0x1p25%3") && out.contains("=0x1p25f;") && out.contains("%3);"),
        "the const pair must not become an abstract const-expression: {out}"
    );
}

/// An override-expression array size is naga's anonymous override; it is
/// spelled back in place, never declared under a minted name (a
/// pipeline-constant key the source never had, and once the positional
/// `O<n>` colliding with a user `override O2`).
#[test]
fn an_override_expression_array_size_is_spelled_in_place() {
    let src = "override O2 : i32 = 3;\
        override N : i32;\
        var<workgroup> w : array<i32, N * 2>;\
        @group(0) @binding(0) var<storage, read_write> out : array<i32>;\
        @compute @workgroup_size(1) fn main() {\
          w[0] = O2; out[0] = workgroupUniformLoad(&w[0]) + i32(arrayLength(&out)); }";
    let out = minify(src);
    assert!(
        out.contains("override O2:i32=3;override N:i32;")
            && out.matches("override ").count() == 2
            && out.contains("array<i32,(N*2)>"),
        "user overrides keep their names, the anonymous one is its expression: {out}"
    );
    let module = crate::io::parse_wgsl(&out).expect("re-parses");
    let anonymous = module
        .overrides
        .iter()
        .filter(|(_, o)| o.name.is_none())
        .count();
    assert_eq!(
        anonymous, 1,
        "the size re-parses to one anonymous override: {out}"
    );
}

/// The spelling is in the name slot before the alias plan reads it: a
/// source `alias` gives an override-sized array one type handle, four
/// declarations make its alias pay, and the alias body spells the size.
/// Spelled at emission instead, the plan read an empty slot and declared
/// `alias c=array<u32,>;` - a runtime-sized array, which WGSL's trailing
/// comma admits - so the text failed the self-check, and naga's writer,
/// which cannot spell an anonymous override, could not stand in: the run
/// erred where the release before it shipped.
#[test]
fn an_aliased_override_sized_array_spells_its_size_in_the_alias() {
    let src = "override N: u32 = 4u;        alias T = array<u32, N * 2>;        var<workgroup> w1: T; var<workgroup> w2: T; var<workgroup> w3: T; var<workgroup> w4: T;        @group(0) @binding(0) var<storage, read_write> out: array<u32>;        fn f(p: ptr<workgroup, T>) -> u32 { return (*p)[0] + (*p)[1]; }        @compute @workgroup_size(1) fn main() {          w1[0] = 1u; w2[1] = 2u; w3[2] = 3u; w4[3] = 4u;          out[0] = f(&w1) + f(&w2) + f(&w3) + f(&w4); }";
    let out = minify(src);
    assert!(
        out.starts_with("alias ")
            && out.contains("=array<u32,(N*2)>;")
            && !out.contains("array<u32,>")
            && out.matches("override ").count() == 1,
        "the alias spells the anonymous override's expression: {out}"
    );
    let module = crate::io::parse_wgsl(&out).expect("re-parses");
    assert!(
        module.global_variables.iter().all(|(_, g)| {
            g.space != naga::AddressSpace::WorkGroup
                || matches!(
                    module.types[g.ty].inner,
                    naga::TypeInner::Array {
                        size: naga::ArraySize::Pending(_),
                        ..
                    }
                )
        }),
        "every workgroup array keeps its override size: {out}"
    );
}

/// naga's `@early_depth_test(...)` (an extension tint does not read) is a
/// fragment entry point's contract with the depth test: dropped, a forced
/// early test or a conservative-depth promise vanishes on wgpu.
#[test]
fn a_fragment_entry_point_keeps_its_early_depth_test() {
    use naga::{ConservativeDepth as D, EarlyDepthTest as T};
    for (attr, expected) in [
        ("force", T::Force),
        (
            "greater_equal",
            T::Allow {
                conservative: D::GreaterEqual,
            },
        ),
        (
            "less_equal",
            T::Allow {
                conservative: D::LessEqual,
            },
        ),
        (
            "unchanged",
            T::Allow {
                conservative: D::Unchanged,
            },
        ),
    ] {
        let src = format!(
            "@fragment @early_depth_test({attr}) fn main(@builtin(position) p: vec4f) \
             -> @builtin(frag_depth) f32 {{ return p.z - 0.1; }}"
        );
        let out = minify(&src);
        assert!(out.contains(&format!("@early_depth_test({attr})")), "{out}");
        let module = crate::io::parse_wgsl(&out).expect("re-parses");
        assert_eq!(
            module.entry_points[0].early_depth_test,
            Some(expected),
            "{out}"
        );
    }
}

/// A matrix local passed by address keeps `: T`: an inferred `var b = a * b`
/// is the unnamed twin of the alias the pointer parameter is spelled with,
/// and naga compares those by handle.
#[test]
fn matrix_local_passed_by_pointer_keeps_its_declared_type() {
    let src = "struct S { m: mat3x3<f32>, n: mat3x3<f32> }\
        @group(0) @binding(0) var<uniform> u: S;\
        fn f(p: ptr<function, mat3x3<f32>>) -> f32 { return (*p)[0][0]; }\
        @compute @workgroup_size(1) fn main() {\
          var a: mat3x3<f32> = u.m;  var b: mat3x3<f32> = u.n;\
          var p: mat3x3<f32> = a * b;  var q: mat3x3<f32> = b * a;  var r: mat3x3<f32> = a + b;\
          _ = f(&p) + f(&q) + f(&r); }";
    // Preserved: three clones of `(*p)[0][0]` would price cheaper than the
    // definition and dissolve the pointer calls this fixture is about.
    let config = Config {
        preserve_symbols: vec!["f".to_string()],
        ..Config::default()
    };
    let out = on_big_stack(move || crate::run(src, &config).expect("run failed"));
    assert!(
        out.report.fallback.is_none() && !out.source.contains("_e"),
        "the generator's text must survive the re-parse: {}",
        out.source
    );
    // Name-agnostic: every `var` initialised from a product keeps `: T`.
    let typed_var_count = out
        .source
        .split("var ")
        .skip(1)
        .filter(|rest| {
            rest.split('=')
                .next()
                .is_some_and(|decl| decl.contains(':'))
        })
        .count();
    assert!(
        typed_var_count >= 3,
        "the pointer-argument locals are declared with their type: {}",
        out.source
    );
}

/// The copy `var b = a` and a `for`-header `var` are declaration sites too;
/// without an alias both sides are the one unnamed type and no `: T` is
/// spent.
#[test]
fn pointer_argument_locals_spell_the_alias_at_every_declaration_site() {
    let head = "@group(0) @binding(0) var<storage, read_write> out: array<mat4x4<f32>>;\
        @group(0) @binding(1) var<storage, read> inm: array<mat4x4<f32>>;\
        fn f(p: ptr<function, mat4x4<f32>>) { (*p)[0][0] += 1.0; }\
        fn g1(a: mat4x4<f32>, b: mat4x4<f32>) -> mat4x4<f32> { return a * b; }\
        fn g2(a: mat4x4<f32>, b: mat4x4<f32>) -> mat4x4<f32> { return a + b; }\
        fn g3(a: mat4x4<f32>, b: mat4x4<f32>) -> mat4x4<f32> { return a - b; }";
    let copy = format!(
        "{head} @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {{\
           var a = inm[id.x] * inm[id.y]; var b = a; a[1][1] = 7.0; f(&b);\
           out[0] = g1(a, b); out[1] = g2(a, b); out[2] = g3(a, b); out[4] = b; out[5] = a; }}"
    );
    let header = format!(
        "{head} @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {{\
           var acc = inm[0];\
           for (var m = inm[id.x] * inm[id.y]; m[0][0] < 10.0; m = m * inm[1]) {{\
             f(&m); acc = g2(acc, g3(m, m)); }}\
           out[0] = acc; }}"
    );
    // No inlining (preserved helpers, zero node budget) keeps every
    // `mat4x4<f32>` spelling, so an alias is minted.
    let config = Config {
        max_inline_node_count: Some(0),
        preserve_symbols: ["f", "g1", "g2", "g3"].map(String::from).to_vec(),
        ..Config::default()
    };
    for (src, site) in [(copy, "var "), (header, "for(var ")] {
        let config = config.clone();
        let out = on_big_stack(move || crate::run(&src, &config).expect("run failed"));
        assert!(
            out.report.fallback.is_none() && !out.source.contains("_e"),
            "the generator's text must survive the re-parse: {}",
            out.source
        );
        let alias = out
            .source
            .strip_prefix("alias ")
            .and_then(|s| s.split('=').next())
            .expect("an alias is minted");
        assert!(
            out.source.contains(site) && out.source.contains(&format!(":{alias}=")),
            "the `{site}` declaration spells the alias: {}",
            out.source
        );
    }
    // Inlined, nothing is aliased: the copy stays untyped.
    let plain = "@group(0) @binding(0) var<storage, read_write> out: array<mat4x4<f32>>;\
        fn f(p: ptr<function, mat4x4<f32>>) { (*p)[0][0] += 1.0; }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          var a = mat4x4<f32>(vec4f(1.0), vec4f(2.0), vec4f(3.0), vec4f(f32(id.x)));\
          var b = a; f(&b); out[0] = a * b; out[1] = b; }";
    let out = on_big_stack(move || crate::run(plain, &Config::default()).expect("run failed"));
    assert!(
        out.report.fallback.is_none()
            && !out.source.contains("alias ")
            && !out.source.contains(":mat4x4f="),
        "no alias, no declared type: {}",
        out.source
    );
}

/// A `let` whose only textual use is a swizzle the emitter merges
/// (`vec4(a.x,a.y,0,1)` -> `vec4(a.xy,0,1)`) must not be bound: the
/// slot-per-reference census needs `discount_compose_folds`, or a second pass
/// reading the merged spelling drops the `let` (non-idempotent).
#[test]
fn swizzle_merged_base_is_not_let_bound() {
    let src = "@group(0) @binding(0) var<uniform> u : vec4f;\
        @fragment fn fs() -> @location(0) vec4f {\
        let a = u * 2.0; return vec4f(a.x, a.y, 0.0, 1.0);}";
    let out = minify(src);
    assert!(
        out.contains("vec4((A*2).xy,0,1)"),
        "the swizzle-merged base was still let-bound: {out}"
    );
    assert!(
        !out.contains("let "),
        "a single-use let survived the swizzle-merge discount: {out}"
    );
    assert_eq!(
        minify(&out),
        out,
        "swizzle-merge pricing must be idempotent"
    );
}

/// A splat collapse (`vec4f(d,d,d,d)` -> `vec4f(d)`) is four slots in the
/// arena and one rendered operand, so `d` is single-use in the text and must
/// not be let-bound.
#[test]
fn splat_collapsed_value_is_not_let_bound() {
    let src = "@fragment fn fs(@builtin(position) p : vec4f) -> @location(0) vec4f {\
        let d = pow(p.x, 1.7); return vec4f(d, d, d, d);}";
    let out = minify(src);
    assert!(
        out.contains("return vec4(pow("),
        "the splat-collapsed value was still let-bound: {out}"
    );
    assert_eq!(minify(&out), out, "splat pricing must be idempotent");
}

/// The swizzle discount declines a base whose cone holds a call result
/// (`cone_has_call_or_image`): an over-discount there is a second execution
/// of the call's effects, not a byte loss.
#[test]
fn swizzle_base_over_a_call_stays_bound() {
    let src = "@group(0) @binding(0) var<storage, read_write> o : array<vec4f>;\
        var<private> n : i32;\
        fn tick(x : f32) -> vec4f { n = n + 1; o[n] = vec4f(x);\
        return vec4f(x, x * 2.0, x * 3.0, x * 4.0); }\
        @fragment fn fs(@builtin(position) p : vec4f) -> @location(0) vec4f {\
        let a = tick(p.x) * 2.0; return vec4f(a.x, a.y, 0.0, 1.0);}";
    let out = minify_preserving(src, &["tick"]);
    assert_eq!(
        out.matches("tick(").count(),
        2,
        "the call must be spelled once at its definition and once at its site: {out}"
    );
    assert!(
        out.contains("=tick("),
        "a base whose cone holds a call must stay let-bound: {out}"
    );
}

/// The splat discount skips a column the matrix arm may flatten
/// (`discount_compose_folds`): flattened, a single-call result priced at one
/// use is spelled - and run - twice.
#[test]
fn splat_column_of_a_flattened_matrix_keeps_its_call_bound() {
    let src = "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\
        var<private> n : u32;\
        fn tick() -> f32 { n = n + 1u; return f32(n); }\
        @compute @workgroup_size(1) fn main() {\
        let v = tick();\
        let m = mat2x2f(vec2f(v, v), vec2f(1.0, 2.0));\
        o[0] = m[0][0] + m[1][1];}";
    let out = minify_preserving(src, &["tick"]);
    assert_eq!(
        out.matches("tick(").count(),
        2,
        "the call must be spelled once at its definition and once at its site: {out}"
    );
    assert!(
        out.contains("=tick("),
        "a splat column the matrix arm may flatten must keep its call bound: {out}"
    );
}

/// A splat over an image cone keeps its slots (`cone_has_call_or_image`):
/// discounted to one use, the fetch `loop_sunk_work` priced as bound would
/// render at its use inside the loop, and the shorter text is why the
/// never-grow guard cannot catch it.
#[test]
fn splat_over_an_image_op_stays_bound_outside_a_loop() {
    let src = "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\
        @group(0) @binding(2) var t : texture_2d<f32>;\
        @compute @workgroup_size(1) fn main() {\
        let s = textureLoad(t, vec2i(0), 0).x;\
        let v = vec4f(s, s, s, s);\
        var acc = 0.0;\
        for (var i = 0; i < 4; i++) { acc = acc + dot(v, vec4f(f32(i))); }\
        o[0] = acc;}";
    let out = minify(src);
    let (before_loop, in_loop) = out.split_once("for(").expect("a loop survives");
    assert!(
        before_loop.contains("textureLoad("),
        "the fetch must stay bound outside the loop: {out}"
    );
    assert!(
        !in_loop.contains("textureLoad("),
        "the fetch must not sink into the loop body: {out}"
    );
}

// MARK: Splice-gate and emitter regressions

/// The IN-direction splice gates read the caller's arena through the
/// earlier splices' replacements: `h(f() + 0.0)` with `f` spliced first
/// hands `h` a const `0.0 + 0.0`, so `-p` / `p * -1.0` would fold to `-0.`
/// (Dawn on Metal flushes the literal to `+0.0` where the input's runtime
/// negation kept the sign).  Judged on the `CallResult` the statement still
/// spelled, the crossing went unseen.
#[test]
fn a_nested_call_argument_is_judged_on_the_inner_splices_value() {
    for (body, spelled) in [
        ("fn h(p: f32) -> f32 { return -p; }", "-B"),
        ("fn h(p: f32) -> f32 { return p * -1.0; }", "B*-1"),
    ] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\
             fn f() -> f32 {{ return 0.0; }}\
             {body}\
             @compute @workgroup_size(1) fn main() {{ o[0] = bitcast<u32>(h(f() + 0.0)); }}"
        );
        let out = minify(&src);
        assert!(
            !out.contains("-0.") && !out.contains("0f*-1"),
            "the negation must stay a runtime operation: {out}"
        );
        assert!(
            out.contains(spelled),
            "the sign-sensitive operator survives: {out}"
        );
    }
}

/// The static-error twin: `h(f() - 1u)` with `f` returning `1u` manufactures
/// `x / (1u - 1u)`, which rolled the whole inlining pass back every sweep
/// when the argument was read unresolved.
#[test]
fn a_nested_call_argument_that_zeroes_a_divisor_keeps_the_outer_call() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\
        fn f() -> u32 { return 1u; }\
        fn h(d: u32) -> u32 { return o[1] / d; }\
        @compute @workgroup_size(1) fn main() { o[0] = h(f() - 1u); }";
    let out = minify(src);
    assert!(
        out.contains("fn ") && !out.contains("/(1-1)") && !out.contains("/0"),
        "the outer call must survive with a runtime divisor: {out}"
    );
}

/// A negative zero the cone COMPUTES crosses a sign-sensitive operator as
/// surely as a zero leaf: `-f32(p)` with `p = 0u`, `-(p - 1.0)` with
/// `p = 1.0`.  The gate evaluates the substituted cone and declines on a
/// `-0.0` lane; the forwarder (`var p = 1f; -(p - 1f)`) does the same.
#[test]
fn a_computed_zero_crossing_a_sign_sensitive_operator_stays_runtime() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\
        fn f(p: u32) -> f32 { if (o[2] > 0u) { o[1] = 1u; } return -f32(p); }\
        fn g(p: f32) -> f32 { if (o[3] > 0u) { o[1] = 1u; } return -(p - 1.0); }\
        @compute @workgroup_size(1) fn main() {\
          o[0] = bitcast<u32>(f(0u));\
          o[4] = bitcast<u32>(g(1.0));\
          var q = 1.0;\
          o[5] = bitcast<u32>(-(q - 1.0));\
          var r = 3.0;\
          o[6] = bitcast<u32>(-(r - 1.0));\
        }";
    let out = minify(src);
    assert!(
        !out.contains("-0."),
        "no negative-zero literal may appear: {out}"
    );
    assert!(
        out.contains("-f32(") && out.contains("-(") && out.contains("var "),
        "the negations stay runtime, the forwarded var survives: {out}"
    );
    assert!(
        out.contains("3221225472"),
        "a non-zero result still folds (`-(3-1)` is `-2`): {out}"
    );
}

/// An f32 at or past 2^24 under an integer conversion keeps its suffix
/// whatever the token's shape: bare `i32(3e9)` converts as an ABSTRACT
/// float to 2147483647 where the typed `i32(3e9f)` the input computed
/// saturates to 2147483520 (GPU-verified).  A `Compose` operand pins its
/// constructor type under the same rule, and so does an integer lane the
/// elided constructor would leave abstract (`vec4u(vec4(37,-7,..))` errors).
#[test]
fn an_out_of_range_conversion_operand_keeps_its_type() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        fn conv(x: f32) -> i32 { return i32(x) + 1; }\
        fn convs(x: f32) -> vec2i { return vec2i(vec2f(x)) + vec2i(1); }\
        @compute @workgroup_size(1) fn main() {\
          var f = 3e9; out[0] = bitcast<u32>(i32(f));\
          var g = 5e9; out[1] = u32(g);\
          out[2] = bitcast<u32>(conv(3e9));\
          out[3] = bitcast<u32>(convs(3e9).x);\
          out[4] = k(1.0);\
        }\
        fn k(p: f32) -> u32 { var v0: i32 = i32(p) - 7; v0 = v0 * v0 + 1;\
          return vec4u(vec4i(v0, -7, -7, -7)).y; }";
    let out = minify(src);
    assert!(
        out.contains("i32(3e9f)") && out.contains("u32(5e9f)"),
        "scalar operands keep the f32 suffix: {out}"
    );
    assert!(
        out.contains("vec2i(vec2f(3e9))"),
        "a splat operand spells its constructor type: {out}"
    );
    assert!(
        out.contains("vec4u(vec4i(37,-7,-7,-7))") || out.contains("4294967289"),
        "an all-literal operand with a negative i32 lane spells its type: {out}"
    );
}

/// A single-use call whose result sits in a POINTER chain used more than
/// once (`let p = &a[f()]; *p = 1; x = *p;`) must stay `let`-bound: the
/// emitter never binds a pointer-typed expression and spells the chain at
/// every use, so the stashed call ran once per use (GPU-verified: the
/// counter read 5 for 2 calls).
#[test]
fn a_call_inside_a_shared_pointer_chain_stays_bound() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        fn f() -> u32 { out[8] = out[8] + 1u; return out[8] % 8u; }\
        @compute @workgroup_size(1) fn main() {\
          let p = &out[f()]; *p = 100u; out[16] = *p;\
          let q = &out[f()]; *q = 300u; *q = 500u; out[17] = *q;\
        }";
    let out = minify_preserving(src, &["f"]);
    assert_eq!(
        out.matches("=f()").count(),
        2,
        "each call is spelled once, bound at its own statement: {out}"
    );
}

/// An override-sized array's element count is a pipeline-time value: a
/// const index past 0 made by a splice (`wr(8u)` into `wg[i]`) or naga's
/// let-erasure (`let x = 8u; wg[x]`) fails Dawn's pipeline creation for a
/// host value at or below it, where the input's runtime index was clamped.
/// The writers decline it and the generator binds what the parse left.
#[test]
fn a_const_index_into_an_override_sized_array_stays_runtime() {
    let src = "override N: u32 = 4u;\
        var<workgroup> wg: array<u32, N>;\
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        fn wr(i: u32) { wg[i] = 1u; }\
        @compute @workgroup_size(1) fn main() {\
          wr(8u);\
          let x = 8u; wg[x] = 2u;\
          var i = 0u; i = 5u; wg[i] = 3u;\
          wg[0] = 4u;\
          out[0] = wg[0] + wg[1];\
        }";
    let out = minify(src);
    assert!(!out.contains("[8]") && !out.contains("[5]"), "{out}");
    assert!(
        out.contains("[0]=4"),
        "index 0 is in bounds for every count: {out}"
    );
}

/// A run of equal lanes prints as a sub-vector splat (`vec4(vec3f(2),x)`),
/// a type no expression of the function resolves to; the alias planner
/// must count that spelling or a local of the function takes the alias's
/// letter and the constructor reads as a call of the local.
#[test]
fn a_subsplat_spelling_counts_as_a_use_of_its_alias() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        @group(0) @binding(1) var<storage, read_write> p: array<vec3f>;\
        @compute @workgroup_size(1) fn other(@builtin(global_invocation_id) id: vec3u) {\
          var q: vec3f = p[id.x] * 2.0;\
          var r: vec3f = q + vec3f(1.0, 2.0, 3.0);\
          p[id.y] = q + r + vec3f(4.0, 5.0, 6.0) + vec3f(7.0, 8.0, 9.0) + vec3f(10.0, 11.0, 12.0)\
            + vec3f(13.0, 14.0, 15.0);\
        }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          o[id.y] = vec4f(2.0, 2.0, 2.0, f32(id.x));\
        }";
    let config = Config::default();
    let out = crate::run(src, &config).expect("run failed");
    assert!(
        out.report.fallback.is_none(),
        "the generator's text must pass the self-check: {}",
        out.source
    );
}

/// The const-hazard binder's tree walk stops at depth 16; an operand deeper
/// than that must read as opaque (bind), not as runtime (pass): the
/// single-call splice manufactures divisors of any depth, and one whose
/// value tint computes as zero shipped as a shader-creation error.
#[test]
fn a_deep_manufactured_divisor_is_bound() {
    let terms = "+1u".repeat(40);
    let src = format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
         fn f(v: u32) -> u32 {{ return out[1] / (unpack4xU8(v).x{terms}-40u); }}\
         @compute @workgroup_size(1) fn main() {{ out[0] = f(256u); }}"
    );
    let out = minify(&src);
    assert!(out.contains("let "), "the divisor must be let-bound: {out}");
}

/// naga's front-end folds `dot`, `length`, `any` / `all`, a composite
/// `const` member and a vector conversion that nagami's evaluator does not,
/// so the `let` the generator would bind for such a divisor is evaluated
/// away at the self-check's re-parse; the gate declines instead of admitting.
#[test]
fn a_divisor_only_naga_can_fold_keeps_its_call() {
    for (helper, call) in [
        (
            "fn f() -> vec2u { return vec2u(4u, 4u); }",
            "out[1] << dot(f(), f())",
        ),
        (
            "const V = vec4u(0u, 1u, 2u, 3u); fn f() -> vec4u { return V; }",
            "out[1] / f().x",
        ),
        (
            "fn f(v: vec2f) -> u32 { return u32(length(v)); }",
            "out[1] / f(vec2f(0.0))",
        ),
        (
            "fn f(v: vec2f) -> u32 { return u32(vec2u(v).x); }",
            "out[1] / f(vec2f(0.0))",
        ),
    ] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
             {helper}\
             @compute @workgroup_size(1) fn main() {{ out[0] = {call}; }}"
        );
        let config = Config::default();
        let result = crate::run(&src, &config).expect("run failed");
        assert!(
            result.report.bailout.is_none(),
            "the module must not ship compacted: {}",
            result.source
        );
    }
}

/// A multi-site clone lands at every site or at none: the pre-check judges
/// each site on the caller before any splice, the gate at the site reads
/// the caller through its earlier splices (`dot(f(), f())` turns const at
/// the second site and naga folds `dot`), and a definition kept beside its
/// clones is pure loss, so the candidate is withdrawn and the run redone.
#[test]
fn a_multi_site_clone_declined_at_one_site_lands_at_none() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        fn f() -> vec2u { return vec2u(4u, 4u); }\
        @compute @workgroup_size(1) fn main() { out[0] = out[1] << dot(f(), f()); }";
    let out = minify(src);
    assert_eq!(
        out.matches("vec2u(4)").count(),
        1,
        "the body is spelled once, in the definition: {out}"
    );
    assert!(out.contains("dot(a(),a())"), "both calls stay: {out}");
}

/// An image op bound before a loop and read only by the loop's `break if`
/// runs in the `continuing` block, once per iteration: it is rendered at
/// the loop's depth, so it stays bound outside.
#[test]
fn an_image_op_read_by_break_if_stays_bound_outside_the_loop() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        @group(0) @binding(2) var t: texture_2d<f32>;\
        @compute @workgroup_size(1) fn main() {\
          let lim = textureLoad(t, vec2i(1, 2), 0);\
          var i = 0u;\
          loop { o[i] = vec4f(1.0); i++; continuing { break if f32(i) > lim.x; } }\
        }";
    let out = minify(src);
    let (before_loop, in_loop) = out.split_once("loop{").expect("a loop survives");
    assert!(before_loop.contains("textureLoad("), "{out}");
    assert!(!in_loop.contains("textureLoad("), "{out}");
}
