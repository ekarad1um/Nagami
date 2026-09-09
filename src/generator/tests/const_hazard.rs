//! Const-expression hazards (`const_hazard`) and literal positions nothing
//! pins: every input is valid, its naive emission is tint-rejected while
//! naga accepts it, so assertions are on the emitted text.

use super::helpers::*;

fn minify(src: &str) -> String {
    let out = crate::run(src, &Config::default()).expect("run failed");
    assert!(
        out.report.bailout.is_none(),
        "unexpected bailout: {}",
        out.source
    );
    assert_valid_wgsl(&out.source);
    out.source
}

const HEAD: &str = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\
    @group(0)@binding(1) var<storage,read> inp: array<u32>;\
    @compute @workgroup_size(1) fn main() {";

fn body(stmts: &str) -> String {
    format!("{HEAD}{stmts}}}")
}

/// Binding names depend on the module's other declarations, so tests key on
/// the initializer.
fn bound_name(out: &str, value: &str) -> String {
    let needle = format!("={value};");
    let end = out
        .find(&needle)
        .unwrap_or_else(|| panic!("no `let ..={value};` in {out}"));
    let start = out[..end].rfind("let ").expect("binding is a let") + 4;
    out[start..end].split(':').next().unwrap().to_string()
}

#[test]
fn bitcast_of_inf_bits_binds_the_literal() {
    // The canonical way to build inf/NaN in WGSL; `bitcast<f32>(2139095040u)`
    // is a const-expression tint rejects ("value inf cannot be represented").
    let out = minify(&body(
        "let bits = 0x7f800000u; out[0] = bitcast<u32>(bitcast<f32>(bits) * f32(inp[1]));",
    ));
    let n = bound_name(&out, "2139095040u");
    assert!(
        out.contains(&format!("bitcast<f32>({n})")),
        "inf bits must be let-bound before the bitcast: {out}"
    );
}

#[test]
fn vector_bitcast_checks_every_lane() {
    let out = minify(&body(
        "let bits = vec2u(0x7f800000u, 1u); \
         out[0] = bitcast<u32>(bitcast<vec2f>(bits).x * f32(inp[1]));",
    ));
    let n = bound_name(&out, "vec2u(2139095040,1)");
    assert!(
        out.contains(&format!("let {n}:vec2u=")) && out.contains(&format!("bitcast<vec2f>({n})")),
        "a constructor operand binds with an explicit type: {out}"
    );
}

#[test]
fn binding_is_scoped_to_its_block() {
    // One literal handle serves both arms and the tail; an arm's name must
    // not reach the tail.
    let out = minify(&body(
        "let a = 0x7f800000u; \
         if (inp[1] > 3u) { out[0] = bitcast<u32>(bitcast<f32>(a)); } \
         else { out[1] = bitcast<u32>(bitcast<f32>(a) * 2.0); } \
         out[2] = a;",
    ));
    assert!(
        out.ends_with("A[2]=2139095040;}"),
        "tail use after the arms must inline the literal again: {out}"
    );
}

#[test]
fn float_overflow_and_domain_errors_bind() {
    for (stmt, value, consumer) in [
        (
            "let big = 1e38; out[0] = bitcast<u32>(big * 10.0 + f32(inp[1]));",
            "1e38f",
            "bitcast<u32>({n}*10+",
        ),
        (
            "let s = -1.0; out[0] = bitcast<u32>(sqrt(s)) + inp[1];",
            "-1f",
            "sqrt({n})",
        ),
        (
            "let l = 0.0; out[0] = bitcast<u32>(log(l)) + inp[1];",
            "0f",
            "log({n})",
        ),
        ("let z = 0.0; out[0] = u32(1.0 / z);", "1f", "u32({n}/0)"),
        (
            "let q = 1e10; out[0] = bitcast<u32>(quantizeToF16(q)) + inp[1];",
            "1e10f",
            "quantizeToF16({n})",
        ),
        (
            "let b = 0x7c007c00u; out[0] = bitcast<u32>(unpack2x16float(b).x) + inp[1];",
            "2080406528u",
            "unpack2x16float({n})",
        ),
    ] {
        let out = minify(&body(stmt));
        let n = bound_name(&out, value);
        let needle = consumer.replace("{n}", &n);
        assert!(
            out.contains(&needle),
            "{stmt}\n  expected {needle}\n  got {out}"
        );
    }
}

#[test]
fn benign_all_constant_expressions_stay_inline() {
    // A vector comparison and a scalar-by-vector product `const_fold` leaves
    // alone cannot fail at shader creation and must not pay for a `let`.
    let out = minify(&body(
        "let v = vec2f(3.0); out[0] = u32(select(0.0, 1.0, all(v > vec2f(5.0))) + f32(inp[1]));",
    ));
    assert!(
        !out.contains("let "),
        "no binding for a benign comparison: {out}"
    );
}

#[test]
fn builtin_argument_rules_bind_the_checked_operand() {
    for (stmt, value, consumer) in [
        (
            "let o = 40u; out[0] = extractBits(inp[1], o, 1u);",
            "40u",
            ",{n},1)",
        ),
        (
            "let o = 30u; out[0] = extractBits(inp[1], o, 10u);",
            "30u",
            ",{n},10)",
        ),
        (
            "let o = 30u; out[0] = insertBits(inp[1], 1u, o, 10u);",
            "30u",
            ",1u,{n},10)",
        ),
        (
            "let lo = 5u; out[0] = clamp(inp[1], lo, 1u);",
            "5u",
            ",{n},1)",
        ),
        (
            "let e0 = 1.0; out[0] = u32(smoothstep(e0, 1.0, f32(inp[1])));",
            "1f",
            "smoothstep({n},1,",
        ),
        (
            "let ex = 200i; out[0] = bitcast<u32>(ldexp(f32(inp[1]), ex));",
            "200i",
            ",{n}))",
        ),
        // All-constant call: the checked operand (not the first) binds.
        (
            "let ex = 200i; out[0] = bitcast<u32>(ldexp(1.0, ex)) + inp[1];",
            "200i",
            "ldexp(1,{n})",
        ),
    ] {
        let out = minify(&body(stmt));
        let n = bound_name(&out, value);
        let needle = consumer.replace("{n}", &n);
        assert!(
            out.contains(&needle),
            "{stmt}\n  expected {needle}\n  got {out}"
        );
    }
}

#[test]
fn f16_conversion_out_of_range_binds() {
    let out = minify(
        "enable f16; @group(0)@binding(0) var<storage,read_write> out: array<u32>;\
         @compute @workgroup_size(1) fn main() { let w = 65536.0; \
         out[0] = bitcast<u32>(f32(f16(w) + f16(out[1]))); }",
    );
    let n = bound_name(&out, "65536f");
    assert!(
        out.contains(&format!("f16({n})")),
        "65536 exceeds f16 range: the operand must bind: {out}"
    );
}

#[test]
fn f16_conversion_of_an_opaque_integer_binds() {
    // `unpack4xI8` is beyond both evaluators, so the integer converted is
    // unknown; only f16 can overflow on an integer, so the f32 conversion
    // of the same tree stays inline.
    let out = minify(
        "enable f16; @group(0)@binding(0) var<storage,read_write> out: array<f16>;\
         @compute @workgroup_size(1) fn main() { var v = unpack4xI8(0x7f7f7f7fu).x; \
         out[0] = f16(v); }",
    );
    assert!(
        out.contains("let ") && !out.contains("f16(unpack4xI8"),
        "an opaque integer must bind before its conversion to f16: {out}"
    );
    let out = minify(&body(
        "var v = unpack4xI8(0x7f7f7f7fu).x; out[0] = u32(f32(v));",
    ));
    assert!(
        !out.contains("let ") && out.contains("f32(unpack4xI8("),
        "f32 spans every integer type, so nothing binds: {out}"
    );
}

#[test]
fn conversion_of_out_of_range_literal_keeps_its_suffix() {
    // `i32(3000000000)` converts from the ABSTRACT value and is rejected;
    // `i32(3000000000u)` wraps like the runtime conversion did.
    let out = minify(&body("let w = 3000000000u; out[0] = u32(i32(w)) + inp[1];"));
    assert!(out.contains("i32(3000000000u)"), "{out}");
    let out = minify(&body("let w = -1i; out[0] = u32(w) + inp[1];"));
    assert!(out.contains("u32(-1i)"), "{out}");
}

#[test]
fn shift_left_operand_literal_keeps_its_suffix() {
    // A shift types as its left operand alone; bare `4294967295>>x` is an
    // abstract-int shift tint concretizes to i32 and rejects.
    let out = minify(&body("let a = 4294967295u; out[0] = a >> (inp[1] & 31u);"));
    assert!(out.contains("4294967295u>>"), "{out}");
    let out = minify(&body("let a = 100u; out[0] = a >> (inp[1] & 31u);"));
    assert!(out.contains("100u>>"), "{out}");
}

#[test]
fn extract_bits_value_literal_keeps_its_suffix() {
    // `extractBits(e, ...)` types as `e`; a bare `1` would be i32.
    let out = minify(&body(
        "let e = 5u; out[0] = extractBits(e, inp[1] & 7u, 3u);",
    ));
    assert!(out.contains("extractBits(5u,"), "{out}");
    let out = minify(&body(
        "let e = 5u; out[0] = insertBits(inp[2], e, inp[1] & 7u, 3u);",
    ));
    assert!(out.contains(",5u,"), "{out}");
}

#[test]
fn hazard_bound_output_is_idempotent() {
    let first = minify(&body(
        "let bits = 0x7fc00000u; let f = bitcast<f32>(bits); out[0] = u32(f != f);",
    ));
    let second = minify(&first);
    assert_eq!(first, second);
}

#[test]
fn for_header_hazard_binds_before_the_loop() {
    // Header expressions render inline, so the binding must precede the
    // `for`.
    let out = minify(&body(
        "let bits = 0x7f800000u;          for (var i = 0u; bitcast<f32>(bits) > f32(i) && i < 4u; i++) { out[i] = i + inp[1]; }",
    ));
    let n = bound_name(&out, "2139095040u");
    let bind = out.find(&format!("let {n}=")).unwrap();
    let head = out.find("for(").unwrap();
    assert!(
        bind < head && out.contains(&format!("bitcast<f32>({n})>")),
        "{out}"
    );
}

#[test]
fn for_shaped_loop_body_releases_its_bindings() {
    // A for-shaped `loop` has no `Block` around its body, so the for-loop
    // emitter's own statement path must release the binding.
    let out = minify(&body(
        "let a = 0x7f800000u; var i = 0u; \
         loop { if (i >= 2u) { break; } \
         out[i] = bitcast<u32>(bitcast<f32>(a) * f32(i)); \
         continuing { i++; } } \
         out[3] = bitcast<u32>(bitcast<f32>(a));",
    ));
    assert_eq!(
        out.matches("=2139095040u;").count(),
        2,
        "one binding inside the loop and a fresh one for the tail: {out}"
    );
}

#[test]
fn ldexp_limit_follows_the_float_width() {
    let out = minify(
        "enable f16; @group(0)@binding(0) var<storage,read_write> out: array<u32>; \
         @compute @workgroup_size(1) fn main() { let e = 20i; \
         out[0] = u32(ldexp(f16(out[1]), e)) + out[1]; }",
    );
    let n = bound_name(&out, "20i");
    assert!(
        out.contains(&format!(",{n}))")),
        "f16 ldexp allows 16 at most: {out}"
    );
    let out = minify(&body("let e = 20i; out[0] = u32(ldexp(f32(inp[1]), e));"));
    assert!(
        out.contains("ldexp(f32(a[1]),20)"),
        "f32 ldexp allows 128: {out}"
    );
}

/// An inlined argument can make both shift operands constant while the
/// amount hides behind a builtin the hazard evaluator does not model; the
/// signed overflow is rejected at const-evaluation, so the operand binds.
/// The literal arrives by inlining: `load_dedup` declines every forward
/// inside a shift-AMOUNT slot, so the `var v = -100i;` spelling of this
/// never reaches the hazard.
#[test]
fn unmodeled_constant_shift_amount_binds_the_shifted_literal() {
    let out: String = compact_with_passes(
        "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\
         fn f(v: i32) -> u32 { return u32(v << (reverseBits(u32(v)) & 31u)); }\
         @compute @workgroup_size(1) fn main() { out[0] = f(-100i); }",
        Profile::Max,
    )
    .split_whitespace()
    .collect();
    assert!(out.contains("let"), "{out}");
    assert!(!out.contains("-100i<<"), "{out}");
}

/// Const-ness entering a failable slot must be declined for the WHOLE slot:
/// the top declines (its value IS the error) and an interior node then folds
/// anyway, leaving `5u / (1u - 1u)`, which no emitter can spell, so the
/// module ships lexically compacted.  `const_fold` reaches this through a
/// never-written local, `load_dedup` through a forwarded initialiser.
#[test]
fn interior_fold_under_a_divisor_keeps_the_slot_runtime() {
    for stmts in [
        "var s: u32; out[0] = 5u / (s + 1u - 1u);",
        "var s: u32; out[0] = 5u % (s + 1u - 1u);",
        "var s: u32; out[0] = 5u >> (s + 40u - 4u);",
        "var s: u32; out[0] = 5u / ((s + 2u) * 3u - 6u);",
        "var v: vec2u; out[0] = (vec2u(5u) / (v + vec2u(1u) - vec2u(1u))).x;",
    ] {
        // `minify` rejects a bailout, which is the regression; the local
        // surviving says the slot stayed runtime rather than the statement
        // dying.
        let out = minify(&body(stmts));
        assert!(out.contains("var "), "{out}");
    }
}

#[test]
fn interior_forward_under_a_divisor_keeps_the_slot_runtime() {
    // The forwarded literal's own value proves nothing: `0u` errors one
    // slot away through `+ 1u - 1u`, `1u` errors through `- 1u`.
    for stmts in [
        "var s: u32 = 0u; out[0] = 5u / (s + 1u - 1u);",
        "var s: u32 = 1u; out[0] = 5u / (s - 1u);",
        "var s: u32 = 40u; out[0] = 5u >> (s - 4u);",
    ] {
        minify(&body(stmts));
    }
}

/// The forward walk must read the slot AS IT WILL BE after the rewrite.
/// Stopping at the root hid every forward whose target is COMPOUND: `d`
/// inlines to `a - 1u` and `a` to `1u` in the SAME pass, shipping
/// `inp[0] / (1u - 1u)`.  A compound target is also unjudgeable in itself -
/// `u32(length(vec2f()))` is const the moment it is inlined, with no literal
/// anywhere and the evaluator that would fold it living in naga - so the
/// forward AT the slot is what has to go.
#[test]
fn forward_of_a_compound_into_a_divisor_keeps_the_slot_runtime() {
    for stmts in [
        "var a: u32 = 1u; var d: u32; d = a - 1u; out[0] = inp[0] / d; out[1] = d;",
        "var a: u32 = 1u; var d: u32; d = a - 1u; out[0] = inp[0] % d; out[1] = d;",
        "var a: u32 = 40u; var d: u32; d = a - 4u; out[0] = inp[0] >> d; out[1] = d;",
        "var a: u32 = 40u; var d: u32; d = a - 4u; out[0] = inp[0] << d; out[1] = d;",
        "var d: u32; d = u32(length(vec2f(0.0, 0.0))); out[0] = inp[0] / d; out[1] = d;",
        // A lane of a forwarded `Compose` is a value position one hop past
        // where the walk used to stop.
        "var a: u32 = 1u; var d: vec2u; d = vec2u(a - 1u, 3u); \
         out[0] = (vec2u(inp[0]) / d).x; out[1] = d.y;",
    ] {
        let out = minify(&body(stmts));
        assert!(out.contains("var "), "{out}");
    }
}

/// The half the fix must not trade away: in the value position the slot IS
/// the forwarded literal, so a safe one survives.
#[test]
fn a_safe_literal_still_forwards_into_a_divisor() {
    let out = minify(&body("var k: u32 = 5u; out[0] = inp[0] / k;"));
    assert!(out.contains("/5"), "{out}");
    assert!(!out.contains("var "), "{out}");
}

/// A runtime slot is never walked: `inp[i]` is a global load, so no forward
/// under it can raise a creation error and the index must still fold.
#[test]
fn a_runtime_divisor_keeps_the_forwards_under_it() {
    let out = minify(&body("var i: u32 = 1u; out[0] = inp[0] / (inp[i] + 1u);"));
    assert!(out.contains("inp[1]") || out.contains("[1]"), "{out}");
    assert!(!out.contains("var "), "{out}");
}

/// Dawn's MSL for `~u32(i32(x))` is `(~(uint(int(v))) & 3u)`, which Metal
/// parses as a C-style cast of `&3u`; the inner conversion binds so the
/// unary operand is `uint(a)`.

#[test]
fn unary_over_nested_vector_constructors_binds_the_inner_one() {
    // Dawn's Metal backend reads `-(float4(int4(v)))` as a function type, so
    // a splat under a conversion needs the same binding as `~u32(i32(v))`.
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
               @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {\n\
                 let v = vec4i(i32(i));\n\
                 out[0] = u32((-vec4f(v) * 2.0).x);\n\
               }";
    let bound: String = compact_with_passes(src, Profile::Max)
        .split_whitespace()
        .collect();
    assert!(!bound.contains("vec4f(vec4i("), "{bound}");
}

#[test]
fn unary_over_three_conversions_binds_below_the_outer_one() {
    // Binding only the innermost would leave `uint(int(name))`, ambiguous
    // again; the operand one level down leaves a single constructor.
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
               @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {\n\
                 let a = u32(i);\n\
                 out[0] = ~u32(i32(u32(a))) & 3u;\n\
               }";
    let bound: String = compact_with_passes(src, Profile::Max)
        .split_whitespace()
        .collect();
    assert!(!bound.contains("u32(i32("), "{bound}");
}

#[test]
fn a_for_header_declines_rather_than_render_the_metal_cast_shape() {
    // The header cannot `let`-bind, and hoisting the binding above the loop
    // would freeze a value the body updates, so the plain `loop` form wins.
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
               @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {\n\
                 var b = i32(i);\n\
                 var n = 0u;\n\
                 loop {\n\
                   let t = i32(b);\n\
                   if ((~u32(t) & 7u) <= n) { break; }\n\
                   n += 1u; b += 1;\n\
                 }\n\
                 out[0] = n;\n\
               }";
    let out: String = compact_with_passes(src, Profile::Max)
        .split_whitespace()
        .collect();
    assert!(out.contains("loop{") && !out.contains("~u32(i32("), "{out}");
}

#[test]
fn unary_over_nested_conversion_of_an_identifier_binds_the_inner_cast() {
    // An indexed leaf (`int(out[1u])`) is no declarator, so only the bare
    // identifier form binds.
    let src = |arg: &str| {
        format!(
            "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
             @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {{\n\
               out[0] = ~u32(i32({arg})) & 3u;\n\
             }}"
        )
    };
    let bound: String = compact_with_passes(&src("i"), Profile::Max)
        .split_whitespace()
        .collect();
    assert!(
        bound.contains("let") && !bound.contains("~u32(i32("),
        "{bound}"
    );
    let inline: String = compact_with_passes(&src("out[1]"), Profile::Max)
        .split_whitespace()
        .collect();
    assert!(inline.contains("~u32(i32("), "{inline}");
}

/// Integer `/` `%` `<<` `>>` are keyed on the RIGHT operand alone: a const
/// divisor of zero or a shift amount at the width or beyond is rejected
/// whatever the left operand is.  naga cannot evaluate `bitcast`, so the
/// `let` survives lowering and the inlined text is exactly the rejected one;
/// the right operand is what binds.
#[test]
fn a_const_divisor_under_a_runtime_dividend_binds_the_divisor() {
    for (stmt, value, op) in [
        (
            "let z = bitcast<u32>(0.0); out[0] = inp[1] / z;",
            "bitcast<u32>(0f)",
            "/",
        ),
        (
            "let z = bitcast<u32>(0.0); out[0] = inp[1] % z;",
            "bitcast<u32>(0f)",
            "%",
        ),
        (
            "let s = bitcast<u32>(1e-40); out[1] = inp[2] << s;",
            "bitcast<u32>(1e-40f)",
            "<<",
        ),
    ] {
        let out = minify(&body(stmt));
        let n = bound_name(&out, value);
        assert!(
            out.contains(&format!("{op}{n};")),
            "{stmt}\n  the right operand must bind: {out}"
        );
    }
}

/// An argument-rule operand tint evaluates but nagami cannot (`unpack4xU8`)
/// binds where the rule would read it: `clamp` with an opaque `low` bound
/// against a const `high` is rejected as `low > high`.
#[test]
fn an_opaque_argument_rule_operand_binds() {
    let out = minify(&body(
        "let lo = unpack4xU8(84215045u).x; out[0] = clamp(inp[1], lo, 1u);",
    ));
    let n = bound_name(&out, "unpack4xU8(84215045).x");
    assert!(out.contains(&format!(",{n},1)")), "{out}");
}

/// `break if` is the last statement INSIDE `continuing`, so the hazard `let`
/// its condition needs is still in scope when it renders; releasing the
/// block's bindings first re-inlined the rejected const-expression next to a
/// dead `let`.
#[test]
fn a_break_if_condition_uses_the_hazard_binding() {
    let out = minify(&body(
        "var i: u32 = 0u; let bits = 0x7f800000u; \
         loop { i = i + 1u; continuing { break if bitcast<f32>(bits) < f32(i); } } \
         out[0] = i;",
    ));
    let n = bound_name(&out, "2139095040u");
    assert!(
        out.contains(&format!("break if bitcast<f32>({n})")),
        "{out}"
    );
}
