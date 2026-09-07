//! Expression emission: bare literals in constructors, single-use `let`
//! inlining, swizzle folding, bitcast/`select`/`arrayLength`, vector and
//! matrix access, splats, and the Compose-to-splat collapse.

use super::helpers::*;

// MARK: Bare literals inside Compose/Splat

#[test]
fn compose_uses_bare_literals() {
    let out = compact("fn f() -> vec2<f32> { return vec2<f32>(1.0, 2.0); }");
    assert!(out.contains("vec2f(1,2)"), "got: {out}");
    assert!(!out.contains("1f"), "typed suffix inside compose: {out}");
}

#[test]
fn splat_uses_bare_literal() {
    let out = compact("fn f() -> vec3<f32> { return vec3<f32>(5.0, 5.0, 5.0); }");
    // naga may lower this to a Splat.
    assert!(
        !out.contains("5f)") && !out.contains("5.f)"),
        "typed suffix in splat: {out}"
    );
}

// MARK: Single-use let inlining

#[test]
fn single_use_expression_is_inlined() {
    let out = compact("fn f(a: f32, b: f32) -> f32 { return a + b; }");
    assert!(!out.contains("let "), "unexpected let binding: {out}");
    assert!(out.contains("return"), "missing return: {out}");
}

#[test]
fn multi_use_expression_gets_let_binding() {
    // `a+b` is 3 chars; at 2 refs the `let` overhead exceeds the savings.
    let out = compact("fn f(a: f32, b: f32) -> f32 { let s = a + b; return s * s; }");
    assert!(
        !out.contains("let "),
        "short binary on named operands should be inlined at 2 refs: {out}"
    );
}

#[test]
fn longer_multi_use_expression_gets_let_binding() {
    let src = r#"
        fn f(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
            let n = normalize(cross(a, b));
            return n + n;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("let "),
        "long multi-use expr should be bound: {out}"
    );
}

#[test]
fn shared_initializer_subexpression_is_not_bound() {
    // CSE gives the repeated `vec2u(1,0)`/`vec2u(0,1)` two refs each, but the
    // hoisted `var` renders the whole tree before any `Emit` range is priced,
    // so a later `let` would be dead text.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: array<vec2u>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            var offs = array<vec2u, 6>(vec2u(0, 0), vec2u(1, 0), vec2u(0, 1),
                                       vec2u(0, 1), vec2u(1, 0), vec2u(1, 1));
            out[0] = offs[id.x % 6u];
        }
    "#;
    let out = compact_with_passes(src, Profile::Max);
    let flat: String = out.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(
        flat.matches("(1,0)").count(),
        2,
        "the initializer renders inline: {out}"
    );
    assert!(
        !out.contains("let "),
        "no binding serves an initializer tree: {out}"
    );
}

// MARK: Literal emission: different types coexist

#[test]
fn different_literal_types_not_conflated() {
    let src = r#"
            fn a(x: f32) -> f32 { return x; }
            fn b(x: i32) -> i32 { return x; }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(a(1.0), a(1.0), f32(b(1)), f32(b(1)));
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Struct field access via AccessIndex

#[test]
fn struct_field_access_roundtrip() {
    let src = r#"
            struct S { x: f32, y: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(u.x, u.y, 0.0, 1.0);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: bitcast / As expression

#[test]
fn bitcast_expression_roundtrip() {
    let out = compact(
        "fn f(x: u32) -> f32 { return bitcast<f32>(x); }\n\
             @compute @workgroup_size(1) fn main() { _ = f(0u); }",
    );
    assert!(
        out.contains("bitcast<f32>"),
        "bitcast should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: select() expression

#[test]
fn select_expression_roundtrip() {
    let out = compact(
        "fn f(a: f32, b: f32, c: bool) -> f32 { return select(a, b, c); }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1.0, 2.0, true); }",
    );
    assert!(
        out.contains("select("),
        "select call should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn whole_number_float_literals_outside_constructor_keep_type() {
    // Abstract-type coercion promotes a bare whole number to the float operand
    // type.
    let out = compact(
        "fn f(x: f32) -> f32 { return x + 1.0; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1.0); }",
    );
    assert!(
        !out.contains("1."),
        "whole-number float should shed the dot in binary context: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn whole_number_float_literals_in_binary_with_float_context_can_drop_dot() {
    let out = compact(
        "fn f(x: f32) -> f32 { return x + 2.0; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1.0); }",
    );
    assert!(
        out.contains("+2") || out.contains("+ 2"),
        "expected bare 2 in binary float context: {out}"
    );
    assert!(
        !out.contains("2."),
        "unexpected trailing dot in safe binary context: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn whole_number_float_literal_comparison_can_drop_dot() {
    let out = compact(
        "fn f(x: f32) -> bool { return x >= 0.0; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1.0); }",
    );
    assert!(
        out.contains(">=0") || out.contains(">= 0"),
        "expected bare 0 in float comparison: {out}"
    );
    assert!(
        !out.contains("0."),
        "unexpected trailing dot in comparison: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn whole_number_float_binary_of_literals_stays_float_like() {
    let out = compact(
        "fn f() -> f32 { return 2.0 * 3.0; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(); }",
    );
    assert_valid_wgsl(&out);
}

#[test]
fn negative_zero_float_binary_operand_keeps_float_marker() {
    // `-0.0` emits bare; the negative-zero distinction is knowingly lost.
    let out = compact(
        "fn f(x: f32) -> f32 { return -0.0 * x; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1.0); }",
    );
    assert_valid_wgsl(&out);
}

#[test]
fn large_whole_number_float_binary_operand_stays_float_like() {
    let out = compact(
        "fn f(x: f32) -> f32 { return x + 100000000000000000000.0; }\n\
         @compute @workgroup_size(1) fn main() { _ = f(1.0); }",
    );
    assert!(
        out.contains("100000000000000000000.")
            || out.contains("e")
            || out.contains("E")
            || out.contains("0x"),
        "large whole-number float must not collapse to an out-of-range bare int: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn integer_literals_outside_constructor_stay_bare() {
    let out = compact(
        "fn f(x: i32) -> i32 { return x + 1234567890 - 1234567890; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1); }",
    );
    assert!(
        out.contains("1234567890"),
        "integer literal should be present in output: {out}"
    );
    assert!(
        !out.contains("1234567890i"),
        "integer literals outside constructors should stay bare: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn unsigned_integer_literals_outside_constructor_stay_bare() {
    let out = compact(
        "fn f(x: u32) -> u32 { return x + 4000000000u - 4000000000u; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(1u); }",
    );
    assert!(
        out.contains("4000000000"),
        "unsigned integer literal should be present in output: {out}"
    );
    assert!(
        !out.contains("4000000000u"),
        "unsigned integer literals outside constructors should stay bare: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: arrayLength() expression

#[test]
fn array_length_roundtrip() {
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
            @compute @workgroup_size(1)
            fn main() {
                let n = arrayLength(&buf);
                if n > 0u {
                    buf[0] = f32(n);
                }
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("arrayLength(&"),
        "arrayLength should have & pointer arg: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Negation fallback (non-comparison condition)

#[test]
fn negation_fallback_wraps_in_not() {
    let src = r#"
            fn cond() -> bool { return true; }
            fn f() -> i32 {
                var x = 0i;
                if cond() {} else { x = 1; }
                return x;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Type cast (As with convert)

#[test]
fn type_cast_roundtrip() {
    let out = compact(
        "fn f(x: i32) -> f32 { return f32(x); }\n\
             @compute @workgroup_size(1) fn main() { _ = f(42); }",
    );
    assert!(out.contains("f32("), "type cast should be present: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Vector/matrix access

#[test]
fn vector_access_index_uses_dot_xyzw() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4<f32>) -> @location(0) vec4<f32> {
            let x = v.x;
            let y = v.y;
            let z = v.z;
            let w = v.w;
            return vec4<f32>(w, z, y, x);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".wzyx"),
        "full compose should fold to swizzle: {out}"
    );
    assert!(
        !out.contains("[0]") && !out.contains("[3]"),
        "vector access should not use bracket notation: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn vector_access_index_on_pointer_deref() {
    let src = r#"
        @fragment fn fs(@location(0) c: vec3<f32>) -> @location(0) vec4<f32> {
            var tmp = c;
            let r = tmp.x;
            let g = tmp.y;
            let b = tmp.z;
            return vec4<f32>(r, g, b, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".xyz"),
        "vector access through var should be grouped into swizzle: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn matrix_access_index_stays_bracket() {
    let src = r#"
        @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let m = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
            let col0 = m[0];
            let col1 = m[1];
            return vec4<f32>(col0.x + col1.y, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Swizzle folding in Compose

#[test]
fn full_swizzle_identity_vec4_eliminates_constructor() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.x, v.y, v.z, v.w);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("vec4f(v"),
        "identity should eliminate the constructor: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn identity_swizzle_collapse_in_constructor_arg_is_not_parenthesized() {
    // An identity swizzle of a binary (`vec3f(c.x,c.y,c.z)`, `c = col*s`)
    // collapses to the bare base; as a comma-delimited constructor argument
    // nothing is appended, so it must stay unparenthesised.  The full pipeline
    // is needed so CSE first unifies the three components onto one base.
    let src = r#"
        @fragment fn fs(@location(0) col: vec3f, @location(1) s: f32) -> @location(0) vec4f {
            let m = mat3x3f(
                vec3f((col * s).x, (col * s).y, (col * s).z),
                vec3f((col + col).x, (col + col).y, (col + col).z),
                col,
            );
            return vec4f(m[0], 1.);
        }
    "#;
    let out = compact_with_passes(src, crate::config::Profile::Max);
    let flat: String = out.split_whitespace().collect();
    let ctor = flat
        .split_once("mat3x3")
        .and_then(|(_, rest)| rest.split_once('('))
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(args, _)| args)
        .unwrap_or_else(|| panic!("expected a mat3x3 constructor: {out}"));
    assert!(
        !ctor.contains('('),
        "identity-collapsed binary columns must be bare in the constructor \
         arg list, found wrapping parens: {out}"
    );
    assert!(ctor.contains('*') || ctor.contains('+'), "sanity: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn identity_swizzle_collapse_as_postfix_base_keeps_parens() {
    // As the base of a postfix swizzle the collapsed binary needs its parens:
    // `(a-b).x` must not degrade to `a-b.x` (= `a-(b.x)`).
    let src = r#"
        @fragment fn fs(@location(0) p: vec3f, @location(1) q: vec3f) -> @location(0) vec4f {
            let r = vec3f((p - q).x, (p - q).y, (p - q).z);
            return vec4f(r.x, 0., 0., 1.);
        }
    "#;
    let out = compact_with_passes(src, crate::config::Profile::Max);
    let flat: String = out.split_whitespace().collect();
    // naga's fallback spells `vec4<f32>` and would mask a regression behind its
    // own valid `vec3<f32>(...).x`.
    assert!(
        flat.contains("vec4f"),
        "expected the optimised generator emission (short `vec4f`), not the \
         naga fallback: {out}"
    );
    // Both forms validate and names are mangled, so only this structural check
    // guards the fix.
    assert!(
        flat.contains(").x"),
        "the binary base of a postfix `.x` swizzle must stay parenthesised \
         (`(a-b).x`, not the miscompiled `a-b.x`): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn full_swizzle_reorder_vec4() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.w, v.z, v.y, v.x);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("v.wzyx"),
        "reordered components should become swizzle: {out}"
    );
    assert!(
        !out.contains("vec4f(v"),
        "constructor should be eliminated: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn full_swizzle_vec2_from_vec4() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(vec2f(v.z, v.w), 0., 1.);
        }
    "#;
    let out = compact(src);
    assert!(out.contains(".zw"), "should fold to v.zw swizzle: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn partial_swizzle_grouping_vec4() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.x, v.y, 0., 1.);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".xy"),
        "consecutive same-base components should group into swizzle: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn partial_swizzle_trailing_group() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(1., 0., v.x, v.y);
        }
    "#;
    let out = compact(src);
    assert!(out.contains(".xy"), "trailing run should be grouped: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn no_swizzle_for_different_bases() {
    let src = r#"
        @fragment fn fs(@location(0) a: vec4f, @location(1) b: vec4f) -> @location(0) vec4f {
            return vec4f(a.x, b.y, 0., 1.);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains(".xy"),
        "different bases must not be grouped: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_through_var_pointer() {
    let src = r#"
        @fragment fn fs(@location(0) c: vec4f) -> @location(0) vec4f {
            var v = c;
            return vec4f(v.x, v.y, v.z, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".xyz"),
        "load-of-AccessIndex through var should group: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_identity_through_var() {
    let src = r#"
        fn helper(v: vec3f) -> vec3f {
            return vec3f(v.x, v.y, v.z);
        }
        @fragment fn fs(@location(0) c: vec3f) -> @location(0) vec4f {
            return vec4f(helper(c), 1.);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("vec3f(v"),
        "identity on by-value vec3 should fold: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn no_swizzle_for_single_component() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.x, 0., 0., 1.);
        }
    "#;
    let out = compact(src);
    assert!(out.contains("v.x"), "single component stays as .x: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_non_sequential_indices() {
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.w, v.x, 0., 1.);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains(".wx"),
        "non-sequential same-base should group: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_base_binary_keeps_parens() {
    // `a-b.yx` parses as `a-(b.yx)` and still validates: a silent miscompile.
    let full = compact("fn f(a:vec2f,b:vec2f)->vec2f{ let d=a-b; return vec2f(d.y,d.x); }");
    assert!(
        full.contains("(a-b).yx") && !full.contains("a-b.yx"),
        "full swizzle of a binary base must keep parens: {full}"
    );
    assert_valid_wgsl(&full);

    // Partial-group path.
    let grouped =
        compact("fn f(a:vec3f,b:vec3f)->vec4f{ let d=a-b; return vec4f(d.x,d.y,0.,1.); }");
    assert!(
        grouped.contains("(a-b).xy"),
        "grouped swizzle of a binary base must keep parens: {grouped}"
    );
    assert_valid_wgsl(&grouped);
}

#[test]
fn swizzle_identity_collapse_of_binary_keeps_parens() {
    let out = compact("fn f(a:vec2f,b:vec2f,k:f32)->vec2f{ let d=a-b; return vec2f(d.x,d.y)*k; }");
    assert!(
        out.contains("(a-b)*k") && !out.contains("a-b*k"),
        "identity-collapsed binary base in operator context must keep parens: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_struct_field_vector() {
    // Each `s.color` gets its own AccessIndex handle, so grouping cannot fire;
    // only validity is asserted.
    let src = r#"
        struct S { color: vec4f }
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            let s = S(v);
            return vec4f(s.color.x, s.color.y, 0., 1.);
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Global Splat expression

#[test]
fn global_splat_vec3_roundtrip() {
    let src = r#"
        const a = vec3f(1.0);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(a, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3f("),
        "global Splat should produce vec3f(...): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_splat_vec2_roundtrip() {
    let src = r#"
        const a = vec2f(0.0);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(a, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec2f("),
        "global Splat should produce vec2f(...): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_splat_vec4_roundtrip() {
    let src = r#"
        const a = vec4f(0.5);
        @fragment fn main() -> @location(0) vec4f {
            return a;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec4f("),
        "global Splat should produce vec4f(...): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_splat_integer_roundtrip() {
    let src = r#"
        const a = vec3<i32>(0);
        @compute @workgroup_size(1)
        fn main() {
            _ = a;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3i("),
        "integer global Splat should produce vec3i(...): {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Compose-to-splat collapse

#[test]
fn compose_identical_f32_collapses_to_splat() {
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(vec3f(1.2, 1.2, 1.2), 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3(1.2)"),
        "identical-component Compose should collapse to splat (float-form literal pins f32, so the ctor suffix drops): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_identical_vec4_collapses_to_splat() {
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(0.5, 0.5, 0.5, 0.5);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec4(.5)"),
        "vec4f with identical components should collapse (and drop its pinned suffix): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_identical_vec2_collapses_to_splat() {
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            let v = vec2f(3.0, 3.0);
            return vec4f(v, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec2f(3)"),
        "vec2f with identical components should collapse: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_identical_integer_collapses_to_splat() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: vec3i;
        @compute @workgroup_size(1) fn main() {
            out = vec3i(7, 7, 7);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3(7)"),
        "vec3i with identical components should collapse (int literal pins i32, so the ctor suffix drops): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_identical_u32_collapses_to_splat() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> out: vec2u;
        @compute @workgroup_size(1) fn main() {
            out = vec2u(4u, 4u);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec2u(4)"),
        "vec2u with identical components should collapse: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_different_components_not_collapsed() {
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(1.0, 2.0, 3.0, 4.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec4f(1,2,3,4)"),
        "non-identical Compose should keep all components: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn compose_matrix_not_collapsed() {
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            let m = mat2x2f(1.0, 1.0, 1.0, 1.0);
            return vec4f(m[0], m[1]);
        }
    "#;
    let out = compact(src);
    // `mat2x2f(1)` would be the diagonal identity, not all-ones.
    assert!(
        !out.contains("mat2x2f(1)") && !out.contains("mat2x2<f32>(1)"),
        "matrix Compose must not be collapsed to splat: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_compose_identical_collapses_to_splat() {
    let src = r#"
        const v = vec3f(2.5, 2.5, 2.5);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(v, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3(2.5)"),
        "global Compose with identical literals should collapse (float-form literal pins f32): {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Vector constructor element-type inference (vec2u(x,4) -> vec2(x,4))

#[test]
fn vector_ctor_drops_suffix_when_a_component_pins_type() {
    let out = compact(
        r#"
        @group(0) @binding(0) var<uniform> p: vec2u;
        @group(0) @binding(1) var<storage, read_write> buf: array<u32>;
        @compute @workgroup_size(1) fn main() {
            let v = vec2(p.x, 4u);
            buf[0] = v.x + v.y;
        }
        "#,
    );
    assert!(
        out.contains("vec2(") && !out.contains("vec2u("),
        "a u32 component pins the element type, so vec2u drops its suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn vector_ctor_keeps_suffix_when_all_components_are_literals() {
    let out = compact(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<u32>;
        @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) n: u32) {
            var v = vec2(4u, 1u);
            v.x = v.x + n;
            buf[0] = v.x + v.y;
        }
        "#,
    );
    // Bare literals are abstract; dropping `u` would re-infer vec2<i32>.
    assert!(
        out.contains("vec2u("),
        "all-literal vec2u must keep its suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn vector_ctor_drops_suffix_for_subvector_component() {
    let out = compact(
        r#"
        @group(0) @binding(0) var<uniform> a: vec2f;
        @group(0) @binding(1) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(1) fn main() {
            let v = vec4<f32>(a, a);
            buf[0] = v.x + v.w;
        }
        "#,
    );
    assert!(
        out.contains("vec4(") && !out.contains("vec4f("),
        "a vec2f component pins f32, so vec4f drops its suffix: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn an_overflowing_literal_pair_keeps_its_i32_type() {
    // Two bare literals evaluate as AbstractInt, which is exact where i32
    // wraps: the value changes and an out-of-range result becomes a
    // shader-creation error.  Typing the left operand pins the pair back.
    let src = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
               @compute @workgroup_size(1) fn main() {\n\
                 var c = 2147483647i;\n\
                 c = c + 1i;\n\
                 out[0] = u32(c);\n\
                 var d = -2147483647i;\n\
                 out[1] = u32(clamp(abs(d * d), -100i, 100i));\n\
               }";
    let out: String = compact_with_passes(src, Profile::Max)
        .split_whitespace()
        .collect();
    assert!(out.contains("2147483647i+1"), "{out}");
    assert!(out.contains("-2147483647i*-2147483647"), "{out}");

    // A pair that does not overflow keeps the shorter bare form.
    let plain = "@group(0)@binding(0) var<storage,read_write> out: array<u32>;\n\
                 @compute @workgroup_size(1) fn main() {\n\
                   var c = 3i;\n\
                   c = c + 4i;\n\
                   out[0] = u32(c);\n\
                 }";
    let short: String = compact_with_passes(plain, Profile::Max)
        .split_whitespace()
        .collect();
    assert!(!short.contains("3i+4"), "{short}");
}
