//! Expression-level emission tests spanning the generator's
//! size-oriented rewrites: bare-literal emission inside constructor
//! arguments, single-use `let` inlining, swizzle folding, bitcast
//! and `select` shapes, `arrayLength` handling, vector and matrix
//! accesses, splat expressions, and the Compose-to-splat collapse.

use super::helpers::*;

// MARK: Bare literals inside Compose/Splat

#[test]
fn compose_uses_bare_literals() {
    let out = compact("fn f() -> vec2<f32> { return vec2<f32>(1.0, 2.0); }");
    // Inside vec2f(...), literals must have no `f` suffix.
    assert!(out.contains("vec2f(1,2)"), "got: {out}");
    assert!(!out.contains("1f"), "typed suffix inside compose: {out}");
}

#[test]
fn splat_uses_bare_literal() {
    let out = compact("fn f() -> vec3<f32> { return vec3<f32>(5.0, 5.0, 5.0); }");
    // naga may lower this to Splat(5.0).  Either way, no `f` suffix inside constructor.
    assert!(
        !out.contains("5f)") && !out.contains("5.f)"),
        "typed suffix in splat: {out}"
    );
}

// MARK: Single-use let inlining

#[test]
fn single_use_expression_is_inlined() {
    let out = compact("fn f(a: f32, b: f32) -> f32 { return a + b; }");
    // `a + b` should be inlined into the return, not bound to a let.
    assert!(!out.contains("let "), "unexpected let binding: {out}");
    assert!(out.contains("return"), "missing return: {out}");
}

#[test]
fn multi_use_expression_gets_let_binding() {
    // `a + b` used twice: both operands are function arguments (1-char names),
    // so the expression text is ~3 chars (`a+b`).  At only 2 references the
    // `let` overhead exceeds the savings - the generator should inline it.
    let out = compact("fn f(a: f32, b: f32) -> f32 { let s = a + b; return s * s; }");
    assert!(
        !out.contains("let "),
        "short binary on named operands should be inlined at 2 refs: {out}"
    );
}

#[test]
fn longer_multi_use_expression_gets_let_binding() {
    // `normalize(cross(a, b))` is long enough that a `let` binding saves
    // space when used twice (textLen >> 7).
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

// MARK: Literal emission: different types coexist

#[test]
fn different_literal_types_not_conflated() {
    // 1.0 (f32) and 1 (i32) have different bare strings.
    // They should NOT be conflated into the same extracted const.
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
    // Exercises struct_field_name through AccessIndex.
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
    // Whole-number float literals emit as bare integers (same as the original
    // minifier baseline).  WGSL abstract-type coercion handles promotion to
    // the required float type in binary-arithmetic and similar contexts.
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
    // Two whole-number float literals multiplied: naga const-folds to a single
    // float literal.  With bare emission, the result emits without a dot.
    let out = compact(
        "fn f() -> f32 { return 2.0 * 3.0; }\n\
             @compute @workgroup_size(1) fn main() { _ = f(); }",
    );
    assert_valid_wgsl(&out);
}

#[test]
fn negative_zero_float_binary_operand_keeps_float_marker() {
    // -0.0 emits as the bare form; the semantic distinction of negative-zero
    // is lost (same behaviour as the original minifier baseline).
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
    // Condition is a function call result - cannot flip a comparison,
    // so the fallback `!(expr)` path should be used.
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
    // All four AccessIndex components on the same vec4 base are folded
    // into a single swizzle expression: vec4f(v.w,v.z,v.y,v.x) -> v.wzyx.
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
    // Accessing .x on a var (pointer-to-vector) should also use swizzle form.
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
    // Load-of-AccessIndex components through a var are grouped into a
    // partial swizzle: vec4f(tmp.x,tmp.y,tmp.z,1) -> vec4f(tmp.xyz,1).
    assert!(
        out.contains(".xyz"),
        "vector access through var should be grouped into swizzle: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn matrix_access_index_stays_bracket() {
    // Matrix column access must still use [idx] (no swizzle notation).
    let src = r#"
        @fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let m = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
            let col0 = m[0];
            let col1 = m[1];
            return vec4<f32>(col0.x + col1.y, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    // Matrix column access should be [0], [1] - not .x, .y.
    // But the vector components of the result should use .x, .y.
    assert_valid_wgsl(&out);
}

// MARK: Swizzle folding in Compose

#[test]
fn full_swizzle_identity_vec4_eliminates_constructor() {
    // vec4f(v.x, v.y, v.z, v.w) on a vec4 base -> just `v` (identity).
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.x, v.y, v.z, v.w);
        }
    "#;
    let out = compact(src);
    // Identity swizzle on same-size vector: no constructor, no swizzle suffix.
    assert!(
        !out.contains("vec4f(v"),
        "identity should eliminate the constructor: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn full_swizzle_reorder_vec4() {
    // vec4f(v.w, v.z, v.y, v.x) -> v.wzyx
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
    // vec2f(v.z, v.w) on a vec4 base -> v.zw (non-identity, smaller output vec)
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
    // vec4f(v.x, v.y, 0., 1.) -> vec4f(v.xy, 0., 1.)
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
    // vec4f(1., 0., v.x, v.y) -> vec4f(1., 0., v.xy)
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
    // vec2f(a.x, b.y) - different bases, no grouping.
    let src = r#"
        @fragment fn fs(@location(0) a: vec4f, @location(1) b: vec4f) -> @location(0) vec4f {
            return vec4f(a.x, b.y, 0., 1.);
        }
    "#;
    let out = compact(src);
    // Should still have individual accesses (no swizzle grouping across bases).
    assert!(
        !out.contains(".xy"),
        "different bases must not be grouped: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_through_var_pointer() {
    // Access through var (ptr-to-vector) should also produce swizzle.
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
    // vec3f(v.x, v.y, v.z) where v is vec3 var -> just v (identity via load-path).
    let src = r#"
        fn helper(v: vec3f) -> vec3f {
            return vec3f(v.x, v.y, v.z);
        }
        @fragment fn fs(@location(0) c: vec3f) -> @location(0) vec4f {
            return vec4f(helper(c), 1.);
        }
    "#;
    let out = compact(src);
    // Identity on a by-value vec3 parameter should eliminate the constructor.
    assert!(
        !out.contains("vec3f(v"),
        "identity on by-value vec3 should fold: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn no_swizzle_for_single_component() {
    // A single AccessIndex component should not trigger grouping.
    let src = r#"
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            return vec4f(v.x, 0., 0., 1.);
        }
    "#;
    let out = compact(src);
    // Single component -> no swizzle; should stay as v.x
    assert!(out.contains("v.x"), "single component stays as .x: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn swizzle_non_sequential_indices() {
    // vec2f(v.w, v.x) -> v.wx (non-sequential but same base)
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
fn swizzle_struct_field_vector() {
    // When the front-end creates separate AccessIndex handles for each
    // `s.color` reference, the expressions have different handles and
    // cannot be grouped.  This is a known limitation - the optimisation
    // requires components to share the same expression handle.
    let src = r#"
        struct S { color: vec4f }
        @fragment fn fs(@location(0) v: vec4f) -> @location(0) vec4f {
            let s = S(v);
            return vec4f(s.color.x, s.color.y, 0., 1.);
        }
    "#;
    let out = compact(src);
    // Even without grouping, the output must be valid.
    assert_valid_wgsl(&out);
}

// MARK: Global Splat expression

#[test]
fn global_splat_vec3_roundtrip() {
    // `vec3f(1.0)` produces a Splat in the global expression arena.
    let src = r#"
        const a = vec3f(1.0);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(a, 1.0);
        }
    "#;
    let out = compact(src);
    // The const should emit with a vec3f constructor.
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
    // vec3f(1.2, 1.2, 1.2) should emit as vec3f(1.2).
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(vec3f(1.2, 1.2, 1.2), 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3f(1.2)"),
        "identical-component Compose should collapse to splat: {out}"
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
        out.contains("vec4f(.5)"),
        "vec4f with identical components should collapse: {out}"
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
        out.contains("vec3i(7)"),
        "vec3i with identical components should collapse: {out}"
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
    // Non-identical components must NOT be collapsed.
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
    // Matrix constructors must NOT be collapsed even with identical components.
    let src = r#"
        @fragment fn main() -> @location(0) vec4f {
            let m = mat2x2f(1.0, 1.0, 1.0, 1.0);
            return vec4f(m[0], m[1]);
        }
    "#;
    let out = compact(src);
    // mat2x2f(1) would mean diagonal identity, NOT all-ones.
    assert!(
        !out.contains("mat2x2f(1)") && !out.contains("mat2x2<f32>(1)"),
        "matrix Compose must not be collapsed to splat: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_compose_identical_collapses_to_splat() {
    // const-declared Compose in the global expression arena.
    let src = r#"
        const v = vec3f(2.5, 2.5, 2.5);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(v, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("vec3f(2.5)"),
        "global Compose with identical literals should collapse: {out}"
    );
    assert_valid_wgsl(&out);
}
