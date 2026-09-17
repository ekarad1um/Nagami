//! Identifier allocation and mode-specific output: short binding names,
//! mangling with preserved symbols, shared literal extraction, deferred
//! variables, struct layout attributes, workgroup-size trimming, override
//! `@id`, float precision, beautify with mangling, and parenthesisation
//! edge cases.

use super::super::{GenerateOptions, generate};
use super::helpers::*;

// MARK: Short expression names

#[test]
fn expr_names_avoid_collision_with_args() {
    // "A" is the first mangle name.
    let src = r#"
            fn f(A: f32) -> f32 {
                let x = A + 1.0;
                return x + x + x;
            }
            @compute @workgroup_size(1)
            fn main() { _ = f(1.0); }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn expr_names_are_short() {
    let src = r#"
            fn f(x: f32, y: f32) -> f32 {
                let a = x * y + 1.0;
                let b = a * a;
                return b * b;
            }
            @compute @workgroup_size(1)
            fn main() { _ = f(1.0, 2.0); }
        "#;
    let out = compact(src);
    assert!(
        !out.contains("_e"),
        "expression names should not use _eN pattern: {out}"
    );
}

#[test]
fn expr_names_do_not_shadow_globals() {
    let src = r#"
            @group(0) @binding(0) var tex: texture_2d<f32>;
            @group(0) @binding(1) var samp: sampler;
            fn helper(uv: vec2f) -> vec4f {
                let a = uv.x + uv.y;
                let b = a * a;
                let c = b * b;
                return textureSampleLevel(tex, samp, vec2f(c, c), 0.0);
            }
            @fragment fn main() -> @location(0) vec4f {
                return helper(vec2f(0.5, 0.5));
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Mangle mode

#[test]
fn mangle_renames_struct_types_and_members() {
    let out = compact_mangled(
        r#"
            struct MyStruct {
                longFieldName: f32,
                anotherField: f32,
            }
            @fragment
            fn fs_main() -> @location(0) vec4f {
                var s: MyStruct;
                s.longFieldName = 1.0;
                s.anotherField = 2.0;
                return vec4f(s.longFieldName, s.anotherField, 0.0, 1.0);
            }
        "#,
    );
    assert!(
        !out.contains("MyStruct"),
        "struct type name should be mangled: {out}"
    );
    assert!(
        !out.contains("longFieldName"),
        "struct field name should be mangled: {out}"
    );
    assert!(
        !out.contains("anotherField"),
        "struct field name should be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_produces_shorter_output() {
    let src = r#"
            struct Uniforms {
                resolution: vec2<f32>,
                time: f32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(uniforms.resolution, uniforms.time, 1.0);
            }
        "#;
    let normal = compact(src);
    let mangled = compact_mangled(src);
    assert!(
        mangled.len() < normal.len(),
        "mangled output ({}) should be shorter than normal ({})",
        mangled.len(),
        normal.len()
    );
}

#[test]
fn mangle_round_trip_complex_shader() {
    let src = r#"
            struct Material {
                color: vec3<f32>,
                roughness: f32,
            }
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) normal: vec3<f32>,
            }
            @group(0) @binding(0) var<uniform> material: Material;
            fn lighting(n: vec3<f32>, l: vec3<f32>) -> f32 {
                return max(dot(n, l), 0.0);
            }
            @fragment fn main(input: VertexOutput) -> @location(0) vec4f {
                let d = lighting(input.normal, vec3f(0.0, 1.0, 0.0));
                return vec4f(material.color * d, 1.0);
            }
        "#;
    let out = compact_mangled(src);
    assert!(
        !out.contains("Material"),
        "Material should be mangled: {out}"
    );
    assert!(
        !out.contains("roughness"),
        "roughness should be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Mangle with preserve_symbols

#[test]
fn mangle_preserves_struct_type_name() {
    let out = compact_mangled_preserved(
        r#"
            struct Uniforms {
                resolution: vec2<f32>,
                time: f32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(uniforms.resolution, uniforms.time, 1.0);
            }
        "#,
        &["Uniforms"],
    );
    assert!(
        out.contains("Uniforms"),
        "preserved struct type name should survive mangling: {out}"
    );
    assert!(
        !out.contains("resolution"),
        "non-preserved member should be mangled: {out}"
    );
    assert!(
        !out.contains("time"),
        "non-preserved member should be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_preserves_struct_member_name() {
    let out = compact_mangled_preserved(
        r#"
            struct Uniforms {
                resolution: vec2<f32>,
                time: f32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(uniforms.resolution, uniforms.time, 1.0);
            }
        "#,
        &["resolution"],
    );
    assert!(
        out.contains("resolution"),
        "preserved member name should survive mangling: {out}"
    );
    assert!(
        !out.contains("Uniforms"),
        "non-preserved struct type should be mangled: {out}"
    );
    assert!(
        !out.contains("time"),
        "non-preserved member should be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_preserves_both_type_and_member() {
    let out = compact_mangled_preserved(
        r#"
            struct Material {
                color: vec3<f32>,
                roughness: f32,
            }
            @group(0) @binding(0) var<uniform> mat: Material;
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(mat.color, mat.roughness);
            }
        "#,
        &["Material", "color"],
    );
    assert!(
        out.contains("Material"),
        "preserved struct type should survive: {out}"
    );
    assert!(
        out.contains("color"),
        "preserved member should survive: {out}"
    );
    assert!(
        !out.contains("roughness"),
        "non-preserved member should be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_no_preserve_still_mangles_everything() {
    let out = compact_mangled_preserved(
        r#"
            struct Params { scale: f32, offset: f32 }
            fn transform(p: Params) -> f32 {
                return p.scale + p.offset;
            }
            @compute @workgroup_size(1)
            fn main() { _ = transform(Params(2.0, 3.0)); }
        "#,
        &[],
    );
    assert!(
        !out.contains("Params"),
        "struct type should be mangled with empty preserve: {out}"
    );
    assert!(
        !out.contains("scale"),
        "member should be mangled with empty preserve: {out}"
    );
    assert!(
        !out.contains("offset"),
        "member should be mangled with empty preserve: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_preserve_short_type_no_collision_with_next_struct() {
    // A preserved "A" must sit in `used_names` or the counter hands "A" to
    // struct B as well; B is declared first so the counter reaches "A" before
    // the preserved one.
    let out = compact_mangled_preserved(
        r#"
            struct B { y: f32 }
            struct A { x: f32 }
            @group(0) @binding(0) var<uniform> g0: B;
            @group(0) @binding(1) var<uniform> g1: A;
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(g0.y, g1.x, 0.0, 1.0);
            }
        "#,
        &["A"],
    );
    assert!(
        out.contains("struct A"),
        "preserved struct A must survive: {out}"
    );
    let struct_a_count = out.matches("struct A").count();
    assert_eq!(
        struct_a_count, 1,
        "only one struct A definition should exist (no collision): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_preserve_short_member_no_collision_within_struct() {
    let out = compact_mangled_preserved(
        r#"
            struct S { A: f32, other: f32 }
            @group(0) @binding(0) var<uniform> s: S;
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(s.A, s.other, 0.0, 1.0);
            }
        "#,
        &["A"],
    );
    assert!(
        out.contains(".A"),
        "preserved member A must survive in access: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_struct_names_avoid_collision_with_short_param_names() {
    // Parameters may already carry short names (rename pass); the struct
    // counter must skip them.
    let out = compact_mangled(
        r#"
            struct S { val: f32 }
            fn f(A: f32) -> S {
                var s: S;
                s.val = A;
                return s;
            }
            @fragment fn main() -> @location(0) vec4f {
                let r = f(1.0);
                return vec4f(r.val, 0.0, 0.0, 1.0);
            }
        "#,
    );
    assert_valid_wgsl(&out);
}

// MARK: Shared literal extraction

#[test]
fn extracts_repeated_long_literal_into_const() {
    let src = r#"
            fn a() -> f32 { return 3.333333; }
            fn b() -> f32 { return 3.333333; }
            fn c() -> f32 { return 3.333333; }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(a(), b(), c(), 1.0);
            }
        "#;
    let out = compact(src);
    let fn_body_count = out.matches("3.333333").count();
    assert_eq!(
        fn_body_count, 1,
        "expected literal extracted to shared const (1 declaration), got {fn_body_count} occurrences: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn no_extraction_for_short_or_rare_literal() {
    let src = r#"
            fn a() -> i32 { return 42; }
            fn b() -> i32 { return 42; }
            @compute @workgroup_size(1)
            fn main() { _ = a(); _ = b(); }
        "#;
    let out = compact(src);
    assert!(!out.contains("const"), "unexpected const extraction: {out}");
}

#[test]
fn extracts_typed_only_f64_literal_at_lower_break_even() {
    // Standalone (non-constructor) uses emit the typed `123.456lf` (9 chars),
    // so three uses break even (3*(9-1) - (8+1+9) = 6 > 0); priced at the bare
    // length (3*(7-1) - 18 = 0) the extraction would be missed.
    let src = r#"
            fn a() -> f64 { return 123.456lf; }
            fn b() -> f64 { return 123.456lf; }
            fn c() -> f64 { return 123.456lf; }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(f32(a()), f32(b()), f32(c()), 1.0);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert_eq!(
        out.matches("123.456").count(),
        1,
        "typed-only f64 literal should be extracted to a single const: {out}"
    );
    assert!(out.contains("const"), "expected an extracted const: {out}");
}

#[test]
fn does_not_over_extract_bare_heavy_needs_typed_literal() {
    // Bare constructor uses emit the short `1.5`; pricing them at the typed
    // `1.5lf` length would over-extract and grow the output.  With any bare
    // use the bare price holds: 5*(3-1) - (8+1+5) = -4 <= 0.  Distinct second
    // components keep the constructors from being one value.
    let src = r#"
            fn a() -> vec2<f64> { return vec2<f64>(1.5lf, 2lf); }
            fn b() -> vec2<f64> { return vec2<f64>(1.5lf, 3lf); }
            fn c() -> vec2<f64> { return vec2<f64>(1.5lf, 4lf); }
            fn d() -> vec2<f64> { return vec2<f64>(1.5lf, 5lf); }
            fn e() -> vec2<f64> { return vec2<f64>(1.5lf, 6lf); }
            @fragment fn main() -> @location(0) vec4f {
                let s = a() + b() + c() + d() + e();
                return vec4f(f32(s.x), f32(s.y), 0.0, 1.0);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert!(
        !out.contains("const"),
        "bare-heavy f64 literal must not be extracted (would grow output): {out}"
    );
}

#[test]
fn extracted_literal_no_collision_with_locals() {
    let src = r#"
            fn f(A: f32) -> f32 { return A + 3.333333; }
            fn g() -> f32 { return 3.333333; }
            fn h() -> f32 { return 3.333333; }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(f(1.0), g(), h(), 1.0);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn extracted_literal_in_entry_point() {
    let src = r#"
            fn a() -> f32 { return 3.333333; }
            fn b() -> f32 { return 3.333333; }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(a(), b(), 3.333333, 1.0);
            }
        "#;
    let out = compact(src);
    let count = out.matches("3.333333").count();
    assert_eq!(
        count, 1,
        "expected exactly 1 const declaration for extracted literal, got {count}: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn extracts_repeated_long_integer_literal_into_valid_const() {
    let src = r#"
            fn a(x: i32) -> i32 { return x + 1234567890; }
            fn b(x: i32) -> i32 { return x + 1234567890; }
            fn c(x: i32) -> i32 { return x + 1234567890; }
            @compute @workgroup_size(1)
            fn main() { _ = a(1) + b(1) + c(1); }
        "#;
    let out = compact(src);
    assert!(out.contains("const "), "expected extracted const: {out}");
    assert!(
        out.contains("=1234567890i;")
            || out.contains("=0x499602d2i;")
            || out.contains("=1234567890;")
            || out.contains("=0x499602d2;"),
        "extracted integer literal declaration should be valid WGSL: {out}"
    );
    assert_eq!(
        out.matches("1234567890").count() + out.matches("0x499602d2").count(),
        1,
        "literal should appear only once after extraction: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn extracted_large_u64_literal_remains_valid() {
    let src = r#"
            var<private> out: u64;
            fn a(x: u64) -> u64 { return x + 18446744073709551615lu; }
            fn b(x: u64) -> u64 { return x + 18446744073709551615lu; }
            fn c(x: u64) -> u64 { return x + 18446744073709551615lu; }
            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                out = a(u64(id.x)) + b(u64(id.x)) + c(u64(id.x));
            }
        "#;
    let out = compact(src);
    assert!(out.contains("const "), "expected extracted const: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn literal_extraction_no_collision_with_mangled_names() {
    let src = r#"
        fn f1(x: f32) -> f32 { return x + 1.23456f; }
        fn f2(x: f32) -> f32 { return x + 1.23456f; }
        fn f3(x: f32) -> f32 { return x + 1.23456f; }
        fn f4(x: f32) -> f32 { return x * 2.0f + 1.23456f; }
        fn f5(x: f32) -> f32 { return x * 3.0f + 1.23456f; }
        @compute @workgroup_size(1)
        fn main() {
            _ = f1(0.0) + f2(0.0) + f3(0.0) + f4(0.0) + f5(0.0);
        }
    "#;
    let out = compact_mangled(src);
    assert_valid_wgsl(&out);
}

// MARK: count_literals adjustment regressions

// Emission-bypass paths where `ref_counts` overstates the textual emission
// count `count_literals` must price: splat-collapsed vector `Compose` (only
// `components[0]` is emitted), `Select`/`Derivative` direct-literal type-pins,
// and integer-literal atomic operands.  An uncorrected count extracts a
// `const` the typed-form emission never references.

#[test]
fn count_literals_does_not_extract_splat_only_literal() {
    // Naive counting sees 6 refs (3 per Compose); only 2 splats are emitted.
    let src = r#"
            fn h(p: vec3f) -> vec3f {
                return p + vec3f(1.234567, 1.234567, 1.234567);
            }
            fn k(p: vec3f) -> vec3f {
                return p * vec3f(1.234567, 1.234567, 1.234567);
            }
            @compute @workgroup_size(1)
            fn main() {
                _ = length(h(vec3f(0.0)) + k(vec3f(0.0)));
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("1.234567"),
        "literal should still appear in output: {out}"
    );
    assert!(
        !out.contains("=1.234567") && !out.contains("= 1.234567"),
        "literal should NOT have been extracted into a const decl: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn count_literals_does_not_over_extract_deferred_var_store_literal() {
    // The deferred first store emits `var acc = 3.333333f` in typed form;
    // discounting it leaves 2 shrinking uses, below break-even, so extraction
    // would grow the output.
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<f32, 4>;
            @compute @workgroup_size(1)
            fn main() {
                var acc: f32;
                acc = 3.333333;
                buf[0] = 3.333333;
                buf[1] = 3.333333;
                buf[2] = acc;
            }
        "#;
    let out = compact(src);
    assert!(
        !out.contains("const"),
        "deferred-var-store literal must not be counted as an extraction-\
         shrinking use; 2 real uses are below break-even, so no const: {out}"
    );
    assert!(
        out.contains("3.333333"),
        "literal should still appear inline: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn count_literals_does_not_extract_select_literal_operand() {
    // `Select`'s direct literal operands emit typed regardless of
    // `extracted_literals`.
    let src = r#"
            fn p(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            fn q(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            fn r(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            @compute @workgroup_size(1)
            fn main() { _ = p(true) + q(false) + r(true); }
        "#;
    let out = compact(src);
    assert!(
        out.contains(".1234567"),
        "literal should still appear in output: {out}"
    );
    assert!(
        !out.contains("=.1234567") && !out.contains("= .1234567"),
        "literal should NOT have been extracted into a const decl: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn count_literals_does_not_extract_atomic_int_literal() {
    // `emit_expr_for_atomic` type-pins integer operands, bypassing
    // `extracted_literals`.
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> ai: atomic<i32>;
            fn t1() { atomicAdd(&ai, 12345678); }
            fn t2() { atomicAdd(&ai, 12345678); }
            fn t3() { atomicAdd(&ai, 12345678); }
            @compute @workgroup_size(1)
            fn main() { t1(); t2(); t3(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("12345678"),
        "literal should still appear in output: {out}"
    );
    assert!(
        !out.contains("=12345678") && !out.contains("= 12345678"),
        "literal should NOT have been extracted into a const decl: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn count_literals_does_not_extract_atomic_store_int_literal() {
    // `emit_atomic_store` type-pins its value like `atomicAdd`.
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> au: atomic<u32>;
            fn t1() { atomicStore(&au, 12345678u); }
            fn t2() { atomicStore(&au, 12345678u); }
            fn t3() { atomicStore(&au, 12345678u); }
            @compute @workgroup_size(1)
            fn main() { t1(); t2(); t3(); }
        "#;
    let out = compact(src);
    assert!(
        out.contains("12345678"),
        "literal should still appear in output: {out}"
    );
    assert!(
        !out.contains("=12345678") && !out.contains("= 12345678"),
        "atomicStore literal should NOT have been extracted into a const decl: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Deferred variable edge cases

#[test]
fn deferred_var_not_in_nested_block() {
    let src = r#"
            fn f(cond: bool) -> f32 {
                var x: f32;
                if cond {
                    x = 1.0;
                } else {
                    x = 2.0;
                }
                return x;
            }
            @compute @workgroup_size(1)
            fn main() { _ = f(true); }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn multiple_deferred_vars() {
    let out = compact(
        r#"
            fn f(a: f32) -> f32 {
                var x: f32;
                var y: f32;
                x = a + 1.0;
                y = a + 2.0;
                return x + y;
            }
        "#,
    );
    assert!(
        out.contains("var x") && out.contains("var y"),
        "both vars should be deferred: {out}"
    );
}

// MARK: Struct @size / @align layout attributes

#[test]
fn struct_size_attr_preserved() {
    let src = r#"
        struct S {
            @size(32) x: f32,
            y: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1) fn main() { _ = u.x + u.y; }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@size(32)"),
        "explicit @size(32) must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn struct_natural_layout_no_attrs() {
    let src = r#"
        struct S {
            x: f32,
            y: f32,
            z: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1) fn main() { _ = u.x + u.y + u.z; }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("@size("),
        "natural layout should not emit @size: {out}"
    );
    assert!(
        !out.contains("@align("),
        "natural layout should not emit @align: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn struct_align_preserved_via_size() {
    let src = r#"
        struct S {
            x: f32,
            @align(16) y: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1) fn main() { _ = u.x + u.y; }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@align(16)") || out.contains("@size(16)"),
        "alignment must be preserved via @size or @align: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Workgroup size trimming

#[test]
fn workgroup_size_trims_trailing_ones() {
    let src = r#"
        @compute @workgroup_size(64, 1, 1)
        fn main() {}
    "#;
    let out = compact(src);
    assert!(
        out.contains("@workgroup_size(64)") || out.contains("@workgroup_size(64) "),
        "trailing 1s should be trimmed: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_size_keeps_second_when_not_one() {
    let src = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {}
    "#;
    let out = compact(src);
    assert!(
        out.contains("@workgroup_size(8,8)") || out.contains("@workgroup_size(8, 8)"),
        "second component should be kept: {out}"
    );
    assert!(!out.contains(",1)"), "trailing 1 should be trimmed: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_size_keeps_all_three_when_needed() {
    let src = r#"
        @compute @workgroup_size(4, 4, 4)
        fn main() {}
    "#;
    let out = compact(src);
    assert!(
        out.contains("4,4,4") || out.contains("4, 4, 4"),
        "all three components should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Override with @id

#[test]
fn override_with_id_roundtrip() {
    let src = r#"
        @id(0) override brightness: f32 = 1.0;
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(brightness, brightness, brightness, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@id(0)"),
        "@id attribute should be preserved: {out}"
    );
    assert!(
        out.contains("override"),
        "override keyword should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn override_without_init_roundtrip() {
    let src = r#"
        @id(1) override scale: f32;
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(scale, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@id(1)"),
        "@id attribute should be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Workgroup size override expressions

#[test]
fn workgroup_size_override_expr() {
    let src = r#"
        override block_size: u32 = 64;
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(block_size) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            buf[gid.x] = 1.0;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("block_size") || out.contains("@workgroup_size("),
        "override expression in workgroup_size must be preserved: {out}"
    );
    assert!(
        !out.contains("@workgroup_size(64)"),
        "should use override name, not literal value: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn workgroup_size_mixed_override_literal() {
    let src = r#"
        override sx: u32 = 8;
        override sy: u32 = 4;
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(sx, sy, 2) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            buf[gid.x] = 1.0;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("sx"),
        "first override dim must be preserved: {out}"
    );
    assert!(
        out.contains("sy"),
        "second override dim must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn global_override_expressions_emit_access_swizzle_math_cast_and_relational() {
    let src = r#"
        const TABLE: array<vec4<f32>, 2> = array<vec4<f32>, 2>(
            vec4<f32>(1.0, 2.0, 3.0, 4.0),
            vec4<f32>(5.0, 6.0, 7.0, 8.0),
        );

        override IDX: u32 = 1u;
        override SX: i32 = -1i;
        override OX: f32 = -1.0;
        override OY: f32 = 2.0;

        var<private> pick_y: f32 = TABLE[IDX].y;
        var<private> pick_yz: vec2<f32> = TABLE[IDX].yz;
        var<private> abs_ox: f32 = abs(OX);
        var<private> cast_sx: u32 = u32(SX);
        var<private> any_pos: bool = any(vec2<bool>(OX > 0.0, OY > 0.0));

        @compute @workgroup_size(1)
        fn main() {
            _ = pick_y;
            _ = pick_yz;
            _ = abs_ox;
            _ = cast_sx;
            _ = any_pos;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("[IDX].y"),
        "expected access-index emission: {out}"
    );
    assert!(out.contains(".yz"), "expected swizzle emission: {out}");
    assert!(out.contains("abs("), "expected math-call emission: {out}");
    assert!(out.contains("u32("), "expected cast emission: {out}");
    assert!(out.contains("any("), "expected relational emission: {out}");
    assert_valid_wgsl(&out);
}

// MARK: float_precision round-trip

#[test]
fn precision_rounds_float() {
    let src = r#"
        @compute @workgroup_size(1)
        fn main() {
            var x: f32 = 3.14159265f;
            _ = x;
        }
    "#;
    let full = compact(src);
    let trunc = compact_with_precision(src, 2);
    assert!(
        full.len() >= trunc.len(),
        "truncated should be no longer than full: full={full}, trunc={trunc}"
    );
    assert_valid_wgsl(&trunc);
}

#[test]
fn significant_figures_round_trips_f16_near_max() {
    // f16 emits through f32 widening, so rounding 65504 up (-> 70000) can
    // produce an out-of-range `70000h`; driven through `generate` so no
    // `run()` fallback masks it.
    let src = r#"
        enable f16;
        @compute @workgroup_size(1)
        fn main() {
            var x: f16 = 65504h;
            _ = x;
        }
    "#;
    for s in [1u8, 2, 3] {
        let out = compact_with_float_precision(
            src,
            FloatPrecision {
                f16: PrecisionMode::SignificantFigures(s),
                ..Default::default()
            },
        );
        assert!(!out.contains("inf"), "f16 sf={s} produced inf token: {out}");
        assert_valid_wgsl(&out);
    }
}

#[test]
fn significant_figures_round_trips_near_type_max() {
    // A literal at the type maximum must not round up across the leading
    // decade into `inff`/`inflf`.
    let src = r#"
        @compute @workgroup_size(1)
        fn main() {
            var a: f32 = 3.4028235e38f;
            _ = a;
        }
    "#;
    for s in [1u8, 2, 3, 4, 5] {
        let out = compact_with_float_precision(
            src,
            FloatPrecision::all(PrecisionMode::SignificantFigures(s)),
        );
        assert!(!out.contains("inf"), "f32 sf={s} produced inf token: {out}");
        assert_valid_wgsl(&out);
    }
}

#[test]
fn precision_preserves_integers() {
    let src = r#"
        @compute @workgroup_size(1)
        fn main() {
            var x: i32 = 123456;
            _ = x;
        }
    "#;
    let out = compact_with_precision(src, 2);
    assert!(
        out.contains("123456"),
        "integer should be preserved exactly: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn typed_f64_and_f16_short_suffix_forms_round_trip() {
    // The hex-float (`0x1p50lf`) and scientific (`1e15lf`, `1e4h`) typed forms
    // are re-parsed so a naga grammar change trips here rather than shipping
    // invalid output.
    let src = r#"
        enable f16;
        @compute @workgroup_size(1)
        fn main() {
            var a: f64 = 1125899906842624.0lf;   // 2^50 -> 0x1p50lf
            var b: f64 = 1e15lf;                  // clean power of ten
            var c: f16 = 10000h;                  // -> 1e4h
            _ = a; _ = b; _ = c;
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert!(out.contains("0x1p50lf"), "expected hex f64 form: {out}");
    assert!(
        out.contains("1e15lf"),
        "expected scientific f64 form: {out}"
    );
    assert!(out.contains("1e4h"), "expected scientific f16 form: {out}");
}

// MARK: beautify + mangle combo

#[test]
fn beautify_mangle_roundtrip() {
    let src = r#"
        struct Params { scale: f32, offset: f32 }
        fn transform(p: Params) -> f32 {
            return p.scale + p.offset;
        }
        @compute @workgroup_size(1)
        fn main() {
            let r = transform(Params(2.0, 3.0));
            _ = r;
        }
    "#;
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    let out = generate(
        &module,
        &info,
        GenerateOptions {
            beautify: true,
            indent: 2,
            mangle: true,
            float_precision: FloatPrecision::default(),
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source;
    assert!(
        out.contains('\n'),
        "beautified output should have newlines: {out}"
    );
    assert!(
        !out.contains("Params") && !out.contains("scale") && !out.contains("offset"),
        "mangled output should not have original struct/field names: {out}"
    );
    assert_valid_wgsl(&out);
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

// MARK: Postfix parenthesization

#[test]
fn postfix_access_on_binary_base_is_parenthesized() {
    let src = r#"
            fn f(a: vec2f, b: vec2f) -> f32 {
                return (a - b).x + (a * b).y;
            }
            @compute @workgroup_size(1)
            fn main() { _ = f(vec2f(1.0), vec2f(2.0)); }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Comparison in type constructor

#[test]
fn less_than_comparison_in_vec_bool_constructor() {
    // A bare `<` inside `vec3<bool>(...)` is ambiguous with template syntax.
    let src = r#"
            fn f(p: vec2f, a: vec2f, b: vec2f) -> f32 {
                let c = vec3<bool>(p.y >= a.y, (p.y < b.y), (a.x > b.x));
                if all(c) { return 1.0; }
                return 0.0;
            }
            @compute @workgroup_size(1)
            fn main() { _ = f(vec2f(0.0), vec2f(1.0), vec2f(2.0)); }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

/// A bare `<` in a non-final argument pairs with a later argument's `>` in
/// the template-list scanner (`f(a<b,c>d)` is the template `a<b,c>` applied
/// to `d`).  A self-check failure ships the input verbatim, which the newline
/// assertion detects.
#[test]
fn less_than_call_and_select_arguments_get_template_guard_parens() {
    let src = r#"
            var<private> a: u32; var<private> b: u32;
            var<private> c: u32; var<private> d: u32;
            var<private> r: u32; var<private> rb: bool;
            fn f(p: bool, q: bool) -> u32 {
                if p { return 1u; }
                if q { return 2u; }
                return 0u;
            }
            @compute @workgroup_size(1)
            fn main() {
                r = f((a < b), c > d) + f((a < b), c > d);
                rb = select((a < b), c > d, a == c);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert!(
        !out.contains('\n'),
        "multi-line output means the whole-file fallback fired: {out}"
    );
    assert!(
        out.contains("((") && out.contains("select(("),
        "both `<` arguments must be parenthesised: {out}"
    );
}

/// Parenthesisation and template guards classify children by arena variant,
/// so a `Unary` rendered as a bare comparison would ship atom-tight into a
/// comparison parent (`a<b==c`: tint rejects, naga's self-check accepts).
/// The flip is condition-only (if/break_if).
#[test]
fn value_context_negated_comparison_keeps_unary_shape() {
    let src = r#"
            var<private> i: i32; var<private> j: i32;
            var<private> c: bool; var<private> r1: bool; var<private> r2: bool;
            @compute @workgroup_size(1)
            fn main() {
                r1 = !(i < j);
                r2 = (!(i >= j)) == c;
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    // A regressed flip renders r2's negation as a bare comparison.
    assert!(
        out.matches("!(").count() >= 2,
        "value-context negations must keep the prefix form: {out}"
    );
}

/// `a<b>>c` scans as the template `a<b>` plus `>c`; precedence alone leaves
/// the tighter-binding shift bare, so the emitter needs a dedicated wrap.
#[test]
fn shift_right_under_less_gets_template_guard_parens() {
    let src = r#"
            var<private> a: u32; var<private> b: u32;
            var<private> c: u32; var<private> rc: bool;
            @compute @workgroup_size(1)
            fn main() {
                rc = a < (b >> c);
            }
        "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert!(
        !out.contains('\n'),
        "multi-line output means the whole-file fallback fired: {out}"
    );
    assert!(
        out.contains("<("),
        "the shift right-operand must be parenthesised: {out}"
    );
}

/// Per struct of the re-parsed `out`, the member names and the names of
/// the types its members spell must be disjoint: WGSL keeps the two
/// namespaces apart, the C++ naga's MSL writer emits does not, and a
/// member named like the type of a later member hides it.
fn assert_members_dodge_member_types(out: &str) {
    let module = naga::front::wgsl::parse_str(out).expect("re-parse failed");
    for (_, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            continue;
        };
        let names: std::collections::HashSet<&str> =
            members.iter().filter_map(|m| m.name.as_deref()).collect();
        for member in members {
            if let Some(spelled) = module.types[member.ty].name.as_deref() {
                assert!(
                    !names.contains(spelled),
                    "struct {:?} has a member named like its member type {spelled}: {out}",
                    ty.name
                );
            }
        }
    }
}

const MEMBER_TYPE_SRC: &str = r#"
    struct Inner { x: f32, y: f32 }
    struct Outer { p: f32, q: Inner, r: Inner, t: vec3f, u: vec3f, v: vec3f }
    @group(0) @binding(0) var<storage> s: Outer;
    @fragment fn main() -> @location(0) vec4f {
        return vec4f(s.q.x + s.r.y + s.p, s.t.x, s.u.y, s.v.z);
    }
"#;

#[test]
fn mangled_member_names_dodge_member_type_names() {
    let out = compact_mangled(MEMBER_TYPE_SRC);
    assert_valid_wgsl(&out);
    assert_members_dodge_member_types(&out);
    // The aliased `vec3f` members spell a minted alias inside the same body.
    let out = compact_mangled_aliased(MEMBER_TYPE_SRC);
    assert_valid_wgsl(&out);
    assert_members_dodge_member_types(&out);
}

#[test]
fn member_type_dodge_is_idempotent() {
    let once = compact_mangled_aliased(MEMBER_TYPE_SRC);
    let twice = compact_mangled_aliased(&once);
    assert_eq!(
        once, twice,
        "member letters must be a function of the final type names"
    );
}

#[test]
fn preserved_struct_name_keeps_dodging_member_types() {
    let out = compact_mangled_preserved(MEMBER_TYPE_SRC, &["Outer"]);
    assert!(out.contains("struct Outer{"), "{out}");
    assert_valid_wgsl(&out);
    assert_members_dodge_member_types(&out);
}

/// The other direction: a member the host reads by name keeps its letter,
/// so the type letters have to move out of its way.
#[test]
fn minted_type_name_dodges_a_kept_member_name() {
    let src = r#"
        struct Inner { x: f32 }
        struct Outer { A: f32, q: Inner }
        @group(0) @binding(0) var<storage> s: Outer;
        @fragment fn main() -> @location(0) vec4f { return vec4f(s.A + s.q.x); }
    "#;
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    let out = generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: true,
            preserve_members: ["A".to_string()].into_iter().collect(),
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source;
    assert!(out.contains(".A"), "kept member must survive: {out}");
    assert_valid_wgsl(&out);
    assert_members_dodge_member_types(&out);
}
