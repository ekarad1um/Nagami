//! Tests for the generator's identifier-allocation logic and
//! mode-specific output formatting.  Covers short expression-binding
//! names, mangling (with and without preserved symbols), shared
//! literal extraction, deferred-variable emission, struct layout
//! attributes, workgroup-size trimming, override `@id` annotations,
//! the `max_precision` round-trip, beautify combined with mangling,
//! and a handful of parenthesisation edge cases.

use super::super::{GenerateOptions, generate_wgsl};
use super::helpers::*;

// MARK: Short expression names

#[test]
fn expr_names_avoid_collision_with_args() {
    // If a function argument is named "A" (the first mangle name),
    // expression bindings should skip "A" and use the next available name.
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
    // Multi-use expressions should get 1-char names, not _eN patterns.
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
    // The global variable is renamed to a short name (e.g. "A").
    // Inside the function, multi-use expressions get let-bindings
    // via next_expr_name().  Those names must skip any module-scope
    // name to avoid shadowing in the emitted WGSL text.
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
    // The output must re-parse and re-validate successfully.
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
    // Round-trip validation
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
    // Comprehensive mangle mode round-trip test.
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
    // Must not contain original long names.
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
    // Members should still be mangled since they're not in the preserve list.
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
    // Struct type name should still be mangled.
    assert!(
        !out.contains("Uniforms"),
        "non-preserved struct type should be mangled: {out}"
    );
    // Other member should still be mangled.
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
    // Verify the default (empty preserve) still mangles all struct names.
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
    // Regression: preserved struct type name "A" must not collide with the
    // mangled name assigned to another struct.  Before the fix, "A" was
    // not added to the generator's `used_names`, so the counter could
    // generate "A" for struct B -> duplicate struct definitions -> invalid WGSL.
    //
    // Struct B is declared BEFORE struct A so that the counter reaches "A"
    // first, then the preserved "A" duplicates it.
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
    // The mangled name for struct B must NOT be "A".
    let struct_a_count = out.matches("struct A").count();
    assert_eq!(
        struct_a_count, 1,
        "only one struct A definition should exist (no collision): {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mangle_preserve_short_member_no_collision_within_struct() {
    // Regression: preserved member name "A" must not collide with mangled
    // names of other members in the same struct.
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
    // When function parameters already have short names (e.g. after the
    // rename pass), the generator's struct mangle counter must skip those
    // names to avoid shadowing the struct type inside the function body.
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
    // The struct must not be named "A" since the parameter is already "A".
    assert_valid_wgsl(&out);
}

// MARK: Shared literal extraction

#[test]
fn extracts_repeated_long_literal_into_const() {
    // Use a long literal (3.333333 = 8 chars) across multiple functions.
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

// These tests pin the three emission-bypass paths in
// `src/generator/literal_extract.rs::count_literals`:
//   (a) splat-collapse for vector `Compose` (only `components[0]` is emitted)
//   (b) `Select` / `Derivative` direct-`Literal`-operand type-pin
//   (c) integer-literal atomic operand type-pin
//
// In each case `ref_counts[h]` overstates the textual emission count, and
// `count_literals` must compensate.  If the adjustment drifts out of sync
// with the corresponding emission code in `expr_emit.rs` / `stmt_emit.rs`,
// the literal would be wrongly extracted into a `const` that is never
// referenced (because the bypass path emits the typed-suffix form
// directly), bloating output.

#[test]
fn count_literals_does_not_extract_splat_only_literal() {
    // Two splat-collapsable vec3f composes share the same long literal.
    // Naive ref counting would see 6 references (3 per Compose) and
    // greenlight extraction.  Splat-collapse adjustment must drop this
    // to 2 textual emissions so no `const` is created.
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
    // Literal must remain present (twice, once per splat).  No `const`
    // declaration should have been emitted for it.
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
fn count_literals_does_not_extract_select_literal_operand() {
    // Direct `Literal` operands of `Select` are emitted in typed-suffix
    // form regardless of `extracted_literals`.  Without the Select
    // type-pin adjustment, three uses would count as 3 textual emissions
    // and trigger extraction; with it, zero emissions count and no const
    // is created.
    let src = r#"
            fn p(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            fn q(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            fn r(c: bool) -> f32 { return select(0.1234567f, 0.7654321f, c); }
            @compute @workgroup_size(1)
            fn main() { _ = p(true) + q(false) + r(true); }
        "#;
    let out = compact(src);
    // Literal still appears (once per select call site).  No `const`
    // declaration should have been generated for the bare-form key.
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
    // Integer literal operands of atomic statements are type-pinned to
    // the atomic's scalar type by `emit_expr_for_atomic`, bypassing
    // `extracted_literals`.  The integer-literal-only adjustment must
    // suppress extraction here.
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
    // Two independent variables should both be deferred to first store.
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

// MARK: max_precision round-trip

#[test]
fn max_precision_truncates_float() {
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
fn max_precision_preserves_integers() {
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
    let out = generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: true,
            indent: 2,
            mangle: true,
            max_precision: None,
            ..Default::default()
        },
    )
    .expect("generate failed");
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
    // AccessIndex on a Binary base must emit `(a-b).x`, not `a-b.x`.
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
    // Bare `<` inside vec3<bool>() is ambiguous with WGSL template syntax;
    // the generator must parenthesize it.
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
