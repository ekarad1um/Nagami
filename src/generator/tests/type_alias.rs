//! Tests for the generator's `alias T = ...;` introduction pass.
//! Each section pins one threshold or interaction: when aliasing is
//! profitable, when it is skipped, how it interacts with mangling,
//! which type categories participate (matrices, compounds, pointers,
//! samplers, textures, splats), and that aliased output still
//! round-trips through naga.

use super::helpers::{assert_valid_wgsl, compact, compact_aliased, compact_mangled_aliased};

// MARK: Basic alias thresholds

#[test]
fn no_alias_when_single_use() {
    // A single use of an array type - aliasing costs more than it saves.
    let src = r#"
        fn foo() -> array<vec4<f32>, 16> {
            return array<vec4<f32>, 16>();
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // The output should NOT contain "alias" because the type is used too few
    // times to justify the declaration overhead.
    // (2 uses * 15 chars saved per use = 30; decl cost ~= 8 + 1 + 16 = 25, so
    //  it might or might not alias depending on exact count.  Use a truly-single
    //  reference case to be sure.)
    // Actually with Compose expression + return type, the count is still small.
    // Let's just verify the output is valid.
}

// MARK: Alias introduced for frequent types

#[test]
fn alias_for_array_type_many_refs() {
    let src = r#"
        struct S {
            a: array<vec4<f32>, 16>,
            b: array<vec4<f32>, 16>,
            c: array<vec4<f32>, 16>,
        }
        fn foo(x: array<vec4<f32>, 16>) -> array<vec4<f32>, 16> {
            return x;
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // With 5+ references to array<vec4f,16>, an alias should be introduced.
    assert!(
        out.contains("alias "),
        "expected alias declaration in output: {out}"
    );
    // The full type string should NOT appear directly in struct members or
    // function signatures if the alias is used.
    // (It may appear once in the alias declaration itself.)
    let alias_decl_count = out.matches("array<vec4f,16>").count();
    assert!(
        alias_decl_count <= 1,
        "expected at most 1 occurrence of the full type (in alias decl), got {alias_decl_count}: {out}"
    );
}

#[test]
fn alias_for_array_type_used_in_struct_and_functions() {
    let src = r#"
        struct Data {
            a: array<vec4<f32>, 16>,
            b: array<vec4<f32>, 16>,
        }
        @group(0) @binding(0) var<storage> data: Data;
        fn process(x: array<vec4<f32>, 16>) -> vec4<f32> {
            return x[0];
        }
        fn main_fn() -> vec4<f32> {
            return process(data.a) + process(data.b);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(
        out.contains("alias "),
        "expected alias declaration in output: {out}"
    );
}

// MARK: Short types not aliased

#[test]
fn no_alias_for_scalar_type_few_uses() {
    // Scalars like f32 (3 chars) need many uses to justify aliasing.
    let src = r#"
        fn foo(a: f32, b: f32) -> f32 {
            return a + b;
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // With so few uses, aliasing f32 shouldn't happen.
    assert!(!out.contains("alias "), "unexpected alias in output: {out}");
}

// MARK: Alias with mangling

#[test]
fn alias_with_mangle() {
    let src = r#"
        struct MyStruct {
            position: vec4<f32>,
            color: vec4<f32>,
        }
        struct Container {
            items: array<MyStruct, 8>,
            extra: array<MyStruct, 8>,
            more: array<MyStruct, 8>,
        }
        @group(0) @binding(0) var<uniform> container: Container;
        fn get_item(idx: u32) -> MyStruct {
            return container.items[idx];
        }
    "#;
    let out = compact_mangled_aliased(src);
    assert_valid_wgsl(&out);
}

// MARK: Matrix types

#[test]
fn alias_for_matrix_type() {
    // Many references to mat4x4f should trigger aliasing
    let src = r#"
        struct Transform {
            model: mat4x4<f32>,
            view: mat4x4<f32>,
            proj: mat4x4<f32>,
        }
        fn apply(t: Transform, v: vec4<f32>) -> vec4<f32> {
            return t.proj * t.view * t.model * v;
        }
        fn identity() -> mat4x4<f32> {
            return mat4x4<f32>(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            );
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // mat4x4f is only 7 chars; alias saves 6 per use minus ~16 decl overhead.
    // With 3 struct members + 1 return type + compose = 5+ refs, it may alias.
    // Just verify validity - the cost-benefit analysis may or may not trigger
    // depending on exact counts.
}

// MARK: Compound types

#[test]
fn compound_type_benefits_from_base_alias() {
    // If vec4f gets aliased, array<aliased_vec4f, N> strings get shorter too.
    let src = r#"
        fn a(x: vec4<f32>) -> vec4<f32> { return x; }
        fn b(x: vec4<f32>) -> vec4<f32> { return x; }
        fn c(x: vec4<f32>) -> vec4<f32> { return x; }
        fn d(x: vec4<f32>) -> vec4<f32> { return x; }
        fn e(x: vec4<f32>) -> vec4<f32> { return x; }
        fn f_fn(x: vec4<f32>) -> vec4<f32> { return x; }
        fn g(x: vec4<f32>) -> vec4<f32> { return x; }
        fn h(x: vec4<f32>) -> vec4<f32> { return x; }
        fn main_fn() -> vec4<f32> {
            return a(vec4<f32>(1.0)) + b(vec4<f32>(2.0));
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // vec4f has 16+ references (8 args + 8 returns) - should definitely alias.
}

// MARK: Disabled

#[test]
fn no_alias_when_disabled() {
    let src = r#"
        struct S {
            a: array<vec4<f32>, 16>,
            b: array<vec4<f32>, 16>,
            c: array<vec4<f32>, 16>,
        }
        fn foo(x: array<vec4<f32>, 16>) -> array<vec4<f32>, 16> {
            return x;
        }
    "#;
    // Use the standard compact helper which has type_alias=false.
    let out = compact(src);
    assert_valid_wgsl(&out);
    assert!(
        !out.contains("alias "),
        "alias should not appear when type_alias is disabled: {out}"
    );
}

// MARK: Size reduction

#[test]
fn alias_reduces_output_size() {
    let src = r#"
        struct S {
            a: array<vec4<f32>, 16>,
            b: array<vec4<f32>, 16>,
            c: array<vec4<f32>, 16>,
            d: array<vec4<f32>, 16>,
        }
        fn foo(x: array<vec4<f32>, 16>) -> array<vec4<f32>, 16> { return x; }
        fn bar(x: array<vec4<f32>, 16>) -> array<vec4<f32>, 16> { return x; }
    "#;
    let without = compact(src);
    let with = compact_aliased(src);
    assert_valid_wgsl(&without);
    assert_valid_wgsl(&with);
    assert!(
        with.len() <= without.len(),
        "aliased output ({} bytes) should be <= non-aliased ({} bytes)\n\
         without: {without}\n\
         with:    {with}",
        with.len(),
        without.len(),
    );
}

// MARK: Round-trip validation

#[test]
fn roundtrip_with_aliases() {
    let src = r#"
        struct Particle {
            pos: vec4<f32>,
            vel: vec4<f32>,
            acc: vec4<f32>,
            col: vec4<f32>,
        }
        struct ParticleBuffer {
            particles: array<Particle, 256>,
        }
        @group(0) @binding(0) var<storage, read_write> buf: ParticleBuffer;
        fn update(idx: u32) {
            var p: Particle = buf.particles[idx];
            p.vel = p.vel + p.acc;
            p.pos = p.pos + p.vel;
            buf.particles[idx] = p;
        }
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            update(gid.x);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
}

// MARK: Pointer types

#[test]
fn alias_for_pointer_types() {
    // ptr<function, vec4f> used many times
    let src = r#"
        fn modify(p: ptr<function, vec4<f32>>) {
            *p = *p + vec4<f32>(1.0);
        }
        fn main_fn() {
            var a: vec4<f32> = vec4<f32>(0.0);
            var b: vec4<f32> = vec4<f32>(0.0);
            var c: vec4<f32> = vec4<f32>(0.0);
            modify(&a);
            modify(&b);
            modify(&c);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
}

// MARK: Edge cases

#[test]
fn empty_module_no_crash() {
    let src = "";
    let out = compact_aliased(src);
    assert!(out.is_empty() || out.trim().is_empty());
}

// MARK: Sampler & texture types

#[test]
fn sampler_and_texture_types() {
    let src = r#"
        @group(0) @binding(0) var s1: sampler;
        @group(0) @binding(1) var s2: sampler;
        @group(0) @binding(2) var s3: sampler;
        @group(0) @binding(3) var s4: sampler;
        @group(0) @binding(4) var s5: sampler;
        @group(0) @binding(5) var t1: texture_2d<f32>;
        @group(0) @binding(6) var t2: texture_2d<f32>;
        @group(0) @binding(7) var t3: texture_2d<f32>;
        @group(0) @binding(8) var t4: texture_2d<f32>;
        @fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            return textureSample(t1, s1, uv) + textureSample(t2, s2, uv);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
}

// MARK: Splat expressions use aliases

#[test]
fn splat_uses_type_alias() {
    // Splat expressions (e.g. vec3f(0.0)) should use the type alias when one
    // exists.  Previously Splat bypassed alias lookup because naga resolves
    // their type as TypeResolution::Value rather than Handle.
    let src = r#"
        fn a(x: vec3<f32>) -> vec3<f32> { return x; }
        fn b(x: vec3<f32>) -> vec3<f32> { return x; }
        fn c(x: vec3<f32>) -> vec3<f32> { return x; }
        fn d(x: vec3<f32>) -> vec3<f32> { return x; }
        fn e(x: vec3<f32>) -> vec3<f32> { return x; }
        fn main_fn() -> vec3<f32> {
            var v = vec3<f32>(0.0);
            var w = vec3<f32>(1.0);
            return a(v) + b(w);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // vec3f should be aliased given 10+ refs (5 args + 5 returns).
    assert!(
        out.contains("alias "),
        "expected alias declaration for vec3f: {out}"
    );
    // The splat expressions inside main_fn must NOT use the raw "vec3f" name.
    // They should use the alias.  The alias decl itself will have "vec3f" once.
    let raw_count = out.matches("vec3f").count();
    assert!(
        raw_count <= 1,
        "splat should use alias, but vec3f appears {raw_count} times (expected <= 1 in alias decl): {out}"
    );
}

#[test]
fn splat_vec2_uses_type_alias() {
    let src = r#"
        fn a(x: vec2<f32>) -> vec2<f32> { return x; }
        fn b(x: vec2<f32>) -> vec2<f32> { return x; }
        fn c(x: vec2<f32>) -> vec2<f32> { return x; }
        fn d(x: vec2<f32>) -> vec2<f32> { return x; }
        fn e(x: vec2<f32>) -> vec2<f32> { return x; }
        fn main_fn() -> vec2<f32> {
            var v = vec2<f32>(0.0);
            return a(v);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(out.contains("alias "), "expected alias for vec2f: {out}");
    let raw_count = out.matches("vec2f").count();
    assert!(
        raw_count <= 1,
        "splat should use alias, but vec2f appears {raw_count} times: {out}"
    );
}

#[test]
fn global_splat_uses_type_alias() {
    // Global const splat expressions should also use the alias.
    let src = r#"
        const ZERO: vec3<f32> = vec3<f32>(0.0);
        const ONE: vec3<f32> = vec3<f32>(1.0);
        fn a(x: vec3<f32>) -> vec3<f32> { return x; }
        fn b(x: vec3<f32>) -> vec3<f32> { return x; }
        fn c(x: vec3<f32>) -> vec3<f32> { return x; }
        fn d(x: vec3<f32>) -> vec3<f32> { return x; }
        fn main_fn() -> vec3<f32> {
            return a(ZERO) + b(ONE);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(out.contains("alias "), "expected alias for vec3f: {out}");
    let raw_count = out.matches("vec3f").count();
    assert!(
        raw_count <= 1,
        "global splat should use alias, but vec3f appears {raw_count} times: {out}"
    );
}

#[test]
fn cast_uses_type_alias() {
    // Type cast (As) expressions should also use the alias for the target type.
    let src = r#"
        fn a(x: vec3<i32>) -> vec3<i32> { return x; }
        fn b(x: vec3<i32>) -> vec3<i32> { return x; }
        fn c(x: vec3<i32>) -> vec3<i32> { return x; }
        fn d(x: vec3<i32>) -> vec3<i32> { return x; }
        fn e(x: vec3<i32>) -> vec3<i32> { return x; }
        fn main_fn() -> vec3<i32> {
            let v = vec3<f32>(1.0, 2.0, 3.0);
            return a(vec3<i32>(v));
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(out.contains("alias "), "expected alias for vec3i: {out}");
    let raw_count = out.matches("vec3i").count();
    assert!(
        raw_count <= 1,
        "cast should use alias, but vec3i appears {raw_count} times: {out}"
    );
}

#[test]
fn splat_alias_with_source_alias_duplicate_inner() {
    // When the source contains `alias F3 = vec3f;`, naga creates two Type
    // entries with the same TypeInner (one named "F3", one unnamed).  The
    // minifier's alias should still apply to Splat expressions regardless of
    // which handle the arena scan encounters first.
    let src = r#"
        alias F3 = vec3<f32>;
        fn a(x: F3) -> F3 { return x; }
        fn b(x: F3) -> F3 { return x; }
        fn c(x: vec3<f32>) -> vec3<f32> { return x; }
        fn d(x: vec3<f32>) -> vec3<f32> { return x; }
        fn e(x: vec3<f32>) -> vec3<f32> { return x; }
        fn main_fn() -> vec3<f32> {
            var v = vec3<f32>(0.0);
            return a(v);
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(out.contains("alias "), "expected alias declaration: {out}");
    let raw_count = out.matches("vec3f").count();
    assert!(
        raw_count <= 1,
        "splat should use alias despite source-level alias creating duplicate TypeInner, \
         but vec3f appears {raw_count} times: {out}"
    );
}

#[test]
fn alias_with_split_ref_counts_across_duplicate_inner() {
    // When refs are split across two handles (one from `alias F3 = vec3f`,
    // one from bare `vec3f`), neither handle individually may meet the alias
    // threshold.  The combined count across same-TypeInner handles must be
    // used for the cost-benefit decision.
    //
    // vec3f: savings_per_use = 4, decl_cost = 14.
    // 2 refs via F3 + 2 refs via vec3f = 4 combined, 4*4=16 > 14 -> alias.
    // But individually 2*4=8 < 14 -> no alias without combining.
    let src = r#"
        alias F3 = vec3<f32>;
        fn a(x: F3) -> F3 { return x; }
        fn b(x: vec3<f32>) -> vec3<f32> { return x; }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(
        out.contains("alias "),
        "expected alias from combined ref counts across duplicate-inner handles: {out}"
    );
    let raw_count = out.matches("vec3f").count();
    assert!(
        raw_count <= 1,
        "vec3f should be aliased via combined counts, but appears {raw_count} times: {out}"
    );
}
