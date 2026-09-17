//! `alias T = ...;` introduction: profitability thresholds, mangling
//! interaction, which type categories participate, and round-trip validity.

use super::helpers::{assert_valid_wgsl, compact, compact_aliased, compact_mangled_aliased};
use crate::config::Config;

// MARK: Priced by the text

/// The shipped render prices an alias by the spellings the tail's render
/// produced (`GenerateOptions::type_uses`), not the IR census: four
/// constructors a pinning component lets the text spell `vec3(` are 12
/// bytes of alias use against a 14-byte declaration - the census, counting
/// four `vec3f`, declares it and the text grows by two.  The census alone
/// (`compact_aliased`) still declares it.
#[test]
fn an_alias_is_priced_by_what_the_text_spells() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          let x = f32(id.x);\
          o[0] = dot(vec3f(x, 1.0, 2.0), vec3f(3.0, x, 4.0)) + dot(vec3f(5.0, 6.0, x), vec3f(x, x, 7.0));\
        }";
    let census = compact_aliased(src);
    assert!(census.contains("alias"), "{census}");
    let out = crate::run(src, &Config::default())
        .expect("run failed")
        .source;
    assert!(
        !out.contains("alias") && out.contains("vec3(B,1,2)"),
        "{out}"
    );
    assert_valid_wgsl(&out);
}

/// The census counts constructors and declarations only; a conversion
/// (`vec3f(v)`) spells the type as well, and four of them beside one
/// parameter pay for the alias the census never saw.
#[test]
fn a_conversion_counts_as_a_spelling_of_its_type() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\
        fn f(v: vec3f) -> f32 { return v.x * v.y - v.z; }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          o[0] = f(vec3f(id)) + f(vec3f(id + 1)) * f(vec3f(id * 2)) - f(vec3f(id ^ vec3u(3)));\
        }";
    let census = compact_aliased(src);
    assert!(!census.contains("alias"), "{census}");
    let out = crate::run(src, &Config::default())
        .expect("run failed")
        .source;
    assert!(
        out.contains("alias C=vec3f;") && out.contains("A(C(a))"),
        "{out}"
    );
    assert_valid_wgsl(&out);
}

/// naga keeps an alias as a named type, and a bare `array(...)` takes its
/// element type from the first component's: a splat (`D(1)`) resolves to
/// the anonymous `vec4f`, a full constructor or `D()` to `D`, so one `var`
/// stored from both forms fails validation.  With `vec4f` aliased and the
/// array type not, the first form keeps its typed constructor.
#[test]
fn a_bare_array_constructor_keeps_its_type_under_an_aliased_element() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        fn f(a: array<vec4f, 2>) -> vec4f { return a[0] + a[1]; }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          let x = f32(id.x);\
          var b = array<vec4f, 2>(vec4f(1.0), vec4f());\
          if x > 2.0 { b = array<vec4f, 2>(vec4f(x, x, 1.0, 2.0), vec4f(x, 1.0, 2.0, 3.0)); }\
          o[0] = f(b) + vec4f(x, 0.0, 1.0, 2.0) * vec4f(3.0, x, x, x) - vec4f(vec3f(id), 1.0) + vec4f(x);\
        }";
    let out = crate::run(src, &Config::default()).expect("run failed");
    assert!(
        out.report.fallback.is_none(),
        "the generator's text must pass the self-check: {}",
        out.source
    );
    assert!(
        out.source.contains("array<b,2>(b(1),b())") && out.source.contains("=array(b(C,C,1,2)"),
        "{}",
        out.source
    );
    assert_valid_wgsl(&out.source);
}

/// A preamble owns its declarations; the text never spells their types, so
/// an alias for one is a declaration used nowhere.  The tail's render
/// leaves them out as the shipped one does.
#[test]
fn a_type_only_the_preamble_spells_gets_no_alias() {
    let preamble = "@group(0) @binding(0) var t: texture_2d<f32>;\
        @group(0) @binding(1) var s: sampler;\
        @group(0) @binding(2) var u: texture_2d<f32>;\
        @group(0) @binding(3) var v: texture_2d<f32>;";
    let src = "@fragment fn main(@location(0) p: vec2f) -> @location(0) vec4f {\
        return textureSample(t, s, p) + textureSample(u, s, p) + textureSample(v, s, p); }";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Config::default()
    };
    let out = crate::run(src, &config).expect("run failed").source;
    assert!(!out.contains("alias"), "{out}");
}

// MARK: Basic alias thresholds

/// An alias dodges only the locals of functions that spell its type
/// (`plan_type_aliases`), so a same-named local elsewhere leaves it the
/// shortest free name.
#[test]
fn alias_ignores_locals_of_functions_not_spelling_the_type() {
    let src = r#"
        fn f(p: f32) -> f32 { var A = p; var a = A * 2.0; return a; }
        fn g(q: vec4<f32>, r: vec4<f32>) -> vec4<f32> {
            let v = vec4<f32>(q.x, r.y, 0.0, 1.0);
            let w = vec4<f32>(v.x, v.y, 0.0, 1.0);
            return v + w + q * r;
        }
        @fragment fn m() -> @location(0) vec4<f32> {
            let s = f(1.0);
            return g(vec4<f32>(s, s, 0.0, 1.0), vec4<f32>(s, 0.0, s, 1.0));
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(
        out.contains("alias A=vec4f"),
        "`A` is a local only in f, which never spells vec4f: {out}"
    );
    // The same local inside a spelling function shadows it.
    let src = src.replace("let v = vec4<f32>(q.x", "var A = q.x; let v = vec4<f32>(A");
    let out = compact_aliased(&src);
    assert_valid_wgsl(&out);
    assert!(
        out.contains("alias a=vec4f"),
        "the alias must dodge g's own local `A`: {out}"
    );
}

/// Dead locals are never printed, so their type's refs must not count toward
/// the alias break-even or the `alias` itself is dead text.
#[test]
fn no_alias_for_type_used_only_by_dead_locals() {
    let src = r#"
        fn helper() {
            var a: vec4<f32>;
            var b: vec4<f32>;
            var c: vec4<f32>;
            var d: vec4<f32>;
            var e: vec4<f32>;
            var f: vec4<f32>;
        }
        @compute @workgroup_size(1) fn main() { helper(); }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    assert!(
        !out.contains("alias"),
        "vec4<f32> appears only in dead locals; it must not be aliased: {out}"
    );
}

#[test]
fn no_alias_when_single_use() {
    let src = r#"
        fn foo() -> array<vec4<f32>, 16> {
            return array<vec4<f32>, 16>();
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
    // Near the break-even, so only validity is asserted.
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
    assert!(
        out.contains("alias "),
        "expected alias declaration in output: {out}"
    );
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
    let src = r#"
        fn foo(a: f32, b: f32) -> f32 {
            return a + b;
        }
    "#;
    let out = compact_aliased(src);
    assert_valid_wgsl(&out);
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
    // Near the break-even, so only validity is asserted.
}

// MARK: Compound types

#[test]
fn compound_type_benefits_from_base_alias() {
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
    // `compact` runs with `type_alias` off.
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
    // A Splat's type resolves as `TypeResolution::Value`, not a handle, so the
    // alias lookup cannot key on the handle alone.
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
    assert!(
        out.contains("alias "),
        "expected alias declaration for vec3f: {out}"
    );
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
    // A source-level `alias F3 = vec3f;` gives naga two Type entries with one
    // TypeInner; the alias must apply whichever handle the arena scan meets
    // first.
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
    // Refs split across the two same-TypeInner handles (2 via F3 + 2 bare)
    // each miss the threshold (2*4=8 < decl cost 14); the combined 4*4=16 > 14
    // must alias.
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
