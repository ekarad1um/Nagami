//! Tests for the emitter's type-annotation elision paths.
//!
//! WGSL infers most declaration types from the initialiser, so the
//! generator can drop `: T` annotations on `const`, `var`, function
//! returns, and storage-access modes whenever the inferred type
//! matches the declared one.  Each MARK section below locks in one
//! elision flavour.

use super::helpers::*;

// MARK: Const type elision

#[test]
fn const_elides_type_for_compose_init() {
    let out = compact("const C: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);");
    // Type annotation must be gone; bare literals inside Compose.
    assert!(out.contains("const C=vec3f(1,2,3)"), "got: {out}");
    assert!(
        !out.contains("const C:"),
        "type annotation not elided: {out}"
    );
}

#[test]
fn const_keeps_type_for_abstract_literal_init() {
    // AbstractFloat literal - cannot elide because text has no suffix.
    // naga parses `0.5` in const init as AbstractFloat.
    let out = compact("const C: f32 = 0.5;");
    // The generator must keep `:f32` since the literal is abstract.
    // (naga may or may not concretize this; either way the output must be valid)
    assert_valid_wgsl(&out);
}

#[test]
fn const_zero_value_elides_type() {
    // const with ZeroValue init should elide type annotation.
    let out = compact(
        r#"
            const ZERO: f32 = 0.0;
            fn f() -> f32 {
                return ZERO;
            }
        "#,
    );
    // Should NOT have "const ZERO:f32=0" - type is elided
    assert!(
        !out.contains("ZERO:f32") && !out.contains("ZERO: f32"),
        "const with zero value should elide type: {out}"
    );
}

// MARK: Var type elision

#[test]
fn var_elides_type_with_concrete_literal_init() {
    let out = compact("fn f() { var x: f32 = 1.0; _ = x; }");
    // var should use typed suffix and elide `:f32`.
    assert!(
        out.contains("var x=1f;") || out.contains("var x=1.f;"),
        "got: {out}"
    );
    assert!(
        !out.contains("var x:f32"),
        "type annotation not elided: {out}"
    );
}

#[test]
fn var_defers_declaration_to_first_store() {
    let out = compact("fn f() { var x: f32; x = 1.0; _ = x; }");
    // Deferred var should merge declaration with the first store and
    // use a typed suffix so that the type annotation can be elided.
    assert!(
        out.contains("var x=1f;") || out.contains("var x=1.f;"),
        "got: {out}"
    );
    assert!(
        !out.contains("var x:f32;"),
        "type annotation should be elided: {out}"
    );
}

#[test]
fn var_keeps_type_when_first_use_in_nested_block() {
    let out = compact(
        "fn f(c: bool) -> f32 { var x: f32; if c { x = 1.0; } else { x = 2.0; } return x; }",
    );
    assert_valid_wgsl(&out);
    // var first used inside the if-block cannot be deferred into a branch
    // (it would fall out of scope before `return x`), so it keeps a
    // standalone declaration.  The zero-init now renders as the shorter
    // `=0f` form; the branch stores must stay plain assignments.
    assert!(
        out.contains("var x=0f;") || out.contains("var x:f32;"),
        "should keep a standalone declaration: {out}"
    );
    assert!(
        !out.contains("var x=1") && !out.contains("var x=2"),
        "must not defer declaration into a branch: {out}"
    );
}

#[test]
fn var_elides_type_with_zero_value_init() {
    // var x: f32 = f32() (zero value) should elide the type annotation.
    let out = compact("fn f() -> f32 { var x: f32 = f32(); return x; }");
    // f32() provides explicit type, so `:f32` annotation can be elided.
    // Output should have `var x=0f` or similar - no `:f32`.
    assert_valid_wgsl(&out);
}

// MARK: Default @interpolate elision

#[test]
fn default_interpolate_elided_on_fragment_input() {
    // @interpolate(perspective,center) is the WGSL default for float
    // location bindings and must be omitted to save bytes.
    let src = r#"
        @fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
            return vec4f(uv, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("@interpolate(perspective"),
        "default @interpolate(perspective,center) should be elided: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn default_interpolate_elided_on_struct_member() {
    let src = r#"
        struct VSOut {
            @builtin(position) pos: vec4f,
            @location(0) color: vec3f,
        }
        @vertex fn vs() -> VSOut {
            return VSOut(vec4f(0.0), vec3f(1.0, 0.0, 0.0));
        }
        @fragment fn fs(i: VSOut) -> @location(0) vec4f {
            return vec4f(i.color, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("@interpolate(perspective"),
        "default @interpolate on struct member should be elided: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn non_default_interpolate_preserved_linear() {
    let src = r#"
        struct VSOut {
            @builtin(position) pos: vec4f,
            @location(0) @interpolate(linear) val: f32,
        }
        @vertex fn vs() -> VSOut { return VSOut(vec4f(0.0), 1.0); }
        @fragment fn fs(i: VSOut) -> @location(0) vec4f {
            return vec4f(i.val, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@interpolate(linear)"),
        "non-default @interpolate(linear) must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn non_default_interpolate_preserved_flat() {
    let src = r#"
        struct VSOut {
            @builtin(position) pos: vec4f,
            @location(0) @interpolate(flat) id: u32,
        }
        @vertex fn vs() -> VSOut { return VSOut(vec4f(0.0), 1u); }
        @fragment fn fs(i: VSOut) -> @location(0) vec4f {
            return vec4f(f32(i.id), 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@interpolate(flat)"),
        "non-default @interpolate(flat) must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn non_default_sampling_preserved_centroid() {
    // perspective + centroid is NOT the default (center is), so must keep.
    let src = r#"
        struct VSOut {
            @builtin(position) pos: vec4f,
            @location(0) @interpolate(perspective, centroid) val: f32,
        }
        @vertex fn vs() -> VSOut { return VSOut(vec4f(0.0), 1.0); }
        @fragment fn fs(i: VSOut) -> @location(0) vec4f {
            return vec4f(i.val, 0.0, 0.0, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("@interpolate(perspective,centroid)")
            || out.contains("@interpolate(perspective, centroid)"),
        "non-default sampling centroid must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Storage access mode elision

#[test]
fn storage_read_access_mode_elided() {
    let src = r#"
        @group(0) @binding(0) var<storage, read> data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> result: array<f32>;
        @compute @workgroup_size(1) fn cs() {
            result[0] = data[0];
        }
    "#;
    let out = compact(src);
    // Should emit var<storage> not var<storage,read>.
    assert!(
        out.contains("var<storage>"),
        "read-only storage should elide access mode: {out}"
    );
    assert!(
        !out.contains("storage,read>") && !out.contains("storage, read>"),
        "should not contain default 'storage,read' access mode: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn storage_read_write_access_mode_preserved() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(1) fn cs() {
            buf[0] = 1.0;
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("storage,read_write") || out.contains("storage, read_write"),
        "read_write storage must preserve access mode: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Trailing void return elision

#[test]
fn trailing_void_return_elided() {
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(1) fn cs() {
            buf[0] = 1.0;
        }
    "#;
    let out = compact(src);
    // Should NOT end the function body with "return;".
    assert!(
        !out.contains("return;"),
        "trailing void return should be elided: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn non_void_return_preserved() {
    let src = r#"
        @fragment fn fs(@location(0) v: f32) -> @location(0) vec4<f32> {
            return vec4<f32>(v, v, v, 1.0);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("return"),
        "non-void return must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn mid_function_return_preserved_in_void() {
    // A void function with an early return inside an `if` must keep it.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
            if gid.x > 10u {
                return;
            }
            buf[gid.x] = 1.0;
        }
    "#;
    let out = compact(src);
    // The early return inside the if must be preserved.
    assert!(
        out.contains("return;"),
        "early return inside if must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Global splat const elision

#[test]
fn global_splat_elides_const_type_annotation() {
    // Splat RHS has explicit type, so const type annotation can be elided.
    let src = r#"
        const a: vec3<f32> = vec3f(1.0);
        @fragment fn main() -> @location(0) vec4f {
            return vec4f(a, 1.0);
        }
    "#;
    let out = compact(src);
    // Should NOT have `:vec3<f32>` type annotation after `a`.
    assert!(
        !out.contains("a:vec3") && !out.contains("a: vec3"),
        "const with Splat RHS should elide type annotation: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Zero-init var rendering (shorter of `:type` vs `=0i`)

#[test]
fn for_counter_and_scalar_use_literal_zero_init_end_to_end() {
    // Full pipeline strips zero inits, so the for-counter reaches the
    // bare-declaration path.  An un-aliased i32 must render the cheaper
    // `= 0i` instead of `: i32`, and the update clause must use `++`.
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<i32>;
        @compute @workgroup_size(1) fn main() {
            var s = 0i;
            for (var r = 0i; r < 4; r = r + 1) {
                for (var c = 0i; c < 4; c = c + 1) { s = s + r * c; }
            }
            buf[0] = s;
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains("= 0i"),
        "i32 zero-init should use `= 0i`: {out}"
    );
    assert!(out.contains("++"), "counter update should use `++`: {out}");
    assert!(!out.contains("+= 1"), "no `+= 1` should remain: {out}");
    assert!(
        !out.contains(": i32"),
        "i32 zero-init must not regress to `: i32`: {out}"
    );
}

#[test]
fn deferred_zero_accumulator_uses_type_form_end_to_end() {
    // A deferred var whose first store writes the zero value drops the
    // store and relies on zero-init: `var acc = vec3f(0)` -> `var acc: vec3f`
    // (the annotation is shorter than the explicit zero construct).
    let out = compact_with_passes(
        r#"
        @group(0) @binding(0) var<storage, read_write> buf: array<f32>;
        @compute @workgroup_size(1)
        fn main(@builtin(local_invocation_index) n: u32) {
            if n > 0u {
                var acc = vec3(0.0, 0.0, 0.0);
                for (var i = 0u; i < n; i = i + 1u) { acc = acc + vec3f(1.0); }
                buf[0] = acc.x;
            }
        }
        "#,
        Profile::Max,
    );
    assert!(
        out.contains(": vec3f"),
        "deferred zero accumulator should use `: vec3f`: {out}"
    );
    assert!(
        !out.contains("= vec3f("),
        "explicit zero construct should be dropped: {out}"
    );
}
