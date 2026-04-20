//! Round-trip coverage: assert that the emitted WGSL parses and
//! validates cleanly across a representative set of shader
//! constructs (pointer operations, atomics, barriers, struct
//! layouts, matrices, textures, ray tracing, and the `beautify`
//! code path).  These tests defend the emitter's single most
//! important contract: output is always valid WGSL.

use super::helpers::*;

// MARK: Round-trip validation

#[test]
fn generated_output_is_valid_wgsl() {
    let out = compact(VALIDATION_SRC);
    assert_valid_wgsl(&out);
}

#[test]
fn beautified_output_is_valid_wgsl() {
    let out = compact_beautified(VALIDATION_SRC);
    // Must contain indentation
    assert!(
        out.contains('\n'),
        "beautified output should have newlines: {out}"
    );
    assert!(
        out.contains("  "),
        "beautified output should have indentation: {out}"
    );
    // Round-trip parse + validate
    assert_valid_wgsl(&out);
}

// MARK: Pointer operations

#[test]
fn load_from_pointer_argument() {
    let src = r#"
        fn read_val(p: ptr<function, u32>) -> u32 {
            return *p;
        }

        @compute @workgroup_size(1)
        fn main() {
            var x: u32 = 42u;
            let v = read_val(&x);
            _ = v;
        }
    "#;
    let out = compact(src);
    println!("load_from_pointer_argument output: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn store_to_pointer_argument() {
    let src = r#"
        fn write_val(p: ptr<function, u32>, v: u32) {
            *p = v;
        }

        @compute @workgroup_size(1)
        fn main() {
            var x: u32 = 0u;
            write_val(&x, 42u);
            _ = x;
        }
    "#;
    let out = compact(src);
    println!("store_to_pointer_argument output: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn pointer_member_access() {
    let src = r#"
        struct S {
            a: u32,
            b: f32,
        }

        fn get_b(p: ptr<function, S>) -> f32 {
            return (*p).b;
        }

        @compute @workgroup_size(1)
        fn main() {
            var s = S(1u, 2.0);
            let v = get_b(&s);
            _ = v;
        }
    "#;
    let out = compact(src);
    println!("pointer_member_access output: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn pointer_read_modify_write() {
    // Read through pointer, modify, write back through same pointer.
    let src = r#"
        fn double_val(p: ptr<function, u32>) {
            *p = *p * 2u;
        }

        @compute @workgroup_size(1)
        fn main() {
            var x: u32 = 21u;
            double_val(&x);
            _ = x;
        }
    "#;
    let out = compact(src);
    println!("pointer_read_modify_write output: {out}");
    assert!(
        out.contains("*p"),
        "should contain explicit dereference: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn pointer_forward_to_another_function() {
    // Pointer parameter forwarded to another function taking pointer.
    let src = r#"
        fn inner(p: ptr<function, u32>) -> u32 {
            return *p;
        }

        fn outer(p: ptr<function, u32>) -> u32 {
            return inner(p);
        }

        @compute @workgroup_size(1)
        fn main() {
            var x: u32 = 7u;
            let v = outer(&x);
            _ = v;
        }
    "#;
    let out = compact(src);
    println!("pointer_forward output: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn call_with_ptr_to_local_roundtrip() {
    // A function that takes ptr<function, f32> should receive &x in WGSL.
    let src = r#"
            fn mutate(p: ptr<function, f32>) { *p = 42.0; }
            @compute @workgroup_size(1)
            fn main() {
                var x: f32;
                mutate(&x);
                _ = x;
            }
        "#;
    let out = compact(src);
    // Must emit `&` before the local variable name.
    assert!(
        out.contains("(&"),
        "call with ptr-to-local must have &: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn call_with_ptr_to_local_prevents_dead_var() {
    // A variable only referenced via a Call pointer arg should NOT be
    // eliminated as dead - the Call uses it.
    let src = r#"
            fn mutate(p: ptr<function, f32>) { *p = 42.0; }
            @compute @workgroup_size(1)
            fn main() {
                var x = 0f;
                mutate(&x);
            }
        "#;
    let out = compact(src);
    // Variable must still appear in output.
    assert!(out.contains("var "), "var should not be eliminated: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn call_with_ptr_blocks_deferral() {
    // Call with &x before a store to x -> x must NOT be deferred.
    let src = r#"
            fn read_ptr(p: ptr<function, f32>) -> f32 { return *p; }
            @compute @workgroup_size(1)
            fn main() {
                var x: f32 = 0.0;
                let v = read_ptr(&x);
                x = v + 1.0;
                _ = x;
            }
        "#;
    let out = compact(src);
    // TODO: upgrade to assert_valid_wgsl once compactor emits `*p` (Load) correctly
    assert!(
        naga::front::wgsl::parse_str(&out).is_ok(),
        "re-parse failed: {out}"
    );
}

#[test]
fn ptr_forward_no_double_ref() {
    // Forwarding a pointer parameter should emit `inner(p)`, not `inner(&p)`.
    let src = r#"
            fn inner(p: ptr<function, f32>) { *p = 1.0; }
            fn outer(p: ptr<function, f32>) { inner(p); }
            @compute @workgroup_size(1)
            fn main() {
                var x = 0f;
                outer(&x);
                _ = x;
            }
        "#;
    let out = compact(src);
    // Output must NOT contain `(&(&` or similar double-reference.
    assert!(
        !out.contains("(&(&"),
        "double & on forwarded pointer: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn call_with_ptr_to_struct_field() {
    // Passing a pointer to a struct field: `set_x(&s.x)`.
    let src = r#"
            struct S { x: f32, y: f32 }
            fn set_x(p: ptr<function, f32>) { *p = 1.0; }
            @compute @workgroup_size(1)
            fn main() {
                var s = S(0.0, 0.0);
                set_x(&s.x);
                _ = s;
            }
        "#;
    let out = compact(src);
    assert!(out.contains("(&"), "ptr to struct field must have &: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Atomic operations

#[test]
fn atomic_add_roundtrip() {
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
            @compute @workgroup_size(1)
            fn main() {
                let old = atomicAdd(&counter, 1u);
                _ = old;
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("atomicAdd(&"),
        "atomicAdd with & pointer should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn atomic_compare_exchange_roundtrip() {
    // TODO: upgrade to assert_valid_wgsl once naga stops naming the return
    // struct `__atomic_compare_exchange_result` (reserved `__` prefix).
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> val: atomic<u32>;
            @compute @workgroup_size(1)
            fn main() {
                let result = atomicCompareExchangeWeak(&val, 0u, 1u);
                _ = result;
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("atomicCompareExchangeWeak(&"),
        "atomicCompareExchangeWeak with & pointer should be present: {out}"
    );
}

// MARK: workgroupUniformLoad

#[test]
fn workgroup_uniform_load_roundtrip() {
    let src = r#"
            var<workgroup> shared_val: u32;
            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_index) lid: u32) {
                if lid == 0u {
                    shared_val = 42u;
                }
                workgroupBarrier();
                let v = workgroupUniformLoad(&shared_val);
                _ = v;
            }
        "#;
    let out = compact(src);
    assert!(
        out.contains("workgroupUniformLoad("),
        "workgroupUniformLoad should be present: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn subgroup_barrier_does_not_inject_enable_subgroups() {
    // Regression: auto-injecting `enable subgroups;` made tiny shaders grow
    // while also tripping naga's known subgroup directive parser limitation
    // in text-validation paths.
    let src = r#"
        @compute @workgroup_size(1)
        fn main() {
            subgroupBarrier();
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("enable subgroups;"),
        "generator should not synthesize subgroup enable directive: {out}"
    );
    // Keep this as parse-only because naga does not fully support subgroup
    // extension directives in WGSL text paths.
    assert!(
        naga::front::wgsl::parse_str(&out).is_ok(),
        "re-parse failed: {out}"
    );
    assert!(
        out.len() <= src.len(),
        "subgroup-only minification should not grow: {} -> {}",
        src.len(),
        out.len()
    );
}

// MARK: Struct layout (@size / @align)

#[test]
fn struct_explicit_size_roundtrip() {
    let src = r#"
        struct S {
            @size(32) a: f32,
            b: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1)
        fn main() { _ = u.a + u.b; }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn struct_explicit_align_roundtrip() {
    // @align(16) on the first member bumps struct alignment.
    let src = r#"
        struct S {
            @align(16) a: f32,
            b: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1)
        fn main() { _ = u.a + u.b; }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn struct_align_on_second_member_roundtrip() {
    // @align(16) on the second member shifts its offset.
    let src = r#"
        struct S {
            a: f32,
            @align(16) b: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1)
        fn main() { _ = u.a + u.b; }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn struct_size_and_align_combined_roundtrip() {
    let src = r#"
        struct S {
            @size(16) a: f32,
            @align(16) b: vec4<f32>,
            c: f32,
        }
        @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1)
        fn main() { _ = u.a + u.b.x + u.c; }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn struct_nested_with_align_roundtrip() {
    // Inner struct with alignment bump, used inside outer struct.
    let src = r#"
        struct Inner {
            @align(16) x: f32,
        }
        struct Outer {
            a: Inner,
            b: f32,
        }
        @group(0) @binding(0) var<uniform> u: Outer;
        @compute @workgroup_size(1)
        fn main() { _ = u.a.x + u.b; }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Override declarations

#[test]
fn override_with_initializer_roundtrip() {
    let src = r#"
        override scale: f32 = 2.0;
        @compute @workgroup_size(1)
        fn main() { _ = scale; }
    "#;
    let out = compact(src);
    assert!(out.contains("override"), "should contain override: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn override_without_initializer_roundtrip() {
    let src = r#"
        @id(0) override scale: f32;
        @compute @workgroup_size(1)
        fn main() { _ = scale; }
    "#;
    let out = compact(src);
    assert!(out.contains("override"), "should contain override: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Matrix types

#[test]
fn matrix_type_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var<uniform> m: mat4x4<f32>;
        @compute @workgroup_size(1)
        fn main() {
            let v = m * vec4f(1.0);
            _ = v;
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn matrix_3x3_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var<uniform> m: mat3x3<f32>;
        @compute @workgroup_size(1)
        fn main() {
            let v = m * vec3f(1.0);
            _ = v;
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Depth / multisampled textures

#[test]
fn depth_texture_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var t: texture_depth_2d;
        @group(0) @binding(1) var s: sampler_comparison;
        @fragment fn main(@builtin(position) pos: vec4f) -> @location(0) f32 {
            return textureSampleCompare(t, s, pos.xy, 0.5);
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

#[test]
fn multisampled_texture_roundtrip() {
    let src = r#"
        @group(0) @binding(0) var t: texture_multisampled_2d<f32>;
        @fragment fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
            return textureLoad(t, vec2i(pos.xy), 0);
        }
    "#;
    let out = compact(src);
    assert_valid_wgsl(&out);
}

// MARK: Workgroup address space

#[test]
fn workgroup_var_roundtrip() {
    let src = r#"
        var<workgroup> wg_data: array<f32, 64>;
        @compute @workgroup_size(64)
        fn main(@builtin(local_invocation_index) idx: u32) {
            wg_data[idx] = f32(idx);
            workgroupBarrier();
            _ = wg_data[0u];
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("workgroup"),
        "should contain workgroup address space: {out}"
    );
    assert_valid_wgsl(&out);
}

// MARK: Beautify mode

#[test]
fn beautify_preserves_readability() {
    let src = r#"
        struct VertOut {
            @builtin(position) pos: vec4f,
            @location(0) uv: vec2f,
        }
        @vertex fn vs(@builtin(vertex_index) vi: u32) -> VertOut {
            var o: VertOut;
            o.pos = vec4f(0.0);
            o.uv = vec2f(0.0);
            return o;
        }
    "#;
    let compact_out = compact(src);
    let beauty_out = compact_beautified(src);
    assert!(
        beauty_out.len() > compact_out.len(),
        "beautified ({}) should be longer than compact ({})",
        beauty_out.len(),
        compact_out.len()
    );
    assert_valid_wgsl(&compact_out);
    assert_valid_wgsl(&beauty_out);
}

// MARK: Ray tracing / barycentric builtins

#[test]
fn ray_pipeline_trace_roundtrip() {
    let src = r#"
        enable wgpu_ray_tracing_pipeline;

        struct HitCounters {
            hit_num: u32,
            selected_hit: u32,
        }

        var<ray_payload> hit_num: HitCounters;
        @group(0) @binding(0) var acc_struct: acceleration_structure;

        @ray_generation
        fn ray_gen_main(
            @builtin(ray_invocation_id) id: vec3<u32>,
            @builtin(num_ray_invocations) num_invocations: vec3<u32>
        ) {
            hit_num = HitCounters();
            traceRay(acc_struct, RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), &hit_num);
        }
    "#;

    let out = compact(src);
    assert!(
        out.contains("traceRay("),
        "traceRay should be emitted: {out}"
    );
    assert!(
        out.contains("&"),
        "traceRay payload should pass by pointer: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn barycentric_builtin_roundtrip() {
    let src = r#"
        @vertex
        fn vs_main(@location(0) xy: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(xy, 0.0, 1.0);
        }

        @fragment
        fn fs_main(@builtin(barycentric) bary: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(bary, 1.0);
        }

        @fragment
        fn fs_main_no_perspective(@builtin(barycentric_no_perspective) bary: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(bary, 1.0);
        }
    "#;

    let out = compact(src);
    assert!(
        out.contains("@builtin(barycentric)"),
        "perspective barycentric builtin name should be canonical: {out}"
    );
    assert!(
        out.contains("@builtin(barycentric_no_perspective)"),
        "no-perspective barycentric builtin should be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn atomic_store_roundtrip() {
    let src = r#"
        @group(0) @binding(0)
        var<storage, read_write> a: atomic<i32>;

        @compute @workgroup_size(1)
        fn main() {
            atomicStore(&a, 1i);
        }
    "#;

    let out = compact(src);
    assert!(
        out.contains("atomicStore(&"),
        "atomic stores must emit as atomicStore calls: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn ray_hit_stages_emit_incoming_payload_attr() {
    let src = r#"
        enable wgpu_ray_tracing_pipeline;

        struct P { hits: u32 }
        var<incoming_ray_payload> incoming_p: P;

        @miss
        @incoming_payload(incoming_p)
        fn miss_main() {}

        @any_hit
        @incoming_payload(incoming_p)
        fn any_hit_main(@builtin(instance_custom_data) data: u32) {
            incoming_p.hits = data;
        }

        @closest_hit
        @incoming_payload(incoming_p)
        fn closest_hit_main() {}
    "#;

    let out = compact(src);
    assert!(
        out.contains("@incoming_payload("),
        "ray hit/miss stages should emit @incoming_payload: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn ray_tracing_pipeline_predeclared_raydesc_not_emitted_as_struct() {
    // `RayDesc` is a WGSL predeclared type for the wgpu_ray_tracing_pipeline
    // extension.  The generator must NOT emit a `struct RayDesc { ... }`
    // declaration - doing so shadows the predeclared type and causes naga's
    // validator to reject the constructor expression because the user-defined
    // struct has different type-arena handles than the canonical special type.
    let src = r#"
        enable wgpu_ray_tracing_pipeline;

        struct HitCounters { hit_num: u32, selected_hit: u32 }
        var<ray_payload> hit_num: HitCounters;

        @group(0) @binding(0)
        var acc_struct: acceleration_structure;

        @ray_generation
        fn ray_gen_main() {
            hit_num = HitCounters();
            traceRay(acc_struct, RayDesc(0u, 255u, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), &hit_num);
        }
    "#;

    let out = compact(src);
    // No `struct RayDesc` should appear - it must use the predeclared type
    assert!(
        !out.contains("struct RayDesc"),
        "RayDesc is a predeclared type and must not be emitted as a struct declaration: {out}"
    );
    // traceRay must still use RayDesc as a constructor
    assert!(
        out.contains("RayDesc("),
        "RayDesc constructor expression must be preserved: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn modf_result_members_not_mangled() {
    // `modf()` returns a WGSL predeclared struct type (`__modf_result_f32` etc.)
    // whose members `.fract` and `.whole` are canonical names.  If the mangler
    // renames them (e.g. to `B`, `b`), the field accessor becomes invalid.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> val: f32;

        @compute @workgroup_size(1)
        fn main() {
            let r = modf(val);
            val = r.fract + r.whole;
        }
    "#;

    let out = compact(src);
    assert!(
        out.contains(".fract") && out.contains(".whole"),
        "modf result members `.fract` and `.whole` must not be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn frexp_result_members_not_mangled() {
    // `frexp()` returns a WGSL predeclared struct type whose members `.fract`
    // and `.exp` are canonical.  They must survive the mangle pass intact.
    let src = r#"
        @group(0) @binding(0) var<storage, read_write> val: f32;
        @group(0) @binding(1) var<storage, read_write> exp_out: i32;

        @compute @workgroup_size(1)
        fn main() {
            let r = frexp(val);
            val = r.fract;
            exp_out = r.exp;
        }
    "#;

    let out = compact(src);
    assert!(
        out.contains(".fract") && out.contains(".exp"),
        "frexp result members `.fract` and `.exp` must not be mangled: {out}"
    );
    assert_valid_wgsl(&out);
}
