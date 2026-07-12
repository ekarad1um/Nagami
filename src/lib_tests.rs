//! Child module of the file that declares it (via `#[path]`), so
//! `use super::*` keeps private items reachable; relocated out of the
//! declaring file purely for size.

use super::*;

/// Run `f` on a large stack (as the CLI runs minification on its big-stack
/// worker) so a deep-chain test does not overflow the default test-thread
/// stack in the debug profile.  A failed assertion still fails the test - its
/// panic is re-raised via `resume_unwind`.
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

const TRIVIAL_SHADER: &str = r#"
        @vertex fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    "#;

#[test]
fn run_module_report_bytes_nonzero() {
    let mut module = io::parse_wgsl(TRIVIAL_SHADER).unwrap();
    let config = Config::default();
    let report = run_module(&mut module, &config).unwrap();
    assert!(report.input_bytes > 0, "input_bytes must be nonzero");
    assert!(report.output_bytes > 0, "output_bytes must be nonzero");
    assert!(
        report.output_bytes <= report.input_bytes,
        "output should not exceed input for a trivial shader"
    );
}

#[test]
fn non_const_initializer_predicate_flags_binary_override_init() {
    // `2.0 * d` (d is an override) cannot be const-folded, so it stays a
    // `Binary` in the override's init - the variant naga's wgsl-out aborts
    // on.  Must be flagged so the naga baseline is skipped.
    let m = io::parse_wgsl(
        "override d: f32;\
             override h = 2.0 * d;\
             @compute @workgroup_size(1) fn m(){ var t = h; }",
    )
    .unwrap();
    assert!(module_has_non_const_global_initializer(&m));
    assert!(module_needs_naga_baseline_skip(&m));
}

#[test]
fn ray_query_predicate_flags_module() {
    // naga's WGSL writer aborts on `Statement::RayQuery`
    // (`unreachable!()`), so any module holding a `ray_query` type must
    // skip the naga baseline/fallback emit.
    let m = io::parse_wgsl(
            "enable wgpu_ray_query;\
             @group(0)@binding(0) var acc: acceleration_structure;\
             @compute @workgroup_size(1) fn m(){\
                 var rq: ray_query;\
                 rayQueryInitialize(&rq, acc, RayDesc(4u,255u,0.1,100.0,vec3<f32>(0.0),vec3<f32>(0.0,1.0,0.0)));\
             }",
        )
        .unwrap();
    assert!(module_has_ray_query(&m));
    assert!(module_needs_naga_baseline_skip(&m));
}

#[test]
fn ray_tracing_pipeline_without_query_keeps_naga_baseline() {
    // A ray-tracing-*pipeline* module without ray queries stays on the
    // naga baseline: the writer handles its stages, payloads, and
    // builtins fine, and skipping needlessly would forgo the byte
    // comparison.
    let m = io::parse_wgsl(
        "enable wgpu_ray_tracing_pipeline;\
             struct P { hit: u32 }\
             var<ray_payload> payload: P;\
             @ray_generation fn rgen(){ payload = P(0u); }",
    )
    .unwrap();
    assert!(!module_has_ray_query(&m));
    assert!(!module_needs_naga_baseline_skip(&m));
}

#[test]
fn non_const_initializer_predicate_flags_binary_global_var_init() {
    let m = io::parse_wgsl(
        "override k: f32;\
             var<private> g: f32 = k * 10.0;\
             @compute @workgroup_size(1) fn m(){ g = g + 1.0; _ = k; }",
    )
    .unwrap();
    assert!(module_has_non_const_global_initializer(&m));
}

#[test]
fn non_const_initializer_predicate_ignores_const_foldable_inits() {
    // `2.0 * 3.0` is folded to a `Literal` by the front-end, and `c`
    // resolves to a `Constant` - both are in naga's writable set, so the
    // module must NOT be flagged (the naga baseline stays available).
    let m = io::parse_wgsl(
        "const c = 2.0 * 3.0;\
             var<private> g: f32 = c;\
             override d: f32 = 1.5;\
             @compute @workgroup_size(1) fn m(){ g = g + d; }",
    )
    .unwrap();
    assert!(!module_has_non_const_global_initializer(&m));
}

#[test]
fn run_produces_generator_emit_pass() {
    let config = Config::default();
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    let gen_pass = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit");
    assert!(
        gen_pass.is_some(),
        "report must include generator_emit pass"
    );
    let gen_pass = gen_pass.unwrap();
    assert!(gen_pass.validation_ok, "generator output must be valid");
    assert!(!gen_pass.rolled_back, "generator should not need rollback");
    assert!(
        output.report.output_bytes > 0,
        "output_bytes must be nonzero"
    );
}

#[test]
fn run_output_matches_report_bytes() {
    let config = Config::default();
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    assert_eq!(
        output.source.len(),
        output.report.output_bytes,
        "report.output_bytes must match source length"
    );
    assert_eq!(
        TRIVIAL_SHADER.len(),
        output.report.input_bytes,
        "report.input_bytes must match original source length"
    );
}

#[test]
fn validate_each_pass_reports_after_bytes() {
    let config = Config {
        trace: config::TraceConfig {
            enabled: false,
            validate_each_pass: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    // IR passes should all have after_bytes when validate_each_pass is on
    for pr in output
        .report
        .pass_reports
        .iter()
        .filter(|p| p.pass_name != "generator_emit")
    {
        assert!(
            pr.after_bytes.is_some(),
            "pass '{}' should have after_bytes when validate_each_pass is on",
            pr.pass_name
        );
    }
}

#[test]
fn generator_emit_report_consistency() {
    // Verify the generator_emit pass report fields are internally consistent.
    let config = Config::default();
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");

    // If not rolled back, final source should equal output
    if !gen_report.rolled_back {
        assert_eq!(
            gen_report.after_bytes,
            Some(output.source.len()),
            "after_bytes must match output source length"
        );
        assert!(gen_report.validation_ok);
        assert_eq!(gen_report.text_validation_ok, Some(true));
    }
    // Whether rolled back or not, before_bytes must be present
    assert!(gen_report.before_bytes.is_some());
    assert!(gen_report.after_bytes.is_some());
    // changed must be false when rolled_back
    if gen_report.rolled_back {
        assert!(!gen_report.changed);
    }
}

// MARK: End-to-end preserve_symbols tests

/// Regression guard for [`literal_to_wgsl_bare`].  Bare literal
/// emission is safe only at its two sanctioned call sites:
///
///   1. inside a type constructor, where the enclosing type pins
///      the component type, and
///   2. as the RHS of an extracted `const NAME = ...;` declaration,
///      where every use of `NAME` re-binds via abstract coercion.
///
/// The shader below stresses concrete-typed whole-number literals
/// (`F32(1.0)`, `F32(2.0)`, `F32(3.0)`) in positions that would
/// break if the emitter ever used the bare form outside those two
/// patterns, e.g. as a standalone `let` initialiser or as an
/// overload-resolution argument to `atan2`.
#[test]
fn e2e_concrete_float_literals_round_trip_after_minification() {
    let src = r#"
            @compute @workgroup_size(1)
            fn m() {
                // F32 literals as standalone let initializers (concretize path).
                let a: f32 = 1.0;
                let b: f32 = 2.0;
                // F32 literals as atan2 arguments (overload-resolution path).
                let c: f32 = atan2(a, b);
                // Vec constructor with repeated concrete-typed float component
                // (splat-collapse + bare-emit path).
                let d: vec3<f32> = vec3<f32>(3.0, 3.0, 3.0);
                // Binary-arithmetic with literal (left & right both typed).
                let e: f32 = c + 1.0;
                // Suppress unused-variable warnings in the emitted code.
                _ = d.x + e;
            }
        "#;
    let config = Config::default();
    let output = run(src, &config).expect("pipeline should succeed");
    // The emitted WGSL must re-parse - this is the whole point of the
    // bare-literal invariant.
    io::parse_wgsl(&output.source).expect("minified output must round-trip");
}

#[test]
fn e2e_preserve_symbols_struct_type_survives_mangle() {
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
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        preserve_symbols: vec!["Uniforms".to_string()],
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    assert!(
        output.source.contains("Uniforms"),
        "preserved struct type name must survive full pipeline: {}",
        output.source
    );
    // Validate the output is valid WGSL.
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

#[test]
fn e2e_preserve_symbols_struct_member_survives_mangle() {
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
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        preserve_symbols: vec!["resolution".to_string()],
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    assert!(
        output.source.contains("resolution"),
        "preserved member name must survive full pipeline: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

#[test]
fn e2e_preserve_symbols_multiple_categories() {
    // Preserve a struct type, a member, and a constant through the full pipeline.
    let src = r#"
            const MY_CONST: f32 = 3.14;
            struct Material {
                color: vec3<f32>,
                roughness: f32,
            }
            @group(0) @binding(0) var<uniform> mat: Material;
            @fragment fn fs_main() -> @location(0) vec4f {
                return vec4f(mat.color * MY_CONST, mat.roughness);
            }
        "#;
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        preserve_symbols: vec![
            "Material".to_string(),
            "color".to_string(),
            "MY_CONST".to_string(),
        ],
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    assert!(
        output.source.contains("Material"),
        "preserved struct type must survive: {}",
        output.source
    );
    assert!(
        output.source.contains("color"),
        "preserved member must survive: {}",
        output.source
    );
    assert!(
        output.source.contains("MY_CONST"),
        "preserved constant must survive: {}",
        output.source
    );
    // Non-preserved member should be mangled.
    assert!(
        !output.source.contains("roughness"),
        "non-preserved member should be mangled: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

/// A flat reassignment chain must not become one unboundedly-deep
/// expression tree: store-to-load forwarding is depth-capped
/// (`SUBSTITUTION_DEPTH_CAP`), because the deep tree overflows recursive
/// consumers (render_depth's measurement, naga's writer, wasm's ~1 MB
/// stack that the CLI's big-stack worker does not cover) and ships in the
/// output, re-charging every re-minification.
#[test]
fn e2e_flat_reassignment_chain_stays_depth_bounded() {
    let mut src = String::from(
        "@group(0) @binding(0) var<storage, read_write> s: f32;\n\
         @compute @workgroup_size(1)\n\
         fn main() {\n    var a: f32 = 1.0;\n",
    );
    for _ in 0..600 {
        src.push_str("    a = a * 2.0 + 1.0;\n");
    }
    src.push_str("    s = a;\n}\n");
    let output = on_big_stack(move || run(&src, &Config::default()).unwrap());
    let (mut depth, mut max_depth) = (0i32, 0i32);
    for ch in output.source.chars() {
        match ch {
            '(' => {
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            ')' => depth -= 1,
            _ => {}
        }
    }
    assert!(
        max_depth <= 200,
        "output nesting must stay near the substitution cap, got {max_depth}"
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

/// naga materialises a dynamically-indexed function-scope `const` array as
/// a full `Compose` at the use site; once load_dedup forwards the index
/// variable's literal, const_fold must pick the element so the composite
/// dies - round one used to ship the whole array inline
/// (`array<u32,2310>(...)[0]`, the corpus large_array 4749-vs-100-byte
/// idempotence gap).
#[test]
fn e2e_const_array_with_forwarded_index_folds_to_element() {
    let src = r#"
            @group(0) @binding(0) var<storage, read_write> s: array<u32>;
            @compute @workgroup_size(1)
            fn main() {
                const kArray = array(10u, 20u, 30u, 40u);
                var q = 2u;
                s[0] = kArray[q];
            }
        "#;
    let output = run(src, &Config::default()).unwrap();
    assert!(
        output.source.contains("=30"),
        "the picked element must be stored directly: {}",
        output.source
    );
    assert!(
        !output.source.contains("40"),
        "the composite (its unpicked elements) must be gone: {}",
        output.source
    );
}

/// A preserved function must keep BOTH its definition and its call sites.
/// Template inlining used to bake the body into callers, after which
/// `naga::compact` culled the now call-less declaration - for a `--preamble`
/// input that body is only a STUB, so the consumer's real definition was
/// silently bypassed.
#[test]
fn e2e_preserve_symbols_function_keeps_definition_and_call_site() {
    let src = r#"
            fn palette(t: f32) -> f32 {
                return t * 2.0 + 1.0;
            }
            @fragment
            fn fs_main() -> @location(0) vec4f {
                let v = palette(0.25);
                return vec4f(v);
            }
        "#;
    let config = Config {
        profile: config::Profile::Max,
        preserve_symbols: vec!["palette".to_string()],
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    assert!(
        output.source.matches("palette(").count() >= 2,
        "preserved function must survive as declaration + intact call: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

// MARK: Struct name collision regression tests

#[test]
fn e2e_struct_name_does_not_collide_with_function_params() {
    // Regression: the rename pass assigns short names to function
    // parameters and locals.  The generator must not pick the same
    // names for struct types/members, because in WGSL function-scope
    // names shadow module-scope type names, making the struct
    // unusable as a type inside that function.
    //
    // This shader has enough globals + function params that the
    // rename pass will consume early short names, creating a
    // collision opportunity for the generator's struct mangling.
    let src = r#"
            struct Data { value: f32, extra: f32 }
            @group(0) @binding(0) var<uniform> d: Data;
            fn helper(a: f32, b: f32, c: f32) -> Data {
                var result: Data;
                result.value = a + b + c + d.value;
                result.extra = a * d.extra;
                return result;
            }
            @fragment fn main() -> @location(0) vec4f {
                let r = helper(1.0, 2.0, 3.0);
                return vec4f(r.value, r.extra, 0.0, 1.0);
            }
        "#;
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    // The generator must not roll back.
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(
        !gen_report.rolled_back,
        "generator should not roll back; struct names must not collide \
             with function parameter names: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

#[test]
fn e2e_struct_name_does_not_collide_with_local_variables() {
    // Similar to the above, but the collision is with local variables
    // rather than function parameters.
    let src = r#"
            struct Result { x: f32, y: f32 }
            @group(0) @binding(0) var<uniform> input: Result;
            fn compute(val: f32) -> Result {
                var a: f32 = val;
                var b: f32 = val * 2.0;
                var c: f32 = a + b;
                var out: Result;
                out.x = c + input.x;
                out.y = a * input.y;
                return out;
            }
            @fragment fn main() -> @location(0) vec4f {
                let r = compute(1.0);
                return vec4f(r.x, r.y, 0.0, 1.0);
            }
        "#;
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(
        !gen_report.rolled_back,
        "generator should not roll back; struct names must not collide \
             with local variable names: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

// MARK: Preamble tests

#[test]
fn preamble_declarations_excluded_from_output() {
    let preamble = "\
            struct Inputs { time: f32, size: vec2f, }\n\
            @group(0) @binding(0) var<uniform> inputs: Inputs;\
        ";
    let source = "\
            @fragment fn main() -> @location(0) vec4f {\
                return vec4f(inputs.time, inputs.size, 1.0);\
            }\
        ";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    assert!(
        !output.source.contains("Inputs"),
        "preamble struct should not appear in output: {}",
        output.source
    );
    assert!(
        output.source.contains("main"),
        "entry point must still appear in output: {}",
        output.source
    );
    // Output alone is incomplete; validate with preamble re-prepended.
    let (emit_dirs, emit_body) = split_directives(&output.source);
    let (pre_dirs, pre_body) = split_directives(preamble);
    let combined = join_with_newline(&[emit_dirs, pre_dirs, pre_body, emit_body]);
    io::validate_wgsl_text(&combined).expect("output + preamble must be valid WGSL");
}

#[test]
fn preamble_names_preserved_from_renaming() {
    let preamble = "\
            struct Inputs { time: f32, size: vec2f, }\n\
            @group(0) @binding(0) var<uniform> inputs: Inputs;\
        ";
    let source = "\
            @fragment fn main() -> @location(0) vec4f {\
                return vec4f(inputs.time, inputs.size, 1.0);\
            }\
        ";
    let config = Config {
        preamble: Some(preamble.to_string()),
        mangle: Some(true),
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    // The preamble member names must survive mangling so that access
    // expressions (inputs.time, inputs.size) remain correct.
    assert!(
        output.source.contains("time"),
        "preamble member 'time' must survive mangling: {}",
        output.source
    );
    assert!(
        output.source.contains("size"),
        "preamble member 'size' must survive mangling: {}",
        output.source
    );
}

#[test]
fn empty_preamble_treated_as_none() {
    let source = TRIVIAL_SHADER;
    let config_empty = Config {
        preamble: Some(String::new()),
        ..Default::default()
    };
    let config_none = Config::default();
    let out_empty = run(source, &config_empty).unwrap();
    let out_none = run(source, &config_none).unwrap();
    assert_eq!(
        out_empty.source, out_none.source,
        "empty preamble should produce same output as no preamble"
    );
}

#[test]
fn preamble_report_input_bytes_excludes_preamble() {
    let preamble = "struct Inputs { time: f32, }";
    let source = "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    assert_eq!(
        output.report.input_bytes,
        source.len(),
        "input_bytes should reflect user source, not preamble"
    );
}

// MARK: Error diagnostic tests

#[test]
fn parse_error_contains_source_annotation() {
    let bad = "@vertex fn bad() -> vec4<f32> { return bad_func(); }";
    let config = Config::default();
    let err = match run(bad, &config) {
        Err(e) => e,
        Ok(_) => panic!("expected parse error"),
    };
    let msg = err.to_string();
    assert_eq!(err.kind(), "parse");
    // Must contain the annotated source line.
    assert!(
        msg.contains("bad_func"),
        "parse error should reference the problematic identifier: {msg}"
    );
    // Must contain line/column info from codespan.
    assert!(
        msg.contains("wgsl:"),
        "parse error should have source location: {msg}"
    );
}

#[test]
fn preamble_parse_error_uses_preamble_label() {
    let bad_preamble = "struct Bad { x: nonexistent_type }";
    let source =
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }";
    let config = Config {
        preamble: Some(bad_preamble.to_string()),
        ..Default::default()
    };
    let err = match run(source, &config) {
        Err(e) => e,
        Ok(_) => panic!("expected preamble parse error"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("<preamble>"),
        "preamble parse error should identify <preamble> as the source: {msg}"
    );
}

#[test]
fn error_kind_and_message_accessors() {
    let bad = "fn oops { }";
    let config = Config::default();
    let err = match run(bad, &config) {
        Err(e) => e,
        Ok(_) => panic!("expected parse error"),
    };
    assert_eq!(err.kind(), "parse");
    // message() should contain the codespan diagnostic.
    assert!(!err.message().is_empty(), "error message must not be empty");
}

#[test]
fn atomic_compare_exchange_members_do_not_trigger_generator_rollback() {
    let src = r#"
            @group(0) @binding(0)
            var<storage, read_write> val: atomic<u32>;

            @compute @workgroup_size(1)
            fn main() {
                let result = atomicCompareExchangeWeak(&val, 0u, 1u);
                let old = result.old_value;
                let exchanged = result.exchanged;
                _ = old;
                _ = exchanged;
            }
        "#;

    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        ..Default::default()
    };
    let output = run(src, &config).expect("run should succeed");

    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(
        !gen_report.rolled_back,
        "generator should not roll back for atomic compare-exchange member access: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

#[test]
fn run_strips_naga_only_binding_array_enable_from_output() {
    // naga 30 requires `enable wgpu_binding_array;` to parse a binding_array,
    // but tint/Dawn reject the naga-specific directive and support binding
    // arrays natively, so the shipped tint-facing output must NOT carry it -
    // yet the minified binding_array itself must survive.
    let src = r#"
            enable wgpu_binding_array;
            @group(0) @binding(0)
            var arr: binding_array<texture_2d<f32>>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return textureLoad(arr[0], vec2<i32>(0, 0), 0);
            }
        "#;
    let output = run(src, &Config::default()).expect("run should succeed");
    assert!(
        !output.source.contains("enable wgpu_binding_array;"),
        "naga-only enable directive must be stripped from tint-facing output: {}",
        output.source
    );
    assert!(
        output.source.contains("binding_array<"),
        "the binding_array type itself must survive minification: {}",
        output.source
    );
}

/// When the source declares an extension naga cannot parse,
/// `run` returns the original input verbatim plus a synthetic
/// `unsupported_extension_bailout` PassReport so downstream
/// tooling can distinguish "ran the pipeline" from "bailed out".
/// `subgroups` remains a parse-time error in naga 30 (its
/// `enable subgroups;` is the sole `UnimplementedEnableExtension`,
/// matched here by `UNSUPPORTED_EXTENSION_PATTERNS`); a future naga
/// release that lands subgroup support will need a different
/// trigger here.
#[test]
fn unsupported_extension_bailout_includes_synthetic_pass_report() {
    let src = "enable subgroups; // naga cannot parse this extension\n\
                   @compute @workgroup_size(1) fn m() {}";
    let output = run(src, &Config::default()).expect("bailout returns Ok");
    // The bailout path never reaches the pipeline, but it still ships the
    // source lexically compacted (comments stripped, whitespace collapsed,
    // token-fusing joins kept apart).
    assert_eq!(
        output.source, "enable subgroups;@compute@workgroup_size(1)fn m(){}",
        "bailout must ship the lexically compacted source"
    );
    let bailout = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "unsupported_extension_bailout");
    assert!(
        bailout.is_some(),
        "synthetic bailout pass report must be present so callers can detect the short-circuit"
    );
    let b = bailout.unwrap();
    assert!(b.changed, "compaction shrank the text, so changed=true");
    assert!(!b.rolled_back);
    assert_eq!(b.before_bytes, Some(src.len()));
    assert_eq!(b.after_bytes, Some(output.source.len()));
}

#[test]
fn compact_wgsl_text_is_token_safe_and_idempotent() {
    // Comments become whitespace; runs collapse; ident-ident keeps one
    // space; token-fusing pairs keep one space; the rest joins.
    let src = "enable f16;  // trailing comment\n\
                   /* block */ fn  m ( a : f32 , p : ptr<function, f32> ) {\n\
                     let b = a - -a;\n\
                     let c = b / *p;\n\
                   }";
    let compacted = compact_wgsl_text(src);
    assert_eq!(
        compacted,
        "enable f16;fn m(a:f32,p:ptr<function,f32>){let b=a- -a;let c=b/ *p;}"
    );
    // `- -` must not fuse into the reserved `--`, nor `/ *` into `/*`.
    assert!(!compacted.contains("--") && !compacted.contains("/*"));
    // Idempotent: a second pass is byte-identical.
    assert_eq!(compact_wgsl_text(&compacted), compacted);
}

#[test]
fn line_comments_end_at_every_wgsl_line_break() {
    // WGSL ends a `//` comment at any of LF/VT/FF/CR/NEL/LS/PS; ending only
    // at `\n` swallowed the statement after a lone `\r` INTO the comment on
    // the bailout paths (silent statement loss with exit 0).
    for brk in [
        '\u{000B}', '\u{000C}', '\r', '\u{0085}', '\u{2028}', '\u{2029}',
    ] {
        let src = format!("// note{brk}live();\n");
        let stripped = strip_wgsl_comments(&src);
        assert!(
            stripped.contains("live();"),
            "statement after {brk:?}-terminated comment was blanked: {stripped:?}"
        );
    }
}

#[test]
fn compact_keeps_space_before_non_ascii_identifier() {
    // U+2118 is XID_Start but fails `is_alphanumeric`; fusing the keyword
    // and identifier into one token shipped tint-invalid text on the
    // bailout paths.
    let compacted = compact_wgsl_text("let \u{2118} = subgroupAdd(1.0);");
    assert!(
        compacted.starts_with("let \u{2118}"),
        "non-ASCII identifier fused with the keyword: {compacted}"
    );
}

#[test]
fn preamble_plus_bailout_with_directives_hard_errors() {
    // The unsupported-extension bailout ships the body verbatim-compacted,
    // keeping its leading `enable subgroups;`; the consumer's
    // [preamble, body] order would misplace it, so preamble mode must
    // refuse loudly instead of shipping a poisoned document with exit 0.
    let body = "enable subgroups;\n@compute @workgroup_size(64)\n\
        fn m(@builtin(subgroup_invocation_id) sid: u32) { _ = subgroupAdd(f32(sid)); }";
    let config = Config {
        preamble: Some("@group(0) @binding(9) var<uniform> pre_u: f32;".to_string()),
        ..Config::default()
    };
    let err = match run(body, &config) {
        Err(e) => e,
        Ok(out) => panic!(
            "directive-carrying bailout must not ship, got: {}",
            out.source
        ),
    };
    assert!(
        err.to_string().contains("preamble"),
        "error should name the preamble conflict: {err}"
    );
}

#[test]
fn int16_fallback_keeps_enable_and_compacted_text_validates() {
    // The generator has no i16/u16 spelling, so int16 modules always ship
    // via the naga fallback; its text genuinely uses 16-bit tokens, so
    // `strip_naga_only_enables` must KEEP `enable wgpu_int16;` (stripping
    // shipped naga-invalid text), and the fallback compaction must
    // round-trip.
    let src = "enable wgpu_int16;\n\
                   @group(0) @binding(0) var<storage, read_write> o: u32;\n\
                   @compute @workgroup_size(1) fn m() {\n  var x: i16;\n  o = u32(x);\n}\n";
    let compacted = compact_wgsl_text(src);
    assert!(
        io::validate_wgsl_text(&compacted).is_ok(),
        "compacted int16 text must re-validate: {compacted}"
    );
    let out = run(src, &Config::default()).expect("int16 module should minify via fallback");
    assert!(
        out.source.contains("enable wgpu_int16;"),
        "load-bearing enable stripped: {}",
        out.source
    );
    assert!(
        io::validate_wgsl_text(&out.source).is_ok(),
        "shipped int16 output must be naga-valid: {}",
        out.source
    );
}

#[test]
fn output_never_exceeds_input_without_beautify() {
    // Hand-pre-minified input where the emit's CSE scaffolding costs more
    // than it saves: the guard must ship the input verbatim.
    let src = "@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f{\
            let m=mat2x2f(cos(p.x),sin(p.x),-sin(p.x),cos(p.x));return vec4f(m[0],m[1]);}";
    let output = run(src, &Config::default()).expect("run succeeds");
    assert!(
        output.source.len() <= src.len(),
        "output must never exceed the input: {} > {}",
        output.source.len(),
        src.len()
    );
    // Beautify mode is exempt - it grows output on purpose.
    let pretty = run(
        src,
        &Config {
            beautify: true,
            ..Default::default()
        },
    )
    .expect("beautify run succeeds");
    assert!(
        pretty.source.len() > src.len(),
        "beautify must not be clamped by the size guard"
    );
}

#[test]
fn run_auto_enables_f16_when_used() {
    let src = r#"
            fn id(x: f16) -> f16 { return x; }
        "#;
    let output = run(src, &Config::default()).expect("run should succeed with auto f16 enable");
    assert!(
        !output.source.is_empty(),
        "output should be emitted after auto f16 enable"
    );
}

#[test]
fn split_directives_extracts_leading_enables() {
    let src = "enable f16;\nenable subgroups;\nstruct S { x: f32 }\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;\nenable subgroups;\n");
    assert_eq!(rest, "struct S { x: f32 }\nfn f() {}\n");
}

#[test]
fn split_directives_handles_comments_and_blanks() {
    let src = "// header\n\nenable f16;\n\nstruct S { x: f32 }\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "// header\n\nenable f16;\n\n");
    assert_eq!(rest, "struct S { x: f32 }\n");
}

#[test]
fn split_directives_terminator_scan_ignores_semicolon_in_block_comment() {
    // A `;` inside a comment BETWEEN the directive keyword and its real
    // terminator must not split the directive mid-comment (which would
    // splice a preamble into the broken comment region).
    let src = "diagnostic /* a;b */ (off, derivative_uniformity);\n\
                   @fragment fn main() -> @location(0) vec4f { return vec4f(0.); }\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "diagnostic /* a;b */ (off, derivative_uniformity);\n");
    assert!(
        rest.starts_with("@fragment"),
        "body must start at the real declaration, not inside the comment: {rest:?}"
    );
}

#[test]
fn split_directives_terminator_scan_ignores_semicolon_in_line_comment() {
    let src = "enable f16; // trailing ; comment\nstruct S { x: f16 }\n";
    let (dirs, rest) = split_directives(src);
    // The directive's own `;` terminates it; the line comment is body
    // trivia.  (The point is the line-comment `;` is not mistaken for a
    // second directive terminator.)
    assert!(dirs.starts_with("enable f16;"));
    assert!(rest.contains("struct S"));
}

#[test]
fn split_directives_no_directives() {
    let src = "struct S { x: f32 }\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "");
    assert_eq!(rest, src);
}

#[test]
fn split_directives_diagnostic() {
    let src = "diagnostic(off, derivative_uniformity);\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
    assert_eq!(rest, "fn f() {}\n");
}

#[test]
fn split_directives_line_comment_ends_at_vertical_tab() {
    // A `//` comment ends at ANY WGSL line break (VT/CR/FF/...), not just `\n`.
    // A directive after such a comment must still be hoisted; otherwise the
    // preamble path ships it after the preamble's declarations (invalid, exit 0).
    for brk in ["\u{0b}", "\r", "\u{0c}"] {
        let src = format!("//x{brk}diagnostic(off, derivative_uniformity);\nfn f() {{}}\n");
        let (dirs, rest) = split_directives(&src);
        assert!(
            dirs.contains("diagnostic(off, derivative_uniformity);"),
            "directive after a line comment ended by {brk:?} must be hoisted: {dirs:?}"
        );
        assert_eq!(rest, "fn f() {}\n");
    }
}

#[test]
fn has_enable_directive_accepts_line_break_after_keyword() {
    // `enable` may be separated from the extension by any blankspace, incl. a
    // line break (`enable\nf16;` is valid WGSL); the guard must not read that as
    // an identifier and miss the directive (which spuriously fires the f16
    // preamble guard on a preamble that DOES enable f16).
    assert!(has_enable_directive("enable\nf16;", "f16"));
    assert!(has_enable_directive("enable\r\nf16;", "f16"));
    assert!(has_enable_directive("enable f16;", "f16"));
    assert!(!has_enable_directive("enablef16;", "f16"));
}

/// Regression: when `split_directives` returns a fragment that
/// lacks a trailing newline (e.g. a source whose last directive
/// is the final byte), splicing it directly in front of the
/// preamble's directives glued them together into one syntax
/// error.  `join_with_newline` must insert a separator.
#[test]
fn join_with_newline_inserts_separator_between_fragments() {
    let joined = join_with_newline(&["enable f16;", "enable subgroups;\n", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nenable subgroups;\nfn body() {}\n");
}

#[test]
fn join_with_newline_skips_empty_fragments() {
    let joined = join_with_newline(&["enable f16;\n", "", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nfn body() {}\n");
}

#[test]
fn join_with_newline_keeps_existing_trailing_newline() {
    let joined = join_with_newline(&["enable f16;\n", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nfn body() {}\n");
}

/// CRLF fragments survive `split_directives` (verified by
/// `split_directives_crlf_line_endings`), so the join helper must
/// preserve them unchanged.  `ends_with('\n')` matches the LF in
/// `\r\n`, so no extra newline is appended after a CRLF-ending
/// fragment.
#[test]
fn join_with_newline_preserves_crlf_fragments() {
    let joined = join_with_newline(&["enable f16;\r\n", "fn body() {}\r\n"]);
    assert_eq!(joined, "enable f16;\r\nfn body() {}\r\n");
}

#[test]
fn split_directives_requires() {
    let src = "requires readonly_and_readwrite_storage_textures;\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "requires readonly_and_readwrite_storage_textures;\n");
    assert_eq!(rest, "fn f() {}\n");
}

#[test]
fn split_directives_crlf_line_endings() {
    let src = "enable f16;\r\nstruct S { x: f32 }\r\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;\r\n");
    assert_eq!(rest, "struct S { x: f32 }\r\n");
}

#[test]
fn split_directives_no_trailing_newline() {
    let src = "enable f16;";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;");
    assert_eq!(rest, "");
}

#[test]
fn split_directives_all_directives() {
    let src = "enable f16;\nrequires something;\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, src);
    assert_eq!(rest, "");
}

#[test]
fn split_directives_empty_source() {
    let (dirs, rest) = split_directives("");
    assert_eq!(dirs, "");
    assert_eq!(rest, "");
}

#[test]
fn split_directives_compact_single_line() {
    // Compact generator output is one physical line; the splitter must
    // still stop right after the directive's `;`, not swallow the body.
    let src = "enable f16;@group(0)@binding(0)var<storage,read_write>A:f16;\
                   @compute @workgroup_size(1) fn m(){A=1h;}";
    let (dirs, body) = split_directives(src);
    assert_eq!(dirs, "enable f16;");
    assert_eq!(
        body,
        "@group(0)@binding(0)var<storage,read_write>A:f16;\
             @compute @workgroup_size(1) fn m(){A=1h;}"
    );
    // Multiple directives run together on one line are all consumed.
    let (dirs, body) =
        split_directives("enable f16;enable dual_source_blending;@fragment fn m(){}");
    assert_eq!(dirs, "enable f16;enable dual_source_blending;");
    assert_eq!(body, "@fragment fn m(){}");
}

#[test]
fn split_directives_word_boundary_single_line() {
    // Identifiers that merely start with a directive keyword must NOT be
    // hoisted, even when the whole input is one line.
    assert_eq!(
        split_directives("enablef16;fn m(){}"),
        ("", "enablef16;fn m(){}")
    );
    assert_eq!(
        split_directives("requires_foo();fn m(){}"),
        ("", "requires_foo();fn m(){}")
    );
    // A real directive followed mid-line by a `diagnostic`-prefixed
    // identifier stops at that identifier.
    assert_eq!(
        split_directives("enable f16;diagnostic_counter_thing fn m(){}"),
        ("enable f16;", "diagnostic_counter_thing fn m(){}")
    );
}

#[test]
fn references_f16_token_ignores_identifiers() {
    // `myf16var` and `f16_test` must NOT be detected as an f16 token.
    assert!(!references_f16_token("var myf16var: i32;"));
    assert!(!references_f16_token("fn f16_test() {}"));
    assert!(!references_f16_token("let x = ff16;"));
}

#[test]
fn references_f16_token_detects_real_use() {
    assert!(references_f16_token("var x: f16 = 1.0h;"));
    assert!(references_f16_token("let v = vec3<f16>(0h);"));
    assert!(references_f16_token("fn f() -> f16 { return 0h; }"));
}

#[test]
fn references_f16_token_detects_aliases_and_suffix() {
    // Predeclared half-precision aliases (no `f16` substring).
    assert!(references_f16_token("var v: vec2h = vec2h(1.0h, 2.0h);"));
    assert!(references_f16_token("let v = vec3h(0h);"));
    assert!(references_f16_token("var m: mat4x4h;"));
    assert!(references_f16_token("let m = mat2x3h();"));
    // `h` float-literal suffix in its various spellings, no alias/keyword.
    assert!(references_f16_token("let x = 1.0h + 2.0h;"));
    assert!(references_f16_token("let x = 0h;"));
    assert!(references_f16_token("let x = 1.5e2h;"));
    assert!(references_f16_token("let x = 1.0e-3h;"));
    assert!(references_f16_token("let x = 0x1p2h;"));
    // Negatives: longer identifiers, other float suffixes, plain ints.
    assert!(!references_f16_token("var width: f32; var height: f32;"));
    assert!(!references_f16_token("let mesh = 1.0;"));
    assert!(!references_f16_token("var vec2hh: i32;"));
    assert!(!references_f16_token("let x = 1.0f + 2u + 3;"));
    assert!(!references_f16_token("let x = 1.0e-3;"));
    assert!(!references_f16_token("fn vec2h_helper() {}"));
}

#[test]
fn references_f16_token_ignores_comments() {
    // f16 mentioned only inside comments should not trigger injection.
    assert!(!references_f16_token("// uses f16 later\nvar x: i32;"));
    assert!(!references_f16_token("/* f16 */ var x: i32;"));
    assert!(!references_f16_token(
        "/* multiline\n   f16\n */\nvar x: i32;"
    ));
}

// Regression: WGSL (https://www.w3.org/TR/WGSL/#comments) permits
// nested block comments.  A non-nesting comment scrubber would close
// at the inner `*/` and expose the trailing `f16` content to the
// token scan; the depth-tracking implementation must absorb the inner
// pair and treat the whole region as commented.
#[test]
fn references_f16_token_ignores_nested_block_comments() {
    assert!(!references_f16_token(
        "/* outer /* inner */ f16 still in outer */ var x: i32;"
    ));
    assert!(!references_f16_token(
        "/* /* /* deeply nested */ */ f16 inside */ var x: i32;"
    ));
    // A real f16 after a properly-closed nested comment still
    // detects.
    assert!(references_f16_token(
        "/* nest /* inner */ done */ var x: f16 = 0h;"
    ));
}

// Regression: classic-Mac (lone `\r`) line endings must be
// normalised before any `.lines()` scan, otherwise the
// `wgpu_*` directive strip and `enable f16;` detection fold the
// entire source into one line and silently fail.
#[test]
fn normalize_line_endings_handles_lone_cr() {
    // Lone `\r` becomes `\n`.
    assert_eq!(normalize_line_endings("a\rb\rc"), "a\nb\nc");
    // `\r\n` stays intact (str::lines already handles it).
    assert_eq!(normalize_line_endings("a\r\nb\r\nc"), "a\r\nb\r\nc");
    // Mixed.
    assert_eq!(normalize_line_endings("a\rb\r\nc\nd"), "a\nb\r\nc\nd");
    // Source with no `\r` returns identical content.
    assert_eq!(normalize_line_endings("a\nb\nc"), "a\nb\nc");
    // Multi-byte UTF-8 around line endings preserved.
    assert_eq!(normalize_line_endings("α\rβ\r\nγ"), "α\nβ\r\nγ");
}

#[test]
fn preprocess_preserves_wgpu_directive_with_cr_only_endings() {
    // naga 30 implements (and requires) `wgpu_binding_array`, so it is no
    // longer stripped; the lone-CR line ending is still normalised to LF so
    // the downstream per-line scans see the break.
    let src =
        "enable wgpu_binding_array;\r@fragment fn m() -> @location(0) vec4f { return vec4f(0); }";
    let out = preprocess_source_for_naga(src);
    assert!(
        out.contains("enable wgpu_binding_array;"),
        "implemented wgpu_* directive must be preserved: {out:?}"
    );
    assert!(
        !out.contains('\r'),
        "lone CR must be normalised to LF: {out:?}"
    );
}

#[test]
fn has_enable_f16_directive_matches_canonical() {
    assert!(has_enable_f16_directive("enable f16;\n"));
    assert!(has_enable_f16_directive("enable f16;"));
}

#[test]
fn has_enable_f16_directive_matches_extra_whitespace() {
    assert!(has_enable_f16_directive("enable  f16;\n"));
    assert!(has_enable_f16_directive("enable\tf16;\n"));
    assert!(has_enable_f16_directive("enable f16 ;\n"));
}

#[test]
fn has_enable_f16_directive_rejects_commented_out() {
    assert!(!has_enable_f16_directive("// enable f16;"));
    assert!(!has_enable_f16_directive("/* enable f16; */"));
}

#[test]
fn has_enable_f16_directive_rejects_non_directive() {
    assert!(!has_enable_f16_directive("var enable_f16: bool;"));
    assert!(!has_enable_f16_directive("fn enable() {} // f16"));
}

#[test]
fn has_enable_f16_directive_matches_comma_separated_list() {
    // WGSL permits a comma-separated enable list; `f16` in any position must
    // be recognised.  A false negative here is not harmless: the preamble
    // guard in `run` turns it into a hard error on valid input.
    assert!(has_enable_f16_directive("enable f16, clip_distances;\n"));
    assert!(has_enable_f16_directive("enable clip_distances, f16;\n"));
    assert!(has_enable_f16_directive(
        "enable dual_source_blending , f16 ;"
    ));
    assert!(!has_enable_f16_directive(
        "enable clip_distances, dual_source_blending;"
    ));
}

#[test]
fn has_enable_f16_directive_matches_second_directive_on_same_line() {
    // Several directives may share one physical line; f16 in any of them
    // must be found (a false negative hard-errors the preamble guard).
    assert!(has_enable_f16_directive(
        "enable dual_source_blending; enable f16;"
    ));
    assert!(has_enable_f16_directive(
        "enable clip_distances; enable f16, subgroups;"
    ));
    // Directives on separate lines (each ends in `;`, all before any decl).
    assert!(has_enable_f16_directive(
        "enable dual_source_blending;\nenable f16;\nstruct S { x: f32 }"
    ));
    assert!(!has_enable_f16_directive(
        "enable dual_source_blending; enable clip_distances;"
    ));
}

#[test]
fn preprocess_does_not_inject_for_identifier_only() {
    // Source references `f16` only as part of identifiers -> no injection.
    let src = "var myf16_var: i32 = 0;\n";
    let out = preprocess_source_for_naga(src);
    assert!(
        !out.contains("enable f16;"),
        "must not inject enable when `f16` appears only in identifiers: {out}"
    );
}

#[test]
fn preprocess_injects_for_real_f16_use() {
    let src = "fn f() -> f16 { return 0h; }\n";
    let out = preprocess_source_for_naga(src);
    assert!(
        out.starts_with("enable f16;\n"),
        "must inject enable directive for real f16 use: {out}"
    );
}

#[test]
fn preprocess_does_not_duplicate_enable_f16_with_extra_whitespace() {
    let src = "enable  f16;\nfn f() -> f16 { return 0h; }\n";
    let out = preprocess_source_for_naga(src);
    // The output must carry exactly the directive that was already
    // present in the source: one occurrence of `enable...f16;` plus
    // one occurrence of the `f16` return type in the function
    // signature.  The `enable` count locks against accidental
    // injection of a second normalised `enable f16;`.
    assert_eq!(
        out.matches("enable").count(),
        1,
        "must not inject a second `enable` directive when the source already enables f16: {out}"
    );
    assert!(
        !out.contains("enable f16;\nenable  f16;"),
        "must not inject a second enable f16 directive: {out}"
    );
}

// MARK: Naga error-message coupling tests

// These tests fail if `UNSUPPORTED_EXTENSION_PATTERNS` or
// `KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS` drift out of sync with
// the naga error phrasings they target.
#[test]
fn unsupported_extension_patterns_match_documented_phrasings() {
    // These are the exact phrasings naga produces today.  All must
    // be recognised by `is_unsupported_extension_parse_error` or
    // the short-circuit return path in `run` falls over into a
    // hard error the moment naga rewords them.
    let samples = [
        "error: enable extension is not enabled",
        "error: the `wgpu_ray_query` enable-extension is not yet supported",
    ];
    for s in samples {
        let err = Error::Parse(s.to_string());
        assert!(
            is_unsupported_extension_parse_error(&err),
            "naga error phrasing should be recognized as unsupported-extension: {s}"
        );
    }
}

#[test]
fn unsupported_extension_patterns_reject_unrelated_errors() {
    let err = Error::Parse("error: expected identifier, found `{`".into());
    assert!(
        !is_unsupported_extension_parse_error(&err),
        "unrelated parse errors must not be treated as unsupported-extension"
    );
}

/// Regression: a parse error whose codespan snippet quotes a user
/// comment containing the unsupported-extension phrasing must NOT
/// trigger the bailout.  Pre-fix, substring matching against the
/// entire rendered message swallowed real failures whenever the
/// user happened to comment-mention an unsupported extension.
#[test]
fn unsupported_extension_patterns_ignore_quoted_source_lines() {
    let rendered = "error: expected identifier, found `{`\n  \
                        ┌─ wgsl:5:1\n  │\n5 │ // TODO: enable extension is not enabled \
                        on our backend\n  │ ^^\n";
    let err = Error::Parse(rendered.into());
    assert!(
        !is_unsupported_extension_parse_error(&err),
        "pattern in a quoted source line must NOT trigger the bailout"
    );

    // Same shape for the subgroups text-validation limitation: a
    // shader quoting the phrase in a comment, surfaced as context
    // around an unrelated parse error, must not opt into the
    // validation-bypass.
    let rendered = "error: expected `;`\n  \
                        ┌─ wgsl:3:1\n  │\n3 │ // subgroups enable-extension is not yet supported\n";
    let err = Error::Parse(rendered.into());
    assert!(
        !is_known_text_validation_limitation(&err),
        "pattern in a quoted source line must NOT trigger the validation bypass"
    );
}

#[test]
fn unsupported_extension_patterns_only_match_parse_errors() {
    // Regression: substring patterns must not match against the
    // rendered message of a non-`Parse` error variant, since a
    // validator/emit failure quoting the same phrasing (or even an
    // I/O error path containing the offending source) would
    // previously short-circuit `run` into the "return input
    // unchanged" branch and silently swallow a real failure.
    for ctor in [
        Error::Validation as fn(String) -> Error,
        Error::Emit as fn(String) -> Error,
        Error::Io as fn(String) -> Error,
        Error::Config as fn(String) -> Error,
    ] {
        let err = ctor("error: enable extension is not enabled".to_string());
        assert!(
            !is_unsupported_extension_parse_error(&err),
            "non-Parse error variants must not be treated as \
                 unsupported-extension even when the message matches: {err:?}"
        );
    }
}

#[test]
fn known_text_validation_limitation_only_matches_parse_or_validation() {
    // Same defensive policy as above for the subgroups-limitation
    // matcher: only `Parse` and `Validation` may opt into the
    // round-trip-validation bypass; `Emit` and `Io` errors quoting
    // the same phrasing must be reported normally.
    for ctor in [
        Error::Emit as fn(String) -> Error,
        Error::Io as fn(String) -> Error,
        Error::Config as fn(String) -> Error,
    ] {
        let err = ctor("error: `subgroups` enable-extension is not yet supported".to_string());
        assert!(
            !is_known_text_validation_limitation(&err),
            "non-Parse/Validation error variants must not opt into the \
                 subgroup text-validation bypass: {err:?}"
        );
    }
}

#[test]
fn known_text_validation_limitation_matches_subgroup_phrasings() {
    let samples = [
        "error: `subgroups` enable-extension is not yet supported",
        "error: subgroups enable-extension is not yet supported",
    ];
    for s in samples {
        let err = Error::Parse(s.to_string());
        assert!(
            is_known_text_validation_limitation(&err),
            "subgroup limitation phrasing should be recognized: {s}"
        );
    }
}

#[test]
fn known_text_validation_limitation_matches_validation_variant_too() {
    // The matcher is intentionally scoped to `Parse | Validation`
    // because `io::validate_wgsl_text` internally calls both
    // `parse_wgsl` (Parse errors) and `validate_module_with_source`
    // (Validation errors); naga can report a not-yet-supported
    // `enable` directive through either path.  This regression pins
    // the Validation branch so a future tightening to "Parse only"
    // fails loudly here.
    let samples = [
        "error: `subgroups` enable-extension is not yet supported",
        "error: subgroups enable-extension is not yet supported",
    ];
    for s in samples {
        let err = Error::Validation(s.to_string());
        assert!(
            is_known_text_validation_limitation(&err),
            "subgroup limitation phrasing must also be recognized when wrapped as Validation: {s}"
        );
    }
}

#[test]
fn known_text_validation_limitation_rejects_unrelated_errors() {
    let err = Error::Validation("error: mismatched types".into());
    assert!(!is_known_text_validation_limitation(&err));
}

#[test]
fn preamble_with_enable_f16_shader() {
    // A preamble that carries its OWN `enable f16;` directive, combined with
    // a source that genuinely USES f16, minified in COMPACT mode (the whole
    // module is one physical line).  Two contracts are verified:
    //
    //   1. The output body is directive-FREE.  With a preamble active the
    //      consumer prepends the preamble in front of this output, so a
    //      directive left in the body would land AFTER the preamble's global
    //      declarations - illegal WGSL ("directives must come before all
    //      global declarations").  The preamble owns the directive.
    //   2. The shipped artifact `[preamble, output]` is valid WGSL.  This is
    //      what exercises the `;`-aware `split_directives`: a line-based
    //      splitter would misclassify the compact one-line output as one big
    //      directive and mis-order the splice.
    let preamble = "\
            enable f16;\n\
            @group(0) @binding(0) var<uniform> bias: f16;\
        ";
    let source = "\
            @group(0) @binding(1) var<storage, read_write> sink: f16;\n\
            @compute @workgroup_size(1) fn main() {\n\
                var h: f16 = 1.0h;\n\
                h = h + bias;\n\
                sink = h;\n\
            }\
        ";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    // Default config => compact mode.
    let output = run(source, &config)
        .expect("enable f16; in preamble + f16 source must minify in compact mode");
    // Contract 1: the preamble owns the directive; the body must not carry a
    // copy (it would be mis-ordered after the preamble's globals).
    assert!(
        !output.source.contains("enable f16;"),
        "output body must not carry the directive when the preamble supplies it: {}",
        output.source
    );
    // Contract 2: the artifact the consumer actually ships re-parses.
    let shipped = format!("{preamble}\n{}", output.source);
    io::validate_wgsl_text(&shipped)
        .expect("the shipped [preamble, output] concatenation must be valid WGSL");
}

#[test]
fn preamble_missing_directive_errors_not_silent_ship() {
    // A source that genuinely uses f16 but whose preamble does NOT supply
    // `enable f16;`.  The directive cannot validly live in the output (it
    // would land after the preamble's globals) nor be dropped (f16 then has
    // no enabling directive anywhere), so there is no valid `[preamble,
    // output]` to emit.  It must ERROR here, naming the missing extension, so
    // the user adds the directive to the preamble.
    let preamble = "\
            @group(0) @binding(0) var<uniform> bias: f32;\
        ";
    // The f16 write to a storage buffer is observable, so DCE keeps it and
    // the emitted module genuinely needs `enable f16;`.
    let source = "\
            @group(0) @binding(1) var<storage, read_write> sink: f16;\n\
            @compute @workgroup_size(1) fn main() {\n\
                sink = 1.0h;\n\
            }\
        ";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let msg = match run(source, &config) {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "f16 source with a preamble lacking `enable f16;` must error, \
                 not silently ship invalid WGSL"
        ),
    };
    assert!(
        msg.contains("f16"),
        "the error should name the missing f16 extension: {msg}"
    );
}

#[test]
fn preamble_declares_f16_in_comma_separated_enable_list() {
    // A preamble that declares f16 as one entry of a comma-separated enable
    // list (valid WGSL) supplies the directive just as a lone `enable f16;`
    // does, so an f16 body must minify - NOT trip the missing-directive
    // guard.  Regression: the guard's single-extension-only detection used to
    // hard-error this valid input.
    let preamble = "\
            enable f16, clip_distances;\n\
            @group(0) @binding(0) var<uniform> bias: f16;\
        ";
    let source = "\
            @group(0) @binding(1) var<storage, read_write> sink: f16;\n\
            @compute @workgroup_size(1) fn main() {\n\
                sink = 1.0h + bias;\n\
            }\
        ";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(source, &config)
        .expect("f16 declared in a comma-separated enable list must minify, not error");
    assert!(
        !output.source.contains("enable f16;"),
        "the preamble owns the directive; the body must not carry it: {}",
        output.source
    );
    let shipped = format!("{preamble}\n{}", output.source);
    io::validate_wgsl_text(&shipped)
        .expect("the shipped [preamble, output] concatenation must be valid WGSL");
}

#[test]
fn binding_array_minifies_under_preamble_without_naga_only_enable() {
    // naga needs `enable wgpu_binding_array;` to PARSE a binding_array, but
    // tint supports them natively (no enable) so the shipped body omits it and
    // the preamble need not declare it.  The preamble self-check injects that
    // naga-only enable for its naga re-parse ONLY.  Regression: the self-check
    // validated the enable-less body against naga and hard-errored, with no
    // preamble spelling that both minified and shipped tint-valid output.
    let preamble = "const SCALE: f32 = 2.0;";
    let source = "\
            @group(0) @binding(0) var tex: binding_array<texture_2d<f32>, 4>;\n\
            @fragment fn m() -> @location(0) vec4f {\n\
                return textureLoad(tex[0], vec2i(0), 0) * SCALE;\n\
            }";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(source, &config)
        .expect("binding_array + preamble must minify, not hard-error on the self-check");
    assert!(
        !output.source.contains("enable wgpu_binding_array;"),
        "shipped body must not carry the naga-only enable: {}",
        output.source
    );
    assert!(
        output.source.contains("binding_array<"),
        "the binding_array type must survive: {}",
        output.source
    );
    // The shipped body is tint-valid without the enable; naga needs it, so
    // prepend it to confirm [preamble, body] round-trips through naga.
    let shipped = format!("enable wgpu_binding_array;\n{preamble}\n{}", output.source);
    io::validate_wgsl_text(&shipped).expect("[preamble, body] must round-trip through naga");
}

// MARK: Splat elision tests

/// Minify with the `Max` profile (mangle + inline) and assert the
/// result is valid WGSL.  Returns the minified source for the
/// caller's own assertions.
fn minify_and_validate(src: &str) -> String {
    let config = Config {
        profile: config::Profile::Max,
        mangle: Some(true),
        ..Default::default()
    };
    let output = run(src, &config).unwrap();
    io::validate_wgsl_text(&output.source)
        .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
    output.source
}

#[test]
fn run_never_ships_invalid_wgsl_when_fallback_is_also_invalid() {
    // Regression for the rollback re-validation guard: when BOTH the
    // custom generator AND naga's own wgsl-out fallback emit text that
    // naga's frontend rejects, run() must surface a diagnosable error
    // rather than silently shipping output that does not re-parse.
    // Invariant under test: run() either errors, or returns output that
    // round-trips.  (Robust to future naga changes - if a naga release
    // accepts these forms, the Ok branch simply validates clean.)
    let inputs = [
        // f64 literal narrowed to f32: naga const-substitutes the literal
        // under the cast, then rejects `f32(<F64 literal>)`.
        "@group(0) @binding(0) var<storage, read_write> s: f32;\n\
             @compute @workgroup_size(1) fn m() { let a: f64 = 0.5lf; s = f32(a); }",
        // f16 literal in a cast whose `enable f16;` naga's backend drops.
        "enable f16;\n\
             @fragment fn m() -> @location(0) vec4f { let h: f16 = 1.0h; return vec4f(f32(h)); }",
    ];
    for src in inputs {
        if let Ok(output) = run(src, &Config::default()) {
            io::validate_wgsl_text(&output.source).unwrap_or_else(|e| {
                panic!(
                    "run() shipped invalid WGSL (it should have errored): {e}\n{}",
                    output.source
                )
            });
        }
    }
}

#[test]
fn folds_f64_literal_narrowing_cast_to_valid_literal() {
    // Regression for issue A: naga const-substitutes `let a: f64 = 2.5lf`
    // into the cast, yielding `As { Literal(F64), convert }`, which the
    // emitter rendered as `f32(2.5lf)` - a token naga rejects on re-parse.
    // const_fold now folds the narrowing cast of an F64 literal to the
    // converted scalar literal, so the output round-trips (and shrinks).
    for (decl, stmt) in [
        ("var<storage, read_write> s: f32;", "s = f32(a);"),
        ("var<storage, read_write> s: i32;", "s = i32(a);"),
        ("var<storage, read_write> s: u32;", "s = u32(a);"),
    ] {
        let src = format!(
            "@group(0) @binding(0) {decl}\n\
                 @compute @workgroup_size(1) fn m() {{ let a: f64 = 2.5lf; {stmt} }}"
        );
        let output = run(&src, &Config::default())
            .unwrap_or_else(|e| panic!("f64 narrowing cast must minify, got error: {e}"));
        io::validate_wgsl_text(&output.source)
            .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
        // The `T(<F64 literal>)` cast must be folded away, not emitted
        // (no `lf)` token - the f64 literal no longer appears in a cast).
        assert!(
            !output.source.contains("lf)"),
            "f64 cast literal should be folded, not emitted as a cast: {}",
            output.source
        );
    }
}

#[test]
fn folds_f64_vector_narrowing_cast_to_valid_constructor() {
    // A const f64 VECTOR narrowing cast (`vec2<f32>(vec2<f64>(.5lf,1.5lf))`)
    // is rejected by naga on re-parse, and const_fold can't materialize the
    // converted vector (the converted F32 component literals don't exist as
    // arena handles).  The generator folds it to a converted constructor.
    for (store_ty, decl_a, cast) in [
        ("vec2<f32>", "vec2<f64>(0.5lf, 1.5lf)", "vec2<f32>(a)"),
        (
            "vec3<f32>",
            "vec3<f64>(0.5lf, 1.5lf, 2.5lf)",
            "vec3<f32>(a)",
        ),
        ("vec2<i32>", "vec2<f64>(2.5lf, 3.5lf)", "vec2<i32>(a)"),
    ] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: {store_ty};\n\
                 @compute @workgroup_size(1) fn m() {{ let a = {decl_a}; o = {cast}; }}"
        );
        let output = run(&src, &Config::default()).unwrap_or_else(|e| {
            panic!("f64 vector narrowing cast must minify, got error: {e}\nsrc:{src}")
        });
        io::validate_wgsl_text(&output.source)
            .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
        // The `vecN<f32>(vecN<f64>(..lf..))` cast must be folded away.
        assert!(
            !output.source.contains("lf)"),
            "f64 vector cast should be folded, not emitted: {}",
            output.source
        );
    }
}

#[test]
fn folds_int64_narrowing_cast_to_valid_literal() {
    // The u64/i64 (`lu`/`li`) narrowing cast is the exact analogue of the
    // f64 cast: naga rejects re-parsing `u32(<U64 literal>)`, and naga's own
    // backend emits the same token, so it used to hard-error.  const_fold
    // (scalar) and the generator (vector) now fold it, WRAPPING on
    // narrowing - `u32(i64(-1))` is `4294967295`, NOT a clamped `0` -
    // matching naga's value-conversion semantics.
    let cases = [
        // (decl, cast, expected substring in the folded output)
        ("let a: u64 = 107lu;", "o = u32(a);", "107"),
        ("let a: u64 = 4294967296lu;", "o = u32(a);", "0"), // 2^32 wraps to 0
        ("let a: i64 = -1li;", "o = u32(a);", "4294967295"), // wrap, not clamp
        (
            "let a = vec2<u64>(107lu, 4294967296lu);",
            "o = vec2<u32>(a);",
            "vec2u(107,0)",
        ),
    ];
    for (decl, cast, expect) in cases {
        let store_ty = if cast.contains("vec2") {
            "vec2<u32>"
        } else {
            "u32"
        };
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: {store_ty};\n\
                 @compute @workgroup_size(1) fn m() {{ {decl} {cast} }}"
        );
        let output = run(&src, &Config::default()).unwrap_or_else(|e| {
            panic!("int64 narrowing cast must minify, got error: {e}\nsrc:{src}")
        });
        io::validate_wgsl_text(&output.source)
            .unwrap_or_else(|e| panic!("output is invalid WGSL: {e}\n{}", output.source));
        // The `T(<width-8 literal>)` cast must be folded (no `lu)`/`li)`).
        assert!(
            !output.source.contains("lu)") && !output.source.contains("li)"),
            "int64 cast should be folded, not emitted: {}",
            output.source
        );
        assert!(
            output.source.contains(expect),
            "expected wrapped value {expect:?} in output: {}",
            output.source
        );
    }
}

#[test]
fn splat_elision_add_vec3f() {
    // vec3f(1) + vec3f_var  ->  bare `1` via scalar-vector broadcasting
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var c = vec3f(0.5, 0.6, 0.7);
                c = vec3f(1.0) + c;
                return vec4f(c, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // The splat `vec3f(1)` (or its alias) should NOT appear;
    // instead the bare scalar should be used in the addition.
    assert!(
        !out.contains("vec3f(1)") && !out.contains("vec3<f32>(1"),
        "splat should be elided in addition: {out}"
    );
}

#[test]
fn splat_elision_subtract_vec2f() {
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var uv = vec2f(0.3, 0.7);
                uv = uv - vec2f(0.5);
                return vec4f(uv, 0.0, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        !out.contains("vec2f(.5)") && !out.contains("vec2<f32>(.5"),
        "splat should be elided in subtraction: {out}"
    );
}

#[test]
fn splat_elision_multiply_vec4f() {
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var c = vec4f(0.1, 0.2, 0.3, 0.4);
                c = vec4f(2.0) * c;
                return c;
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        !out.contains("vec4f(2") && !out.contains("vec4<f32>(2"),
        "splat should be elided in multiplication: {out}"
    );
}

#[test]
fn splat_elision_divide_by_splat() {
    // vector / splat  ->  vector / scalar
    let src = r#"
            fn helper(v: vec3f) -> vec3f {
                return v / vec3f(dot(v, v));
            }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(helper(vec3f(1.0, 2.0, 3.0)), 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // The vec3f(dot(...)) should be elided to just dot(...)
    assert!(
        !out.contains("vec3f(dot") && !out.contains("vec3<f32>(dot"),
        "splat wrapping dot() should be elided in division: {out}"
    );
}

#[test]
fn splat_elision_compound_assign() {
    // v -= vec2f(0.5)  ->  v -= .5
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var uv = vec2f(1.0, 1.0);
                uv -= vec2f(0.5);
                return vec4f(uv, 0.0, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        !out.contains("vec2f(.5)") && !out.contains("vec2<f32>(.5"),
        "splat should be elided in compound assignment: {out}"
    );
}

#[test]
fn splat_elision_no_double_elide() {
    // vec3f(a) + vec3f(b): at most one side should be elided so the
    // result stays a vector (not scalar + scalar = scalar).
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let a = 1.0;
                let b = 2.0;
                let c = vec3f(a) + vec3f(b);
                return vec4f(c, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // Output must be valid WGSL - validation above ensures the type is correct.
    assert!(!out.is_empty());
}

#[test]
fn splat_elision_skipped_when_other_is_scalar() {
    // vec4f(1.0) * scalar must NOT elide the Splat, because
    // 1.0 * scalar = scalar, not vec4f.
    let src = r#"
            struct S { b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @vertex fn main() -> @builtin(position) vec4f {
                return vec4f(1.0) * u.b;
            }
        "#;
    let out = minify_and_validate(src);
    // The output must still contain vec4f - the splat can't be elided.
    assert!(
        out.contains("vec4"),
        "splat must not be elided when other operand is scalar: {out}"
    );
}

#[test]
fn splat_elision_skipped_when_other_is_scalar_rhs() {
    // scalar * vec4f(1.0) must NOT elide the Splat, because
    // scalar * 1.0 = scalar, not vec4f.  (Reverse of the LHS test.)
    let src = r#"
            struct S { b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @vertex fn main() -> @builtin(position) vec4f {
                return u.b * vec4f(1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        out.contains("vec4"),
        "splat must not be elided when other operand is scalar (RHS): {out}"
    );
}

#[test]
fn splat_elision_non_arithmetic_unchanged() {
    // Comparison operators should NOT elide splats.
    // vec3f(0) == vec3f_var is NOT valid as 0 == vec3f_var.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let v = vec3f(1.0, 2.0, 3.0);
                let mask = v > vec3f(1.5);
                return select(vec4f(0), vec4f(1), mask.x);
            }
        "#;
    // Just verify it's valid - comparison ops shouldn't trigger elision.
    minify_and_validate(src);
}

// MARK: Last-store inlining tests

#[test]
fn last_store_inlined_when_earlier_loads_keep_var_alive() {
    // Pattern: var m is loaded before AND after the last store.
    // The pre-store load keeps m alive (non-dead).  The post-store
    // load should still be inlined to the stored expression, and
    // the store itself should be removed.
    let src = r#"
            fn helper(v: vec3<f32>) -> vec3<f32> { return v; }
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let pre = m;           // load before store - keeps m alive
                m = helper(pre);       // store complex expr
                let post = m;          // load after store - should be inlined
                return vec4f(post, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // The variable `m` should still exist (pre-store load keeps it alive),
    // but the last store's value should be inlined into the return.
    // Check that the output doesn't contain a redundant store+load pattern.
    // The output should be shorter than without inlining.
    assert!(!out.is_empty(), "output should not be empty");
}

#[test]
fn last_store_not_inlined_when_escaped() {
    // When a variable's pointer escapes via a function call,
    // its Stores must be preserved - the callee may read through
    // the pointer at any time.
    let src = r#"
            fn consume(p: ptr<function, vec3f>) -> vec3f { return *p; }
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let pre = m;
                m = pre + vec3f(1.0);
                let post = consume(&m);
                return vec4f(post, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(!out.is_empty(), "output should not be empty");
}

#[test]
fn last_store_not_inlined_with_partial_stores() {
    // When a variable has partial stores (field access), the whole-variable
    // store must be preserved because the partial store reads the full value.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var v = vec3f(1.0, 2.0, 3.0);
                let pre = v;
                v = pre + vec3f(1.0);
                v.x = 0.0;             // partial store - depends on full v
                let post = v;
                return vec4f(post, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(!out.is_empty(), "output should not be empty");
}

#[test]
fn last_store_preserves_other_stores_to_same_var() {
    // Regression: last-store inlining must only remove the specific
    // Store that was proven dead, not ALL stores to the same variable.
    // Here, the conditional store inside `if` must be preserved because
    // the final `log(1+m)` reads `m` which depends on it.
    let src = r#"
            fn heavy(a: vec3f, b: vec3f) -> vec3f { return a + b; }
            struct U { v: f32 }
            @group(0) @binding(0) var<uniform> u: U;
            @fragment fn main() -> @location(0) vec4f {
                var m = vec3f(0.0);
                let ray = vec3f(1.0, 2.0, 3.0);
                if u.v >= 0.0 {
                    m = heavy(ray, vec3f(0.5));
                }
                m = 0.5 * log(1.0 + m);
                return vec4f(m, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // The conditional store `m = heavy(...)` must be preserved.
    // Without it, m stays vec3f(0) and the result is always black.
    assert!(
        out.contains("if"),
        "conditional branch must be preserved (store to m is live): {out}"
    );
}

#[test]
fn last_store_init_preserved_when_loop_reads_var() {
    // Regression: when a variable's init Store has a seeded load
    // that gets forwarded, but the variable is also read inside a
    // subsequent loop body (where cache is cleared), the init Store
    // must NOT be removed.
    let src = r#"
            fn transform(v: vec3f) -> vec3f { return abs(v) - vec3f(0.7); }
            @fragment fn main() -> @location(0) vec4f {
                var p = vec3f(1.0, 2.0, 3.0);
                let ip = p;  // seeded load from init Store, gets forwarded
                for (var i = 0u; i < 4u; i++) {
                    p = transform(p);  // loop body reads p (needs init on 1st iter)
                }
                return vec4f(p + ip, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // The init `p = vec3f(1,2,3)` must be preserved.  Without it, the
    // loop reads p=vec3f(0) on the first iteration, producing wrong results.
    // Verify the output produces valid WGSL (validation above) and that
    // the transform call appears (loop body is not dead).
    assert!(
        out.contains("abs"),
        "loop body with transform must be preserved: {out}"
    );
}

#[test]
fn last_store_in_loop_not_removed() {
    // Regression: a Store inside a loop body must NOT be removed by
    // last-store inlining, even if all seeded loads are forwarded.
    // On the next iteration, loads BEFORE the Store in the loop body
    // observe the Store's value via the loop back-edge.
    let src = r#"
            fn complexSquare(a: vec2f) -> vec2f {
                return vec2f(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y);
            }
            @fragment fn main() -> @location(0) vec4f {
                var p = vec3f(1.0, 2.0, 3.0);
                for (var i = 0u; i < 10u; i++) {
                    p = 0.7 * abs(p) / dot(p, p) - vec3f(0.7);
                    p = vec3f(p.x, complexSquare(p.yz)).zxy;
                }
                return vec4f(p, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    // Both stores to p in the loop must be preserved.
    // The .zxy swizzle is the telltale of the second store.
    assert!(
        out.contains(".zxy"),
        "second store in loop (with .zxy swizzle) must be preserved: {out}"
    );
}

// Regression: `split_directives` must word-boundary-match the
// `diagnostic` keyword.  A bare `starts_with("diagnostic")` would
// misclassify user identifiers such as `diagnostic_counter`.
#[test]
fn split_directives_recognizes_paren_diagnostic() {
    let source = "diagnostic(off, derivative_uniformity);\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
    assert_eq!(body, "fn main(){}\n");
}

#[test]
fn split_directives_recognizes_space_diagnostic() {
    let source = "diagnostic (off, derivative_uniformity);\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(dirs, "diagnostic (off, derivative_uniformity);\n");
    assert_eq!(body, "fn main(){}\n");
}

#[test]
fn split_directives_does_not_capture_diagnostic_prefixed_identifier() {
    // `diagnostic_counter` is a plain identifier; it is not a
    // directive and must not be hoisted above the preamble.  But the
    // realistic failure path is a top-level declaration whose RHS
    // happens to start with `diagnostic_...`.  In WGSL the top-level
    // line would start with `const`/`var`/etc., so the split would
    // terminate there.  Still, guard against pathological inputs
    // that begin a line with an identifier-looking token.
    let source = "diagnostic_counter_alias\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(
        dirs, "",
        "identifier starting with 'diagnostic' must not be treated as a directive"
    );
    assert_eq!(body, source);
}
