//! `#[path]` child of `lib.rs`, so `use super::*` reaches private items.

use super::*;

/// The CLI minifies on a big-stack worker; a deep-chain test overflows the
/// default test-thread stack in debug.  A panic inside `f` is re-raised.
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
    // `2.0 * d` stays a `Binary` init, the variant naga's wgsl-out aborts on.
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
    // naga's WGSL writer has `Statement::RayQuery => unreachable!()`.
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
    // The writer handles pipeline stages, payloads, and builtins; skipping
    // needlessly forgoes the byte comparison.
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
    // The front-end folds `2.0 * 3.0` to a `Literal` and `c` is a `Constant`,
    // both in naga's writable set.
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
    let config = Config::default();
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");

    if !gen_report.rolled_back {
        assert_eq!(
            gen_report.after_bytes,
            Some(output.source.len()),
            "after_bytes must match output source length"
        );
        assert!(gen_report.validation_ok);
        assert_eq!(gen_report.text_validation_ok, Some(true));
    }
    assert!(gen_report.before_bytes.is_some());
    assert!(gen_report.after_bytes.is_some());
    if gen_report.rolled_back {
        assert!(!gen_report.changed);
    }
}

// MARK: End-to-end preserve_symbols tests

/// Bare literal emission is safe only inside a type constructor (the type
/// pins the component) and as the RHS of an extracted `const` (every use
/// re-binds via abstract coercion); this stresses concrete whole-number
/// literals in positions that break if the bare form leaks elsewhere (a
/// standalone `let`, an `atan2` overload argument).
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
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

#[test]
fn e2e_later_declarations_see_earlier_constant_names() {
    // The declaration sections share one emission context and name each
    // constant in it as it is emitted, so a global initializer that reuses a
    // constant's tree prints the name instead of re-rendering it.
    let src = "const K: vec2<i32> = vec2<i32>(42, 43);\n\
               var<private> P: vec2<i32> = K;\n\
               @group(0) @binding(0) var<storage, read_write> out: array<i32>;\n\
               @compute @workgroup_size(1) fn main() { out[0] = P.x + P.y; }";
    let config = Config {
        preserve_symbols: vec!["K".to_string()],
        ..Default::default()
    };
    let out = run(src, &config).expect("minifies").source;
    assert!(
        out.contains("=K;") && out.matches("vec2i(42,43)").count() == 1,
        "the global initializer must reuse the constant's name: {out}"
    );
    io::validate_wgsl_text(&out).expect("output must be valid WGSL");
}

#[test]
fn e2e_library_constants_keep_their_concrete_type() {
    // `const NAME = <init>` prints no `: T`, so the init text must spell the
    // type: an abstract `1` or `vec2(42,43)` is a different constant that
    // naga's front end folds away instead of declaring.
    let src = "const PI: f32 = 3.1415927;\n\
               const IX: i32 = 7;\n\
               const CF: f32 = 1.0;\n\
               const VI: vec2<i32> = vec2<i32>(42, 43);\n\
               fn use_them(x: f32) -> f32 { return x * PI + f32(IX) + CF + f32(VI.x); }";
    let config = Config {
        profile: config::Profile::Max,
        preserve_symbols: ["PI", "IX", "CF", "VI"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        ..Default::default()
    };
    let once = run(src, &config).expect("library module minifies").source;
    let twice = run(&once, &config).expect("output re-minifies").source;
    assert_eq!(once, twice, "constants must survive a second pass");

    let module = io::parse_wgsl(&once).expect("output parses");
    let info = io::validate_module(&module).expect("output validates");
    for (name, expected) in [
        ("PI", naga::ScalarKind::Float),
        ("IX", naga::ScalarKind::Sint),
        ("CF", naga::ScalarKind::Float),
    ] {
        let (_, c) = module
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("`{name}` must survive as a constant: {once}"));
        assert_eq!(
            module.types[c.ty].inner.scalar_kind(),
            Some(expected),
            "`{name}` changed type: {once}"
        );
    }
    let _ = info;
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
    assert!(
        !output.source.contains("roughness"),
        "non-preserved member should be mangled: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must be valid WGSL");
}

/// Store-to-load forwarding is depth-capped (`SUBSTITUTION_DEPTH_CAP`): an
/// unbounded reassignment tree overflows recursive consumers (render_depth,
/// naga's writer, wasm's ~1 MB stack the big-stack worker does not cover)
/// and ships in the output, re-charging every re-minification.
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
/// literal, const_fold must pick the element so the composite dies instead
/// of shipping inline.
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

/// A preserved function keeps both definition and call sites: inlining the
/// body lets `naga::compact` cull the call-less declaration, and for a
/// `--preamble` input that body is only a stub, so the consumer's real
/// definition would be bypassed.
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
    // Function-scope names shadow module-scope type names in WGSL, so the
    // generator's struct names must avoid the rename pass's short
    // parameter/local names; enough globals and params here consume the
    // early names.
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
    // The output alone is incomplete; re-prepend the preamble to validate.
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
    assert!(
        msg.contains("bad_func"),
        "parse error should reference the problematic identifier: {msg}"
    );
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
fn run_keeps_the_binding_array_enable_the_output_still_needs() {
    // A source that opts into the wgpu extension is wgpu-facing: naga cannot
    // parse the output without the directive, and tint rejects this shape
    // (`binding_array` there takes two template arguments and a
    // sampled-texture element), so stripping would serve neither.
    let body = |enable: &str| {
        format!(
            "{enable}\n\
             @group(0) @binding(0) var arr: binding_array<texture_2d<f32>>;\n\
             @fragment fn main() -> @location(0) vec4<f32> {{\n\
               return textureLoad(arr[0], vec2<i32>(0, 0), 0);\n\
             }}"
        )
    };
    let output =
        run(&body("enable wgpu_binding_array;"), &Config::default()).expect("run should succeed");
    assert!(
        output.source.contains("enable wgpu_binding_array;"),
        "an opted-in extension the output still uses must survive: {}",
        output.source
    );
    assert!(
        output.source.contains("binding_array<"),
        "the binding_array type itself must survive minification: {}",
        output.source
    );
    io::validate_wgsl_text(&output.source).expect("output must re-parse");

    // Without the opt-in the shader targets tint, which needs no directive.
    let implicit = run(&body(""), &Config::default()).expect("run should succeed");
    assert!(
        !implicit.source.contains("enable wgpu_binding_array;"),
        "a tint-facing source must not gain a naga-only directive: {}",
        implicit.source
    );
}

/// Unparseable extension: ship the input compacted, no pass reports,
/// naga's error in `report.bailout`.  `subgroups` is naga 30's sole
/// `UnimplementedEnableExtension`; a release that lands it needs a new
/// trigger.
#[test]
fn unsupported_extension_bailout_sets_reason() {
    let src = "enable subgroups; // naga cannot parse this extension\n\
                   @compute @workgroup_size(1) fn m() {}";
    let output = run(src, &Config::default()).expect("bailout returns Ok");
    // Compacted, not verbatim: comments stripped, token-fusing joins kept.
    assert_eq!(
        output.source, "enable subgroups;@compute@workgroup_size(1)fn m(){}",
        "bailout must ship the lexically compacted source"
    );
    assert!(
        output.report.pass_reports.is_empty(),
        "the IR pipeline must not have run"
    );
    let reason = output
        .report
        .bailout
        .as_deref()
        .expect("bailout runs must carry the triggering naga error");
    assert!(
        reason.starts_with("naga cannot parse the input: "),
        "reason must name the stage that gave up: {reason}"
    );
}

/// Validator-reject twin: const division by zero parses but fails
/// validation; ship compacted with the reason, not `Err`.
#[test]
fn validation_bailout_sets_reason_and_ships_compacted() {
    let src = "@compute @workgroup_size(1) fn m() { var x = 1; let d = x / 0; }";
    let output = run(src, &Config::default()).expect("bailout returns Ok");
    assert_eq!(
        output.source, "@compute@workgroup_size(1)fn m(){var x=1;let d=x/0;}",
        "bailout must ship the lexically compacted source"
    );
    assert!(
        output.report.pass_reports.is_empty(),
        "the IR pipeline must not have run"
    );
    let reason = output
        .report
        .bailout
        .as_deref()
        .expect("bailout runs must carry the triggering naga error");
    assert!(
        reason.starts_with("naga rejects the input: "),
        "reason must name the stage that gave up: {reason}"
    );

    // Control: `Some` alone is the degradation signal.
    let ok = run(
        "@compute @workgroup_size(1) fn m() { var x = 1; let d = x / 2; }",
        &Config::default(),
    )
    .expect("valid shader minifies");
    assert!(ok.report.bailout.is_none());
}

/// The preamble guard fires on the validation bailout too: a body-leading
/// directive would land after the preamble's declarations.
#[test]
fn preamble_plus_validation_bailout_with_directives_hard_errors() {
    let src = "enable f16;\n@compute @workgroup_size(1) fn m() { var x = 1; let d = x / 0; }";
    let config = Config {
        preamble: Some("const K: f32 = 1.0;".to_string()),
        ..Default::default()
    };
    assert!(
        run(src, &config).is_err(),
        "directive-carrying validation bailout must not ship"
    );
}

// MARK: Pointer-parameter recovery
//
// naga's validator rejects workgroup/storage/uniform pointer arguments tint
// accepts (unrestricted_pointer_parameters); `specialize_ptr_params` recovers
// whole-variable call sites and the validation stand-in carries every other
// shape, so all run the full pipeline.

/// No bailout, the pass on the report, and the helper name mangled away
/// (a surviving name would prove the pipeline was skipped).
fn assert_ptr_param_recovered(src: &str, helper_name: &str) {
    let output = run(src, &Config::default()).expect("recovered run returns Ok");
    assert!(
        output.report.bailout.is_none(),
        "ptr-param shader must be specialized, not bailed out"
    );
    assert!(
        output
            .report
            .pass_reports
            .iter()
            .any(|p| p.pass_name == "specialize_ptr_params"),
        "recovery must be visible on the report"
    );
    assert!(
        !output.source.contains(helper_name),
        "full pipeline (incl. mangling) must have run, got: {}",
        output.source
    );
}

#[test]
fn ptr_workgroup_param_whole_var_root_is_recovered() {
    assert_ptr_param_recovered(
        "var<workgroup> sh: array<vec2f, 256>;\n\
         fn touch(a: ptr<workgroup, array<vec2f, 256>>, i: u32) { (*a)[i] = vec2f(1.0); }\n\
         @compute @workgroup_size(64) fn m(@builtin(local_invocation_id) lid: vec3u) {\n\
           touch(&sh, lid.x);\n\
           workgroupBarrier();\n\
         }",
        "touch",
    );
}

/// The spec's own directive for pointer parameters must reach the pass
/// instead of stopping at naga's parser (wgpu #5158).
#[test]
fn requires_unrestricted_pointer_parameters_shader_is_recovered() {
    assert_ptr_param_recovered(
        "requires unrestricted_pointer_parameters;\n\
         var<workgroup> sh: array<f32, 8>;\n\
         fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
         @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) { touch(&sh, l.x); }",
        "touch",
    );
}

/// The first clone of a specialized helper carries the helper's own name,
/// so the name map keys it by the original and `--preserve-symbol` keeps
/// it verbatim; a second root's clone carries a `_sp` suffix.
#[test]
fn ptr_param_clone_keeps_the_original_name() {
    let src = "var<workgroup> a: array<f32, 8>;\n\
               var<workgroup> b: array<f32, 8>;\n\
               fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }\n\
               @compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
                 touch(&a, l.x);\n\
                 touch(&b, l.x);\n\
               }";
    let output = run(src, &Config::default()).expect("recovers");
    let functions = output.name_map.expect("recovered run has a map").functions;
    assert!(
        functions.contains_key("touch") && functions.keys().any(|k| k.starts_with("touch_sp")),
        "one clone keyed by the original name, one suffixed: {functions:?}"
    );

    let preserved = run(
        src,
        &Config {
            preserve_symbols: vec!["touch".to_string()],
            ..Default::default()
        },
    )
    .expect("recovers");
    assert!(
        preserved.source.contains("fn touch("),
        "preserved helper must survive by name: {}",
        preserved.source
    );
    assert_eq!(preserved.name_map.expect("map").functions["touch"], "touch");
}

#[test]
fn ptr_param_element_chain_root_runs_the_full_pipeline() {
    // An element-rooted pointer (`&a[i]`) carries a call-site-dependent index
    // whole-var specialization cannot express, so the parameter survives and
    // the stand-in validates it.
    let src = "var<workgroup> a: array<f32, 64>;\n\
               fn setf(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
               @compute @workgroup_size(64) fn m(@builtin(local_invocation_id) lid: vec3u) {\n\
                 setf(&a[lid.x]);\n\
               }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(
        output.report.bailout.is_none(),
        "{:?}",
        output.report.bailout
    );
    assert!(
        !output.source.contains("setf") && output.source.contains("ptr<workgroup,f32>"),
        "mangled helper keeps its pointer parameter: {}",
        output.source
    );
    assert!(
        output.source.contains("(&") && output.source.contains("*"),
        "call passes the element address, the callee derefs: {}",
        output.source
    );
    assert!(output.name_map.is_some());
}

#[test]
fn library_module_pointer_helper_runs_the_full_pipeline() {
    // No entry point, so nothing can be specialized: the stand-in is the only
    // way in.  Shape of vgpu's fft-core module.
    let src = "const N: u32 = 8u;\n\
               fn cmul(a: vec2f, b: vec2f) -> vec2f {\n\
                 return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n\
               }\n\
               fn stage(a: ptr<workgroup, array<vec2f, 256>>, b: ptr<workgroup, array<vec2f, 256>>, lid: u32) {\n\
                 for (var s: u32 = 0u; s < N; s = s + 1u) {\n\
                   let half = 1u << s;\n\
                   if (lid < 128u) {\n\
                     let i0 = ((lid >> s) << (s + 1u)) + (lid & (half - 1u));\n\
                     let i1 = i0 + half;\n\
                     let w = vec2f(cos(f32(i0)), sin(f32(i0)));\n\
                     let a0 = (*a)[i0];\n\
                     let a1 = cmul(w, (*a)[i1]);\n\
                     (*a)[i0] = a0 + a1;\n\
                     (*a)[i1] = a0 - a1;\n\
                     (*b)[i0] = (*b)[i0] + cmul(w, (*b)[i1]);\n\
                   }\n\
                   workgroupBarrier();\n\
                 }\n\
               }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(
        output.report.bailout.is_none(),
        "{:?}",
        output.report.bailout
    );
    assert!(
        output.source.contains("ptr<workgroup,") && output.source.contains("(*"),
        "pointer parameters and their explicit derefs survive: {}",
        output.source
    );
    assert!(
        !output.source.contains("stage("),
        "mangled: {}",
        output.source
    );
    assert!(
        output.report.pass_reports.iter().all(|p| !p.rolled_back),
        "every pass must handle the pointer-parameter shape"
    );
    let again = run(&output.source, &Config::default()).expect("re-runs");
    assert!(again.report.bailout.is_none());
    assert!(again.source.len() <= output.source.len());
}

#[test]
fn storage_pointer_parameter_keeps_its_access_mode_and_array_length() {
    let src = "struct Buf { data: array<f32> }\n\
               @group(0) @binding(0) var<storage, read_write> buf: Buf;\n\
               fn bump(p: ptr<storage, array<f32>, read_write>, i: u32) {\n\
                 if (i < arrayLength(p)) { (*p)[i] += 1.0; }\n\
               }\n\
               @compute @workgroup_size(1) fn m(@builtin(global_invocation_id) g: vec3u) {\n\
                 bump(&buf.data, g.x);\n\
               }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(
        output.report.bailout.is_none(),
        "{:?}",
        output.report.bailout
    );
    assert!(
        output.source.contains(",read_write>"),
        "a writable storage pointer spells its access mode: {}",
        output.source
    );
    assert!(
        output.source.contains("arrayLength(") && !output.source.contains("arrayLength(&"),
        "a pointer parameter is already the pointer operand: {}",
        output.source
    );
}

#[test]
fn line_comments_end_at_every_wgsl_line_break() {
    // A `//` comment ends at any of LF/VT/FF/CR/NEL/LS/PS; ending only at `\n`
    // swallows the statement after a lone `\r` on the bailout paths (silent
    // loss, exit 0).
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
    // U+2118 is XID_Start but fails `is_alphanumeric`; fusing keyword and
    // identifier ships tint-invalid text on the bailout paths.
    let compacted = compact_wgsl_text("let \u{2118} = subgroupAdd(1.0);");
    assert!(
        compacted.starts_with("let \u{2118}"),
        "non-ASCII identifier fused with the keyword: {compacted}"
    );
}

/// A directive naga declines inside the PREAMBLE takes the same bailout as
/// one in a single file: the consumer's preamble carries it, so the compacted
/// body concatenates into a shader their compiler accepts.
#[test]
fn preamble_declared_unknown_directive_bails_out_with_body_compacted() {
    let config = Config {
        preamble: Some(
            "enable chromium_experimental_subgroup_matrix;\n\
             @group(0) @binding(0) var<storage, read_write> buf: array<i32>;"
                .to_string(),
        ),
        ..Config::default()
    };
    let body = "@compute @workgroup_size(64) fn m() { // uses the preamble's extension\n\
                subgroupMatrixStore(&buf, 0, subgroup_matrix_left<i8, 8, 8>(), false, 64); }";
    let output = run(body, &config).expect("bailout returns Ok");
    assert_eq!(
        output.source,
        "@compute@workgroup_size(64)fn m(){subgroupMatrixStore(&buf,0,subgroup_matrix_left<i8,8,8>(),false,64);}"
    );
    let reason = output.report.bailout.as_deref().expect("reason carried");
    assert!(
        reason.starts_with("naga cannot parse the preamble: "),
        "{reason}"
    );
    assert!(output.name_map.is_none());
}

#[test]
fn preamble_plus_bailout_with_directives_hard_errors() {
    // The bailout ships the body compacted with its leading `enable
    // subgroups;`, misplaced in the consumer's [preamble, body] order, so
    // preamble mode must refuse rather than ship a poisoned document with
    // exit 0.
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
    // The generator has no i16/u16 spelling, so int16 modules ship via the
    // naga fallback, whose text genuinely uses 16-bit tokens:
    // `strip_naga_only_enables` must keep `enable wgpu_int16;`.
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
fn preprocess_preserves_wgpu_directive_with_cr_only_endings() {
    // Lone `\r` endings must be normalised before any `.lines()` scan, or the
    // directive strip and `enable f16;` detection see one line; naga
    // implements `wgpu_binding_array`, so the directive itself is kept.
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
fn preprocess_does_not_inject_for_identifier_only() {
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

/// Blanked, not cut: every other byte keeps its offset so naga's diagnostics
/// still point at the user's columns.  Comments are not scanned.
#[test]
fn preprocess_blanks_requires_unrestricted_pointer_parameters() {
    let src = "requires unrestricted_pointer_parameters;\nfn f() {}\n";
    let out = preprocess_source_for_naga(src);
    assert_eq!(out.len(), src.len());
    assert!(!out.contains("requires"), "{out:?}");

    let src = "// requires unrestricted_pointer_parameters;\n\
               requires pointer_composite_access, unrestricted_pointer_parameters;\n";
    let out = preprocess_source_for_naga(src);
    assert_eq!(out.len(), src.len());
    assert!(out.contains("requires pointer_composite_access"), "{out:?}");
    assert_eq!(
        out.matches("unrestricted_pointer_parameters").count(),
        1,
        "only the comment's copy survives: {out:?}"
    );

    // A list broken across lines keeps its line count.
    let src =
        "requires\n  unrestricted_pointer_parameters,\n  pointer_composite_access;\nfn f() {}\n";
    let out = preprocess_source_for_naga(src);
    assert_eq!(out.len(), src.len());
    assert_eq!(out.lines().count(), src.lines().count(), "{out:?}");
    assert!(out.contains("pointer_composite_access;"), "{out:?}");
}

#[test]
fn preprocess_does_not_duplicate_enable_f16_with_extra_whitespace() {
    let src = "enable  f16;\nfn f() -> f16 { return 0h; }\n";
    let out = preprocess_source_for_naga(src);
    // The `enable` count locks against injecting a second normalised
    // `enable f16;` beside the source's own spelling.
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

// Fail when `UNSUPPORTED_EXTENSION_PATTERNS` or
// `KNOWN_TEXT_VALIDATION_LIMITATION_PATTERNS` drift from naga's phrasings.

/// Real parses, so a naga rewording fails here instead of silently turning
/// a bailout into a hard error.  `EnableExtensionNotSupported` is unreachable
/// through `parse_str` (every capability granted) and shares the
/// `extension is not` key by construction.
#[test]
fn unsupported_extension_patterns_track_naga_phrasings() {
    let declined = [
        "enable no_such_extension;",
        "requires no_such_extension;",
        "enable subgroups;",
        "requires unrestricted_pointer_parameters;",
        "@fragment fn m() -> @location(0) @blend_src(0) vec4f { return vec4f(); }",
    ];
    for src in declined {
        let err = io::parse_wgsl(src).expect_err("naga declines the directive");
        assert!(is_unsupported_extension_parse_error(&err), "{err}");
    }
    // naga quotes user identifiers on the first line: the bare word is no key.
    let err =
        io::parse_wgsl("fn m() { let x = extension_of_life; }").expect_err("unknown identifier");
    assert!(!is_unsupported_extension_parse_error(&err), "{err}");
}

/// A codespan snippet quoting a user comment that contains the phrasing must
/// not trigger the bailout; matching the whole rendered message swallows
/// real failures.
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

    // Same for the subgroups text-validation limitation.
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
    // A validator/emit/IO error quoting the phrasing must not short-circuit
    // `run` into the return-input branch and swallow a real failure.
    for ctor in [
        Error::Validation as fn(String) -> Error,
        Error::Emit as fn(String) -> Error,
        Error::Io as fn(String) -> Error,
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
    // Only `Parse` and `Validation` may opt into the round-trip bypass.
    for ctor in [
        Error::Emit as fn(String) -> Error,
        Error::Io as fn(String) -> Error,
    ] {
        let err = ctor("error: `subgroups` enable-extension is not yet supported".to_string());
        assert!(
            !is_known_text_validation_limitation(&err),
            "non-Parse/Validation error variants must not opt into the \
                 subgroup text-validation bypass: {err:?}"
        );
    }
}

/// Real parse, so a naga rewording (or subgroup support landing) fails here.
#[test]
fn known_text_validation_limitation_matches_subgroup_phrasing() {
    let err = io::parse_wgsl("enable subgroups;").expect_err("naga 30 declines subgroups");
    assert!(is_known_text_validation_limitation(&err), "{err}");
}

#[test]
fn known_text_validation_limitation_matches_validation_variant_too() {
    // `io::validate_wgsl_text` reports a not-yet-supported `enable` through
    // either `parse_wgsl` (Parse) or `validate_module_with_source`
    // (Validation); pins the Validation branch against a "Parse only"
    // tightening.
    let err = Error::Validation("error: `subgroups` enable-extension is not yet supported".into());
    assert!(
        is_known_text_validation_limitation(&err),
        "subgroup limitation phrasing must also be recognized when wrapped as Validation: {err}"
    );
}

#[test]
fn known_text_validation_limitation_rejects_unrelated_errors() {
    let err = Error::Validation("error: mismatched types".into());
    assert!(!is_known_text_validation_limitation(&err));
}

#[test]
fn preamble_with_enable_f16_shader() {
    // A preamble owning `enable f16;` plus an f16 body in compact mode: the
    // body must be directive-free (a directive after the preamble's globals
    // is illegal WGSL), and the shipped `[preamble, output]` must validate,
    // which exercises the `;`-aware `split_directives` on one-line output.
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
    let output = run(source, &config)
        .expect("enable f16; in preamble + f16 source must minify in compact mode");
    assert!(
        !output.source.contains("enable f16;"),
        "output body must not carry the directive when the preamble supplies it: {}",
        output.source
    );
    let shipped = format!("{preamble}\n{}", output.source);
    io::validate_wgsl_text(&shipped)
        .expect("the shipped [preamble, output] concatenation must be valid WGSL");
}

#[test]
fn preamble_missing_directive_errors_not_silent_ship() {
    // The directive can neither live in the output (after the preamble's
    // globals) nor be dropped, so no valid `[preamble, output]` exists; the
    // error must name the extension.
    let preamble = "\
            @group(0) @binding(0) var<uniform> bias: f32;\
        ";
    // The storage write keeps the f16 use alive through DCE.
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
    // `f16` in a comma-separated enable list supplies the directive like a
    // lone `enable f16;`; the missing-directive guard must not hard-error it.
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
fn an_opted_in_binding_array_needs_the_enable_in_the_preamble() {
    // The preamble owns every directive, so a wgpu-facing body whose preamble
    // never declares the extension would ship a document naga cannot parse.
    let source = "\
            enable wgpu_binding_array;\n\
            @group(0) @binding(0) var tex: binding_array<texture_2d<f32>>;\n\
            @fragment fn m() -> @location(0) vec4f {\n\
                return textureLoad(tex[0], vec2i(0), 0) * SCALE;\n\
            }";
    let err = run(
        source,
        &Config {
            preamble: Some("const SCALE: f32 = 2.0;".to_string()),
            ..Default::default()
        },
    )
    .err()
    .expect("a preamble without the extension cannot carry this body");
    assert!(
        err.to_string().contains("wgpu_binding_array"),
        "the error must name the missing directive: {err}"
    );

    let ok = run(
        source,
        &Config {
            preamble: Some("enable wgpu_binding_array;\nconst SCALE: f32 = 2.0;".to_string()),
            ..Default::default()
        },
    )
    .expect("a preamble that declares it minifies");
    assert!(
        !ok.source.contains("enable wgpu_binding_array;"),
        "the preamble owns the directive: {}",
        ok.source
    );
}

#[test]
fn binding_array_minifies_under_preamble_without_naga_only_enable() {
    // tint needs no enable for binding arrays, so the shipped body omits the
    // naga-only `enable wgpu_binding_array;` and the preamble need not
    // declare it; the self-check must inject it for its naga re-parse only.
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
    // naga needs the enable, so prepend it for the round-trip.
    let shipped = format!("enable wgpu_binding_array;\n{preamble}\n{}", output.source);
    io::validate_wgsl_text(&shipped).expect("[preamble, body] must round-trip through naga");
}

// MARK: Splat elision tests

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
    // When both the generator and naga's wgsl-out fallback emit text naga's
    // frontend rejects, `run()` must error rather than ship; if a naga release
    // accepts these forms the Ok branch simply validates clean.
    let inputs = [
        // naga const-substitutes the f64 literal under the cast, then rejects
        // `f32(<F64 literal>)`.
        "@group(0) @binding(0) var<storage, read_write> s: f32;\n\
             @compute @workgroup_size(1) fn m() { let a: f64 = 0.5lf; s = f32(a); }",
        // An f16 cast whose `enable f16;` naga's backend drops.
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
    // naga const-substitutes `let a: f64 = 2.5lf` into the cast, yielding
    // `As { Literal(F64), convert }` whose `f32(2.5lf)` rendering naga
    // rejects on re-parse; const_fold folds it to the converted literal.
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
        assert!(
            !output.source.contains("lf)"),
            "f64 cast literal should be folded, not emitted as a cast: {}",
            output.source
        );
    }
}

#[test]
fn folds_f64_vector_narrowing_cast_to_valid_constructor() {
    // const_fold cannot materialise the converted vector (no arena handles
    // for the F32 components), so the generator folds the cast to a converted
    // constructor.
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
        assert!(
            !output.source.contains("lf)"),
            "f64 vector cast should be folded, not emitted: {}",
            output.source
        );
    }
}

#[test]
fn folds_int64_narrowing_cast_to_valid_literal() {
    // `u32(<U64 literal>)` is rejected on re-parse like the f64 cast and
    // naga's backend emits the same token; the fold wraps on narrowing
    // (`u32(i64(-1))` is `4294967295`, not a clamped `0`), matching naga's
    // conversion semantics.
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
    // Scalar-vector broadcasting makes the splat redundant.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                var c = vec3f(0.5, 0.6, 0.7);
                c = vec3f(1.0) + c;
                return vec4f(c, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
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
    let src = r#"
            fn helper(v: vec3f) -> vec3f {
                return v / vec3f(dot(v, v));
            }
            @fragment fn main() -> @location(0) vec4f {
                return vec4f(helper(vec3f(1.0, 2.0, 3.0)), 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        !out.contains("vec3f(dot") && !out.contains("vec3<f32>(dot"),
        "splat wrapping dot() should be elided in division: {out}"
    );
}

#[test]
fn splat_elision_compound_assign() {
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
    // At most one side may elide or the sum becomes a scalar.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let a = 1.0;
                let b = 2.0;
                let c = vec3f(a) + vec3f(b);
                return vec4f(c, 1.0);
            }
        "#;
    let out = minify_and_validate(src);
    assert!(!out.is_empty());
}

#[test]
fn splat_elision_skipped_when_other_is_scalar() {
    // `1.0 * scalar` would be a scalar.
    let src = r#"
            struct S { b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            @vertex fn main() -> @builtin(position) vec4f {
                return vec4f(1.0) * u.b;
            }
        "#;
    let out = minify_and_validate(src);
    assert!(
        out.contains("vec4"),
        "splat must not be elided when other operand is scalar: {out}"
    );
}

#[test]
fn splat_elision_skipped_when_other_is_scalar_rhs() {
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
    // A comparison against a bare scalar is not valid WGSL.
    let src = r#"
            @fragment fn main() -> @location(0) vec4f {
                let v = vec3f(1.0, 2.0, 3.0);
                let mask = v > vec3f(1.5);
                return select(vec4f(0), vec4f(1), mask.x);
            }
        "#;
    minify_and_validate(src);
}

// MARK: Last-store inlining tests

#[test]
fn last_store_inlined_when_earlier_loads_keep_var_alive() {
    // The pre-store load keeps `m` alive; the post-store load still inlines
    // the stored expression.
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
    assert!(!out.is_empty(), "output should not be empty");
}

#[test]
fn last_store_not_inlined_when_escaped() {
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
    // The partial store reads the full value.
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
    // Only the specific dead Store may go: the conditional store feeds the
    // final `log(1+m)` read.
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
    assert!(
        out.contains("if"),
        "conditional branch must be preserved (store to m is live): {out}"
    );
}

#[test]
fn last_store_init_preserved_when_loop_reads_var() {
    // The init's seeded load is forwarded, but the loop body (cache cleared)
    // still reads the init on its first iteration.
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
    assert!(
        out.contains("abs"),
        "loop body with transform must be preserved: {out}"
    );
}

#[test]
fn last_store_in_loop_not_removed() {
    // Loads before the Store in the loop body observe it via the back-edge on
    // the next iteration.
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
    // `.zxy` is the second store's telltale.
    assert!(
        out.contains(".zxy"),
        "second store in loop (with .zxy swizzle) must be preserved: {out}"
    );
}

// MARK: Name map

/// Every surviving module-scope symbol keyed by ORIGINAL name, values
/// present in the shipped text, entry points identity, members through
/// the generator's tables.
#[test]
fn name_map_maps_originals_to_shipped_names() {
    let src = "struct Params { scale_factor: f32, offset_amount: f32, }\n\
               @group(0) @binding(0) var<uniform> long_params: Params;\n\
               fn compute_value(x: f32) -> f32 { return x * long_params.scale_factor + long_params.offset_amount; }\n\
               @fragment fn fs_main() -> @location(0) vec4f {\n\
                 return vec4f(compute_value(1.0));\n\
               }";
    let output = run(src, &Config::default()).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("pipeline runs carry a map");

    let global = map.globals.get("long_params").expect("binding is mapped");
    assert!(
        output.source.contains(global.as_str()),
        "mapped global name {global} must appear in the output: {}",
        output.source
    );
    assert_ne!(global, "long_params", "max profile mangles the binding");

    assert_eq!(
        map.entry_points.get("fs_main").map(String::as_str),
        Some("fs_main"),
        "entry points are identity"
    );

    // `compute_value` may be renamed or inlined away; either way the map
    // must agree with the text.
    match map.functions.get("compute_value") {
        Some(new) => assert!(output.source.contains(new.as_str())),
        None => assert!(!output.source.contains("compute_value")),
    }

    let params = map.structs.get("Params").expect("uniform struct is mapped");
    assert!(output.source.contains(params.name.as_str()));
    let member = params
        .members
        .get("scale_factor")
        .expect("member is mapped");
    assert!(
        output.source.contains(member.as_str()),
        "mapped member {member} must appear in the output: {}",
        output.source
    );
}

/// Preserved symbols appear as identity entries.
#[test]
fn name_map_preserved_symbols_are_identity() {
    let src = "@group(0) @binding(0) var<uniform> kept_name: f32;\n\
               @fragment fn fs_main() -> @location(0) vec4f { return vec4f(kept_name); }";
    let config = Config {
        preserve_symbols: vec!["kept_name".to_string()],
        ..Default::default()
    };
    let output = run(src, &config).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("map present");
    assert_eq!(
        map.globals.get("kept_name").map(String::as_str),
        Some("kept_name")
    );
}

/// Absence means eliminated, never unchanged.
#[test]
fn name_map_omits_eliminated_declarations() {
    let src = "fn never_called() -> f32 { return 1.0; }\n\
               @fragment fn fs_main() -> @location(0) vec4f { return vec4f(0.0); }";
    let output = run(src, &Config::default()).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("map present");
    assert!(
        !map.functions.contains_key("never_called"),
        "dead function must be absent from the map"
    );
}

/// Bailouts ship input names: no map.
#[test]
fn name_map_is_none_on_bailout() {
    let src = "@compute @workgroup_size(1) fn m() { var x = 1; let d = x / 0; }";
    let output = run(src, &Config::default()).expect("bailout returns Ok");
    assert!(output.report.bailout.is_some(), "fixture must bail");
    assert!(output.name_map.is_none());
}

// MARK: Review-round regressions

/// A preamble-declared banned helper is frozen (its text re-ships verbatim,
/// so no specialization), and the stand-in still lets the body minify
/// against it.
#[test]
fn preamble_declared_ptr_param_helper_runs_the_pipeline() {
    let preamble = "var<workgroup> sh: array<f32, 8>;\n\
                    fn touch(p: ptr<workgroup, array<f32, 8>>, i: u32) { (*p)[i] = 1.0; }";
    let src = "@compute @workgroup_size(8) fn m(@builtin(local_invocation_id) l: vec3u) {\n\
                 touch(&sh, l.x);\n\
               }";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(src, &config).expect("runs");
    assert!(
        output.report.bailout.is_none(),
        "{:?}",
        output.report.bailout
    );
    assert!(
        output.source.contains("touch(&sh,"),
        "the body keeps calling the preamble's helper by name: {}",
        output.source
    );
    assert!(
        !output.source.contains("fn touch("),
        "preamble-owned text is not re-emitted: {}",
        output.source
    );
}

/// A constant whose every use folds away gets no declaration and must not
/// be mapped.
#[test]
fn name_map_omits_constants_folded_out_of_the_text() {
    let src = "const MODULE_CONST: f32 = 3.25;\n\
               @fragment fn fs_main() -> @location(0) vec4f { return vec4f(MODULE_CONST); }";
    let output = run(src, &Config::default()).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("map present");
    for (orig, new) in &map.constants {
        assert!(
            output.source.contains(new.as_str()),
            "constants entry {orig}->{new} missing from output"
        );
    }
}

/// Dead structs and naga-predeclared result types are never declared, so
/// never mapped.
#[test]
fn name_map_omits_undeclared_structs() {
    let src = "struct Dead { x: f32, }\n\
               @fragment fn fs_main() -> @location(0) vec4f {\n\
                 let r = frexp(1.5);\n\
                 return vec4f(r.fract);\n\
               }";
    let output = run(src, &Config::default()).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("map present");
    assert!(
        !map.structs.contains_key("Dead"),
        "dead struct must be absent from the map"
    );
    assert!(
        !map.structs.keys().any(|k| k.starts_with("__")),
        "predeclared result types must be absent, got {:?}",
        map.structs.keys().collect::<Vec<_>>()
    );
}

/// An `@id` override is renameable and mapped; an `@id`-less one is the
/// host's pipeline-constant key and must stay identity.
#[test]
fn name_map_covers_both_override_flavors() {
    let src = "@id(7) override numbered_override: f32 = 1.0;\n\
               override named_override: f32 = 2.0;\n\
               @fragment fn fs_main() -> @location(0) vec4f {\n\
                 return vec4f(numbered_override + named_override);\n\
               }";
    let output = run(src, &Config::default()).expect("valid shader minifies");
    let map = output.name_map.as_ref().expect("map present");
    let numbered = map
        .overrides
        .get("numbered_override")
        .expect("@id override mapped");
    assert!(output.source.contains(numbered.as_str()));
    assert_eq!(
        map.overrides.get("named_override").map(String::as_str),
        Some("named_override"),
        "@id-less override is the host's pipeline-constant key and must be identity"
    );
}

#[test]
fn an_extracted_literal_never_takes_a_preserved_name() {
    // A preamble binding that DCE prunes leaves no arena entry, so only the
    // preserve list stands between the minted `const` and a redefinition.
    let preamble = "var<private> A: f32;\n\
                    @group(0) @binding(0) var<storage, read_write> Buf: array<f32>;";
    let src = "@compute @workgroup_size(1) fn main() {\n\
                 Buf[0] = 1234.5678; Buf[1] = 1234.5678; Buf[2] = 1234.5678;\n\
                 Buf[3] = 1234.5678; Buf[4] = 1234.5678;\n\
               }";
    let output = run(
        src,
        &Config {
            preamble: Some(preamble.to_string()),
            ..Default::default()
        },
    )
    .expect("a pruned preamble binding must not block minification");
    assert!(
        !output.source.contains("const A="),
        "the minted constant must avoid the preamble name: {}",
        output.source
    );
    io::validate_wgsl_text(&format!("{preamble}\n{}", output.source))
        .expect("the shipped concatenation must be valid");
}
