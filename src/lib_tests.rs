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
    assert!(
        gen_report.before_bytes.is_none(),
        "the plain path renders no naga baseline"
    );
    assert!(gen_report.after_bytes.is_some());
    assert_eq!(
        gen_report.changed,
        output.source != TRIVIAL_SHADER,
        "changed says whether the output differs from the input"
    );
}

/// A full-fidelity run renders naga's baseline and reports the emit
/// against it; `changed` means the output differs from the input.
#[test]
fn generator_emit_reports_before_bytes_only_under_full_fidelity() {
    let config = Config {
        trace: config::TraceConfig {
            enabled: false,
            validate_each_pass: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let output = run(TRIVIAL_SHADER, &config).unwrap();
    let gen_report = output
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(gen_report.before_bytes.is_some_and(|n| n > 0));
    assert_eq!(gen_report.after_bytes, Some(output.source.len()));
    assert_eq!(gen_report.changed, output.source != TRIVIAL_SHADER);
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

/// A body enabling `f16` behind a preamble that does not: the shipped
/// [preamble, body] document would lack the directive, and the error says
/// which one to move rather than quoting the self-check's re-parse.
#[test]
fn preamble_missing_the_bodys_enable_names_the_directive() {
    let config = Config {
        preamble: Some("const K: u32 = 1u;".to_string()),
        ..Default::default()
    };
    let source = "enable f16;\n@group(0) @binding(0) var<storage, read_write> o: f16;\n\
                  @compute @workgroup_size(1) fn main() { o = f16(K) + 1h; }";
    let err = match run(source, &config) {
        Err(e) => e.to_string(),
        Ok(out) => panic!("accepted: {}", out.source),
    };
    assert!(err.contains("add `enable f16;` to the preamble"), "{err}");
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

/// A directive naga's front end declines fails the run.  Deciding otherwise
/// would take reading naga's prose, and a message it never promised to keep
/// cannot be allowed to choose between "ship" and "fail".
#[test]
fn a_directive_naga_cannot_parse_is_a_hard_error() {
    let src = "enable subgroups;\n@compute @workgroup_size(1) fn m() {}";
    let Err(err) = run(src, &Config::default()) else {
        panic!("naga 30 declines `subgroups`");
    };
    assert!(matches!(err, Error::Parse(_)), "{err:?}");
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
                 touch(&a, l.y);\n\
                 touch(&b, l.x);\n\
                 touch(&b, l.y);\n\
               }";
    // Each clone is called twice: called once, it would be spliced away.
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

/// Minted type names go to the most-used types first: an alias spelled
/// dozens of times takes the earlier letter over a struct declared first
/// but spelled once, and the output re-minifies to itself.
#[test]
fn type_names_are_assigned_by_use_count() {
    // Thirty typed globals spell `vec4f` thirty times; the struct is spelled
    // once, by the uniform.
    let globals: String = (0..30)
        .map(|i| format!("var<private> g{i}: vec4f;\n"))
        .collect();
    let sum: String = (0..30).map(|i| format!("acc += g{i};\n")).collect();
    let src = format!(
        "struct S {{ x: f32 }}\n@group(0) @binding(0) var<uniform> s: S;\n{globals}\
         @fragment fn m(@location(0) k: f32) -> @location(0) vec4f {{\n\
         var acc = vec4f(k);\n{sum}return acc + vec4f(s.x);\n}}"
    );
    let output = run(&src, &Config::default()).expect("runs");
    let alias = output
        .source
        .split("alias ")
        .nth(1)
        .and_then(|rest| rest.split('=').next())
        .expect("an alias for vec4f is minted");
    let strukt = output
        .source
        .split("struct ")
        .nth(1)
        .and_then(|rest| rest.split('{').next())
        .expect("the struct survives");
    let position = |name: &str| {
        let mut counter = 0;
        std::iter::repeat_with(|| crate::name_gen::next_name(&mut counter))
            .position(|n| n == name)
            .expect("a minted name")
    };
    assert!(
        position(alias) < position(strukt),
        "the alias ({alias}) must take the earlier letter over the struct ({strukt}): {}",
        output.source
    );
    let again = run(&output.source, &Config::default()).expect("re-runs");
    assert_eq!(
        again.source, output.source,
        "the output re-minifies to itself"
    );
}

/// A const composite read through a chain of element picks - arrays of
/// arrays, an array of structs, an array of matrices - folds to its leaf
/// once the splice puts the composite under the chain, where naga's
/// front-end no longer sees it; a runtime index keeps its access.
#[test]
fn composite_element_chains_fold_after_splicing() {
    let src = "struct P { a: vec2u, b: u32 }\n\
               @group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
               @group(0) @binding(1) var<storage, read_write> f: array<f32>;\n\
               fn zeros(a: array<array<u32, 4>, 3>) -> u32 { return a[2][3]; }\n\
               fn member(a: array<P, 2>) -> u32 { return a[1].a.y; }\n\
               fn column(a: array<mat2x2f, 2>) -> f32 { return a[1][1].x; }\n\
               fn dynamic(a: array<u32, 2>, i: u32) -> u32 { return a[i]; }\n\
               @compute @workgroup_size(1) fn m(@builtin(global_invocation_id) id: vec3u) {\n\
                 o[0] = zeros(array<array<u32, 4>, 3>());\n\
                 o[1] = member(array(P(vec2u(1u, 2u), 3u), P(vec2u(4u, 5u), 6u)));\n\
                 f[0] = column(array(mat2x2f(0.0, 1.0, 2.0, 3.0), mat2x2f(4.0, 5.0, 6.0, 7.0)));\n\
                 o[2] = dynamic(array(7u, 8u), id.x);\n\
               }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(output.report.bailout.is_none());
    for expected in ["[0]=0;", "[1]=5;", "[0]=6;"] {
        assert!(
            output.source.contains(expected),
            "{expected}: {}",
            output.source
        );
    }
    assert!(
        output.source.contains("array(7,8)[") || output.source.contains("array<u32,2>(7,8)["),
        "a runtime index keeps its access: {}",
        output.source
    );
}

/// A guard folded into `if !c { .. }` spells the negation without
/// parentheses around a call, as the same text re-minified would.
#[test]
fn negated_guard_condition_spells_a_call_bare() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
               @compute @workgroup_size(1) fn m(@builtin(global_invocation_id) id: vec3u) {\n\
                 if (any(id.xy >= vec2u(4u))) { return; }\n\
                 o[id.x] = 1u;\n\
               }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(
        output.source.contains("if !any(") && !output.source.contains("!("),
        "{}",
        output.source
    );
}

/// A void helper whose tail `if` carries the returns naga's front-end
/// synthesises is spliced once the IR drops them as the text would: one
/// `fn` survives.
#[test]
fn void_helper_with_a_tail_if_is_spliced() {
    let src = "var<private> acc: u32;\n\
               fn bump(c: bool) { if (c) { acc = acc + 1u; } }\n\
               @compute @workgroup_size(1) fn m(@builtin(global_invocation_id) id: vec3u) { bump(id.x > 4u); }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(output.report.bailout.is_none());
    assert_eq!(output.source.matches("fn ").count(), 1, "{}", output.source);
}

/// A single-call helper whose only early return is a pure guard is merged
/// into one `select` return and then spliced: no definition survives, and
/// the guard is spelled as the `select`.
#[test]
fn guarded_single_call_helper_is_merged_and_spliced() {
    let src = "fn safe_sqrt(x: f32) -> f32 { if (x < 0.0) { return 0.0; } return sqrt(x); }\n\
               @fragment fn m(@location(0) x: f32) -> @location(0) vec4f { return vec4f(safe_sqrt(x)); }";
    let output = run(src, &Config::default()).expect("runs");
    assert!(output.report.bailout.is_none());
    assert_eq!(output.source.matches("fn ").count(), 1, "{}", output.source);
    assert!(output.source.contains("select("), "{}", output.source);
}

#[test]
fn ptr_param_element_chain_root_runs_the_full_pipeline() {
    // An element-rooted pointer (`&a[i]`) carries a call-site-dependent index
    // whole-var specialization cannot express, so the parameter survives and
    // the stand-in validates it.  Two call sites: called once, the helper
    // would be spliced and the parameter dissolved.
    let src = "var<workgroup> a: array<f32, 64>;\n\
               fn setf(p: ptr<workgroup, f32>) { *p = 1.0; }\n\
               @compute @workgroup_size(64) fn m(@builtin(local_invocation_id) lid: vec3u) {\n\
                 setf(&a[lid.x]);\n\
                 setf(&a[lid.y]);\n\
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
    // way in.  The shape of an FFT library module: `ptr<workgroup>` stage
    // helpers behind a `const` size.
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
    // Two call sites: called once, the helper would be spliced away.
    let src = "struct Buf { data: array<f32> }\n\
               @group(0) @binding(0) var<storage, read_write> buf: Buf;\n\
               fn bump(p: ptr<storage, array<f32>, read_write>, i: u32) {\n\
                 if (i < arrayLength(p)) { (*p)[i] += 1.0; }\n\
               }\n\
               @compute @workgroup_size(1) fn m(@builtin(global_invocation_id) g: vec3u) {\n\
                 bump(&buf.data, g.x);\n\
                 bump(&buf.data, g.y);\n\
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

/// The preamble parses at its own call site, which fails the run the same way
/// the body's does.
#[test]
fn a_directive_naga_cannot_parse_in_the_preamble_is_a_hard_error() {
    let config = Config {
        preamble: Some(
            "enable chromium_experimental_subgroup_matrix;\n\
             @group(0) @binding(0) var<storage, read_write> buf: array<i32>;"
                .to_string(),
        ),
        ..Config::default()
    };
    let body = "@compute @workgroup_size(64) fn m() { \
                subgroupMatrixStore(&buf, 0, subgroup_matrix_left<i8, 8, 8>(), false, 64); }";
    let Err(err) = run(body, &config) else {
        panic!("naga declines the preamble's directive");
    };
    assert!(matches!(err, Error::Parse(_)), "{err:?}");
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

/// A preamble struct MEMBER name lives in its struct's namespace; it must not
/// hide a same-named user declaration.  An entry point is the silent case:
/// nothing in the shader references it by name, so the self-check passed on
/// the shortened body and the run reported a saving.
#[test]
fn preamble_member_names_do_not_hide_user_declarations() {
    let preamble = "struct Inputs { main: f32, size: vec2f }\n\
                    @group(0) @binding(0) var<uniform> inputs: Inputs;";
    let source = "@fragment fn main() -> @location(0) vec4f { \
                  return vec4f(inputs.main, inputs.size, 1.0); }";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    assert!(
        output.source.contains("fn main"),
        "the entry point must survive: {}",
        output.source
    );
}

/// The preamble owns every directive, so a module-level `diagnostic(...)`
/// only the body declares would be dropped with the directive block - and
/// naga's validator has no uniformity analysis to notice.  It is an error;
/// the same directive in the preamble is fine.
#[test]
fn preamble_run_rejects_a_body_only_diagnostic_directive() {
    let decls = "@group(0) @binding(0) var t: texture_2d<f32>;\n\
                 @group(0) @binding(1) var s: sampler;";
    let body = "@fragment fn main(@location(0) uv: vec2f, @location(1) k: f32) -> @location(0) vec4f {\n\
                  if (k > 0.5) { return textureSample(t, s, uv); }\n  return vec4f(0.0);\n}";
    let directive = "diagnostic(off, derivative_uniformity);";
    let config = |preamble: String| Config {
        preamble: Some(preamble),
        ..Default::default()
    };
    let err = run(&format!("{directive}\n{body}"), &config(decls.to_string()))
        .err()
        .expect("a body-only directive must be refused");
    assert!(err.to_string().contains("diagnostic("), "{err}");
    let output = run(body, &config(format!("{directive}\n{decls}"))).unwrap();
    assert!(
        !output.source.contains("diagnostic("),
        "the preamble's directive is not re-emitted: {}",
        output.source
    );
}

/// A resource stays in the entry point's interface after nagami's own
/// passes kill its last read - preserved by name or not, as a phony
/// assignment - and an unused named override stays a valid
/// pipeline-constant key.
#[test]
fn preserved_resources_and_named_overrides_survive_dce() {
    let source = "@group(0) @binding(0) var<storage, read_write> a: array<u32>;\n\
                  @group(0) @binding(1) var<storage, read> b: array<u32>;\n\
                  override unused_ov: f32 = 1.0;\n\
                  @compute @workgroup_size(1) fn main() { let x = b[0]; a[0] = 1u; }";
    let config = Config {
        preserve_symbols: vec!["b".to_string()],
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    assert!(
        output.source.contains("b:array<u32>")
            && output.source.contains("{_=&b;")
            && output.source.contains("override unused_ov"),
        "{}",
        output.source
    );
    let output = run(source, &Config::default()).unwrap();
    assert!(
        output.source.contains("binding(1)")
            && output.source.contains("{_=&")
            && output.source.contains("override unused_ov"),
        "{}",
        output.source
    );
}

/// The spec's idiom for putting a resource in the interface: every phony
/// assignment keeps its binding, spelled `_=N;`, or `_=&N;` where the store
/// type cannot be loaded.  The text passes the self-check, so no rung of
/// the ladder was taken.
#[test]
fn phony_assignments_keep_their_bindings() {
    let source = "@group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @group(0) @binding(2) var<uniform> u: vec4f;
        @group(0) @binding(3) var<storage, read_write> big: array<u32>;
        @group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;
        @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            _ = t; _ = s; _ = u; _ = &big; _ = &counter;
            return vec4f(uv, 0.0, 1.0);
        }";
    let output = run(source, &Config::default()).unwrap();
    assert!(output.report.fallback.is_none() && output.report.bailout.is_none());
    for binding in 0..5 {
        assert!(
            output.source.contains(&format!("binding({binding})")),
            "{}",
            output.source
        );
    }
    assert_eq!(output.source.matches("_=").count(), 5, "{}", output.source);
    assert_eq!(output.source.matches("_=&").count(), 2, "{}", output.source);
    let again = run(&output.source, &Config::default()).unwrap();
    assert_eq!(again.source, output.source);
}

/// A read the passes fold away was a static access: the binding, its
/// struct and one `_=N;` stay, per entry point that had it.
#[test]
fn a_dead_read_keeps_the_binding_accessed_per_entry_point() {
    let source = "struct Inputs { a: f32, b: f32, c: f32 }
        @group(0) @binding(0) var<uniform> inputs: Inputs;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(1) fn dead() { let x = inputs.a; let y = inputs.b; o[0] = 1.0; }
        @compute @workgroup_size(1) fn live() { o[1] = inputs.c; }
        @compute @workgroup_size(1) fn none() { o[2] = 2.0; }";
    let output = run(source, &Config::default()).unwrap();
    assert!(output.report.fallback.is_none() && output.report.bailout.is_none());
    assert!(output.source.contains("binding(0)"), "{}", output.source);
    assert_eq!(output.source.matches("_=").count(), 1, "{}", output.source);
    let dead = output.source.split("fn dead(").nth(1).unwrap();
    assert!(dead.starts_with(")"), "{}", output.source);
    assert!(
        dead.split('}').next().unwrap().contains("{_="),
        "{}",
        output.source
    );
}

/// What the input never referenced is no access; what it referenced only
/// beside a live reference needs no phony of its own.
#[test]
fn an_unreferenced_binding_goes_and_a_redundant_phony_goes() {
    let unreferenced = "@group(0) @binding(0) var<uniform> u: vec4f;
        @group(0) @binding(1) var<storage, read_write> o: vec4f;
        @compute @workgroup_size(1) fn main() { o = vec4f(1.0); }";
    let output = run(unreferenced, &Config::default()).unwrap();
    assert!(
        !output.source.contains("binding(0)") && !output.source.contains("_="),
        "{}",
        output.source
    );
    let redundant = "@group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var s: sampler;
        @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
            _ = t;
            return textureSample(t, s, uv);
        }";
    let output = run(redundant, &Config::default()).unwrap();
    assert!(!output.source.contains("_="), "{}", output.source);
    assert!(output.source.len() < redundant.len());
}

/// dead_branch rebuilds the arena from the body; the pin, which no
/// statement reaches, is rooted like naga's compactor roots it.
#[test]
fn a_dead_branch_keeps_its_binding_accessed() {
    let source = "const DEBUG = false;
        @group(0) @binding(0) var<storage, read> probe: array<u32, 4>;
        @group(0) @binding(1) var<storage, read_write> o: array<u32>;
        @compute @workgroup_size(1) fn main() {
            var x = 1u;
            if DEBUG { x = probe[0]; }
            o[0] = x;
        }";
    let output = run(source, &Config::default()).unwrap();
    assert!(output.report.fallback.is_none() && output.report.bailout.is_none());
    assert!(!output.source.contains("if"), "{}", output.source);
    assert!(
        output.source.contains("binding(0)") && output.source.contains("{_="),
        "{}",
        output.source
    );
    assert!(!output.source.contains("_=&"), "{}", output.source);
}

/// A helper's dead read is pinned where the helper is: one phony serves
/// every entry point that calls it, and none appears in the entry points.
#[test]
fn a_helper_pins_for_its_callers() {
    let source = "@group(0) @binding(0) var<uniform> u: vec4f;
        @group(0) @binding(1) var<storage, read_write> o: array<vec4f>;
        fn helper(i: u32) -> vec4f { let dead = u; return vec4f(f32(i)); }
        @compute @workgroup_size(1) fn a() { o[0] = helper(0u); }
        @compute @workgroup_size(1) fn b() { o[1] = helper(1u); }
        @compute @workgroup_size(1) fn c() { o[2] = helper(2u); }";
    let config = Config {
        // The helper stays a function.
        preserve_symbols: vec!["helper".to_string()],
        ..Default::default()
    };
    let output = run(source, &config).unwrap();
    assert!(output.report.fallback.is_none() && output.report.bailout.is_none());
    assert_eq!(output.source.matches("_=").count(), 1, "{}", output.source);
    assert!(
        output
            .source
            .split("fn helper(")
            .nth(1)
            .unwrap()
            .contains("{_="),
        "{}",
        output.source
    );
}

/// A library function keeps the accesses of its call graph: the host
/// composes it under an entry point whose interface they join.
#[test]
fn a_library_function_keeps_its_accesses() {
    let source = "@group(0) @binding(0) var t: texture_2d<f32>;
        @group(0) @binding(1) var<uniform> u: vec4f;
        fn leaf() -> f32 { _ = t; return 1.0; }
        fn root(x: f32) -> f32 { let dead = u.x; return leaf() + x; }";
    let output = run(source, &Config::default()).unwrap();
    assert!(
        output.report.fallback.is_none() && output.report.bailout.is_none(),
        "{:?}",
        output.report
    );
    assert_eq!(output.source.matches("_=").count(), 2, "{}", output.source);
}

/// A binding array is neither a texture nor a pointer, so its pin is its
/// first element, `_=N[0];`, which tint accepts.
#[test]
fn a_binding_array_pins_by_its_first_element() {
    let source = "@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
        @fragment fn fs() { let dead = textureLoad(textures[0], vec2(0, 0), 0); }";
    let output = run(source, &Config::default()).unwrap();
    assert!(output.report.fallback.is_none() && output.report.bailout.is_none());
    assert!(
        output.source.contains("{_=") && output.source.contains("[0];}"),
        "{}",
        output.source
    );
    let again = run(&output.source, &Config::default()).unwrap();
    assert_eq!(again.source, output.source);
}

/// naga allows an immediate no bare reference, so its pin is the loaded
/// form, `_=pc;` all the same; a helper's serves every caller.
#[test]
fn an_immediate_pins_as_its_load() {
    let source = "var<immediate> a: i32;
        var<immediate> b: i32;
        var<immediate> c: i32;
        fn uses_a() { let foo = a; }
        fn uses_uses_a() { uses_a(); }
        fn uses_b() { let foo = b; }
        @compute @workgroup_size(1) fn main1() { uses_a(); }
        @compute @workgroup_size(1) fn main2() { uses_uses_a(); }
        @compute @workgroup_size(1) fn main3() { uses_b(); }
        @compute @workgroup_size(1) fn main4() { }";
    let output = run(source, &Config::default()).unwrap();
    assert!(
        output.report.fallback.is_none() && output.report.bailout.is_none(),
        "{:?}",
        output.report
    );
    assert_eq!(output.source.matches("_=").count(), 2, "{}", output.source);
    assert_eq!(
        output.source.matches("var<immediate>").count(),
        2,
        "{}",
        output.source
    );
    let again = run(&output.source, &Config::default()).unwrap();
    assert_eq!(again.source, output.source);
}

/// The emitter drops a statement of its own accord - the no-op `p = p`,
/// where naga's front end left it - so the pin is decided by what the
/// text rendered, not by what the arena holds: the generator's own text
/// keeps the access, and no rung of the ladder is taken.
#[test]
fn an_identity_store_the_emitter_drops_keeps_the_binding_accessed() {
    let source = "struct S { m: mat2x2<f32> }
        @group(0) @binding(0) var<storage, read_write> ssbo: S;
        @compute @workgroup_size(1) fn f() { let v = ssbo.m; ssbo.m = v; }";
    let output = run(source, &Config::default()).unwrap();
    assert!(
        output.report.fallback.is_none() && output.report.bailout.is_none(),
        "{:?}",
        output.report
    );
    assert!(
        output.source.contains("binding(0)") && output.source.ends_with("fn f(){_=A;}"),
        "{}",
        output.source
    );
}

/// The module API carries the accesses in the IR.
#[test]
fn run_module_keeps_a_dead_read_accessed() {
    let source = "@group(0) @binding(0) var<uniform> u: vec4f;
        @group(0) @binding(1) var<storage, read_write> o: vec4f;
        @compute @workgroup_size(1) fn main() { let dead = u; o = vec4f(1.0); }";
    let mut module = io::parse_wgsl(source).unwrap();
    run_module(&mut module, &Config::default()).unwrap();
    assert_eq!(module.global_variables.len(), 2);
    let pins = crate::pins::of(&module.global_variables, &module.entry_points[0].function);
    assert_eq!(pins.len(), 2);
}

/// A mesh shader: the generator refuses the stage, so naga's emitter prints
/// the renamed IR (a known loss) - the fixture for the fallback rung.
const MESH_FALLBACK_SRC: &str = "enable wgpu_mesh_shader;\n\
    @group(0) @binding(0) var<uniform> tint: vec4f;\n\
    struct V { @builtin(position) p: vec4f, @location(0) c: vec3f }\n\
    struct P { @builtin(point_index) i: u32 }\n\
    struct M { @builtin(vertices) v: array<V, 1>, @builtin(primitives) q: array<P, 1>, \
    @builtin(vertex_count) nv: u32, @builtin(primitive_count) np: u32 }\n\
    var<workgroup> mo: M;\n\
    @mesh(mo) @workgroup_size(1) fn ms() { mo.v[0] = V(tint, tint.xyz); mo.nv = 1u; mo.np = 1u; }\n";

/// The naga-emitter fallback prints the RENAMED module: the name map applies
/// and the report names the fallback (both were `null`; the web build had no
/// signal), and the `generator_emit` entry reports the fallback as rolled
/// back and the output as changed, which the renamed text is.
#[test]
fn naga_fallback_reports_itself_and_keeps_the_name_map() {
    let src = MESH_FALLBACK_SRC;
    let out = run(src, &Config::default()).expect("the module minifies via the fallback");
    assert!(out.report.fallback.is_some(), "the fallback is reported");
    let gen_report = out
        .report
        .pass_reports
        .iter()
        .find(|p| p.pass_name == "generator_emit")
        .expect("generator_emit pass must exist");
    assert!(gen_report.rolled_back, "the fallback shipped");
    assert_ne!(out.source, src, "the fallback text carries the renames");
    assert!(
        gen_report.changed,
        "changed reports the output differing from the input on the fallback arm too"
    );
    let map = out.name_map.expect("the fallback text carries the renames");
    let renamed = map.globals.get("tint").expect("the resource is mapped");
    assert!(
        out.source.contains(&format!("{renamed}:vec4<f32>")),
        "the map names the declaration in the shipped text: {renamed} in {}",
        out.source
    );
    assert_eq!(
        map.structs.get("V").map(|s| s.name.as_str()),
        Some("V"),
        "the surviving struct has its entry"
    );
}

/// naga's writer failing on the fallback rung is the second diagnosis:
/// the error names the generator's own failure too, which is the one to
/// act on.
#[test]
fn a_failing_writer_on_the_fallback_rung_keeps_the_generators_diagnosis() {
    let writer_fails = || Err(Error::Emit("writer: no arm for this expression".into()));
    let not_blocked = || None;
    let self_check = |text: &str| io::validate_wgsl_text(text);
    let err = resolve_generator_output(
        Err(Error::Emit("generator: unsupported image class".into())),
        &writer_fails,
        None,
        None,
        "fn f() {}",
        false,
        &not_blocked,
        &self_check,
    )
    .err()
    .expect("no rung ships");
    let text = err.to_string();
    assert!(
        text.contains("generator: unsupported image class")
            && text.contains("writer: no arm for this expression"),
        "{text}"
    );
}

/// naga's namer prints `fs1` as `fs1_`: a fallback that respells an entry
/// point, a named override or a preserved symbol is refused.
#[test]
fn naga_fallback_never_respells_the_interface() {
    let src = MESH_FALLBACK_SRC.replace("fn ms()", "fn ms1()");
    let err = run(&src, &Config::default())
        .err()
        .expect("no rung ships a respelled entry point");
    assert!(err.to_string().contains("`ms1` as `ms1_`"), "{err}");
    let with_override = MESH_FALLBACK_SRC
        .replace("V(tint,", "V(tint * o1,")
        .replace("@group(0)", "override o1: f32 = 1.0;\n@group(0)");
    let err = run(&with_override, &Config::default())
        .err()
        .expect("no rung ships a respelled override");
    assert!(err.to_string().contains("`o1` as `o1_`"), "{err}");
}

/// `@interpolate(per_vertex)` needs naga's `wgpu_per_vertex` extension, and
/// the generator's directive census carries it: the text stands without
/// the fallback.
#[test]
fn per_vertex_interpolation_keeps_its_enable() {
    let src = "enable wgpu_per_vertex;\n\
               @group(0) @binding(0) var<uniform> tint: vec4f;\n\
               struct V { @builtin(position) p: vec4f, @location(0) @interpolate(per_vertex) c: array<vec3f, 3> }\n\
               @fragment fn fs(v: V) -> @location(0) vec4f { return vec4f(v.c[0], 1.0) * tint; }\n";
    let out = run(src, &Config::default()).expect("run failed");
    assert!(
        out.report.fallback.is_none(),
        "generator text stands: {}",
        out.source
    );
    assert!(
        out.source.starts_with("enable wgpu_per_vertex;")
            && out.source.contains("@interpolate(per_vertex)"),
        "{}",
        out.source
    );
}

/// Preamble member names are preserved as MEMBER names only: a body function
/// sharing one is renamed and its pointer parameter still specialized.
#[test]
fn preamble_member_names_do_not_freeze_body_functions() {
    let preamble = "struct Params { bump: f32, k: u32 }\n\
                    @group(0) @binding(1) var<uniform> params: Params;\n";
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
               fn bump(q: ptr<storage, array<u32>, read_write>, k: u32) { (*q)[0] = k + params.k; }\n\
               @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3u) { bump(&out, g.x); }\n";
    let config = Config {
        preamble: Some(preamble.to_string()),
        ..Config::default()
    };
    let out = run(src, &config).expect("run failed");
    assert!(
        !out.source.contains("fn bump(") && !out.source.contains("ptr<storage"),
        "the body function is renamed and specialized: {}",
        out.source
    );
    assert!(
        out.source.contains("params.k"),
        "the preamble member keeps its name: {}",
        out.source
    );
}

/// A parse label reports the caller's coordinates: the `enable f16;` line
/// `run` injects and a spliced preamble are subtracted out; the preamble's
/// own labels stay relative to the preamble text.
#[test]
fn parse_error_location_is_in_the_callers_coordinates() {
    let Err(err) = run("fn bad { }", &Config::default()) else {
        panic!("syntax error must fail");
    };
    let loc = err.location().expect("labelled");
    assert_eq!((loc.line_number, loc.line_position), (1, 8), "{err}");

    let config = Config {
        preamble: Some("struct P { t: f32 }\n@group(0) @binding(0) var<uniform> p: P;\n".into()),
        ..Config::default()
    };
    let src = "fn g() -> f16 { return 1h; }\n@compute @workgroup_size(1) fn m() {\n  let x = ;\n}";
    let Err(err) = run(src, &config) else {
        panic!("syntax error must fail");
    };
    let loc = err.location().expect("labelled");
    assert_eq!((loc.line_number, loc.line_position), (3, 11), "{err}");

    let config = Config {
        preamble: Some("struct P { t: f32 \n".into()),
        ..Config::default()
    };
    let Err(err) = run("fn m() {}", &config) else {
        panic!("preamble syntax error must fail");
    };
    assert!(err.to_string().contains("<preamble>"), "{err}");
    assert!(err.location().is_some(), "{err}");
}

/// `preserve_interface` keeps what a by-name host reads (bound globals,
/// overrides, reached struct types and members) and nothing else.
#[test]
fn preserve_interface_keeps_host_visible_names_only() {
    let src = "struct Light { color: vec3f, radius: f32 }\n\
        struct Camera { view: mat4x4f, lights: array<Light, 2> }\n\
        struct Local { scratch: f32 }\n\
        @group(0) @binding(0) var<uniform> camera: Camera;\n\
        override BIAS: f32 = 1.0;\n\
        @id(3) override SCALE: f32 = 2.0;\n\
        var<private> counter: f32;\n\
        fn helper(p: vec3f) -> f32 { var l: Local; l.scratch = p.x; counter += l.scratch; return counter; }\n\
        @fragment fn main() -> @location(0) vec4f {\n\
            let c = camera.lights[1].color * camera.lights[0].radius + BIAS * SCALE;\n\
            return camera.view * vec4f(c, helper(c));\n\
        }";
    let plain = run(src, &Config::default()).expect("minifies").source;
    // An `@id`-less override is already kept by role (`BIAS`); the rest is
    // renamed under the default profile.
    for name in ["camera", "Camera", "lights", "SCALE"] {
        assert!(
            !plain.contains(name),
            "{name} is renamed by default: {plain}"
        );
    }
    assert!(plain.contains("override BIAS"), "{plain}");
    let config = Config {
        preserve_interface: true,
        ..Config::default()
    };
    let kept = run(src, &config).expect("minifies").source;
    for name in [
        "var<uniform>camera:Camera",
        "struct Camera{view:",
        "lights:array<Light,2>",
        "struct Light{color:",
        "radius:f32",
        "override BIAS",
        "override SCALE",
    ] {
        assert!(kept.contains(name), "{name} must be preserved: {kept}");
    }
    for name in ["helper", "counter", "Local", "scratch"] {
        assert!(!kept.contains(name), "{name} is not interface: {kept}");
    }
}

// MARK: Pass-order and driver regressions

/// `resugar_short_circuits` rebuilds the arena (renumbering every handle),
/// so the forwarder's index bounds must be sized after it: a stale table
/// let a const out-of-bounds forward through and the whole dead_branch run
/// rolled back, every sweep, until `compact` had culled the dead handle.
#[test]
fn dead_branch_sizes_its_index_bounds_after_the_resugar() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        @group(0) @binding(1) var<storage, read> inp: array<u32>;\
        @compute @workgroup_size(1) fn main() {\
          var unused: u32 = 5u;\
          var i: i32; i = 4;\
          let a = array<u32, 4>(1u, 2u, 3u, 4u);\
          if (inp[0] > 1u && inp[1] > 2u) { out[0] = a[i + 1]; }\
        }";
    let output = run(src, &Config::default()).expect("minifies");
    assert!(
        output.report.pass_reports.iter().all(|p| !p.rolled_back),
        "no pass run may roll back: {}",
        output.source
    );
}

/// An unevaluable index slot is declined only where a zero-init local's
/// fold could make it const: `a[select(z + 4u, 0u, runtime)]` stays
/// runtime whatever `z` folds to, so the fold inside it is free (and the
/// output is what a second pass would have produced anyway).
#[test]
fn an_index_slot_with_a_runtime_leaf_folds_its_zero_local() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\
        @group(0) @binding(1) var<storage, read> inp: array<u32>;\
        @compute @workgroup_size(1) fn main() {\
          var z: u32; var y: u32;\
          let a = array<u32, 4>(1u, 2u, 3u, 4u);\
          out[0] = a[select(z + 4u, 0u, inp[0] > 3u)];\
          out[1] = a[(y + 4u) & inp[1]];\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(
        first.contains("select(4u,0u,") && first.contains("[4&"),
        "the zero locals fold: {first}"
    );
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// `const_hoist` counts a vector only where it renders: one `const_fold`
/// materialised at `arr[1]` and then folded away with its `.z` reader is
/// emitted but consumed by nothing, and hoisting it made a `const` with a
/// single live use.
#[test]
fn const_hoist_ignores_an_unreferenced_materialised_vector() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\
        @group(0) @binding(1) var<storage, read> inp: array<u32>;\
        @compute @workgroup_size(1) fn main() {\
          let arr = array<vec4f, 2>(vec4f(1.0), vec4f(5.0, 6.0, 7.0, 8.0));\
          out[0] = arr[1].z; out[1] = arr[1].w; out[2] = arr[1].x;\
          out[3] = arr[inp[0]].y;\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(!first.contains("const "), "nothing to hoist: {first}");
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// Two sites of one long vector pay for a shared `const` once the render
/// confirms it, so `const_hoist` no longer waits for a third site.
#[test]
fn a_two_site_vector_constant_is_hoisted_when_the_output_shrinks() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        fn f(v: vec4f) -> vec4f { return v * vec4f(7.25, 2.125, 9.5, 3.75); }\
        @compute @workgroup_size(1) fn main() {\
          o[0] = f(o[1]) + vec4f(7.25, 2.125, 9.5, 3.75);\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(
        first.contains("const ") && first.matches("7.25").count() == 1,
        "the vector is declared once and referenced twice: {first}"
    );
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// The zero of a vector type is a `ZeroValue`: pre-emit, so the emitter
/// never `let`-binds it and every use spells `vec4f()` (or the alias's
/// `B()`).  Six such uses share one `const`; the shortlist prices them at
/// the full spelling because the alias they alone justified goes with
/// them, and the render confirms.  A re-parse keeps the constant.
#[test]
fn repeated_zero_vectors_are_hoisted_to_one_constant() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        fn a(i: u32) { o[i] = vec4f(); o[i + 1u] = vec4f(); }\
        fn b(i: u32) { o[i + 2u] = vec4f(); o[i + 3u] = vec4f(); }\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          a(id.x); b(id.y); o[id.z + 4u] = vec4f(); o[id.z + 5u] = vec4f();\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(
        first.contains("const ") && first.matches("vec4f()").count() == 1,
        "one declaration serves six zero vectors: {first}"
    );
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// `let z = vec2f();` names one `ZeroValue` the body uses five times; the
/// emitter cannot bind it (no `Emit`), so one site rendered five times
/// hoists as five sites would - the model must not take the `let` the
/// emitter never writes.
#[test]
fn a_zero_vector_bound_in_the_source_is_hoisted() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec2f>;\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          let z = vec2f();\
          o[id.x] = z; o[id.y] = z; o[id.z] = z; o[id.x + 1u] = z; o[id.y + 1u] = z;\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(
        first.contains("const ") && first.matches("vec2f()").count() == 1,
        "one declaration serves the five uses: {first}"
    );
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// The same five uses with the zero spelled lane by lane: the fold makes
/// one `ZeroValue` of the `Compose`, and the constant follows.
#[test]
fn a_zero_vector_spelled_with_lanes_hoists_alike() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec2f>;\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          let z = vec2f(0.0, 0.0);\
          o[id.x] = z; o[id.y] = z; o[id.z] = z; o[id.x + 1u] = z; o[id.y + 1u] = z;\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(
        first.contains("const ") && first.matches("vec2f()").count() == 1,
        "one declaration serves the five uses: {first}"
    );
    let second = run(&first, &Config::default()).expect("minifies").source;
    assert_eq!(first, second, "idempotent");
}

/// Two zero vectors cannot pay for a declaration; no render is spent.
#[test]
fn two_zero_vectors_stay_inline() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec4f>;\
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {\
          o[id.x] = vec4f(); o[id.y] = vec4f();\
        }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(!first.contains("const "), "nothing pays: {first}");
}

/// A short vector at two sites costs more as a declaration than it saves;
/// the per-site model declines it before any render.
#[test]
fn a_two_site_hoist_that_cannot_pay_is_declined() {
    let src = "@group(0) @binding(0) var<storage, read_write> o: array<vec2f>;\
        fn f(v: vec2f) -> vec2f { return v * vec2f(1.0, 2.0); }\
        @compute @workgroup_size(1) fn main() { o[0] = f(o[1]) + vec2f(1.0, 2.0); }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert!(!first.contains("const "), "nothing pays: {first}");
}

/// Four `vec3f(.125)` share a `const` (beside a vector operand such a
/// value collapses to its scalar, which the render sees and the per-site
/// model does not - that group is declined by the trial), and a matrix
/// built from constant columns hoists as ONE constant: its columns render
/// through it, never on their own.
#[test]
fn repeated_splat_values_and_nested_constant_constructors_are_hoisted_whole() {
    let splat = "@group(0) @binding(0) var<storage, read_write> o: array<vec3f>;\
        fn f(v: vec3f) -> vec3f { return select(v, vec3f(0.125), v.x > 1.0); }\
        @compute @workgroup_size(1) fn main() {\
          o[0] = f(vec3f(0.125)); o[1] = f(vec3f(0.125)); o[2] = vec3f(0.125);\
        }";
    let first = run(splat, &Config::default()).expect("minifies").source;
    assert_eq!(
        first.matches(".125").count(),
        1,
        "one declaration serves the four splats: {first}"
    );
    assert_eq!(
        run(&first, &Config::default()).expect("minifies").source,
        first
    );

    let matrix = "@group(0) @binding(0) var<storage, read_write> o: array<vec2f>;\
        fn f(v: vec2f) -> vec2f { return mat2x2f(vec2f(1.5, 0.25), vec2f(0.75, 1.25)) * v; }\
        @compute @workgroup_size(1) fn main() {\
          o[0] = f(o[1]) + mat2x2f(vec2f(1.5, 0.25), vec2f(0.75, 1.25)) * o[2];\
        }";
    let first = run(matrix, &Config::default()).expect("minifies").source;
    assert_eq!(first.matches("const ").count(), 1, "one constant: {first}");
    assert_eq!(first.matches("1.5").count(), 1, "declared once: {first}");
    assert_eq!(
        run(&first, &Config::default()).expect("minifies").source,
        first
    );
}

/// A value the module already declares as a `const` anchors its sites: they
/// reference it and no twin is minted - the source's own constant here, and
/// on a re-minification the constant the first pass hoisted, whose value a
/// fold over it rebuilds.
#[test]
fn sites_of_a_declared_constant_reference_it_instead_of_a_twin() {
    let src = "const K = vec3f(1.5, 2.5, 3.5);\
        @group(0) @binding(0) var<storage, read_write> o: array<vec3f>;\
        fn f(v: vec3f) -> vec3f { return v * vec3f(1.5, 2.5, 3.5); }\
        @compute @workgroup_size(1) fn main() { o[0] = f(o[1]) + K; o[2] = vec3f(1.5, 2.5, 3.5); }";
    let first = run(src, &Config::default()).expect("minifies").source;
    assert_eq!(first.matches("const ").count(), 1, "one constant: {first}");
    assert_eq!(first.matches("1.5").count(), 1, "declared once: {first}");
    assert_eq!(
        run(&first, &Config::default()).expect("minifies").source,
        first
    );
}

/// `var v = K;` needs no `: T` when `K` is a named constant of a concrete
/// type: its declaration fixes the type the way a constructor would.
#[test]
fn a_var_initialised_from_a_named_constant_elides_its_type() {
    let src = "const K = vec3f(1.5, 2.5, 3.5);\
        @group(0) @binding(0) var<storage, read_write> o: array<vec3f>;\
        @compute @workgroup_size(1) fn main() { var v = K; v.x += o[1].x; o[0] = v + K; }";
    let first = run(src, &Config::default()).expect("minifies").source;
    let var_decl = first
        .split(';')
        .find(|stmt| stmt.contains("var ") && !stmt.contains("var<"))
        .expect("the local survives");
    assert!(
        !var_decl.contains(':'),
        "the constant's type is concrete, so the var infers it: {first}"
    );
    assert_eq!(
        run(&first, &Config::default()).expect("minifies").source,
        first
    );
}

/// A module-level `diagnostic(...)` directive hands every function the
/// module's filter leaf, which is the caller's too: the splice is still
/// free.  Only a helper carrying its own `@diagnostic` keeps its call.
#[test]
fn a_module_diagnostic_directive_does_not_block_splicing() {
    let src = "diagnostic(off, derivative_uniformity);\
        @group(0) @binding(0) var<storage, read_write> out: array<f32>;\
        fn helper(x: f32) -> f32 { var t = x * 2.0; t += 1.0; return t; }\
        @compute @workgroup_size(1) fn main() { out[0] = helper(out[1]); }";
    let output = run(src, &Config::default()).expect("minifies").source;
    assert!(
        !output.contains("fn helper") && !output.contains("fn a("),
        "the helper is spliced: {output}"
    );
    assert_eq!(
        output.matches("fn ").count(),
        1,
        "only the entry point remains: {output}"
    );
}

/// A parse label at the end of a body a preamble splice terminated with a
/// newline clamps to the source's end instead of dropping the location, and
/// a lone-CR source resolves its position on the normalised text (the one
/// naga counted lines in).
#[test]
fn parse_locations_survive_a_trailing_label_and_lone_cr_endings() {
    let config = Config {
        preamble: Some("struct P { a: f32 }\n@group(0) @binding(0) var<uniform> u: P;\n".into()),
        ..Config::default()
    };
    let Err(err) = run("fn f() -> f32 { return u.a; ", &config) else {
        panic!("an unterminated body parses");
    };
    let loc = err.location().expect("a location in the user's text");
    assert_eq!((loc.line_number, loc.line_position), (1, 29));
    let Err(err) = run(
        "const x = 1;\rconst y = 2;\rlet z = ;\r",
        &Config::default(),
    ) else {
        panic!("a statement at module scope parses");
    };
    let loc = err.location().expect("a location");
    assert_eq!(loc.line_number, 3);
}

/// A run cut short by `opt_bisect_limit` is no fixed point.
#[test]
fn a_bisect_cut_reports_not_converged() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\
        fn helper(x: f32) -> f32 { var t = x * 2.0; t += 1.0; return t; }\
        @compute @workgroup_size(1) fn main() { out[0] = helper(out[1]) + 0.0; }";
    let config = Config {
        trace: config::TraceConfig {
            opt_bisect_limit: Some(1),
            ..Default::default()
        },
        ..Config::default()
    };
    let output = run(src, &config).expect("minifies");
    assert!(!output.report.converged, "{:?}", output.report.sweeps);
}

/// naga is `no_std` and folds `f32` builtins through the `libm` crate; if
/// any dependency in its closure links `std` (`half` with its default
/// features would), rustc resolves the same calls to the platform's libm,
/// whose `coshf(1)` is a last ULP off on macOS - a parse-time fold the
/// wasm build cannot follow.  `Cargo.toml` keeps `half` at
/// `default-features = false`; this pins the fold that showed it.
#[test]
fn nagas_f32_builtins_fold_through_the_libm_crate() {
    let module = io::parse_wgsl(
        "@group(0) @binding(0) var<storage, read_write> o: vec4<f32>;\
         @compute @workgroup_size(1) fn m() { o = cosh(vec4<f32>(1f, 1f, 1f, 1f)); }",
    )
    .unwrap();
    let folded: Vec<u32> = module.entry_points[0]
        .function
        .expressions
        .iter()
        .filter_map(|(_, e)| match e {
            naga::Expression::Literal(naga::Literal::F32(v)) => Some(v.to_bits()),
            _ => None,
        })
        .collect();
    assert_eq!(
        folded, [0x3fc5_83aa; 4],
        "cosh(1f) must fold to 1.5430806 (libm), not the platform's 1.5430807"
    );
}
