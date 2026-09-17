//! `generate_reusing`: the prior render ships again only when it is the
//! text this render would produce, so the result always equals a fresh
//! render under the same options.  `generate_after`: the prior render's
//! analyses serve a render of the same arenas under other names, so the
//! result equals a fresh render of those.

use super::super::{GenerateOptions, TypeUses, generate, generate_after, generate_reusing};

const SRC: &str = "struct P { a: vec3<f32>, b: vec3<f32>, c: vec3<f32> }
@group(0) @binding(0) var<storage, read_write> buf: array<vec3<f32>>;
fn f(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> { var z: vec3<f32> = x + y; return z; }
@compute @workgroup_size(1) fn main() { buf[0] = f(vec3<f32>(1.0), vec3<f32>(2.0)); }";

fn fixture() -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(SRC).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    (module, info)
}

fn options() -> GenerateOptions {
    GenerateOptions {
        mangle: true,
        type_alias: true,
        ..Default::default()
    }
}

#[test]
fn reuse_matches_a_fresh_render_under_the_measured_spellings() {
    let (module, info) = fixture();
    let prior = generate(&module, &info, options()).expect("generate failed");
    let prior_source = prior.source.clone();
    assert!(
        prior_source.contains("alias "),
        "the fixture must mint an alias: {prior_source}"
    );
    let measured = GenerateOptions {
        type_uses: Some(prior.type_uses.clone()),
        ..options()
    };
    let fresh = generate(&module, &info, measured.clone())
        .expect("generate failed")
        .source;
    let reused = generate_reusing(&module, &info, measured, prior)
        .expect("generate failed")
        .source;
    assert_eq!(reused, fresh);
    assert_eq!(
        reused, prior_source,
        "the measured spellings keep the census plan here"
    );
}

#[test]
fn reuse_declines_when_the_spellings_change_the_type_plan() {
    let (module, info) = fixture();
    let prior = generate(&module, &info, options()).expect("generate failed");
    let prior_source = prior.source.clone();
    // No spelling counted: no alias pays.
    let none = GenerateOptions {
        type_uses: Some(TypeUses::default()),
        ..options()
    };
    let fresh = generate(&module, &info, none.clone())
        .expect("generate failed")
        .source;
    let reused = generate_reusing(&module, &info, none, prior)
        .expect("generate failed")
        .source;
    assert_eq!(reused, fresh);
    assert_ne!(reused, prior_source);
    assert!(!reused.contains("alias "), "{reused}");
}

#[test]
fn reuse_declines_when_the_options_differ() {
    let (module, info) = fixture();
    let prior = generate(&module, &info, options()).expect("generate failed");
    let prior_source = prior.source.clone();
    let pretty = GenerateOptions {
        beautify: true,
        type_uses: Some(prior.type_uses.clone()),
        ..options()
    };
    let fresh = generate(&module, &info, pretty.clone())
        .expect("generate failed")
        .source;
    let reused = generate_reusing(&module, &info, pretty, prior)
        .expect("generate failed")
        .source;
    assert_eq!(reused, fresh);
    assert_ne!(reused, prior_source);
}

/// Every analysis a body's context is built from has work here: a single
/// use inside a loop of work done before it and a `for` header (must-bind,
/// for-loop vars), a value spelled twice (twins), a splat under a compose
/// (compose-fold discount), a single-use call (inlineable calls), and a
/// literal spelled often enough to extract.
const ANALYSED: &str = "struct P { a: vec3<f32>, b: vec3<f32> }
@group(0) @binding(0) var<storage, read_write> buf: array<vec3<f32>>;
@group(0) @binding(1) var<uniform> k: vec4<f32>;
fn g(x: f32) -> f32 { return x * 0.375 + 0.375; }
fn f(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    var z: vec3<f32> = x + y;
    var acc = vec3<f32>(0.375);
    let w = normalize(k.xyz) * 0.375;
    for (var i = 0u; i < 4u; i++) {
        acc += w * z + vec3<f32>(g(k.w), 0.375, 0.375);
    }
    let s = normalize(x) * y + x;
    return acc + s * s;
}
@compute @workgroup_size(1) fn main() { buf[0] = f(vec3<f32>(1.0), vec3<f32>(2.0)); }";

/// `ANALYSED` and a copy of it under the names the rename would give it.
fn renamed_fixture() -> (
    naga::Module,
    naga::valid::ModuleInfo,
    naga::Module,
    naga::valid::ModuleInfo,
) {
    let module = naga::front::wgsl::parse_str(ANALYSED).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    let renamed =
        crate::passes::rename::plan_names(&module, &Default::default(), true).applied(&module);
    let renamed_info = crate::io::validate_module(&renamed).expect("validation failed");
    (module, info, renamed, renamed_info)
}

#[test]
fn a_render_after_the_prior_one_matches_a_fresh_render_under_other_names() {
    let (module, info, renamed, renamed_info) = renamed_fixture();
    let prior = generate(&module, &info, options()).expect("generate failed");
    assert_eq!(
        prior.analyses.bodies_built(),
        3,
        "every body leaves its analyses"
    );
    let prior_source = prior.source.clone();
    for (marker, what) in [("for(", "a for header"), ("let ", "a binding")] {
        assert!(
            prior_source.contains(marker),
            "the fixture must render {what}: {prior_source}"
        );
    }
    let measured = GenerateOptions {
        type_uses: Some(prior.type_uses.clone()),
        ..options()
    };
    let fresh = generate(&renamed, &renamed_info, measured.clone())
        .expect("generate failed")
        .source;
    let after = generate_after(&renamed, &renamed_info, measured, prior)
        .expect("generate failed")
        .source;
    assert_eq!(after, fresh);
    assert_ne!(after, prior_source, "the names moved");
}

#[test]
fn a_render_after_the_prior_one_declines_its_analyses_when_the_options_differ() {
    let (module, info, renamed, renamed_info) = renamed_fixture();
    let prior = generate(&module, &info, options()).expect("generate failed");
    let pretty = GenerateOptions {
        beautify: true,
        type_uses: Some(prior.type_uses.clone()),
        ..options()
    };
    let fresh = generate(&renamed, &renamed_info, pretty.clone())
        .expect("generate failed")
        .source;
    let after = generate_after(&renamed, &renamed_info, pretty, prior)
        .expect("generate failed")
        .source;
    assert_eq!(after, fresh);
}
