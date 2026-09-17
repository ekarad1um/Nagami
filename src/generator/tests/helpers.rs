//! Shared fixtures: each helper runs `parse -> validate -> generate` under one
//! [`GenerateOptions`] preset; [`compact_with_passes`] runs the full IR pipeline.
//! Validation goes through `crate::io::validate_module`, so pointer parameters
//! naga alone rejects are fixtures too.

use super::super::{GenerateOptions, generate};
pub use crate::config::{Config, FloatPrecision, PrecisionMode, Profile};

pub fn compact(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            float_precision: FloatPrecision::default(),
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

pub fn compact_mangled(src: &str) -> String {
    compact_mangled_preserved(src, &[])
}

pub fn compact_aliased(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            float_precision: FloatPrecision::default(),
            type_alias: true,
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

pub fn compact_mangled_aliased(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: true,
            float_precision: FloatPrecision::default(),
            type_alias: true,
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

pub fn compact_mangled_preserved(src: &str, preserve: &[&str]) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: true,
            float_precision: FloatPrecision::default(),
            preserve_symbols: preserve.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

#[track_caller]
pub fn assert_valid_wgsl(out: &str) {
    // Shipped output omits the naga-only `enable wgpu_binding_array;` (tint
    // rejects it; `run` strips it), so re-inject it for the naga re-parse as the
    // pipeline self-check does.
    let injected;
    let to_parse = if out.contains("binding_array") && !out.contains("enable wgpu_binding_array;") {
        injected = format!("enable wgpu_binding_array;\n{out}");
        injected.as_str()
    } else {
        out
    };
    let module = naga::front::wgsl::parse_str(to_parse).unwrap_or_else(|e| {
        panic!("re-parse failed for:\n{out}\nerror: {e:?}");
    });
    crate::io::validate_module(&module).unwrap_or_else(|e| {
        panic!("re-validation failed for:\n{out}\nerror: {e:?}");
    });
}

pub fn compact_beautified(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: true,
            indent: 2,
            mangle: false,
            float_precision: FloatPrecision::default(),
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

pub fn compact_with_precision(src: &str, prec: u8) -> String {
    compact_with_float_precision(src, FloatPrecision::all(PrecisionMode::DecimalPlaces(prec)))
}

pub fn compact_with_float_precision(src: &str, float_precision: FloatPrecision) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            float_precision,
            ..Default::default()
        },
    )
    .expect("generate failed")
    .source
}

pub fn compact_with_passes(src: &str, profile: Profile) -> String {
    let config = Config {
        profile,
        beautify: true,
        ..Config::default()
    };
    let output = crate::run(src, &config).expect("run failed");
    assert_valid_wgsl(&output.source);
    output.source
}

pub const VALIDATION_SRC: &str = r#"
    const COLOR: vec3<f32> = vec3<f32>(0.5, 0.8, 1.0);
    fn helper(x: f32) -> f32 {
        var t: f32 = 0.0;
        t = x * 2.0;
        return t + 1.0;
    }
    fn main_fn(a: f32, b: f32) -> f32 {
        let sum = a + b;
        return helper(sum) * helper(sum);
    }
"#;
