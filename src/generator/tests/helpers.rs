//! Shared fixtures for the generator test suite.
//!
//! Every helper here runs the `parse -> validate -> generate` chain
//! with a different [`GenerateOptions`] preset, letting each test
//! focus on asserting the emitted text instead of repeating the
//! boilerplate.  [`assert_valid_wgsl`] re-parses the generated output
//! to confirm it round-trips cleanly, and [`compact_with_passes`]
//! layers the full IR-pass pipeline on top for end-to-end assertions.

use super::super::{GenerateOptions, generate_wgsl};
pub use crate::config::{Config, Profile};

/// Parse, validate, and emit `src` with the baseline
/// compact-mode options: no beautify, no mangling, full precision.
pub fn compact(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            max_precision: None,
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Compact variant with mangling enabled and no preserved symbols.
/// Convenience wrapper over [`compact_mangled_preserved`].
pub fn compact_mangled(src: &str) -> String {
    compact_mangled_preserved(src, &[])
}

/// Compact variant with `type_alias` enabled, mangling off.  Used by
/// alias-emission tests.
pub fn compact_aliased(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            max_precision: None,
            type_alias: true,
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Compact variant with both mangling and type aliasing active;
/// exercises the interaction between alias rewriting and mangled type
/// names.
pub fn compact_mangled_aliased(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: true,
            max_precision: None,
            type_alias: true,
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Compact mangled variant that preserves the given symbol names.
/// Used to pin the preserve-symbols contract exercised by the
/// higher-level tests.
pub fn compact_mangled_preserved(src: &str, preserve: &[&str]) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: true,
            max_precision: None,
            preserve_symbols: preserve.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Assert that `out` parses and validates as WGSL: the emitter's
/// most important contract.  `#[track_caller]` ensures panics point
/// at the offending test rather than this helper.
#[track_caller]
pub fn assert_valid_wgsl(out: &str) {
    let module = naga::front::wgsl::parse_str(out).unwrap_or_else(|e| {
        panic!("re-parse failed for:\n{out}\nerror: {e:?}");
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!("re-validation failed for:\n{out}\nerror: {e:?}");
    });
}

/// Beautified variant (indent = 2, mangling off) used by tests that
/// assert on human-readable output.
pub fn compact_beautified(src: &str) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: true,
            indent: 2,
            mangle: false,
            max_precision: None,
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Compact variant with a bounded `max_precision`.  Used by tests
/// that exercise lossy float trimming.
pub fn compact_with_precision(src: &str, prec: u8) -> String {
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validation failed");
    generate_wgsl(
        &module,
        &info,
        GenerateOptions {
            beautify: false,
            indent: 0,
            mangle: false,
            max_precision: Some(prec),
            ..Default::default()
        },
    )
    .expect("generate failed")
}

/// Run the full `parse -> validate -> IR passes -> generate`
/// pipeline at `profile` with beautify on, asserting the output
/// round-trips through [`assert_valid_wgsl`] before returning it.
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
