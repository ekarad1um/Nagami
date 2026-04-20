//! JavaScript / browser binding layer exposed via `wasm-bindgen`.
//!
//! Mirrors the native [`crate::run`] entry point as a `JsValue -> JsValue`
//! boundary: JS config objects are validated and decoded into [`Config`],
//! the pipeline runs unchanged, and [`crate::pipeline::Report`] is
//! projected back to a plain JS object.  TypeScript declarations for
//! the wire types live in the `TS_TYPES` block at the bottom of the file.

use wasm_bindgen::prelude::*;

use crate::config::{Config, Profile};

// MARK: JS value extraction

/// Read `obj[key]`, collapsing `undefined` and `null` to `None` so
/// partial config objects work without the caller noticing the
/// distinction between "missing" and "explicitly null".
fn get_opt(obj: &JsValue, key: &str) -> Option<JsValue> {
    let val = js_sys::Reflect::get(obj, &JsValue::from_str(key)).ok()?;
    if val.is_undefined() || val.is_null() {
        None
    } else {
        Some(val)
    }
}

fn get_string(obj: &JsValue, key: &str) -> Option<String> {
    get_opt(obj, key)?.as_string()
}

fn get_bool(obj: &JsValue, key: &str) -> Option<bool> {
    get_opt(obj, key)?.as_bool()
}

fn get_f64(obj: &JsValue, key: &str) -> Option<f64> {
    get_opt(obj, key)?.as_f64()
}

/// Read `obj[key]` as an array of strings, dropping non-string entries
/// silently.  Returning `None` only on a missing/non-array field keeps
/// downstream code free of "present but empty vs absent" forks.
fn get_string_array(obj: &JsValue, key: &str) -> Option<Vec<String>> {
    let val = get_opt(obj, key)?;
    let arr: js_sys::Array = val.dyn_into().ok()?;
    Some(arr.iter().filter_map(|v| v.as_string()).collect())
}

/// Decode a JS number as a `u8`, rejecting NaN, negatives, overflow,
/// and non-integer values with a structured [`JsError`].
fn require_u8(v: f64, key: &str) -> Result<u8, JsError> {
    if v.is_nan() || v < 0.0 || v > u8::MAX as f64 || v.fract() != 0.0 {
        return Err(JsError::new(&format!(
            "\"{key}\" must be an integer in 0..255, got {v}"
        )));
    }
    Ok(v as u8)
}

/// Decode a JS number as a `usize`, capped at `2^53`.
///
/// JavaScript numbers are IEEE-754 doubles and can represent integers
/// exactly only up to 2^53; values past that threshold silently lose
/// precision when cast to `usize`.  The cap is deliberately tighter
/// than `usize::MAX` (which itself is not exactly representable as
/// `f64`) so callers see a crisp error instead of a latent wrap.
fn require_usize(v: f64, key: &str) -> Result<usize, JsError> {
    const MAX_SAFE: f64 = (1u64 << 53) as f64; // 9_007_199_254_740_992
    if v.is_nan() || !(0.0..=MAX_SAFE).contains(&v) || v.fract() != 0.0 {
        return Err(JsError::new(&format!(
            "\"{key}\" must be a non-negative integer, got {v}"
        )));
    }
    Ok(v as usize)
}

// MARK: Config decoding

/// Decode a JS config object into [`Config`], returning [`Config::default`]
/// when the caller passes `undefined` or `null`.  Unknown fields are
/// ignored so adding config options on the Rust side never breaks older
/// JS callers.
fn parse_config(config: JsValue) -> Result<Config, JsError> {
    if config.is_undefined() || config.is_null() {
        return Ok(Config::default());
    }

    let mut cfg = Config::default();

    if let Some(p) = get_string(&config, "profile") {
        cfg.profile = match p.as_str() {
            "baseline" => Profile::Baseline,
            "aggressive" => Profile::Aggressive,
            "max" => Profile::Max,
            other => return Err(JsError::new(&format!("unknown profile: \"{other}\""))),
        };
    }
    if let Some(symbols) = get_string_array(&config, "preserveSymbols") {
        cfg.preserve_symbols = symbols;
    }
    if let Some(mangle) = get_bool(&config, "mangle") {
        cfg.mangle = Some(mangle);
    }
    if let Some(beautify) = get_bool(&config, "beautify") {
        cfg.beautify = beautify;
    }
    if let Some(indent) = get_f64(&config, "indent") {
        cfg.indent = require_u8(indent, "indent")?;
    }
    if let Some(v) = get_f64(&config, "maxPrecision") {
        cfg.max_precision = Some(require_u8(v, "maxPrecision")?);
    }
    if let Some(v) = get_f64(&config, "maxInlineNodeCount") {
        cfg.max_inline_node_count = Some(require_usize(v, "maxInlineNodeCount")?);
    }
    if let Some(v) = get_f64(&config, "maxInlineCallSites") {
        cfg.max_inline_call_sites = Some(require_usize(v, "maxInlineCallSites")?);
    }
    if let Some(preamble) = get_string(&config, "preamble") {
        cfg.preamble = Some(preamble);
    }
    if let Some(validate) = get_bool(&config, "validateEachPass") {
        cfg.trace.validate_each_pass = validate;
    }

    Ok(cfg)
}

// MARK: Report projection

/// Project a [`crate::pipeline::PassReport`] into a plain JS object.
/// Numeric fields that are `None` become `null` rather than missing so
/// TypeScript consumers can rely on a closed field set.
fn pass_report_to_js(pr: &crate::pipeline::PassReport) -> JsValue {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), &v).unwrap_or(false);
    };
    set("passName", JsValue::from_str(&pr.pass_name));
    set(
        "beforeBytes",
        match pr.before_bytes {
            Some(v) => JsValue::from_f64(v as f64),
            None => JsValue::NULL,
        },
    );
    set(
        "afterBytes",
        match pr.after_bytes {
            Some(v) => JsValue::from_f64(v as f64),
            None => JsValue::NULL,
        },
    );
    set("changed", JsValue::from_bool(pr.changed));
    set("durationUs", JsValue::from_f64(pr.duration_us as f64));
    set("validationOk", JsValue::from_bool(pr.validation_ok));
    set(
        "textValidationOk",
        match pr.text_validation_ok {
            Some(v) => JsValue::from_bool(v),
            None => JsValue::NULL,
        },
    );
    set("rolledBack", JsValue::from_bool(pr.rolled_back));
    obj.into()
}

/// Project a [`crate::pipeline::Report`] into a plain JS object,
/// inlining every pass report so consumers need only a single
/// `Reflect.get` traversal.
fn report_to_js(report: &crate::pipeline::Report) -> JsValue {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), &v).unwrap_or(false);
    };
    set("inputBytes", JsValue::from_f64(report.input_bytes as f64));
    set("outputBytes", JsValue::from_f64(report.output_bytes as f64));
    set("converged", JsValue::from_bool(report.converged));
    set("sweeps", JsValue::from_f64(report.sweeps as f64));

    let passes = js_sys::Array::new();
    for pr in &report.pass_reports {
        passes.push(&pass_report_to_js(pr));
    }
    set("passReports", passes.into());

    obj.into()
}

// MARK: Public entry points

/// Minify a WGSL source string.  Returns a JS object matching the
/// TypeScript `Output` interface declared below.
///
/// # Errors
///
/// Propagates any [`crate::error::Error`] from the pipeline as a
/// [`JsError`] carrying the rendered message.
#[wasm_bindgen(skip_typescript)]
pub fn run(source: &str, config: JsValue) -> Result<JsValue, JsError> {
    let config = parse_config(config)?;
    let output = crate::run(source, &config).map_err(|e| JsError::new(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("source"),
        &JsValue::from_str(&output.source),
    )
    .unwrap_or(false);
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("report"),
        &report_to_js(&output.report),
    )
    .unwrap_or(false);
    Ok(obj.into())
}

/// Return the `CARGO_PKG_VERSION` baked into the wasm bundle.
#[wasm_bindgen(skip_typescript)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &str = r#"
export interface Config {
    profile?: "baseline" | "aggressive" | "max";
    preserveSymbols?: string[];
    mangle?: boolean;
    beautify?: boolean;
    indent?: number;
    maxPrecision?: number;
    maxInlineNodeCount?: number;
    maxInlineCallSites?: number;
    preamble?: string;
    validateEachPass?: boolean;
}

export interface PassReport {
    passName: string;
    beforeBytes: number | null;
    afterBytes: number | null;
    changed: boolean;
    durationUs: number;
    validationOk: boolean;
    textValidationOk: boolean | null;
    rolledBack: boolean;
}

export interface Report {
    inputBytes: number;
    outputBytes: number;
    converged: boolean;
    sweeps: number;
    passReports: PassReport[];
}

export interface Output {
    source: string;
    report: Report;
}

export function run(source: string, config?: Config): Output;
export function version(): string;
"#;
