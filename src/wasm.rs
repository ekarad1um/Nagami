//! JavaScript / browser binding layer exposed via `wasm-bindgen`.
//!
//! Mirrors the native [`crate::run`] entry point as a `JsValue -> JsValue`
//! boundary: JS config objects are validated and decoded into [`Config`],
//! the pipeline runs unchanged, and [`crate::pipeline::Report`] is
//! projected back to a plain JS object.  TypeScript declarations for
//! the wire types live in the `TS_TYPES` block at the bottom of the file.
//!
//! Only useful when compiled for the wasm32 target: the `wasm-bindgen`
//! ABI emits intrinsics that are linker errors on native.  `lib.rs`
//! already gates this module behind `cfg(feature = "wasm")`, but
//! belt-and-suspenders the same gate on `target_arch = "wasm32"`
//! here so a user who toggles the feature on for a native build gets
//! a graceful skip rather than the mysterious wasm-bindgen linker
//! errors.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::config::{Config, FloatPrecision, PrecisionMode, Profile};

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

/// Read `obj[key]` as an array of strings.  Returns `Ok(None)` when
/// the field is missing or not an array (callers default to "no
/// override"), `Err` when the array contains a non-string element so
/// the caller cannot silently misinterpret bad input.  Previously the
/// helper dropped non-string entries silently, which let
/// `preserveSymbols: ["main", 42]` slip through as just `["main"]` -
/// the caller's preserve-list then lacked an entry the user thought
/// they had set, producing surprising renames.
fn get_string_array(obj: &JsValue, key: &str) -> Result<Option<Vec<String>>, JsError> {
    let Some(val) = get_opt(obj, key) else {
        return Ok(None);
    };
    let arr: js_sys::Array = val
        .dyn_into()
        .map_err(|_| JsError::new(&format!("\"{key}\" must be an array of strings")))?;
    let mut out = Vec::with_capacity(arr.length() as usize);
    for (index, entry) in arr.iter().enumerate() {
        match entry.as_string() {
            Some(s) => out.push(s),
            None => {
                return Err(JsError::new(&format!(
                    "\"{key}\"[{index}] must be a string"
                )));
            }
        }
    }
    Ok(Some(out))
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

/// Decode a JS number as a `usize`, capped at `min(2^53, usize::MAX)`.
///
/// JavaScript numbers are IEEE-754 doubles and can represent integers
/// exactly only up to 2^53; values past that threshold silently lose
/// precision when cast to `usize`.  On wasm32 the platform `usize` is
/// `u32`, so an additional cap at `usize::MAX` is required - without
/// it, `v as usize` performs a saturating cast (Rust 1.45+) and any
/// `u32::MAX < v <= 2^53` silently clamps to `u32::MAX` rather than
/// erroring.  The cap chosen below picks whichever bound is tighter
/// for the build target so callers always see a crisp error instead
/// of a latent wrap or silent clamp.
fn require_usize(v: f64, key: &str) -> Result<usize, JsError> {
    const MAX_SAFE_F64: f64 = (1u64 << 53) as f64; // 9_007_199_254_740_992
    // Compare in `u128` so the bound choice is exact: `usize::MAX as f64`
    // rounds on native 64-bit (u64::MAX has no exact f64 representation),
    // which would skew a direct f64 comparison.
    let max = if (usize::MAX as u128) < (1u128 << 53) {
        usize::MAX as f64
    } else {
        MAX_SAFE_F64
    };
    if v.is_nan() || !(0.0..=max).contains(&v) || v.fract() != 0.0 {
        return Err(JsError::new(&format!(
            "\"{key}\" must be a non-negative integer in 0..={max}, got {v}"
        )));
    }
    Ok(v as usize)
}

// MARK: Precision decoding

/// Decode a single per-type precision slot from a JS value.  Accepts:
///
/// * `null` / `undefined` / `"full"` -> [`PrecisionMode::Full`]
/// * a bare number `N` -> shorthand for `{ decimalPlaces: N }`
/// * `{ decimalPlaces: N }` -> [`PrecisionMode::DecimalPlaces`]
/// * `{ significantFigures: N }` or `{ sigFigs: N }` -> [`PrecisionMode::SignificantFigures`]
///
/// `field_path` is used in error messages to point the caller at the
/// offending slot (e.g. `"floatPrecision.f32"`).
fn parse_precision_mode(val: &JsValue, field_path: &str) -> Result<PrecisionMode, JsError> {
    if val.is_undefined() || val.is_null() {
        return Ok(PrecisionMode::Full);
    }
    if let Some(s) = val.as_string() {
        return match s.as_str() {
            "full" => Ok(PrecisionMode::Full),
            other => Err(JsError::new(&format!(
                "{field_path}: unknown string mode \"{other}\" (expected \"full\")"
            ))),
        };
    }
    if let Some(n) = val.as_f64() {
        let p = require_u8(n, field_path)?;
        return Ok(PrecisionMode::DecimalPlaces(p));
    }
    if val.is_object() {
        let dp = get_f64(val, "decimalPlaces");
        let sf = get_f64(val, "significantFigures").or_else(|| get_f64(val, "sigFigs"));
        return match (dp, sf) {
            (Some(_), Some(_)) => Err(JsError::new(&format!(
                "{field_path}: decimalPlaces and significantFigures are mutually exclusive"
            ))),
            (Some(p), None) => Ok(PrecisionMode::DecimalPlaces(require_u8(
                p,
                &format!("{field_path}.decimalPlaces"),
            )?)),
            (None, Some(s)) => Ok(PrecisionMode::SignificantFigures(require_u8(
                s,
                &format!("{field_path}.significantFigures"),
            )?)),
            (None, None) => Ok(PrecisionMode::Full),
        };
    }
    Err(JsError::new(&format!(
        "{field_path} must be \"full\", a number, or an object with decimalPlaces / significantFigures"
    )))
}

/// Decode the `floatPrecision` field into a [`FloatPrecision`].  The JS
/// value may be:
///
/// * absent / `null` -> all kinds default to [`PrecisionMode::Full`]
/// * a single precision-mode spec (see [`parse_precision_mode`]) ->
///   applied uniformly to every float kind, mirroring
///   [`FloatPrecision::all`]
/// * an object with optional `f16` / `f32` / `f64` / `abstractFloat`
///   keys, each accepting any spec form parsed by [`parse_precision_mode`]
fn parse_float_precision(config: &JsValue) -> Result<FloatPrecision, JsError> {
    let Some(val) = get_opt(config, "floatPrecision") else {
        return Ok(FloatPrecision::default());
    };

    // Per-type form: object that carries at least one recognised slot.
    // Read each slot once and parse it in place; otherwise fall through
    // to the uniform-mode interpretation (so `floatPrecision: 6` and
    // `floatPrecision: { decimalPlaces: 6 }` both work).
    if val.is_object() {
        let f16 = get_opt(&val, "f16");
        let f32 = get_opt(&val, "f32");
        let f64 = get_opt(&val, "f64");
        let abs = get_opt(&val, "abstractFloat");
        if f16.is_some() || f32.is_some() || f64.is_some() || abs.is_some() {
            // Reject mixing per-type keys with a uniform-mode key on the
            // same object (e.g. `{ f32: 6, decimalPlaces: 3 }`): the
            // top-level `decimalPlaces` would be silently dropped, which
            // mirrors exactly the "silently misinterpret bad input" trap
            // `get_string_array` was deliberately hardened against.
            if get_opt(&val, "decimalPlaces").is_some()
                || get_opt(&val, "significantFigures").is_some()
                || get_opt(&val, "sigFigs").is_some()
            {
                return Err(JsError::new(
                    "floatPrecision: cannot mix per-type keys (f16/f32/f64/abstractFloat) \
                     with uniform keys (decimalPlaces/significantFigures/sigFigs) on the same object",
                ));
            }
            let mut out = FloatPrecision::default();
            if let Some(v) = f16 {
                out.f16 = parse_precision_mode(&v, "floatPrecision.f16")?;
            }
            if let Some(v) = f32 {
                out.f32 = parse_precision_mode(&v, "floatPrecision.f32")?;
            }
            if let Some(v) = f64 {
                out.f64 = parse_precision_mode(&v, "floatPrecision.f64")?;
            }
            if let Some(v) = abs {
                out.abstract_float = parse_precision_mode(&v, "floatPrecision.abstractFloat")?;
            }
            return Ok(out);
        }
    }

    // Uniform mode: same spec applied to every float kind.
    let mode = parse_precision_mode(&val, "floatPrecision")?;
    Ok(FloatPrecision::all(mode))
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
    if let Some(symbols) = get_string_array(&config, "preserveSymbols")? {
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
    cfg.float_precision = parse_float_precision(&config)?;
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
export type PrecisionMode =
    | "full"
    | number                                  // shorthand: number = decimal places
    | { decimalPlaces: number }
    | { significantFigures: number }
    | { sigFigs: number };                    // alias of significantFigures

export type FloatPrecision =
    | PrecisionMode                           // applied uniformly to every float kind
    | {
        f16?: PrecisionMode;
        f32?: PrecisionMode;
        f64?: PrecisionMode;
        abstractFloat?: PrecisionMode;
      };

export interface Config {
    profile?: "baseline" | "aggressive" | "max";
    preserveSymbols?: string[];
    mangle?: boolean;
    beautify?: boolean;
    indent?: number;
    floatPrecision?: FloatPrecision;
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
