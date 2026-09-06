//! JavaScript / browser binding layer exposed via `wasm-bindgen`.
//!
//! Mirrors the native [`crate::run`] entry point as a `JsValue -> JsValue`
//! boundary: JS config objects are validated and decoded into [`Config`],
//! the pipeline runs unchanged, and the result comes back as the parsed
//! [`crate::json`] document.  TypeScript declarations for the wire types
//! live in the `TS_TYPES` block at the bottom of the file.
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

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(msg: &str);
}

/// Route Rust panic info to `console.error` at module instantiation:
/// under `panic = "abort"` the hook still runs first, so browsers get
/// the panic text and file:line instead of an opaque
/// `RuntimeError: unreachable executed`.  Hand-rolled because the
/// `console_error_panic_hook` crate adds only a JS stack trace of
/// mangled wasm frame indices.
#[wasm_bindgen(start)]
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| console_error(&info.to_string())));
}

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

// The scalar getters return `Ok(None)` only when the key is ABSENT
// (undefined/null).  A present-but-wrong-typed value is an `Err`, not a
// silently-dropped `None` - otherwise a stringly-typed `{ mangle: "false" }`
// would be ignored and the default silently used.  This mirrors
// `get_string_array` / `parse_float_precision`, which the file twice hardens
// against the same "silently misinterpret bad input" trap.
fn get_string(obj: &JsValue, key: &str) -> Result<Option<String>, JsError> {
    match get_opt(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_string()
            .map(Some)
            .ok_or_else(|| JsError::new(&format!("\"{key}\" must be a string"))),
    }
}

fn get_bool(obj: &JsValue, key: &str) -> Result<Option<bool>, JsError> {
    match get_opt(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| JsError::new(&format!("\"{key}\" must be a boolean"))),
    }
}

fn get_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, JsError> {
    match get_opt(obj, key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| JsError::new(&format!("\"{key}\" must be a number"))),
    }
}

/// Read `obj[key]` as an array of strings.  Returns `Ok(None)` only when
/// the field is absent (callers default to "no override"); returns `Err`
/// when it is present but not an array, or is an array with a non-string
/// element, so a malformed value is never silently misinterpreted (e.g.
/// `preserveSymbols: ["main", 42]` errors rather than dropping `42`).
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

/// Decode a JS number as a `usize`.  A double is exact only up to 2^53, and
/// wasm32's `usize` is narrower still, so anything else is an error rather
/// than a silent clamp or precision loss.
fn require_usize(v: f64, key: &str) -> Result<usize, JsError> {
    const MAX_SAFE_F64: f64 = (1u64 << 53) as f64;
    if v.is_nan() || v.fract() != 0.0 || !(0.0..=MAX_SAFE_F64).contains(&v) {
        return Err(JsError::new(&format!(
            "\"{key}\" must be a non-negative integer in 0..=2^53, got {v}"
        )));
    }
    usize::try_from(v as u64).map_err(|_| {
        JsError::new(&format!(
            "\"{key}\" must fit the platform address range, got {v}"
        ))
    })
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
    // Shared across the two "wrong type" diagnostics below so the accepted
    // forms cannot drift apart.
    const EXPECTED: &str =
        "\"full\", a number, or an object with decimalPlaces / significantFigures";
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
    // `Array.isArray(x)` implies `typeof x === "object"`, so a JS array
    // would otherwise reach the `is_object()` branch, match no
    // decimalPlaces/significantFigures key, and be silently coerced to
    // `PrecisionMode::Full` - hiding a malformed slot.  Reject genuine
    // arrays explicitly (covers both `floatPrecision: [..]` and per-type
    // `{ f32: [..] }`, since each slot funnels through this fn).  Typed
    // arrays are not `Array.isArray` and still hit the object path.
    if js_sys::Array::is_array(val) {
        return Err(JsError::new(&format!(
            "{field_path} must be {EXPECTED} (received an array)"
        )));
    }
    if val.is_object() {
        let dp = get_f64(val, "decimalPlaces")?;
        let sf = match get_f64(val, "significantFigures")? {
            Some(s) => Some(s),
            None => get_f64(val, "sigFigs")?,
        };
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
    Err(JsError::new(&format!("{field_path} must be {EXPECTED}")))
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

    if let Some(p) = get_string(&config, "profile")? {
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
    if let Some(mangle) = get_bool(&config, "mangle")? {
        cfg.mangle = Some(mangle);
    }
    if let Some(beautify) = get_bool(&config, "beautify")? {
        cfg.beautify = beautify;
    }
    if let Some(indent) = get_f64(&config, "indent")? {
        cfg.indent = require_u8(indent, "indent")?;
    }
    cfg.float_precision = parse_float_precision(&config)?;
    if let Some(v) = get_f64(&config, "maxInlineNodeCount")? {
        cfg.max_inline_node_count = Some(require_usize(v, "maxInlineNodeCount")?);
    }
    if let Some(v) = get_f64(&config, "maxInlineCallSites")? {
        cfg.max_inline_call_sites = Some(require_usize(v, "maxInlineCallSites")?);
    }
    if let Some(preamble) = get_string(&config, "preamble")? {
        cfg.preamble = Some(preamble);
    }
    if let Some(validate) = get_bool(&config, "validateEachPass")? {
        cfg.trace.validate_each_pass = validate;
    }

    Ok(cfg)
}

// MARK: Public entry points

/// Minify `source`; the result is the parsed [`crate::json`] document
/// (TS `Output` below).  Pipeline errors surface as a thrown `JsError`.
#[wasm_bindgen(skip_typescript)]
pub fn run(source: &str, config: JsValue) -> Result<JsValue, JsError> {
    let config = parse_config(config)?;
    let output = crate::run(source, &config).map_err(|e| JsError::new(&e.to_string()))?;

    // The CLI prints this string and the binding parses it, so the TS
    // interfaces cannot drift from `--format json`.
    js_sys::JSON::parse(&crate::json::render_output(&output))
        .map_err(|e| JsError::new(&e.as_string().unwrap_or_else(|| format!("{e:?}"))))
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
    | number
    | { decimalPlaces: number }
    | { significantFigures: number }
    | { sigFigs: number };

export type FloatPrecision =
    | PrecisionMode
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
    /** naga's error when the output is the input lexically compacted only. */
    bailout: string | null;
    passReports: PassReport[];
}

export interface StructRename {
    name: string;
    members: Record<string, string>;
}

export interface NameMap {
    entryPoints: Record<string, string>;
    globals: Record<string, string>;
    functions: Record<string, string>;
    constants: Record<string, string>;
    overrides: Record<string, string>;
    structs: Record<string, StructRename>;
}

export interface Output {
    source: string;
    report: Report;
    nameMap: NameMap | null;
}

export function run(source: string, config?: Config): Output;
export function version(): string;
"#;
