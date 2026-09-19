//! Grammar-aware helpers every emit site consults: literal formatting (the
//! shortest round-tripping decimal/hex/scientific form under the per-type
//! [`PrecisionMode`] rounding) with the extraction key
//! [`super::literal_extract`] shares, and type/attribute/enum-name rendering
//! (alias lookup, `binding_attrs`, builtins, math names).  Struct-member
//! `@align`/`@size` layout belongs to the struct emitter, operator precedence
//! to `super::expr_emit`.

use crate::config::{FloatPrecision, PrecisionMode};
use crate::error::Error;
use crate::handle_set::HandleMap;
use naga::proc::TypeResolution;

// MARK: Literal formatting

/// Deduplication key for extracted literals: expression text plus
/// declaration text, so literals that render alike share one `const`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct LiteralExtractKey {
    pub(super) expr_text: String,
    pub(super) decl_text: String,
}

/// Strip redundant zeros from a float token (`-0.50f` -> `-.5f`) while
/// keeping suffix, sign, exponent and decimal point.
fn compact_float_literal_token(token: String) -> String {
    let s = token.as_str();
    let (core, suffix) = if let Some(stripped) = s.strip_suffix("lf") {
        (stripped, "lf")
    } else if let Some(stripped) = s.strip_suffix('f') {
        (stripped, "f")
    } else if let Some(stripped) = s.strip_suffix('h') {
        (stripped, "h")
    } else {
        (s, "")
    };

    let (sign, unsigned) = if let Some(stripped) = core.strip_prefix('-') {
        ("-", stripped)
    } else {
        ("", core)
    };

    let (mantissa, exponent) = if let Some(idx) = unsigned.find(['e', 'E']) {
        (&unsigned[..idx], &unsigned[idx..])
    } else {
        (unsigned, "")
    };

    let Some(dot_idx) = mantissa.find('.') else {
        return token;
    };

    let int_part = &mantissa[..dot_idx];
    let frac_part = &mantissa[dot_idx + 1..];
    if frac_part.is_empty() {
        return token;
    }

    let frac_trimmed = frac_part.trim_end_matches('0');

    let compact_mantissa = if int_part == "0" && frac_trimmed.is_empty() {
        "0.".to_string()
    } else if int_part.is_empty() && frac_trimmed.is_empty() {
        ".0".to_string()
    } else if frac_trimmed.is_empty() {
        format!("{int_part}.")
    } else if int_part == "0" || int_part.is_empty() {
        format!(".{frac_trimmed}")
    } else {
        format!("{int_part}.{frac_trimmed}")
    };

    format!("{sign}{compact_mantissa}{exponent}{suffix}")
}

/// Round `v` (finite, non-zero) to `sig_figs >= 1` significant figures.
///
/// With `scale_exp` in `0..=22`, `10^scale_exp` is exact, so the value is
/// scaled to put the requested figures left of the decimal point, rounded
/// half-away-from-zero and scaled back cleanly.  Outside that window the
/// power is inexact: both the scale and the scale-back leave binary noise
/// that mis-rounds (`1.495e-308` at 2 figures lands near `1.0e-308`, not
/// `1.5e-308`) and bloats the mantissa (`1.2300000000000007e300`), so the
/// rounding goes through `{:.*e}` with `sig_figs - 1` mantissa digits, which
/// `core::fmt` computes correctly (half-to-even, differing from half-away
/// only on exact ties this extreme range never hits), parsed back to the
/// nearest `f64`.
///
/// The scale-back can overflow near `f64::MAX`, so callers treat a
/// non-finite result as "leave unrounded"; a lossy round must also never
/// collapse a nonzero value to zero, so both paths fall back to `v` if they
/// would.
fn round_sig_figs_f64(v: f64, sig_figs: i32) -> f64 {
    // `sig_figs - 1` becomes an unsigned digit count; `0` would underflow it
    // to an enormous width.
    let sig_figs = sig_figs.max(1);
    let e = v.abs().log10().floor() as i32;
    // Clamp to f64's power-of-ten range even for an f32 caller: the math
    // runs in f64, and an f32 clamp would strip digits from f32 subnormals.
    let scale_exp = ((sig_figs - 1) - e).clamp(-f64::MAX_10_EXP, f64::MAX_10_EXP);

    if !(0..=22).contains(&scale_exp) {
        let rounded: f64 = format!("{:.*e}", (sig_figs - 1) as usize, v)
            .parse()
            .unwrap_or(v);
        return if rounded.is_finite() && rounded != 0.0 {
            rounded
        } else {
            v
        };
    }

    let scale = 10f64.powi(scale_exp);
    let mantissa = (v * scale).round();
    if mantissa == 0.0 {
        return v;
    }
    mantissa / scale
}

/// Round an `f32` per `mode`; `Full` and non-finite inputs pass through.
/// Rounding runs in `f64`, but the final narrowing can still overflow, so a
/// finite input that rounds out of `f32` range keeps its original value - a
/// lossy round must never manufacture an `inf`.  Call once per literal arm
/// and feed the rounded value to every candidate form, or an alternative
/// could emit the precise original of a value the user asked to truncate.
pub(super) fn round_f32(v: f32, mode: PrecisionMode) -> f32 {
    if !v.is_finite() {
        return v;
    }
    match mode {
        PrecisionMode::Full => v,
        PrecisionMode::DecimalPlaces(p) => {
            // Past `f32::MAX_10_EXP` no f32 has a fractional part left to
            // round; the product cannot overflow f64 (`f32::MAX * 1e38 ~=
            // 3.4e76`), so no finite-result guard is needed.
            let exp = (p as i32).min(f32::MAX_10_EXP);
            let scale = 10f64.powi(exp);
            (((v as f64) * scale).round() / scale) as f32
        }
        PrecisionMode::SignificantFigures(s) => {
            if v == 0.0 {
                return v;
            }
            // `0` figures would always round to zero; treat as `1`.
            let rounded = round_sig_figs_f64(v as f64, s.max(1) as i32) as f32;
            if rounded.is_finite() { rounded } else { v }
        }
    }
}

/// `f64` sibling of [`round_f32`]: the `|v| >= f64::MAX / scale` guard keeps
/// `v * scale` from overflowing, and a `SignificantFigures` scale-back that
/// overflowed near `f64::MAX` falls back to the original rather than an
/// `inflf` / `inf` token.
pub(super) fn round_f64(v: f64, mode: PrecisionMode) -> f64 {
    if !v.is_finite() {
        return v;
    }
    match mode {
        PrecisionMode::Full => v,
        PrecisionMode::DecimalPlaces(p) => {
            let exp = (p as i32).min(f64::MAX_10_EXP);
            let scale = 10f64.powi(exp);
            if v.abs() >= f64::MAX / scale {
                return v;
            }
            (v * scale).round() / scale
        }
        PrecisionMode::SignificantFigures(s) => {
            if v == 0.0 {
                return v;
            }
            let rounded = round_sig_figs_f64(v, s.max(1) as i32);
            if rounded.is_finite() { rounded } else { v }
        }
    }
}

/// `f16` sibling of [`round_f32`], on the `f32`-widened value the IR carries.
/// A lossy round can leave `f16`'s finite range (`SignificantFigures(1)` of
/// `65504` is `70000`, finite as `f32`), and `70000h` makes naga reject the
/// whole output, so an out-of-range result falls back to the original.
pub(super) fn round_f16(v: f32, mode: PrecisionMode) -> f32 {
    // `f16::MAX`, inlined to avoid a `half` dependency.
    const F16_MAX: f32 = 65504.0;
    let rounded = round_f32(v, mode);
    if rounded.abs() <= F16_MAX { rounded } else { v }
}

/// Append `.` to a bare float token that would otherwise lex as an integer
/// (`1` -> `1.`); tokens with a dot, exponent, hex marker or letters (`inf`,
/// `NaN`) pass through.
pub(super) fn ensure_bare_float(s: String) -> String {
    if s.contains('.') || s.contains('e') || s.contains('E') || s.contains('x') || s.contains('X') {
        return s;
    }
    if s.bytes().any(|b| b.is_ascii_alphabetic()) {
        return s;
    }
    format!("{s}.")
}

/// `Debug` keeps a whole number's `.0`, which survives
/// [`compact_float_literal_token`] as `1.lf`; naga rejects `1lf`.  `v` must
/// already be rounded.
fn fmt_f64_debug(v: f64) -> String {
    format!("{v:?}")
}

/// Shorter of decimal and lower-case hex, `suffix` included in both.
fn shortest_uint_repr(v: u64, suffix: &str) -> String {
    let dec = format!("{v}{suffix}");
    let hex = format!("0x{v:x}{suffix}");
    if hex.len() < dec.len() { hex } else { dec }
}

/// Signed sibling of [`shortest_uint_repr`].
fn shortest_int_repr(v: i64, suffix: &str) -> String {
    if v >= 0 {
        shortest_uint_repr(v as u64, suffix)
    } else {
        let abs = v.unsigned_abs();
        let dec = format!("-{abs}{suffix}");
        let hex = format!("-0x{abs:x}{suffix}");
        if hex.len() < dec.len() { hex } else { dec }
    }
}

/// `{sign}0x1[.{hex}]p{exp}{suffix}` for a normal `f32`; `None` for zero,
/// subnormal, infinity and NaN, which the decimal path handles.
fn hex_float_f32(v: f32, suffix: &str) -> Option<String> {
    let bits = v.to_bits();
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let exp_biased = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exp_biased == 0 || exp_biased == 255 {
        return None;
    }
    let exp = exp_biased - 127;
    Some(if mantissa == 0 {
        format!("{sign}0x1p{exp}{suffix}")
    } else {
        // 23 mantissa bits -> shift left 1 -> 24 bits = 6 hex digits.
        let hex_str = format!("{:06x}", mantissa << 1);
        let trimmed = hex_str.trim_end_matches('0');
        format!("{sign}0x1.{trimmed}p{exp}{suffix}")
    })
}

/// `f64` sibling of [`hex_float_f32`] (11 exponent bits, 52 mantissa bits).
fn hex_float_f64(v: f64, suffix: &str) -> Option<String> {
    let bits = v.to_bits();
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let exp_biased = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0xF_FFFF_FFFF_FFFF;
    if exp_biased == 0 || exp_biased == 2047 {
        return None;
    }
    let exp = exp_biased - 1023;
    Some(if mantissa == 0 {
        format!("{sign}0x1p{exp}{suffix}")
    } else {
        // 52 mantissa bits = 13 hex digits exactly.
        let hex_str = format!("{mantissa:013x}");
        let trimmed = hex_str.trim_end_matches('0');
        format!("{sign}0x1.{trimmed}p{exp}{suffix}")
    })
}

/// Bit-exact hex for `+/-f32::MAX`, the one f32 whose shortest decimal
/// (`3.4028235e38`) exceeds the true maximum: naga rounds it back and accepts
/// it, so the self-check passes, but tint/Dawn reject a literal whose exact
/// magnitude is out of range - a silent portability miscompile.  Only the
/// largest finite value has a rounding interval reaching past the maximum
/// (its upper neighbour is infinity); `f16::MAX` and `f64::MAX` have shortest
/// decimals at or below their maxima and need no guard.  `None` for every
/// other value.
fn f32_overshoot_safe_hex(v: f32, suffix: &str) -> Option<String> {
    (v.abs() == f32::MAX)
        .then(|| hex_float_f32(v, suffix))
        .flatten()
}

/// `true` when a decimal/scientific candidate recovers `v` bit-for-bit
/// through WGSL's concretization path (lexed as an AbstractFloat `f64`, then
/// narrowed).  Rust's shortest formatting only guarantees the direct
/// `str -> f32` parse (naga's path); tint/Dawn double-round through f64, and
/// for some bit patterns the two differ by 1 ULP (`7.038531e-26` narrows to
/// `0x15ae43fe` via f64, the value is `0x15ae43fd`), shipping a different
/// constant invisibly to the naga self-check.
fn f32_candidate_narrows_exactly(token: &str, suffix: &str, v: f32) -> bool {
    let numeric = token.strip_suffix(suffix).unwrap_or(token);
    numeric.parse::<f64>().map(|d| (d as f32).to_bits()) == Ok(v.to_bits())
}

/// Shortest f32 token that narrows exactly: a decimal/scientific candidate is
/// adopted only when [`f32_candidate_narrows_exactly`], else the exact hex.
/// The final `dec` fallthrough is defensive - every finite f32 narrows
/// exactly (normals via hex, every subnormal via its shortest decimal,
/// verified exhaustively), so only a NaN could reach it.
fn f32_shortest_exact(
    v: f32,
    suffix: &str,
    dec: String,
    hex: Option<String>,
    sci: Option<String>,
) -> String {
    let sci_safe = sci.filter(|s| f32_candidate_narrows_exactly(s, suffix, v));
    if f32_candidate_narrows_exactly(&dec, suffix, v) {
        pick_shortest(dec, [hex, sci_safe])
    } else if let Some(h) = hex {
        pick_shortest(h, [sci_safe])
    } else if let Some(s) = sci_safe {
        s
    } else {
        dec
    }
}

/// `{v:e}{suffix}` for a finite non-zero `f32` (`0e0` never beats `0`).
/// Subnormals are included, unlike [`hex_float_f32`]: their scientific form
/// is valid WGSL and round-trips, while the leading-`1` hex form cannot
/// represent them.  Rust's `{:e}` picks the shortest round-tripping mantissa
/// and omits the exponent's `+`, matching WGSL grammar; it wins where decimal
/// and hex are both long (`1e20f` vs `100000000000000000000f` or
/// `0x1.5af1d8p66f`).
fn scientific_float_f32(v: f32, suffix: &str) -> Option<String> {
    if !v.is_finite() || v == 0.0 {
        return None;
    }
    Some(format!("{v:e}{suffix}"))
}

/// `f64` sibling of [`scientific_float_f32`].
fn scientific_float_f64(v: f64, suffix: &str) -> Option<String> {
    if !v.is_finite() || v == 0.0 {
        return None;
    }
    Some(format!("{v:e}{suffix}"))
}

/// Shortest of `decimal` and the present `alternatives`; an alternative
/// wins only when strictly shorter, so ties keep the decimal.
fn pick_shortest<I>(decimal: String, alternatives: I) -> String
where
    I: IntoIterator<Item = Option<String>>,
{
    let mut best = decimal;
    for alt in alternatives.into_iter().flatten() {
        if alt.len() < best.len() {
            best = alt;
        }
    }
    best
}

/// Decimal candidate for a bare float from its `Display` token.  Whole
/// numbers collapse to bare integers (`1.0` -> `1`) because the enclosing
/// constructor pins the type; negative zero is the exception, since `-0`
/// re-parses as the integer `0` and drops the sign bit (observable through
/// `1.0 / x`), so it keeps a trailing dot (`-0.`).
fn bare_float_decimal(token: String, is_negative_zero: bool) -> String {
    let compact = compact_float_literal_token(token);
    if is_negative_zero {
        ensure_bare_float(compact)
    } else {
        compact
    }
}

/// A literal with its concrete type suffix (`1.5f`, `42i`, `3u`), safe
/// wherever it must carry its own type; floats are rounded per `precision`.
/// [`literal_to_wgsl_bare`] drops the suffix where an enclosing constructor
/// pins the type.
pub(super) fn literal_to_wgsl(literal: naga::Literal, precision: &FloatPrecision) -> String {
    match literal {
        naga::Literal::F16(v) => {
            // naga accepts a scientific `h` literal (`1e4h`) but rejects a
            // hex-float one (`0x1p10h`).
            let v = round_f16(f32::from(v), precision.f16);
            let dec = compact_float_literal_token(format!("{v}h"));
            pick_shortest(dec, [scientific_float_f32(v, "h")])
        }
        naga::Literal::F32(v) => {
            let v = round_f32(v, precision.f32);
            f32_overshoot_safe_hex(v, "f").unwrap_or_else(|| {
                let dec = compact_float_literal_token(format!("{v}f"));
                f32_shortest_exact(
                    v,
                    "f",
                    dec,
                    hex_float_f32(v, "f"),
                    scientific_float_f32(v, "f"),
                )
            })
        }
        naga::Literal::F64(v) => {
            // `Debug` keeps a whole number's `.0` (`1.lf` re-parses as a
            // float, `1lf` is rejected); hex-float and scientific `lf`
            // literals are accepted and cannot collapse to a bare int.
            let v = round_f64(v, precision.f64);
            let dec = compact_float_literal_token(format!("{}lf", fmt_f64_debug(v)));
            pick_shortest(dec, [hex_float_f64(v, "lf"), scientific_float_f64(v, "lf")])
        }
        // WGSL has no 16-bit integer literal; the constructor form matches
        // naga's WGSL backend.
        naga::Literal::U16(v) => format!("u16({v})"),
        naga::Literal::I16(v) => format!("i16({v})"),
        naga::Literal::U32(v) => shortest_uint_repr(v as u64, "u"),
        naga::Literal::I32(v) => {
            if v == i32::MIN {
                format!("i32({v})")
            } else {
                shortest_int_repr(v as i64, "i")
            }
        }
        naga::Literal::U64(v) => shortest_uint_repr(v, "lu"),
        naga::Literal::I64(v) => {
            if v == i64::MIN {
                let inner = shortest_int_repr(v + 1, "");
                format!("i64({inner} - 1)")
            } else {
                shortest_int_repr(v, "li")
            }
        }
        naga::Literal::Bool(v) => v.to_string(),
        naga::Literal::AbstractInt(v) => {
            if v == i64::MIN {
                let inner = shortest_int_repr(v + 1, "");
                format!("({inner} - 1)")
            } else {
                shortest_int_repr(v, "")
            }
        }
        naga::Literal::AbstractFloat(v) => {
            let v = round_f64(v, precision.abstract_float);
            let dec = compact_float_literal_token(format!("{v}"));
            ensure_bare_float(pick_shortest(
                dec,
                [hex_float_f64(v, ""), scientific_float_f64(v, "")],
            ))
        }
    }
}

/// A literal without its type suffix (`1.5`, `42`); floats are rounded per
/// `precision`.
///
/// Invariant: the caller's context must pin the type, because a bare token
/// re-parses as an abstract literal and a whole-number float (`F32(1.0)`)
/// collapses to a bare integer (`1`).  Two positions qualify: inside a
/// concrete type constructor `T(...)`, whose signature types every argument,
/// and the RHS of an extracted `const NAME = ...;`, which is abstract-typed
/// and valid only where each use pins a concrete type by coercion (a use
/// that cannot, such as a scalar `bitcast` source, bypasses the const for
/// the typed inline form).  Every other position - binary operands where
/// either side is a literal, overload-resolution arguments (`atan2(1.0, x)`),
/// standalone `let`/`var` initialisers - takes [`literal_to_wgsl`], or an
/// abstract-coercion surprise could flip overload resolution.
pub(super) fn literal_to_wgsl_bare(literal: naga::Literal, precision: &FloatPrecision) -> String {
    match literal {
        naga::Literal::F16(v) => {
            let v = round_f16(f32::from(v), precision.f16);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f32(v, ""), scientific_float_f32(v, "")])
        }
        naga::Literal::F32(v) => {
            let v = round_f32(v, precision.f32);
            f32_overshoot_safe_hex(v, "").unwrap_or_else(|| {
                let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
                f32_shortest_exact(
                    v,
                    "",
                    dec,
                    hex_float_f32(v, ""),
                    scientific_float_f32(v, ""),
                )
            })
        }
        naga::Literal::F64(v) => {
            let v = round_f64(v, precision.f64);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f64(v, ""), scientific_float_f64(v, "")])
        }
        // The constructor form is the only 16-bit integer spelling; bare
        // equals typed.
        naga::Literal::U16(v) => format!("u16({v})"),
        naga::Literal::I16(v) => format!("i16({v})"),
        naga::Literal::U32(v) => shortest_uint_repr(v as u64, ""),
        naga::Literal::I32(v) => {
            if v == i32::MIN {
                format!("i32({v})")
            } else {
                shortest_int_repr(v as i64, "")
            }
        }
        naga::Literal::U64(v) => shortest_uint_repr(v, ""),
        naga::Literal::I64(v) => {
            if v == i64::MIN {
                let inner = shortest_int_repr(v + 1, "");
                format!("i64({inner} - 1)")
            } else {
                shortest_int_repr(v, "")
            }
        }
        naga::Literal::Bool(v) => v.to_string(),
        naga::Literal::AbstractInt(v) => {
            if v == i64::MIN {
                let inner = shortest_int_repr(v + 1, "");
                format!("({inner} - 1)")
            } else {
                shortest_int_repr(v, "")
            }
        }
        naga::Literal::AbstractFloat(v) => {
            let v = round_f64(v, precision.abstract_float);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f64(v, ""), scientific_float_f64(v, "")])
        }
    }
}

/// Fixed overhead of one `<keyword> <name>=<body>;` module-scope declaration
/// (`const` and `alias` are both 5 characters): compact `const N=D;` = 8,
/// beautify `const N = D;\n` = 11.  Callers pass their actual output style;
/// pricing compact under beautify accepts borderline extractions that
/// net-cost two bytes per use.
fn decl_boilerplate(beautify: bool) -> usize {
    if beautify { 11 } else { 8 }
}

/// Fixed overhead of one body binding around its name and value: compact
/// `let N=E;` = 6, beautify `let N = E;\n` = 9 (its indent is not counted).
fn let_boilerplate(beautify: bool) -> usize {
    if beautify { 9 } else { 6 }
}

/// Bytes a value costs once bound: the declaration around `name` and `body`
/// (`boilerplate` its fixed part) plus the name at each of `uses` sites.
/// Every binding decision - a `let`, an extracted literal, a hoisted
/// constant, a type alias, a call's clone - is the bytes the uses spell
/// today against this and pays when today is the larger.  What a use
/// spells today and what a new name costs the other names differ between
/// the decisions and stay with them; this sum does not.
fn bound_cost(uses: usize, name: usize, boilerplate: usize, body: usize) -> usize {
    boilerplate + name + body + uses * name
}

/// [`bound_cost`] of a module-scope `const` / `alias` declaration.
pub(super) fn decl_cost(uses: usize, name: usize, body: usize, beautify: bool) -> usize {
    bound_cost(uses, name, decl_boilerplate(beautify), body)
}

/// [`bound_cost`] of a body `let`.
pub(super) fn let_cost(uses: usize, name: usize, value: usize, beautify: bool) -> usize {
    bound_cost(uses, name, let_boilerplate(beautify), value)
}

// MARK: Type rendering

/// The `(expr, decl)` [`LiteralExtractKey`] for [`super::literal_extract`]:
/// `expr_text` is the bare form (every use of an extracted `NAME` re-binds
/// by abstract coercion), `decl_text` the RHS of `const NAME = ...;`.
pub(super) fn literal_extract_key(
    literal: naga::Literal,
    precision: &FloatPrecision,
) -> LiteralExtractKey {
    let expr_text = literal_to_wgsl_bare(literal, precision);
    // A bare decl is abstract and re-binds by the i32/f32 defaults, correct
    // for `I32`/`U32`/`F32`/`Bool`; `F16`/`F64`/`I64`/`U64` keep the typed
    // form so the const carries its type (an f32 default refuses an f16
    // context, an i32 default loses i64 range).  Must agree with
    // `literal_needs_typed_form_outside_constructor`.
    let decl_text = match literal {
        naga::Literal::U64(v) if v > i64::MAX as u64 => literal_to_wgsl(literal, precision),
        naga::Literal::F16(_)
        | naga::Literal::F64(_)
        | naga::Literal::I64(_)
        | naga::Literal::U64(_) => literal_to_wgsl(literal, precision),
        _ => expr_text.clone(),
    };
    LiteralExtractKey {
        expr_text,
        decl_text,
    }
}

/// WGSL scalar type name; [`Error::Emit`] for a combination WGSL lacks.
pub(super) fn scalar_name(kind: naga::ScalarKind, width: u8) -> Result<&'static str, Error> {
    Ok(match (kind, width) {
        (naga::ScalarKind::Bool, _) => "bool",
        (naga::ScalarKind::Sint, 4) => "i32",
        (naga::ScalarKind::Sint, 8) => "i64",
        (naga::ScalarKind::Uint, 4) => "u32",
        (naga::ScalarKind::Uint, 8) => "u64",
        (naga::ScalarKind::Float, 2) => "f16",
        (naga::ScalarKind::Float, 4) => "f32",
        (naga::ScalarKind::Float, 8) => "f64",
        (naga::ScalarKind::AbstractInt, _) => "i32",
        (naga::ScalarKind::AbstractFloat, _) => "f32",
        _ => {
            return Err(Error::Emit(format!(
                "unsupported scalar kind/width: {:?}/{width}",
                kind,
            )));
        }
    })
}

/// Predeclared alias spellings a surviving declaration has taken, so
/// emission falls back to the long form instead of naming the user's type:
/// `struct vec4f {...}` is legal WGSL, and after it `vec4f` is that struct.
/// Empty for any shader that shadows nothing, which is nearly all of them.
///
/// std-hashed like the sibling name sets in `core`; `FxHashSet` here would
/// be a second hashbrown instantiation for `String` keys, and measurement
/// separates none of the candidates.
pub(super) type ShadowedAliases = std::collections::HashSet<String>;

/// The suffix set mirrors [`scalar_short_suffix`] rather than WGSL's shorter
/// one, so a new suffix cannot arrive on one side only; the unreachable
/// spellings it admits (WGSL has no integer matrix) cost nothing.
///
/// Only alias spellings need reserving: the long forms (`vec4<f32>`,
/// `array`, `f32`) are grammar or predeclared types naga refuses to see
/// redeclared and then used.
pub(super) fn is_predeclared_type_alias(name: &str) -> bool {
    let suffix = |s: &u8| matches!(s, b'f' | b'h' | b'i' | b'u');
    let dim = |c: &u8| (b'2'..=b'4').contains(c);
    match name.as_bytes() {
        [b'v', b'e', b'c', n, s] => dim(n) && suffix(s),
        [b'm', b'a', b't', c, b'x', r, s] => dim(c) && dim(r) && suffix(s),
        _ => false,
    }
}

/// `short` unless shadowed, in which case the caller's long form stands.
fn unshadowed(short: String, shadowed: &ShadowedAliases) -> Option<String> {
    (!shadowed.contains(&short)).then_some(short)
}

/// Suffix of the predeclared `vec3f` / `mat2x2f` aliases; `None` for
/// bool/i64/u64/f64, which have none.
fn scalar_short_suffix(kind: naga::ScalarKind, width: u8) -> Option<&'static str> {
    match (kind, width) {
        (naga::ScalarKind::Float, 2) => Some("h"),
        (naga::ScalarKind::Float, 4) => Some("f"),
        (naga::ScalarKind::Sint, 4) => Some("i"),
        (naga::ScalarKind::Uint, 4) => Some("u"),
        _ => None,
    }
}

/// Typed zero literal (`0u`, `0f`, `false`).  The abstract kinds map to
/// `0` / `0.0` so an unconcretised literal still round-trips as its kind.
pub(super) fn scalar_zero(kind: naga::ScalarKind, width: u8) -> &'static str {
    match (kind, width) {
        (naga::ScalarKind::Bool, _) => "false",
        (naga::ScalarKind::Sint, 4) => "0i",
        (naga::ScalarKind::Sint, 8) => "0li",
        (naga::ScalarKind::Uint, 4) => "0u",
        (naga::ScalarKind::Uint, 8) => "0lu",
        (naga::ScalarKind::Float, 2) => "0h",
        (naga::ScalarKind::Float, 4) => "0f",
        (naga::ScalarKind::Float, 8) => "0lf",
        (naga::ScalarKind::AbstractInt, _) => "0",
        (naga::ScalarKind::AbstractFloat, _) => "0.0",
        // Non-canonical widths fail naga validation; a bare `0` surfaces
        // through the round-trip validator rather than a panic.
        _ => "0",
    }
}

pub(super) fn vector_size_num(size: naga::VectorSize) -> u8 {
    size as u8
}

/// WGSL name of a [`TypeResolution`], aliases first.
pub(super) fn type_resolution_name(
    resolution: &TypeResolution,
    module: &naga::Module,
    struct_names: &HandleMap<naga::Type, String>,
    override_names: &[String],
    shadowed: &ShadowedAliases,
) -> Result<String, Error> {
    match resolution {
        TypeResolution::Handle(h) => {
            if let Some(name) = struct_names.get(h) {
                return Ok(name.clone());
            }
            type_inner_name(
                &module.types[*h].inner,
                module,
                struct_names,
                override_names,
                shadowed,
            )
        }
        TypeResolution::Value(inner) => {
            // Several handles can share one `TypeInner` (`type_groups`); any
            // of them with an alias wins.
            if let Some(name) = module.types.iter().find_map(|(handle, ty)| {
                (&ty.inner == inner)
                    .then(|| struct_names.get(handle).cloned())
                    .flatten()
            }) {
                return Ok(name);
            }
            type_inner_name(inner, module, struct_names, override_names, shadowed)
        }
    }
}

/// WGSL name of a [`naga::TypeInner`], aliases substituted where a handle
/// has one and `shadowed` has not claimed the spelling.  `shadowed` is
/// module-wide though a function-local shadows only inside its function: a
/// spelling that varied by function would be harder to reason about than the
/// few characters it saves in a shader that already collides.
pub(super) fn type_inner_name(
    inner: &naga::TypeInner,
    module: &naga::Module,
    struct_names: &HandleMap<naga::Type, String>,
    override_names: &[String],
    shadowed: &ShadowedAliases,
) -> Result<String, Error> {
    Ok(match inner {
        naga::TypeInner::Scalar(s) => scalar_name(s.kind, s.width)?.to_string(),
        naga::TypeInner::Vector { size, scalar } => {
            match scalar_short_suffix(scalar.kind, scalar.width)
                .and_then(|s| unshadowed(format!("vec{}{}", vector_size_num(*size), s), shadowed))
            {
                Some(short) => short,
                None => format!(
                    "vec{}<{}>",
                    vector_size_num(*size),
                    scalar_name(scalar.kind, scalar.width)?
                ),
            }
        }
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => match scalar_short_suffix(scalar.kind, scalar.width).and_then(|s| {
            unshadowed(
                format!(
                    "mat{}x{}{}",
                    vector_size_num(*columns),
                    vector_size_num(*rows),
                    s
                ),
                shadowed,
            )
        }) {
            Some(short) => short,
            None => format!(
                "mat{}x{}<{}>",
                vector_size_num(*columns),
                vector_size_num(*rows),
                scalar_name(scalar.kind, scalar.width)?
            ),
        },
        naga::TypeInner::Atomic(s) => {
            format!("atomic<{}>", scalar_name(s.kind, s.width)?)
        }
        naga::TypeInner::Pointer { base, space } => {
            format!(
                "ptr<{},{}{}>",
                address_space(*space),
                type_ref_from_handle(*base, module, struct_names, override_names, shadowed)?,
                pointer_access_suffix(*space)
            )
        }
        naga::TypeInner::ValuePointer {
            size,
            scalar,
            space,
        } => {
            let value_ty = match size {
                Some(v) => match scalar_short_suffix(scalar.kind, scalar.width)
                    .and_then(|s| unshadowed(format!("vec{}{}", vector_size_num(*v), s), shadowed))
                {
                    Some(short) => short,
                    None => format!(
                        "vec{}<{}>",
                        vector_size_num(*v),
                        scalar_name(scalar.kind, scalar.width)?
                    ),
                },
                None => scalar_name(scalar.kind, scalar.width)?.to_string(),
            };
            format!(
                "ptr<{},{}{}>",
                address_space(*space),
                value_ty,
                pointer_access_suffix(*space)
            )
        }
        naga::TypeInner::Array { base, size, .. } => {
            let base_ty =
                type_ref_from_handle(*base, module, struct_names, override_names, shadowed)?;
            match size {
                naga::ArraySize::Constant(n) => format!("array<{},{}>", base_ty, n.get()),
                naga::ArraySize::Dynamic => format!("array<{}>", base_ty),
                naga::ArraySize::Pending(h) => {
                    format!("array<{},{}>", base_ty, override_names[h.index()])
                }
            }
        }
        naga::TypeInner::Struct { .. } => {
            return Err(Error::Emit(
                "anonymous struct type cannot be emitted inline".into(),
            ));
        }
        naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } => image_type(*dim, *arrayed, *class)?,
        naga::TypeInner::Sampler { comparison } => {
            if *comparison {
                "sampler_comparison".to_string()
            } else {
                "sampler".to_string()
            }
        }
        naga::TypeInner::BindingArray { base, size } => {
            let base_ty =
                type_ref_from_handle(*base, module, struct_names, override_names, shadowed)?;
            match size {
                naga::ArraySize::Constant(n) => format!("binding_array<{},{}>", base_ty, n.get()),
                naga::ArraySize::Dynamic => format!("binding_array<{}>", base_ty),
                naga::ArraySize::Pending(h) => {
                    format!("binding_array<{},{}>", base_ty, override_names[h.index()])
                }
            }
        }
        // `vertex_return` is part of the type: dropped, it re-parses as the
        // plain form and silently loses the vertex-position query capability.
        naga::TypeInner::AccelerationStructure { vertex_return } => {
            if *vertex_return {
                "acceleration_structure<vertex_return>".to_string()
            } else {
                "acceleration_structure".to_string()
            }
        }
        // Same `vertex_return` contract.
        naga::TypeInner::RayQuery { vertex_return } => {
            if *vertex_return {
                "ray_query<vertex_return>".to_string()
            } else {
                "ray_query".to_string()
            }
        }
        _ => {
            return Err(Error::Emit(format!(
                "unsupported type: {}",
                type_inner_kind(inner)
            )));
        }
    })
}

/// WGSL name of a type handle, alias first.
pub(super) fn type_ref_from_handle(
    ty: naga::Handle<naga::Type>,
    module: &naga::Module,
    struct_names: &HandleMap<naga::Type, String>,
    override_names: &[String],
    shadowed: &ShadowedAliases,
) -> Result<String, Error> {
    if let Some(name) = struct_names.get(ty) {
        return Ok(name.clone());
    }
    type_inner_name(
        &module.types[ty].inner,
        module,
        struct_names,
        override_names,
        shadowed,
    )
}

// MARK: Diagnostic names

// Variant names for "cannot emit this" errors, so those messages never
// `{:?}` the value: deriving `Debug` for `naga::Expression` / `Statement` /
// `TypeInner` drags the whole recursive formatter into the binary to serve
// paths that already fall back to naga's emitter, and a variant name beats
// the multi-line tree dump as a diagnostic anyway.

pub(super) fn expression_kind(expression: &naga::Expression) -> &'static str {
    use naga::Expression as E;
    match expression {
        E::Literal(_) => "Literal",
        E::Constant(_) => "Constant",
        E::Override(_) => "Override",
        E::ZeroValue(_) => "ZeroValue",
        E::Compose { .. } => "Compose",
        E::Access { .. } => "Access",
        E::AccessIndex { .. } => "AccessIndex",
        E::Splat { .. } => "Splat",
        E::Swizzle { .. } => "Swizzle",
        E::FunctionArgument(_) => "FunctionArgument",
        E::GlobalVariable(_) => "GlobalVariable",
        E::LocalVariable(_) => "LocalVariable",
        E::Load { .. } => "Load",
        E::ImageSample { .. } => "ImageSample",
        E::ImageLoad { .. } => "ImageLoad",
        E::ImageQuery { .. } => "ImageQuery",
        E::Unary { .. } => "Unary",
        E::Binary { .. } => "Binary",
        E::Select { .. } => "Select",
        E::Derivative { .. } => "Derivative",
        E::Relational { .. } => "Relational",
        E::Math { .. } => "Math",
        E::As { .. } => "As",
        E::CallResult(_) => "CallResult",
        E::AtomicResult { .. } => "AtomicResult",
        E::WorkGroupUniformLoadResult { .. } => "WorkGroupUniformLoadResult",
        E::ArrayLength(_) => "ArrayLength",
        E::RayQueryProceedResult => "RayQueryProceedResult",
        E::RayQueryGetIntersection { .. } => "RayQueryGetIntersection",
        E::RayQueryVertexPositions { .. } => "RayQueryVertexPositions",
        E::SubgroupBallotResult => "SubgroupBallotResult",
        E::SubgroupOperationResult { .. } => "SubgroupOperationResult",
        E::CooperativeLoad { .. } => "CooperativeLoad",
        E::CooperativeMultiplyAdd { .. } => "CooperativeMultiplyAdd",
    }
}

pub(super) fn statement_kind(statement: &naga::Statement) -> &'static str {
    use naga::Statement as S;
    match statement {
        S::Emit(_) => "Emit",
        S::Block(_) => "Block",
        S::If { .. } => "If",
        S::Switch { .. } => "Switch",
        S::Loop { .. } => "Loop",
        S::Break => "Break",
        S::Continue => "Continue",
        S::Return { .. } => "Return",
        S::Kill => "Kill",
        S::ControlBarrier(_) => "ControlBarrier",
        S::MemoryBarrier(_) => "MemoryBarrier",
        S::Store { .. } => "Store",
        S::ImageStore { .. } => "ImageStore",
        S::Atomic { .. } => "Atomic",
        S::ImageAtomic { .. } => "ImageAtomic",
        S::WorkGroupUniformLoad { .. } => "WorkGroupUniformLoad",
        S::Call { .. } => "Call",
        S::RayQuery { .. } => "RayQuery",
        S::RayPipelineFunction(_) => "RayPipelineFunction",
        S::SubgroupBallot { .. } => "SubgroupBallot",
        S::SubgroupGather { .. } => "SubgroupGather",
        S::SubgroupCollectiveOperation { .. } => "SubgroupCollectiveOperation",
        S::CooperativeStore { .. } => "CooperativeStore",
    }
}

pub(crate) fn type_inner_kind(inner: &naga::TypeInner) -> &'static str {
    use naga::TypeInner as T;
    match inner {
        T::Scalar(_) => "Scalar",
        T::Vector { .. } => "Vector",
        T::Matrix { .. } => "Matrix",
        T::Atomic(_) => "Atomic",
        T::Pointer { .. } => "Pointer",
        T::ValuePointer { .. } => "ValuePointer",
        T::Array { .. } => "Array",
        T::Struct { .. } => "Struct",
        T::Image { .. } => "Image",
        T::Sampler { .. } => "Sampler",
        T::AccelerationStructure { .. } => "AccelerationStructure",
        T::RayQuery { .. } => "RayQuery",
        T::BindingArray { .. } => "BindingArray",
        T::CooperativeMatrix { .. } => "CooperativeMatrix",
    }
}

// MARK: Attribute and qualifier rendering

/// WGSL keyword.  `Function` renders `function` because `ptr<function, T>`
/// requires it; a `var<function>` declaration leaves the space implicit but
/// does not use this helper.
pub(crate) fn address_space(space: naga::AddressSpace) -> &'static str {
    match space {
        naga::AddressSpace::Function => "function",
        naga::AddressSpace::Private => "private",
        naga::AddressSpace::WorkGroup => "workgroup",
        naga::AddressSpace::Uniform => "uniform",
        naga::AddressSpace::Storage { .. } => "storage",
        naga::AddressSpace::RayPayload => "ray_payload",
        naga::AddressSpace::IncomingRayPayload => "incoming_ray_payload",
        // naga's WGSL frontend spells the push-constant space `immediate`
        // (it rejects `push_constant`); `private` would silently swap host
        // data for zero-init per-invocation memory.
        naga::AddressSpace::Immediate => "immediate",
        // No `var<...>` spelling exists.  `Handle` globals are intercepted
        // upstream (bare `var`) and a `TaskPayload` global fails naga
        // validation, so `private` is an unreachable placeholder, not a
        // semantic pick.
        naga::AddressSpace::Handle | naga::AddressSpace::TaskPayload => "private",
    }
}

/// The `,mode` tail of a `ptr<storage,T,mode>` type.  A storage pointer
/// defaults to `read`, so only a writable one spells its mode; every other
/// space fixes the mode.
fn pointer_access_suffix(space: naga::AddressSpace) -> String {
    match space {
        naga::AddressSpace::Storage { access } if storage_access(access) != "read" => {
            format!(",{}", storage_access(access))
        }
        _ => String::new(),
    }
}

/// WGSL access mode.  `ATOMIC` takes precedence: naga sets it on atomic
/// storage textures (`texture_storage_2d<r32uint, atomic>`) and the frontend
/// rejects any other mode there; `var<storage>` never carries it.
pub(crate) fn storage_access(access: naga::StorageAccess) -> &'static str {
    if access.contains(naga::StorageAccess::ATOMIC) {
        return "atomic";
    }
    let can_load = access.contains(naga::StorageAccess::LOAD);
    let can_store = access.contains(naga::StorageAccess::STORE);
    match (can_load, can_store) {
        (true, true) => "read_write",
        (true, false) => "read",
        (false, true) => "write",
        (false, false) => "read",
    }
}

/// Render the binding attributes attached to a struct member, function
/// parameter, or return type (`@location`, `@builtin`, `@interpolate`,
/// `@invariant`, `@blend_src`, `@per_primitive`), omitting defaults, and
/// gap-terminated for the identifier that follows.  In `compact` mode the
/// attributes join directly - `@` always ends the previous token, so
/// `@invariant@builtin(position)` lexes identically - and the trailing gap
/// survives only after a bare-word attribute (`@per_primitive`), where
/// joining with the following identifier would fuse tokens; a `)` joins
/// safely.
pub(crate) fn binding_attrs(binding: &naga::Binding, compact: bool) -> Result<String, Error> {
    let sep = if compact { "" } else { " " };
    let mut out = match binding {
        naga::Binding::BuiltIn(bi) => {
            if let naga::BuiltIn::Position { invariant: true } = bi {
                format!("@invariant{sep}@builtin({})", builtin_name(*bi)?)
            } else {
                format!("@builtin({})", builtin_name(*bi)?)
            }
        }
        naga::Binding::Location {
            location,
            interpolation,
            sampling,
            blend_src,
            per_primitive,
        } => {
            let mut out = format!("@location({location})");
            if let Some(bs) = blend_src {
                out.push_str(&format!("{sep}@blend_src({bs})"));
            }
            if *per_primitive {
                out.push_str(sep);
                out.push_str("@per_primitive");
            }
            // `@interpolate(perspective,center)` is the WGSL default, which
            // naga stores explicitly.
            let non_default_interp =
                interpolation.is_some_and(|i| i != naga::Interpolation::Perspective);
            let non_default_sampling = sampling.is_some_and(|s| s != naga::Sampling::Center);
            if non_default_interp || non_default_sampling {
                out.push_str(sep);
                out.push_str("@interpolate(");
                out.push_str(interpolation_name(
                    interpolation.unwrap_or(naga::Interpolation::Perspective),
                ));
                if let Some(s) = sampling {
                    out.push(',');
                    out.push_str(sampling_name(*s));
                }
                out.push(')');
            }
            out
        }
    };
    if !compact || !out.ends_with(')') {
        out.push(' ');
    }
    Ok(out)
}

pub(super) fn interpolation_name(i: naga::Interpolation) -> &'static str {
    match i {
        naga::Interpolation::Perspective => "perspective",
        naga::Interpolation::Linear => "linear",
        naga::Interpolation::Flat => "flat",
        naga::Interpolation::PerVertex => "per_vertex",
    }
}

pub(super) fn sampling_name(s: naga::Sampling) -> &'static str {
    match s {
        naga::Sampling::Center => "center",
        naga::Sampling::Centroid => "centroid",
        naga::Sampling::Sample => "sample",
        naga::Sampling::First => "first",
        naga::Sampling::Either => "either",
    }
}

/// `@builtin(...)` keyword.  Infallible for WGSL-frontend input; the `Result`
/// keeps callers uniform with the other renderers.
pub(super) fn builtin_name(bi: naga::BuiltIn) -> Result<&'static str, Error> {
    Ok(match bi {
        naga::BuiltIn::PrimitiveIndex => "primitive_index",
        naga::BuiltIn::Position { .. } => "position",
        naga::BuiltIn::ViewIndex => "view_index",
        naga::BuiltIn::BaseInstance => "base_instance",
        naga::BuiltIn::BaseVertex => "base_vertex",
        naga::BuiltIn::ClipDistances => "clip_distances",
        naga::BuiltIn::CullDistance => "cull_distance",
        naga::BuiltIn::InstanceIndex => "instance_index",
        naga::BuiltIn::PointSize => "point_size",
        naga::BuiltIn::VertexIndex => "vertex_index",
        naga::BuiltIn::DrawIndex => "draw_index",
        naga::BuiltIn::FragDepth => "frag_depth",
        naga::BuiltIn::PointCoord => "point_coord",
        naga::BuiltIn::FrontFacing => "front_facing",
        naga::BuiltIn::Barycentric { perspective: true } => "barycentric",
        naga::BuiltIn::Barycentric { perspective: false } => "barycentric_no_perspective",
        naga::BuiltIn::SampleIndex => "sample_index",
        naga::BuiltIn::SampleMask => "sample_mask",
        naga::BuiltIn::GlobalInvocationId => "global_invocation_id",
        naga::BuiltIn::LocalInvocationId => "local_invocation_id",
        naga::BuiltIn::LocalInvocationIndex => "local_invocation_index",
        naga::BuiltIn::WorkGroupId => "workgroup_id",
        naga::BuiltIn::WorkGroupSize => "workgroup_size",
        naga::BuiltIn::NumWorkGroups => "num_workgroups",
        naga::BuiltIn::NumSubgroups => "num_subgroups",
        naga::BuiltIn::SubgroupId => "subgroup_id",
        naga::BuiltIn::SubgroupSize => "subgroup_size",
        naga::BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
        naga::BuiltIn::MeshTaskSize => "workgroup_size",
        naga::BuiltIn::CullPrimitive => "cull_primitive",
        naga::BuiltIn::PointIndex => "point_index",
        naga::BuiltIn::LineIndices => "line_indices",
        naga::BuiltIn::TriangleIndices => "triangle_indices",
        naga::BuiltIn::VertexCount => "vertex_count",
        naga::BuiltIn::Vertices => "vertices",
        naga::BuiltIn::PrimitiveCount => "primitive_count",
        naga::BuiltIn::Primitives => "primitives",
        naga::BuiltIn::RayInvocationId => "ray_invocation_id",
        naga::BuiltIn::NumRayInvocations => "num_ray_invocations",
        naga::BuiltIn::InstanceCustomData => "instance_custom_data",
        naga::BuiltIn::GeometryIndex => "geometry_index",
        naga::BuiltIn::WorldRayOrigin => "world_ray_origin",
        naga::BuiltIn::WorldRayDirection => "world_ray_direction",
        naga::BuiltIn::ObjectRayOrigin => "object_ray_origin",
        naga::BuiltIn::ObjectRayDirection => "object_ray_direction",
        naga::BuiltIn::RayTmin => "ray_t_min",
        naga::BuiltIn::RayTCurrentMax => "ray_t_current_max",
        naga::BuiltIn::ObjectToWorld => "object_to_world",
        naga::BuiltIn::WorldToObject => "world_to_object",
        naga::BuiltIn::HitKind => "hit_kind",
    })
}

/// WGSL `texture_*` type; [`Error::Emit`] for a combination WGSL lacks.
pub(crate) fn image_type(
    dim: naga::ImageDimension,
    arrayed: bool,
    class: naga::ImageClass,
) -> Result<String, Error> {
    Ok(match class {
        naga::ImageClass::Sampled { kind, multi } => {
            let scalar = match kind {
                naga::ScalarKind::Float => "f32",
                naga::ScalarKind::Sint => "i32",
                naga::ScalarKind::Uint => "u32",
                _ => {
                    return Err(Error::Emit(format!(
                        "unsupported sampled texture scalar kind: {:?}",
                        kind
                    )));
                }
            };
            match (multi, dim, arrayed) {
                (false, naga::ImageDimension::D1, false) => format!("texture_1d<{scalar}>"),
                (false, naga::ImageDimension::D2, false) => format!("texture_2d<{scalar}>"),
                (false, naga::ImageDimension::D2, true) => format!("texture_2d_array<{scalar}>"),
                (false, naga::ImageDimension::D3, false) => format!("texture_3d<{scalar}>"),
                (false, naga::ImageDimension::Cube, false) => format!("texture_cube<{scalar}>"),
                (false, naga::ImageDimension::Cube, true) => {
                    format!("texture_cube_array<{scalar}>")
                }
                (true, naga::ImageDimension::D2, false) => {
                    format!("texture_multisampled_2d<{scalar}>")
                }
                (true, naga::ImageDimension::D2, true) => {
                    format!("texture_multisampled_2d_array<{scalar}>")
                }
                _ => {
                    return Err(Error::Emit(format!(
                        "unsupported sampled texture dimension: {:?} arrayed={} multi={}",
                        dim, arrayed, multi
                    )));
                }
            }
        }
        naga::ImageClass::Depth { multi } => match (multi, dim, arrayed) {
            (false, naga::ImageDimension::D2, false) => "texture_depth_2d".to_string(),
            (false, naga::ImageDimension::D2, true) => "texture_depth_2d_array".to_string(),
            (false, naga::ImageDimension::Cube, false) => "texture_depth_cube".to_string(),
            (false, naga::ImageDimension::Cube, true) => "texture_depth_cube_array".to_string(),
            (true, naga::ImageDimension::D2, false) => "texture_depth_multisampled_2d".to_string(),
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported depth texture dimension: {:?} arrayed={} multi={}",
                    dim, arrayed, multi
                )));
            }
        },
        naga::ImageClass::External => "texture_external".to_string(),
        naga::ImageClass::Storage { format, access } => {
            let dim_name = match (dim, arrayed) {
                (naga::ImageDimension::D1, false) => "texture_storage_1d",
                (naga::ImageDimension::D2, false) => "texture_storage_2d",
                (naga::ImageDimension::D2, true) => "texture_storage_2d_array",
                (naga::ImageDimension::D3, false) => "texture_storage_3d",
                _ => {
                    return Err(Error::Emit(format!(
                        "unsupported storage texture dimension: {:?} arrayed={}",
                        dim, arrayed
                    )));
                }
            };
            format!(
                "{}<{},{}>",
                dim_name,
                storage_format_name(format),
                storage_access(access)
            )
        }
    })
}

pub(super) fn storage_format_name(format: naga::StorageFormat) -> &'static str {
    match format {
        naga::StorageFormat::R8Unorm => "r8unorm",
        naga::StorageFormat::R8Snorm => "r8snorm",
        naga::StorageFormat::R8Uint => "r8uint",
        naga::StorageFormat::R8Sint => "r8sint",
        naga::StorageFormat::R16Uint => "r16uint",
        naga::StorageFormat::R16Sint => "r16sint",
        naga::StorageFormat::R16Float => "r16float",
        naga::StorageFormat::Rg8Unorm => "rg8unorm",
        naga::StorageFormat::Rg8Snorm => "rg8snorm",
        naga::StorageFormat::Rg8Uint => "rg8uint",
        naga::StorageFormat::Rg8Sint => "rg8sint",
        naga::StorageFormat::R32Uint => "r32uint",
        naga::StorageFormat::R32Sint => "r32sint",
        naga::StorageFormat::R32Float => "r32float",
        naga::StorageFormat::Rg16Uint => "rg16uint",
        naga::StorageFormat::Rg16Sint => "rg16sint",
        naga::StorageFormat::Rg16Float => "rg16float",
        naga::StorageFormat::Rgba8Unorm => "rgba8unorm",
        naga::StorageFormat::Rgba8Snorm => "rgba8snorm",
        naga::StorageFormat::Rgba8Uint => "rgba8uint",
        naga::StorageFormat::Rgba8Sint => "rgba8sint",
        naga::StorageFormat::Bgra8Unorm => "bgra8unorm",
        naga::StorageFormat::Rgb10a2Uint => "rgb10a2uint",
        naga::StorageFormat::Rgb10a2Unorm => "rgb10a2unorm",
        naga::StorageFormat::Rg11b10Ufloat => "rg11b10ufloat",
        naga::StorageFormat::R64Uint => "r64uint",
        naga::StorageFormat::Rg32Uint => "rg32uint",
        naga::StorageFormat::Rg32Sint => "rg32sint",
        naga::StorageFormat::Rg32Float => "rg32float",
        naga::StorageFormat::Rgba16Uint => "rgba16uint",
        naga::StorageFormat::Rgba16Sint => "rgba16sint",
        naga::StorageFormat::Rgba16Float => "rgba16float",
        naga::StorageFormat::Rgba32Uint => "rgba32uint",
        naga::StorageFormat::Rgba32Sint => "rgba32sint",
        naga::StorageFormat::Rgba32Float => "rgba32float",
        naga::StorageFormat::R16Unorm => "r16unorm",
        naga::StorageFormat::R16Snorm => "r16snorm",
        naga::StorageFormat::Rg16Unorm => "rg16unorm",
        naga::StorageFormat::Rg16Snorm => "rg16snorm",
        naga::StorageFormat::Rgba16Unorm => "rgba16unorm",
        naga::StorageFormat::Rgba16Snorm => "rgba16snorm",
    }
}

/// WGSL builtin name; exhaustive so a new naga variant fails the build.
pub(crate) fn math_name(fun: naga::MathFunction) -> &'static str {
    use naga::MathFunction as M;
    match fun {
        M::Abs => "abs",
        M::Min => "min",
        M::Max => "max",
        M::Clamp => "clamp",
        M::Saturate => "saturate",
        M::Cos => "cos",
        M::Cosh => "cosh",
        M::Sin => "sin",
        M::Sinh => "sinh",
        M::Tan => "tan",
        M::Tanh => "tanh",
        M::Acos => "acos",
        M::Asin => "asin",
        M::Atan => "atan",
        M::Atan2 => "atan2",
        M::Asinh => "asinh",
        M::Acosh => "acosh",
        M::Atanh => "atanh",
        M::Radians => "radians",
        M::Degrees => "degrees",
        M::Ceil => "ceil",
        M::Floor => "floor",
        M::Round => "round",
        M::Fract => "fract",
        M::Trunc => "trunc",
        M::Modf => "modf",
        M::Frexp => "frexp",
        M::Ldexp => "ldexp",
        M::Exp => "exp",
        M::Exp2 => "exp2",
        M::Log => "log",
        M::Log2 => "log2",
        M::Pow => "pow",
        M::Dot => "dot",
        M::Dot4I8Packed => "dot4I8Packed",
        M::Dot4U8Packed => "dot4U8Packed",
        M::Outer => "outerProduct",
        M::Cross => "cross",
        M::Distance => "distance",
        M::Length => "length",
        M::Normalize => "normalize",
        M::FaceForward => "faceForward",
        M::Reflect => "reflect",
        M::Refract => "refract",
        M::Sign => "sign",
        M::Fma => "fma",
        M::Mix => "mix",
        M::Step => "step",
        M::SmoothStep => "smoothstep",
        M::Sqrt => "sqrt",
        M::InverseSqrt => "inverseSqrt",
        M::Inverse => "inverse",
        M::Transpose => "transpose",
        M::Determinant => "determinant",
        M::QuantizeToF16 => "quantizeToF16",
        M::CountTrailingZeros => "countTrailingZeros",
        M::CountLeadingZeros => "countLeadingZeros",
        M::CountOneBits => "countOneBits",
        M::ReverseBits => "reverseBits",
        M::ExtractBits => "extractBits",
        M::InsertBits => "insertBits",
        M::FirstTrailingBit => "firstTrailingBit",
        M::FirstLeadingBit => "firstLeadingBit",
        M::Pack4x8snorm => "pack4x8snorm",
        M::Pack4x8unorm => "pack4x8unorm",
        M::Pack2x16snorm => "pack2x16snorm",
        M::Pack2x16unorm => "pack2x16unorm",
        M::Pack2x16float => "pack2x16float",
        M::Pack4xI8 => "pack4xI8",
        M::Pack4xU8 => "pack4xU8",
        M::Pack4xI8Clamp => "pack4xI8Clamp",
        M::Pack4xU8Clamp => "pack4xU8Clamp",
        M::Unpack4x8snorm => "unpack4x8snorm",
        M::Unpack4x8unorm => "unpack4x8unorm",
        M::Unpack2x16snorm => "unpack2x16snorm",
        M::Unpack2x16unorm => "unpack2x16unorm",
        M::Unpack2x16float => "unpack2x16float",
        M::Unpack4xI8 => "unpack4xI8",
        M::Unpack4xU8 => "unpack4xU8",
    }
}

/// `@early_depth_test(...)`, naga's extension, spelled as its front-end
/// reads it.
pub(crate) fn early_depth_test_attr(test: naga::EarlyDepthTest) -> &'static str {
    use naga::{ConservativeDepth as D, EarlyDepthTest as T};
    match test {
        T::Force => "@early_depth_test(force)",
        T::Allow {
            conservative: D::GreaterEqual,
        } => "@early_depth_test(greater_equal)",
        T::Allow {
            conservative: D::LessEqual,
        } => "@early_depth_test(less_equal)",
        T::Allow {
            conservative: D::Unchanged,
        } => "@early_depth_test(unchanged)",
    }
}

// MARK: Diagnostic directive rendering

pub(crate) fn severity_name(severity: naga::diagnostic_filter::Severity) -> &'static str {
    use naga::diagnostic_filter::Severity as S;
    match severity {
        S::Off => "off",
        S::Info => "info",
        S::Warning => "warning",
        S::Error => "error",
    }
}

pub(crate) fn triggering_rule_name(
    rule: &naga::diagnostic_filter::FilterableTriggeringRule,
) -> String {
    use naga::diagnostic_filter::FilterableTriggeringRule as R;
    match rule {
        R::Standard(std_rule) => match std_rule {
            naga::diagnostic_filter::StandardFilterableTriggeringRule::DerivativeUniformity => {
                "derivative_uniformity".to_string()
            }
        },
        R::Unknown(name) => name.to_string(),
        R::User(parts) => format!("{}.{}", parts[0], parts[1]),
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::{
        FloatPrecision, PrecisionMode, compact_float_literal_token, decl_cost, ensure_bare_float,
        let_cost, literal_extract_key, literal_to_wgsl, literal_to_wgsl_bare, round_sig_figs_f64,
        scalar_zero,
    };
    use half::f16;

    /// The one sum every binding decision prices against: the declaration
    /// in the output's own style plus the name at every use; the emitter's
    /// `let` rule is its inline renderings, wrapped ones counted twice,
    /// against it.
    #[test]
    fn a_bound_value_costs_its_declaration_and_a_name_per_use() {
        assert_eq!(decl_cost(3, 1, 9, false), 8 + 1 + 9 + 3);
        assert_eq!(decl_cost(3, 1, 9, true), 11 + 1 + 9 + 3);
        assert_eq!(let_cost(2, 1, 5, false), 6 + 1 + 5 + 2);
        assert_eq!(let_cost(0, 2, 7, true), 9 + 2 + 7);
        for refs in 0..5 {
            for parens in 0..=refs {
                for len in 0..12 {
                    for name in 1..3 {
                        let inline = refs * len + 2 * parens;
                        let bound = len + name + 6 + refs * name;
                        assert_eq!(
                            super::super::stmt_emit::binding_pays(refs, parens, len, name, false),
                            inline > bound,
                            "refs {refs} parens {parens} len {len} name {name}"
                        );
                    }
                }
            }
        }
    }

    fn full() -> FloatPrecision {
        FloatPrecision::default()
    }

    fn dp(n: u8) -> FloatPrecision {
        FloatPrecision::all(PrecisionMode::DecimalPlaces(n))
    }

    fn sf(n: u8) -> FloatPrecision {
        FloatPrecision::all(PrecisionMode::SignificantFigures(n))
    }

    #[test]
    fn compacts_unsuffixed_decimal_tokens() {
        assert_eq!(compact_float_literal_token("0.123".into()), ".123");
        assert_eq!(compact_float_literal_token("123.0".into()), "123.");
        assert_eq!(compact_float_literal_token("-0.5000".into()), "-.5");
    }

    #[test]
    fn compacts_suffixed_decimal_tokens() {
        assert_eq!(compact_float_literal_token("0.5f".into()), ".5f");
        assert_eq!(compact_float_literal_token("0.5h".into()), ".5h");
        assert_eq!(compact_float_literal_token("0.5lf".into()), ".5lf");
        assert_eq!(compact_float_literal_token("-0.25f".into()), "-.25f");
    }

    #[test]
    fn compacts_decimal_with_exponent() {
        assert_eq!(compact_float_literal_token("1.50e10".into()), "1.5e10");
        assert_eq!(compact_float_literal_token("0.5E-3".into()), ".5E-3");
        assert_eq!(compact_float_literal_token("-0.50e2f".into()), "-.5e2f");
    }

    #[test]
    fn early_return_no_dot() {
        assert_eq!(compact_float_literal_token("1f".into()), "1f");
        assert_eq!(compact_float_literal_token("42lf".into()), "42lf");
        assert_eq!(compact_float_literal_token("7".into()), "7");
    }

    #[test]
    fn early_return_empty_frac() {
        assert_eq!(compact_float_literal_token("1.".into()), "1.");
        assert_eq!(compact_float_literal_token("1.f".into()), "1.f");
    }

    #[test]
    fn keeps_non_decimal_tokens_unchanged() {
        assert_eq!(compact_float_literal_token("1e-5".into()), "1e-5");
        assert_eq!(compact_float_literal_token("nan".into()), "nan");
    }

    #[test]
    fn compacts_float_literal_variants() {
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.25), &full()), ".25f");
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(0.5), &full()),
            ".5"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F64(0.5), &full()), ".5lf");
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &full()), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), &full()), "7u");

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(0.25), &full()),
            ".25"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.5), &full()), ".5");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U32(7), &full()), "7");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I32(42), &full()), "42");
    }

    #[test]
    fn literal_i32_min_wraps_in_constructor() {
        assert_eq!(
            literal_to_wgsl(naga::Literal::I32(i32::MIN), &full()),
            "i32(-2147483648)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::I32(i32::MIN), &full()),
            "i32(-2147483648)"
        );
    }

    #[test]
    fn literal_i64_typed_and_bare() {
        assert_eq!(literal_to_wgsl(naga::Literal::I64(99), &full()), "99li");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I64(99), &full()), "99");
        // `-9223372036854775808li` would overflow before negation.
        assert_eq!(
            literal_to_wgsl(naga::Literal::I64(i64::MIN), &full()),
            "i64(-0x7fffffffffffffff - 1)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::I64(i64::MIN), &full()),
            "i64(-0x7fffffffffffffff - 1)"
        );
    }

    #[test]
    fn literal_u64_typed_and_bare() {
        assert_eq!(literal_to_wgsl(naga::Literal::U64(100), &full()), "100lu");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::U64(100), &full()),
            "100"
        );
    }

    #[test]
    fn literal_bool_and_abstract_int() {
        assert_eq!(literal_to_wgsl(naga::Literal::Bool(true), &full()), "true");
        assert_eq!(
            literal_to_wgsl(naga::Literal::Bool(false), &full()),
            "false"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(42), &full()),
            "42"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::Bool(true), &full()),
            "true"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractInt(-7), &full()),
            "-7"
        );
        // Same overflow-safe form for `AbstractInt`.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(i64::MIN), &full()),
            "(-0x7fffffffffffffff - 1)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractInt(i64::MIN), &full()),
            "(-0x7fffffffffffffff - 1)"
        );
    }

    #[test]
    fn scalar_zero_values() {
        assert_eq!(scalar_zero(naga::ScalarKind::Bool, 1), "false");
        assert_eq!(scalar_zero(naga::ScalarKind::Sint, 4), "0i");
        assert_eq!(scalar_zero(naga::ScalarKind::Sint, 8), "0li");
        assert_eq!(scalar_zero(naga::ScalarKind::Uint, 4), "0u");
        assert_eq!(scalar_zero(naga::ScalarKind::Uint, 8), "0lu");
        assert_eq!(scalar_zero(naga::ScalarKind::Float, 2), "0h");
        assert_eq!(scalar_zero(naga::ScalarKind::Float, 4), "0f");
        assert_eq!(scalar_zero(naga::ScalarKind::Float, 8), "0lf");
        assert_eq!(scalar_zero(naga::ScalarKind::AbstractInt, 0), "0");
    }

    #[test]
    fn decimal_places_rounds_to_n_places() {
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), &dp(3)),
            ".123f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.876543), &dp(2)),
            ".88f"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(0.876543), &dp(2)),
            ".88"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(7.65432198), &dp(4)),
            "7.6543"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &dp(2)), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), &dp(2)), "7u");
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), &full()),
            ".123456f"
        );
    }

    #[test]
    fn precision_preserves_whole_number_shortest_form() {
        // With precision enabled a whole-number float still drops the `.0`:
        // the suffix pins the type.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &dp(6)), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-2.0), &dp(2)), "-2f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &dp(6)), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &dp(6)), "-0f");

        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), &dp(6)), "1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(-2.0), &dp(2)), "-2");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), &dp(6)), "0");

        // A value that rounds up to a whole number collapses too.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.999), &dp(2)), "1f");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.999), &dp(2)), "1");

        // Half-away-from-zero rounding at the boundary.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &dp(0)), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.49), &dp(0)), "0f");

        // The typed AbstractFloat keeps a dot so it does not re-parse as
        // AbstractInt.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), &dp(6)),
            "1"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1.0), &dp(6)),
            "1."
        );

        // F64 keeps the dot: naga rejects `1lf`.
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1.0), &dp(6)), "1.lf");
    }

    #[test]
    fn precision_hex_path_still_wins_when_shorter() {
        // Rounding leaves 2^20 intact, so hex (`0x1p20f`) still beats the
        // 8-char decimal.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &dp(6)),
            "0x1p20f"
        );
    }

    #[test]
    fn precision_handles_non_finite_values() {
        // No WGSL spelling exists for these either way; the token must at
        // least be mode-independent.
        for v in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(
                literal_to_wgsl(naga::Literal::F32(v), &dp(6)),
                literal_to_wgsl(naga::Literal::F32(v), &full()),
            );
        }
    }

    #[test]
    fn f32_max_emits_exact_hex_not_overshooting_decimal() {
        // `f32::MAX`'s shortest decimal overshoots the maximum and tint/Dawn
        // reject it.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), &full()),
            "0x1.fffffep127f"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(f32::MAX), &full()),
            "0x1.fffffep127"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(-f32::MAX), &full()),
            "-0x1.fffffep127f"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-f32::MAX), &full()),
            "-0x1.fffffep127"
        );
        // The guard is exactly `+/-f32::MAX`.
        let below_max = f32::from_bits(f32::MAX.to_bits() - 1);
        assert_ne!(
            literal_to_wgsl(naga::Literal::F32(below_max), &full()),
            "0x1.fffffep127f"
        );
    }

    #[test]
    fn precision_hex_form_uses_rounded_value() {
        // Every candidate passed to `pick_shortest` must encode the same
        // value: hex of the rounded 1048576 is `0x1p20f`, while hex of the
        // original 1048575.9 would describe a value the user asked to
        // truncate away.
        let s = literal_to_wgsl(naga::Literal::F32(1048575.9), &dp(0));
        assert_eq!(s, "0x1p20f");
        let s = literal_to_wgsl_bare(naga::Literal::F32(1048575.9), &dp(0));
        assert_eq!(s, "0x1p20");
    }

    #[test]
    fn scientific_form_chosen_when_shorter() {
        // `1e6f` beats `1000000f` and `0x1.e848p19f`.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e6), &full()), "1e6f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e10), &full()), "1e10f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e20), &full()), "1e20f");

        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1e-30), &full()),
            "1e-30f"
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1e10), &full()),
            "1e10"
        );

        // `ensure_bare_float` accepts the `e` as the float marker.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1e20), &full()),
            "1e20"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1e100), &full()),
            "1e100"
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(1e15), &full()),
            "1e15"
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(1e4)), &full()),
            "1e4"
        );

        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &full()), ".5f");

        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &full()),
            "0x1p20f"
        );

        // Zero never takes the `0e0f` form.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &full()), "-0f");
    }

    #[test]
    fn scientific_form_aligns_with_rounding() {
        // Rounding precedes candidate formation: 999999.5 rounds away from
        // zero to 1e6 and picks `1e6f`.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(999999.5), &dp(0)),
            "1e6f"
        );
    }

    #[test]
    fn significant_figures_round_independent_of_magnitude() {
        // Significant figures pin the digit count regardless of magnitude.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1234567.9), &sf(3)),
            "1.23e6f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.001234), &sf(3)),
            ".00123f"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F32(789.0), &sf(1)), "800f");
        // `SignificantFigures(0)` is treated as `1`.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(123.0), &sf(0)), "100f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &sf(4)), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &sf(2)), "42i");
    }

    /// The inexact-power range must round correctly and render short, and
    /// `sig_figs <= 0` must not underflow the fractional-digit count.
    #[test]
    fn round_sig_figs_f64_extreme_magnitudes() {
        let tiny = round_sig_figs_f64(1.495e-308, 2);
        assert!(
            (1.4e-308..=1.6e-308).contains(&tiny),
            "1.495e-308 @ 2sf should be ~1.5e-308, got {tiny:e}"
        );
        let min_pos = round_sig_figs_f64(f64::MIN_POSITIVE, 2); // 2.225e-308
        assert!(
            (2.1e-308..=2.3e-308).contains(&min_pos),
            "f64::MIN_POSITIVE @ 2sf should be ~2.2e-308, got {min_pos:e}"
        );
        // Must render short, not as a noise mantissa.
        let clean = round_sig_figs_f64(1.23456e300, 3);
        assert!(
            format!("{clean:e}").len() <= 10,
            "1.23456e300 @ 3sf should render short, got {clean:e}"
        );
        // 9.5e307's nearest f64 is below 9.5e307, so one figure is 9e307.
        let big = round_sig_figs_f64(9.5e307, 1);
        assert!(
            big.is_finite() && format!("{big:e}").len() <= 8,
            "9.5e307 @ 1sf should be a clean finite value, got {big:e}"
        );
        // An overflowing round falls back to the finite original.
        assert!(round_sig_figs_f64(f64::MAX, 1).is_finite());
        assert!(round_sig_figs_f64(123.0, 0).is_finite());
        assert_eq!(round_sig_figs_f64(1.23456, 3), 1.23);
    }

    #[test]
    fn significant_figures_covers_f16_and_f64_kinds() {
        // f16 rounds through `round_f32` with the f16 mode.
        let s = literal_to_wgsl(naga::Literal::F16(f16::from_f32(0.456)), &sf(2));
        assert_eq!(s, ".46h", "got {s:?}");

        // Both f64 paths offer the scientific candidate (naga accepts
        // `...e...lf`).
        let s = literal_to_wgsl(naga::Literal::F64(1234567.89_f64), &sf(3));
        assert_eq!(s, "1.23e6lf");
        let s = literal_to_wgsl_bare(naga::Literal::F64(1234567.89_f64), &sf(3));
        assert_eq!(s, "1.23e6");
    }

    #[test]
    fn significant_figures_preserves_precision_at_f32_boundaries() {
        // Near `f32::MIN_POSITIVE` the figures survive: the scale exponent
        // is clamped to f64's range, not f32's.
        let s = literal_to_wgsl(naga::Literal::F32(1.18e-38), &sf(2));
        assert!(
            s.starts_with("1.2e-38") || s == "1.2e-38f",
            "expected 2 sig figs of 1.18e-38, got {s:?}"
        );
        // A subnormal survives; exact f32 subnormals are not short decimals,
        // so any nearby rendering is accepted.
        let s = literal_to_wgsl(naga::Literal::F32(1e-40), &sf(2));
        assert!(
            !s.starts_with("0") && (s.contains("e-40") || s.contains("e-41")),
            "expected sf=2 of 1e-40 subnormal to be preserved, got {s:?}"
        );
        // An extreme figure count on f32::MAX still yields a finite token.
        let s = literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(255));
        assert!(s.contains("e38f") || s.contains("0x"), "got {s:?}");
    }

    #[test]
    fn per_type_precision_dispatch() {
        let precision = FloatPrecision {
            f32: PrecisionMode::DecimalPlaces(2),
            f64: PrecisionMode::Full,
            ..Default::default()
        };
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), &precision),
            ".12f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(0.123456), &precision),
            ".123456lf"
        );

        // f16 has its own slot, routed through `round_f32`.
        let precision = FloatPrecision {
            f16: PrecisionMode::DecimalPlaces(1),
            f32: PrecisionMode::Full,
            ..Default::default()
        };
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(0.876)), &precision,),
            ".9h",
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.876), &precision),
            ".876f"
        );

        // The AbstractFloat slot is independent of f32.
        let precision = FloatPrecision {
            abstract_float: PrecisionMode::SignificantFigures(2),
            f32: PrecisionMode::Full,
            ..Default::default()
        };
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1234.5678), &precision),
            "1200."
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1234.5678), &precision),
            "1234.5677f"
        );
    }

    #[test]
    fn hex_repr_used_when_shorter() {
        // u64::MAX: 20 decimal digits vs `0x` + 16 hex.
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(u64::MAX), &full()),
            "0xfffffffffffffffflu"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::U64(u64::MAX), &full()),
            "0xffffffffffffffff"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(10_000_000_000_000_000_000), &full()),
            "0x8ac7230489e80000lu"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::I64(i64::MAX), &full()),
            "0x7fffffffffffffffli"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(1_000_000_000_000), &full()),
            "0xe8d4a51000"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::U64(255), &full()), "255lu");
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(42), &full()),
            "42"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::I32(-5), &full()), "-5i");
        // u32: hex is never shorter (10 decimal digits at most).
        assert_eq!(
            literal_to_wgsl(naga::Literal::U32(u32::MAX), &full()),
            "4294967295u"
        );
    }

    #[test]
    fn hex_float_used_when_shorter() {
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &full()),
            "0x1p20f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(16777216.0), &full()),
            "0x1p24f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(-1048576.0), &full()),
            "-0x1p20f"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &full()), ".5f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), &full()), "3f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &full()), "1f");
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(2.0_f32.powi(-14)), &full()),
            "0x1p-14f"
        );
        // f32::MAX is the exception: its shorter decimal overshoots the
        // maximum and tint rejects it, so the bit-exact hex is forced.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), &full()),
            "0x1.fffffep127f"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MIN_POSITIVE), &full()),
            "0x1p-126f"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), &full()), "3f");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), &full()),
            "0x1p20"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), &full()), ".5");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50"
        );
        // naga accepts a hex-float `lf` literal.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50lf"
        );
        // naga rejects hex `h`, so a power of two stays decimal.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(1024.0)), &full()),
            "1024h"
        );
        assert_eq!(
            literal_to_wgsl_bare(
                naga::Literal::F16(f16::from_f32(2.0_f32.powi(-14))),
                &full()
            ),
            "0x1p-14"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1125899906842624.0), &full()),
            "0x1p50"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1125899906842624.0), &full()),
            "0x1p50"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &full()), "-0f");
    }

    #[test]
    fn whole_number_float_literals_are_context_aware() {
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), &full()), "1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), &full()), "0");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-1.0), &full()),
            "-1"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(3.0), &full()), "3");
        // `-0` would re-parse as the integer 0 and drop the sign bit.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-0.0), &full()),
            "-0."
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(1.0)), &full()),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(0.0)), &full()),
            "0"
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), &full()),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(0.0), &full()),
            "0"
        );

        // A standalone AbstractFloat keeps its dot so it is not AbstractInt.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1.0), &full()),
            "1."
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), &full()), "1");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), &full()),
            "1"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I32(42), &full()), "42");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U32(7), &full()), "7");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::U64(100), &full()),
            "100"
        );
        let u64_key = literal_extract_key(naga::Literal::U64(u64::MAX), &full());
        assert_eq!(u64_key.expr_text, "0xffffffffffffffff");
        assert_eq!(u64_key.decl_text, "0xfffffffffffffffflu");

        assert_eq!(ensure_bare_float("1".into()), "1.");
        assert_eq!(ensure_bare_float("-1".into()), "-1.");
        assert_eq!(ensure_bare_float("0".into()), "0.");
        assert_eq!(ensure_bare_float(".5".into()), ".5");
        assert_eq!(ensure_bare_float("1.5".into()), "1.5");
        assert_eq!(ensure_bare_float("0x1p20".into()), "0x1p20");
        assert_eq!(ensure_bare_float("1e5".into()), "1e5");

        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &full()), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-1.0), &full()), "-1f");

        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), &full()), ".5");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1.5), &full()),
            "1.5"
        );

        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), &full()),
            "0x1p20"
        );
    }

    #[test]
    fn significant_figures_never_overflows_finite_input_to_infinity() {
        // A finite literal near the type maximum can round UP across its
        // leading decade (f32::MAX at 4 figures is 3.403e38); the overflow
        // must fall back to the original, never `inff` / `inflf` / `inf`.
        for s in 1..=8u8 {
            let typed = literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(s));
            let bare = literal_to_wgsl_bare(naga::Literal::F32(f32::MAX), &sf(s));
            assert!(!typed.contains("inf"), "F32::MAX sf={s} typed -> {typed:?}");
            assert!(!bare.contains("inf"), "F32::MAX sf={s} bare -> {bare:?}");
        }
        // f64::MAX overflows on the scale-back division, not the multiply.
        for s in 1..=8u8 {
            let typed = literal_to_wgsl(naga::Literal::F64(f64::MAX), &sf(s));
            let bare = literal_to_wgsl_bare(naga::Literal::F64(f64::MAX), &sf(s));
            let abstr = literal_to_wgsl(naga::Literal::AbstractFloat(f64::MAX), &sf(s));
            assert!(!typed.contains("inf"), "F64::MAX sf={s} typed -> {typed:?}");
            assert!(!bare.contains("inf"), "F64::MAX sf={s} bare -> {bare:?}");
            assert!(
                !abstr.contains("inf"),
                "AbstractFloat::MAX sf={s} -> {abstr:?}"
            );
        }
        // The fallback keeps the exact original: `f32::MAX`'s hex form.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(4)),
            "0x1.fffffep127f"
        );
        // Negative extremes keep their sign through the fallback.
        for s in 1..=8u8 {
            let f32n = literal_to_wgsl(naga::Literal::F32(f32::MIN), &sf(s));
            let f64n = literal_to_wgsl(naga::Literal::F64(f64::MIN), &sf(s));
            assert!(
                !f32n.contains("inf") && f32n.starts_with('-'),
                "F32::MIN sf={s} -> {f32n:?}"
            );
            assert!(
                !f64n.contains("inf") && f64n.starts_with('-'),
                "F64::MIN sf={s} -> {f64n:?}"
            );
        }
    }

    #[test]
    fn significant_figures_keeps_f16_within_representable_range() {
        // f16::MAX (65504) rounds to 66000 / 70000, finite as f32 but past
        // f16::MAX, and `66000h` makes naga reject the whole output; the
        // original is kept.
        let max16 = f16::from_f32(65504.0);
        assert_eq!(literal_to_wgsl(naga::Literal::F16(max16), &sf(1)), "65504h");
        assert_eq!(literal_to_wgsl(naga::Literal::F16(max16), &sf(2)), "65504h");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(max16), &sf(1)),
            "65504"
        );
        // In-range rounding is unaffected.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(61234.0)), &sf(2)),
            "61000h"
        );
    }

    #[test]
    fn significant_figures_emits_clean_powers_of_ten() {
        // A power of ten must not pick up scale-back noise (`100000` at one
        // figure as `99999.99999999999lf`).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(100000.0), &sf(1)),
            "1e5lf"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(100000.0), &sf(1)),
            "1e5"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1000000.0), &sf(2)),
            "1e6"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(100000.0), &sf(1)),
            "1e5f"
        );
    }

    #[test]
    fn significant_figures_stays_clean_at_large_magnitudes() {
        // Large magnitudes stay clean too: `2.5e25` at one figure is `3e25`,
        // not `3.0000000000000005e25`.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.5e25), &sf(1)),
            "3e25"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(2.5e25), &sf(1)),
            "3e25"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.5e25), &sf(1)),
            "3e25lf"
        );
        // The rounded token must never be longer than the unrounded input.
        let rounded = literal_to_wgsl_bare(naga::Literal::F64(1.223e21), &sf(4));
        assert!(
            !rounded.contains("000000000"),
            "expected a clean token, got {rounded:?}"
        );

        // Tiny magnitudes (`scale_exp > 22`): `9e-25` must not become
        // `8.999...e-25`.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(9e-25), &sf(1)),
            "9e-25"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(9e-25), &sf(1)),
            "9e-25lf"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(9e-25), &sf(1)),
            "9e-25"
        );
    }

    #[test]
    fn bare_negative_zero_keeps_float_marker() {
        // A bare `-0` re-parses as the integer 0 and drops the sign bit.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-0.0), &full()),
            "-0."
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(-0.0)), &full()),
            "-0."
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(-0.0), &full()),
            "-0."
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-0.0), &full()),
            "-0."
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), &full()), "0");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-2.0), &full()),
            "-2"
        );
        // A small negative that ROUNDS to -0.0 also keeps the marker.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-0.0001), &dp(2)),
            "-0."
        );
    }

    #[test]
    fn bare_f64_collapses_whole_numbers() {
        // The constructor pins the type, so F64 collapses like its siblings.
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(2.0), &full()), "2");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.0), &full()), "0");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-1.0), &full()),
            "-1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-0.0), &full()),
            "-0."
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.5), &full()), ".5");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50"
        );
    }

    #[test]
    fn significant_figures_does_not_zero_f64_subnormals() {
        // A nonzero f64 subnormal scaled by the clamped power rounds its
        // mantissa to 0 and must fall back to the original (f32/f16
        // subnormals widen to f64 normals and never hit this).
        let s = literal_to_wgsl_bare(naga::Literal::F64(1e-310), &sf(1));
        assert!(!s.starts_with('0'), "f64 subnormal zeroed under sf: {s:?}");
        let s = literal_to_wgsl(naga::Literal::AbstractFloat(1e-310), &sf(1));
        assert!(
            !s.starts_with('0'),
            "abstract subnormal zeroed under sf: {s:?}"
        );
    }

    #[test]
    fn typed_f64_and_f16_use_valid_short_suffix_forms() {
        // naga accepts hex-float and scientific `lf` and scientific `h`, but
        // not hex `h`.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50lf"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1e15), &full()), "1e15lf");
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(10000.0)), &full()),
            "1e4h"
        );
        // Whole numbers keep the float type.
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1.0), &full()), "1.lf");
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(1.0)), &full()),
            "1h"
        );
    }
}
