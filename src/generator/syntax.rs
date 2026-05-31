//! Grammar-aware helpers shared across the generator.
//!
//! Holds four clusters of functionality that must stay in one place
//! because every emit site consults them:
//!
//! * Literal formatting (shortest decimal / hex / scientific form
//!   that still round-trips, plus per-type [`PrecisionMode`] rounding).
//! * Type-to-string rendering for WGSL type constructors, including
//!   alias lookup and `@align`/`@size` attribute printing.
//! * Operator precedence and parenthesisation decisions.
//! * Identifier classification (keywords, builtins, extractable
//!   constants) shared between [`super::literal_extract`] and the
//!   statement / expression emitters.

use crate::config::{FloatPrecision, PrecisionMode};
use crate::error::Error;
use naga::proc::TypeResolution;
use std::collections::HashMap;

// MARK: Literal formatting

/// Key used to deduplicate extracted literals: the raw expression
/// text plus the declaration text.  Identical expressions that render
/// the same way share a single extracted `const`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct LiteralExtractKey {
    pub(super) expr_text: String,
    pub(super) decl_text: String,
}

/// Strip redundant zeros from a float literal token while preserving
/// every syntactic cue the parser needs (suffix, sign, exponent,
/// decimal point).  Shortens `-0.50f` to `-.5f` and similar without
/// changing the parsed value.
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

/// Round an `f64` to `sig_figs` significant figures, half-away-from-zero.
/// `sig_figs` must be `>= 1` (callers map `0 -> 1`) and `v` must be
/// finite and non-zero (callers short-circuit both).
///
/// Scales the value so the requested figures sit just left of the
/// decimal point, rounds, then scales back.  The scale-back is exact
/// only while `10^scale_exp` is - i.e. for `scale_exp` in `0..=22`
/// (powers of ten above `10^22` are not representable in binary, and
/// negative powers never are).  Outside that window the down-scale can
/// be undone two ways - multiply by the positive power `10^-scale_exp`,
/// or divide by `scale` - and each leaves binary noise on a *different*
/// magnitude range (dividing by a sub-unity power bloats clean small
/// powers, e.g. `1e5 -> 99999.99999999999`; multiplying by a big power
/// bloats large magnitudes, e.g. `2.5e25 -> 3.0000000000000005e25`; the
/// divide also bloats tiny magnitudes, e.g. `9e-25 -> 8.999...e-25`).
/// Neither direction is universally exact there, so the candidate whose
/// shortest rendering is shorter - i.e. closer to the intended round
/// value - is chosen.
///
/// The scale-back can overflow to infinity for magnitudes near
/// `f64::MAX`; callers must treat a non-finite result as "leave the
/// value unrounded" and fall back to the original.
fn round_sig_figs_f64(v: f64, sig_figs: i32) -> f64 {
    let e = v.abs().log10().floor() as i32;
    // Clamp to f64's representable power-of-ten range (not the tighter
    // f32 range, even when the caller is an f32): the math runs in f64,
    // and clamping at f32's exponent limit would strip significant
    // digits from f32 subnormals and values near `f32::MIN_POSITIVE`.
    let scale_exp = ((sig_figs - 1) - e).clamp(-f64::MAX_10_EXP, f64::MAX_10_EXP);
    let scale = 10f64.powi(scale_exp);
    let mantissa = (v * scale).round();
    if mantissa == 0.0 {
        // The +/-308 scale clamp can leave the up-scaled mantissa below 0.5
        // for a genuine nonzero f64 subnormal (e.g. `1e-310`), rounding it
        // to zero.  A lossy round must never turn a nonzero value into
        // exactly zero, so fall back to the original (callers already
        // short-circuit a true zero before reaching here).
        return v;
    }
    if (0..=22).contains(&scale_exp) {
        // `10^scale_exp` is exact and `>= 1`, so dividing back is clean.
        return mantissa / scale;
    }
    // Inexact-power range (scale_exp < 0 or > 22): pick whichever
    // scale-back renders shorter.  Prefer a finite candidate; if both
    // overflow, the caller's `is_finite` guard falls back to the original.
    let mul = mantissa * 10f64.powi(-scale_exp);
    let div = mantissa / scale;
    match (mul.is_finite(), div.is_finite()) {
        (true, false) => mul,
        (false, true) => div,
        _ if format!("{mul:e}").len() <= format!("{div:e}").len() => mul,
        _ => div,
    }
}

/// Round an `f32` according to a [`PrecisionMode`].  `Full` and
/// non-finite inputs pass through unchanged.  Rounding runs in `f64` so
/// the intermediate never overflows near `f32::MAX`; for
/// `SignificantFigures` the final narrowing cast still can, so a finite
/// input that rounds out of `f32` range falls back to the original -
/// a lossy round must never manufacture an `inf`.
///
/// Call this **once** per literal arm and pass the rounded value to
/// every candidate path (decimal, hex, scientific) so the alternatives
/// fed to [`pick_shortest`] all describe the same numeric value -
/// otherwise an alternative form might silently emit the precise
/// original of a value the user asked to truncate.
fn round_f32(v: f32, mode: PrecisionMode) -> f32 {
    if !v.is_finite() {
        return v;
    }
    match mode {
        PrecisionMode::Full => v,
        PrecisionMode::DecimalPlaces(p) => {
            // Cap at `f32::MAX_10_EXP` - past that there is no f32
            // value with a fractional part of that precision left to
            // round; the multiplication would only waste cycles.  The
            // product cannot overflow f64 (`f32::MAX * 1e38 ~= 3.4e76`),
            // so no finite-result guard is needed here.
            let exp = (p as i32).min(f32::MAX_10_EXP);
            let scale = 10f64.powi(exp);
            (((v as f64) * scale).round() / scale) as f32
        }
        PrecisionMode::SignificantFigures(s) => {
            if v == 0.0 {
                return v;
            }
            // `0` sig figs is treated as `1` - zero would always round
            // to zero, which is rarely what the user wants.
            let rounded = round_sig_figs_f64(v as f64, s.max(1) as i32) as f32;
            // The f64->f32 narrowing saturates to infinity when a value
            // just below `f32::MAX` rounds up across its leading decade;
            // keep the original rather than emit the invalid `inff`.
            if rounded.is_finite() { rounded } else { v }
        }
    }
}

/// `f64` sibling of [`round_f32`].  For `DecimalPlaces` the exponent is
/// capped at `f64::MAX_10_EXP` (a no-op for the `u8` count, which never
/// reaches it) and the `|v| >= f64::MAX / scale` short-circuit keeps the
/// `v * scale` product from overflowing for huge magnitudes.
///
/// `SignificantFigures` shares [`round_sig_figs_f64`] with the f32 path;
/// its scale-back can overflow to infinity near `f64::MAX`, so a
/// non-finite result falls back to the original value - a lossy round
/// must never emit an `inflf` / `inf` token.
fn round_f64(v: f64, mode: PrecisionMode) -> f64 {
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

/// `f16` sibling of [`round_f32`].  The IR widens `f16` to `f32` for
/// emission, so `v` arrives already widened and rounding runs in `f32`.
/// A lossy round can push the magnitude past `f16::MAX` (e.g.
/// `SignificantFigures(1)` of `65504` rounds to `70000`) - finite as
/// `f32`, yet out of range for `f16`.  Emitting `70000h` would make naga
/// reject the whole output, so the rounded value is kept only while it
/// stays within `f16`'s finite range; otherwise the original (already
/// in-range) value is returned.
fn round_f16(v: f32, mode: PrecisionMode) -> f32 {
    // `f16::MAX`, the largest finite value the type holds.  Inlined as a
    // literal so the emitter needs no direct `half` dependency.
    const F16_MAX: f32 = 65504.0;
    let rounded = round_f32(v, mode);
    if rounded.abs() <= F16_MAX { rounded } else { v }
}

/// Guarantee that a bare (unsuffixed) float literal cannot be
/// mistaken for an integer.  Appends a trailing `.` when `s` contains
/// no decimal point, exponent, or hex marker (`1` becomes `1.`).  Non
/// numeric tokens such as `inf` or `NaN` pass through unchanged.
pub(super) fn ensure_bare_float(s: String) -> String {
    if s.contains('.') || s.contains('e') || s.contains('E') || s.contains('x') || s.contains('X') {
        return s;
    }
    // Leave non-numeric representations (inf, NaN, and so on) alone.
    if s.bytes().any(|b| b.is_ascii_alphabetic()) {
        return s;
    }
    format!("{s}.")
}

/// `f64` formatter that defers to `Debug` rendering.  Used by the F64
/// literal arms so whole-number values keep a trailing `.0` that survives
/// [`compact_float_literal_token`] as `1.lf` - without it the `lf` suffix
/// would attach to a bare integer (`1lf`), which the naga parser does
/// not accept.  Callers are responsible for pre-rounding `v` via
/// [`round_f64`] when a non-`Full` [`PrecisionMode`] is in play.
fn fmt_f64_debug(v: f64) -> String {
    format!("{v:?}")
}

/// Return the shortest text form (decimal or lower-case hex) for an
/// unsigned integer, appending `suffix` (for example `u`, `lu`, or
/// empty) to both candidates before comparing lengths.
fn shortest_uint_repr(v: u64, suffix: &str) -> String {
    let dec = format!("{v}{suffix}");
    let hex = format!("0x{v:x}{suffix}");
    if hex.len() < dec.len() { hex } else { dec }
}

/// Signed-integer sibling of [`shortest_uint_repr`].  Non-negative
/// values delegate directly; negative values compare
/// `-{abs}{suffix}` against `-0x{abs:x}{suffix}`.
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

/// Render a normal (non-zero, non-subnormal, finite) `f32` as
/// `{sign}0x1[.{hex}]p{exp}{suffix}`.  Returns `None` for zero,
/// subnormal, infinity, and NaN, which are handled by the decimal
/// path.
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

/// Hex float representation for a normal f64 value.
/// Returns `None` for zero, subnormal, infinity, and NaN.
/// `f64` sibling of [`hex_float_f32`], using IEEE-754 double layout
/// (11 exponent bits, 52 mantissa bits) with the same normal-only
/// precondition.
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

/// Scientific-notation candidate for a finite `f32` value, appending
/// `suffix` (for example `f` or empty).  Returns `None` only for zero,
/// infinity, and NaN - for zero the decimal `"0"` form always wins and
/// `"0e0"` is just noise.  Unlike [`hex_float_f32`], subnormals ARE
/// emitted here: subnormal scientific notation is valid WGSL and
/// round-trips, whereas the leading-`1` hex form cannot represent them.
///
/// Rust's `{:e}` formatter picks the shortest mantissa that round-trips
/// and emits the exponent without a `+` sign, matching WGSL grammar.
/// Useful when neither decimal (long digit run for very large / small
/// magnitudes) nor hex (mantissa bits show up at non-power-of-2 values)
/// produces a short token: `1e20f` is 5 chars vs `100000000000000000000f`
/// (22) or `0x1.5af1d8p66f` (14).
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

/// Choose the shortest of `decimal` and any opportunistic
/// `alternatives` (typically hex and scientific forms returned by
/// [`hex_float_f32`] / [`scientific_float_f32`] and their f64
/// siblings).  `decimal` is always valid; each alternative is adopted
/// only when strictly shorter than the current best so ties prefer
/// the more familiar decimal token.
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

/// Decimal candidate for a *bare* (unsuffixed) float literal, given the
/// `Display`-formatted token and whether the value is negative zero.
///
/// Whole-number bare floats intentionally collapse to bare integer
/// tokens (`1.0 -> 1`) - safe because the enclosing constructor pins the
/// type.  Negative zero is the one exception: `Display` renders it as
/// `-0`, which re-parses as the *integer* `0`, silently dropping the sign
/// bit (a real value change, observable through sign-sensitive ops such
/// as `1.0 / x -> -inf` vs `+inf`).  Keep a trailing dot (`-0.`) so the
/// negative zero survives the round-trip while every other whole number
/// still collapses to its short bare-int form.
fn bare_float_decimal(token: String, is_negative_zero: bool) -> String {
    let compact = compact_float_literal_token(token);
    if is_negative_zero {
        ensure_bare_float(compact)
    } else {
        compact
    }
}

/// Emit a literal with a concrete type suffix (e.g. `1.5f`, `42i`, `3u`).
///
/// Safe to use wherever the literal must carry its own type: standalone
/// expressions, `let` bindings, arithmetic operands.  Float literals are
/// rounded per `precision`'s per-type [`PrecisionMode`] (`Full` preserves
/// the original value).  Use [`literal_to_wgsl_bare`] when an enclosing
/// constructor pins the type and the suffix would just bloat the output.
pub(super) fn literal_to_wgsl(literal: naga::Literal, precision: &FloatPrecision) -> String {
    match literal {
        naga::Literal::F16(v) => {
            // F16 typed: decimal or scientific.  Naga accepts a scientific
            // `h` literal (`1e4h`) but rejects a hex-float `h` literal
            // (`0x1p10h`), so - unlike the bare path inside a `vec3h(...)`
            // constructor - only the scientific alternative is offered.
            let v = round_f16(f32::from(v), precision.f16);
            let dec = compact_float_literal_token(format!("{v}h"));
            pick_shortest(dec, [scientific_float_f32(v, "h")])
        }
        naga::Literal::F32(v) => {
            let v = round_f32(v, precision.f32);
            let dec = compact_float_literal_token(format!("{v}f"));
            pick_shortest(dec, [hex_float_f32(v, "f"), scientific_float_f32(v, "f")])
        }
        naga::Literal::F64(v) => {
            // F64 typed.  The decimal candidate uses Debug rendering so a
            // whole number keeps its trailing `.0` (`1.lf` re-parses as a
            // float, not the rejected bare-int `1lf`).  Naga also accepts
            // hex-float and scientific `lf` literals, so both are offered
            // as shorter alternatives (e.g. `0x1p50lf`, `1e15lf`); neither
            // can collapse to a bare int, so the float type is preserved.
            let v = round_f64(v, precision.f64);
            let dec = compact_float_literal_token(format!("{}lf", fmt_f64_debug(v)));
            pick_shortest(dec, [hex_float_f64(v, "lf"), scientific_float_f64(v, "lf")])
        }
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

/// Emit a literal without any type suffix (e.g. `1.5` instead of `1.5f`).
///
/// **Invariant - callers must pin the type via enclosing context.**
/// Whole-number concrete floats (`F32(1.0)`, `F64(2.0)`, `F16(3.0)`)
/// collapse to bare integer tokens (`1`, `2`, `3`) because WGSL's
/// `.0` stripping picks the shortest decimal, and the resulting token
/// then parses as `AbstractInt` rather than the original float type.
/// (The same holds for non-suffixed integer tokens: `I32(42) -> 42`
/// parses as `AbstractInt`.)
///
/// Approved call sites - ordered by how strongly the enclosing context
/// pins the type:
///
/// 1. Inside a concrete type constructor `T(...)`.  The constructor's
///    signature determines every argument's type.  Covers `Compose` /
///    `Splat` in [`super::expr_emit::Generator::emit_constructor_arg`]
///    and the global-expression Compose/Splat arms in
///    [`super::module_emit`].
/// 2. As the RHS of an extracted `const NAME = ...;` declaration.  The
///    constant takes the literal's (possibly abstract) type, and every
///    use of `NAME` re-binds via normal abstract coercion.  This is
///    [`literal_extract_key`]'s decl path.
///
/// All other sites must use [`literal_to_wgsl`] (typed form).  In
/// particular: binary operands where either side is itself a literal,
/// overload-resolution arguments (e.g. `atan2(1.0, x)`), and standalone
/// `let` / `var` initializers must NOT receive a bare-form literal -
/// an abstract-coercion surprise could flip overload resolution.  See
/// `lib.rs::e2e_concrete_float_literals_round_trip_after_minification`
/// for the round-trip regression test.
///
/// Float literals are rounded per `precision`'s per-type
/// [`PrecisionMode`] (`Full` preserves the original value).
pub(super) fn literal_to_wgsl_bare(literal: naga::Literal, precision: &FloatPrecision) -> String {
    match literal {
        naga::Literal::F16(v) => {
            let v = round_f16(f32::from(v), precision.f16);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f32(v, ""), scientific_float_f32(v, "")])
        }
        naga::Literal::F32(v) => {
            let v = round_f32(v, precision.f32);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f32(v, ""), scientific_float_f32(v, "")])
        }
        naga::Literal::F64(v) => {
            // Mirror the F16/F32 siblings: `Display` collapses whole
            // numbers to bare ints (`2.0 -> 2`), and `pick_shortest`
            // recovers the short hex/scientific forms for large/small
            // magnitudes that `Display` would otherwise expand in full.
            let v = round_f64(v, precision.f64);
            let dec = bare_float_decimal(format!("{v}"), v == 0.0 && v.is_sign_negative());
            pick_shortest(dec, [hex_float_f64(v, ""), scientific_float_f64(v, "")])
        }
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

// MARK: Type rendering

/// Build the `(expr, decl)` [`LiteralExtractKey`] used by
/// [`super::literal_extract`] to canonicalise repeated literals.
///
/// `expr_text` is the shortest valid form at ordinary use sites.
/// `decl_text` is the valid form for `const NAME = ...;`.  For
/// almost all literals the two strings are equal.  The only exception
/// is U64 values that exceed the `AbstractInt` range (>= 2^63): those
/// need the explicit `lu` suffix to stay well-typed in a const
/// declaration.
///
/// `expr_text` uses the bare form per [`literal_to_wgsl_bare`]'s
/// call-site invariant #2: extracted literals appear as
/// `const NAME = <expr_text>;` and every use of `NAME` re-binds via
/// normal abstract coercion.
pub(super) fn literal_extract_key(
    literal: naga::Literal,
    precision: &FloatPrecision,
) -> LiteralExtractKey {
    let expr_text = literal_to_wgsl_bare(literal, precision);
    // The decl_text appears as the RHS of `const NAME = <decl_text>;`.
    // WGSL's abstract-type concretisation defaults `AbstractInt -> i32`
    // and `AbstractFloat -> f32`; for literals whose original concrete
    // type is one of those defaults (`I32`/`U32`/`F32`/`Bool`/abstract
    // already), the bare form re-binds correctly at every use site.
    // For literals whose abstract-default does NOT match the original
    // type (`F16`/`F64`/`I64`/`U64`), force the typed form so the
    // const carries the original type and abstract-coercion at use
    // sites cannot down-cast (`AbstractFloat -> f32` in an f16 context
    // is illegal; `AbstractInt -> i32` in an i64 context loses range).
    // This mirrors the gate used in `expr_emit::literal_needs_typed_form_outside_constructor`.
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

/// Map `(kind, width)` to its WGSL scalar type name (`f32`, `i32`,
/// `u32`, `bool`, `f16`, `f64`, `i64`, `u64`).  Returns [`Error::Emit`]
/// when the combination is unsupported by WGSL.
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

/// Return the single-character shorthand suffix (`f`, `i`, `u`, `h`) used
/// to build the predeclared vector/matrix alias names (e.g. `vec3f`,
/// `mat2x2f`).  Returns `None` for component types without a shorthand
/// alias (bool, i64, u64, f64).
fn scalar_short_suffix(kind: naga::ScalarKind, width: u8) -> Option<&'static str> {
    match (kind, width) {
        (naga::ScalarKind::Float, 2) => Some("h"),
        (naga::ScalarKind::Float, 4) => Some("f"),
        (naga::ScalarKind::Sint, 4) => Some("i"),
        (naga::ScalarKind::Uint, 4) => Some("u"),
        _ => None,
    }
}

/// Return the zero literal for the given scalar type (`0`, `0u`,
/// `0.`, `0f`, `false`, and so on) so callers can splat concrete-typed
/// zeros without re-deriving the suffix logic.
///
/// `AbstractInt`/`AbstractFloat` resolve to `0` / `0.0` respectively;
/// in valid naga IR they should be concretised before reaching the
/// emitter, but the explicit arms here keep round-trip behaviour
/// well-defined (and prevent a future variant from silently falling
/// through to a bare `0` that re-parses as `AbstractInt`).
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
        // Unknown (kind, width) combinations should not occur in valid
        // naga IR (the validator rejects non-canonical widths).  Fall
        // back to bare `0` rather than panicking; any downstream parse
        // failure surfaces through the round-trip validator.
        _ => "0",
    }
}

/// Convert a [`naga::VectorSize`] to its numeric component count.
pub(super) fn vector_size_num(size: naga::VectorSize) -> u8 {
    size as u8
}

/// Render a [`TypeResolution`] to its WGSL type-name string.
/// Resolves handles through the alias map so extracted aliases render
/// by their short name.
pub(super) fn type_resolution_name(
    resolution: &TypeResolution,
    module: &naga::Module,
    struct_names: &HashMap<naga::Handle<naga::Type>, String>,
    override_names: &[String],
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
            )
        }
        TypeResolution::Value(inner) => {
            // Multiple Type entries may share the same TypeInner when the
            // source uses named aliases alongside bare types (UniqueArena
            // deduplicates by the full Type including its name field).
            // Scan for any handle whose inner matches AND has an alias.
            if let Some(name) = module.types.iter().find_map(|(handle, ty)| {
                (&ty.inner == inner)
                    .then(|| struct_names.get(&handle).cloned())
                    .flatten()
            }) {
                return Ok(name);
            }
            type_inner_name(inner, module, struct_names, override_names)
        }
    }
}

/// Render a [`naga::TypeInner`] to its WGSL type-name string.
/// Handles scalars, vectors (choosing alias vs explicit form),
/// matrices, arrays, pointers, atomics, images, samplers, and opaque
/// resource types.  Aliases are substituted when available.
pub(super) fn type_inner_name(
    inner: &naga::TypeInner,
    module: &naga::Module,
    struct_names: &HashMap<naga::Handle<naga::Type>, String>,
    override_names: &[String],
) -> Result<String, Error> {
    Ok(match inner {
        naga::TypeInner::Scalar(s) => scalar_name(s.kind, s.width)?.to_string(),
        naga::TypeInner::Vector { size, scalar } => {
            match scalar_short_suffix(scalar.kind, scalar.width) {
                Some(suffix) => format!("vec{}{}", vector_size_num(*size), suffix),
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
        } => match scalar_short_suffix(scalar.kind, scalar.width) {
            Some(suffix) => format!(
                "mat{}x{}{}",
                vector_size_num(*columns),
                vector_size_num(*rows),
                suffix
            ),
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
                "ptr<{},{}>",
                address_space(*space),
                type_ref_from_handle(*base, module, struct_names, override_names)?
            )
        }
        naga::TypeInner::ValuePointer {
            size,
            scalar,
            space,
        } => {
            let value_ty = match size {
                Some(v) => match scalar_short_suffix(scalar.kind, scalar.width) {
                    Some(suffix) => format!("vec{}{}", vector_size_num(*v), suffix),
                    None => format!(
                        "vec{}<{}>",
                        vector_size_num(*v),
                        scalar_name(scalar.kind, scalar.width)?
                    ),
                },
                None => scalar_name(scalar.kind, scalar.width)?.to_string(),
            };
            format!("ptr<{},{}>", address_space(*space), value_ty)
        }
        naga::TypeInner::Array { base, size, .. } => {
            let base_ty = type_ref_from_handle(*base, module, struct_names, override_names)?;
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
            let base_ty = type_ref_from_handle(*base, module, struct_names, override_names)?;
            match size {
                naga::ArraySize::Constant(n) => format!("binding_array<{},{}>", base_ty, n.get()),
                naga::ArraySize::Dynamic => format!("binding_array<{}>", base_ty),
                naga::ArraySize::Pending(h) => {
                    format!("binding_array<{},{}>", base_ty, override_names[h.index()])
                }
            }
        }
        naga::TypeInner::AccelerationStructure { .. } => "acceleration_structure".to_string(),
        _ => {
            return Err(Error::Emit(format!("unsupported type: {:?}", inner,)));
        }
    })
}

/// Shortcut for rendering a type by handle, dereferencing through
/// the type arena and honouring alias substitution.
pub(super) fn type_ref_from_handle(
    ty: naga::Handle<naga::Type>,
    module: &naga::Module,
    struct_names: &HashMap<naga::Handle<naga::Type>, String>,
    override_names: &[String],
) -> Result<String, Error> {
    if let Some(name) = struct_names.get(&ty) {
        return Ok(name.clone());
    }
    type_inner_name(
        &module.types[ty].inner,
        module,
        struct_names,
        override_names,
    )
}

// MARK: Attribute and qualifier rendering

/// Map a [`naga::AddressSpace`] to its WGSL keyword.  `Function` returns
/// `"function"` (not the empty string): the keyword is mandatory wherever a
/// `ptr<function, T>` type is rendered.  (A `var<function>` *declaration*
/// leaves the space implicit, but that path emits no qualifier via this
/// helper.)
pub(super) fn address_space(space: naga::AddressSpace) -> &'static str {
    match space {
        naga::AddressSpace::Function => "function",
        naga::AddressSpace::Private => "private",
        naga::AddressSpace::WorkGroup => "workgroup",
        naga::AddressSpace::Uniform => "uniform",
        naga::AddressSpace::Storage { .. } => "storage",
        naga::AddressSpace::RayPayload => "ray_payload",
        naga::AddressSpace::IncomingRayPayload => "incoming_ray_payload",
        // naga's WGSL front-end spells push-constant / immediate-data space
        // `immediate` (it rejects `push_constant`); rendering it as `private`
        // would silently swap the host-supplied data source for zero-init
        // per-invocation memory - a miscompile.
        naga::AddressSpace::Immediate => "immediate",
        // No surface-WGSL `var<...>` form: reaching this arm for a global var
        // would silently emit `var<private>`.  Safe ONLY because `Handle` is
        // intercepted upstream (emits a bare `var`) and a `TaskPayload` global
        // fails naga validation before emission; the `private` text is an
        // arbitrary placeholder for that unreachable case, not a semantic pick.
        naga::AddressSpace::Handle | naga::AddressSpace::TaskPayload => "private",
    }
}

/// Map [`naga::StorageAccess`] flags to the WGSL access-mode keyword
/// (`read`, `read_write`, `write`, `atomic`).
///
/// `atomic` is a WGSL access mode for atomic storage textures
/// (`texture_storage_2d<r32uint, atomic>`).  Naga's IR sets the
/// `ATOMIC` flag on those texture bindings and the frontend rejects
/// any other access mode for a texture that participates in atomic
/// ops - so the flag must take precedence here.  For non-texture
/// storage (`var<storage, ...>`) `ATOMIC` is never set, so the
/// branch is exclusive to texture bindings in practice.
pub(super) fn storage_access(access: naga::StorageAccess) -> &'static str {
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

/// Render `@location`, `@builtin`, `@interpolate`, and
/// `@invariant` attributes attached to a struct member or function
/// parameter, omitting defaults to keep the output compact.
pub(super) fn binding_attrs(binding: &naga::Binding) -> Result<String, Error> {
    Ok(match binding {
        naga::Binding::BuiltIn(bi) => {
            if let naga::BuiltIn::Position { invariant: true } = bi {
                format!("@invariant @builtin({}) ", builtin_name(*bi)?)
            } else {
                format!("@builtin({}) ", builtin_name(*bi)?)
            }
        }
        naga::Binding::Location {
            location,
            interpolation,
            sampling,
            blend_src,
            per_primitive,
        } => {
            let mut out = format!("@location({location}) ");
            if let Some(bs) = blend_src {
                out.push_str(&format!("@blend_src({bs}) "));
            }
            if *per_primitive {
                out.push_str("@per_primitive ");
            }
            // Elide @interpolate(perspective,center) - it is the WGSL
            // default for float location bindings.  naga always stores
            // the default explicitly, so we suppress it to save bytes.
            let non_default_interp =
                interpolation.is_some_and(|i| i != naga::Interpolation::Perspective);
            let non_default_sampling = sampling.is_some_and(|s| s != naga::Sampling::Center);
            if non_default_interp || non_default_sampling {
                out.push_str("@interpolate(");
                out.push_str(interpolation_name(
                    interpolation.unwrap_or(naga::Interpolation::Perspective),
                ));
                if let Some(s) = sampling {
                    out.push(',');
                    out.push_str(sampling_name(*s));
                }
                out.push_str(") ");
            }
            out
        }
    })
}

/// Map [`naga::Interpolation`] to its WGSL keyword.
pub(super) fn interpolation_name(i: naga::Interpolation) -> &'static str {
    match i {
        naga::Interpolation::Perspective => "perspective",
        naga::Interpolation::Linear => "linear",
        naga::Interpolation::Flat => "flat",
        naga::Interpolation::PerVertex => "per_vertex",
    }
}

/// Map [`naga::Sampling`] to its WGSL keyword.
pub(super) fn sampling_name(s: naga::Sampling) -> &'static str {
    match s {
        naga::Sampling::Center => "center",
        naga::Sampling::Centroid => "centroid",
        naga::Sampling::Sample => "sample",
        naga::Sampling::First => "first",
        naga::Sampling::Either => "either",
    }
}

/// Map a [`naga::BuiltIn`] to its WGSL `@builtin(...)` keyword.
/// Returns [`Error::Emit`] for builtins WGSL does not expose.
pub(super) fn builtin_name(bi: naga::BuiltIn) -> Result<&'static str, Error> {
    Ok(match bi {
        naga::BuiltIn::PrimitiveIndex => "primitive_index",
        naga::BuiltIn::Position { .. } => "position",
        naga::BuiltIn::ViewIndex => "view_index",
        naga::BuiltIn::BaseInstance => "base_instance",
        naga::BuiltIn::BaseVertex => "base_vertex",
        naga::BuiltIn::ClipDistance => "clip_distances",
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

/// Render a [`naga::ImageClass`] to its WGSL `texture_*` type
/// expression, including element type and storage format where
/// applicable.  Returns [`Error::Emit`] for unsupported combinations.
pub(super) fn image_type(
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

/// Map [`naga::StorageFormat`] to its WGSL texel-format keyword
/// (`rgba8unorm`, `r32float`, and so on).
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

/// Map every [`naga::MathFunction`] variant to its WGSL built-in
/// function name.  Exhaustive by design so a future naga variant
/// fails the build and forces an explicit mapping.
pub(super) fn math_name(fun: naga::MathFunction) -> &'static str {
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

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::{
        FloatPrecision, PrecisionMode, compact_float_literal_token, ensure_bare_float,
        literal_extract_key, literal_to_wgsl, literal_to_wgsl_bare, scalar_zero,
    };
    use half::f16;

    /// All-types-Full precision; the baseline for tests that want the
    /// emitter to preserve every digit of the input value.
    fn full() -> FloatPrecision {
        FloatPrecision::default()
    }

    /// Apply `DecimalPlaces(n)` to every float kind.  Used by the
    /// legacy "single u8" tests that pre-date per-type precision.
    fn dp(n: u8) -> FloatPrecision {
        FloatPrecision::all(PrecisionMode::DecimalPlaces(n))
    }

    /// Apply `SignificantFigures(n)` to every float kind.
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
        // Typed: F32 gets 'f' suffix, AbstractFloat stays bare, F64 keeps 'lf'
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.25), &full()), ".25f");
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(0.5), &full()),
            ".5"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F64(0.5), &full()), ".5lf");
        // Typed: I32 gets 'i' suffix
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &full()), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), &full()), "7u");

        // Bare: all suffixes stripped for use inside Compose
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
        // The i64::MIN uses overflow-safe constructor
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
        // AbstractInt i64::MIN must use overflow-safe subtraction form.
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
        // F32 typed: 0.123456 rounded to 3 decimal places -> 0.123f -> .123f
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), &dp(3)),
            ".123f"
        );
        // F32 typed: 0.876543 rounded to 2 -> 0.88f -> .88f
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.876543), &dp(2)),
            ".88f"
        );
        // F32 bare: same value, no suffix
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(0.876543), &dp(2)),
            ".88"
        );
        // AbstractFloat (f64): precision limiting works
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(7.65432198), &dp(4)),
            "7.6543"
        );
        // Integer literals are unaffected by any float-precision mode.
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &dp(2)), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), &dp(2)), "7u");
        // None preserves full precision (baseline)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), &full()),
            ".123456f"
        );
    }

    #[test]
    fn precision_preserves_whole_number_shortest_form() {
        // Whole-number concrete floats must stay one byte shorter with
        // precision enabled - the suffix pins the type, so no trailing
        // `.0` is needed.  This is the regression that switching from
        // `{:.prec$}` to round-then-shortest fixes.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &dp(6)), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-2.0), &dp(2)), "-2f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &dp(6)), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &dp(6)), "-0f");

        // Bare form (inside `T(...)`) drops the suffix.
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), &dp(6)), "1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(-2.0), &dp(2)), "-2");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), &dp(6)), "0");

        // Values that round *up* to a whole number also collapse: 0.999
        // with precision 2 rounds to 1.0, which formats as "1f"/"1".
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.999), &dp(2)), "1f");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.999), &dp(2)), "1");

        // Half-away-from-zero rounding at the boundary.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &dp(0)), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.49), &dp(0)), "0f");

        // AbstractFloat: bare form mirrors the F32 collapse, while the
        // typed form still keeps a dot via `ensure_bare_float` so it does
        // not reparse as `AbstractInt`.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), &dp(6)),
            "1"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1.0), &dp(6)),
            "1."
        );

        // F64 keeps the trailing `.0` via the Debug path so `1.lf`
        // remains a valid float literal (naga rejects `1lf`).
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1.0), &dp(6)), "1.lf");
    }

    #[test]
    fn precision_hex_path_still_wins_when_shorter() {
        // 2^20 = 1048576: decimal "1048576f" (8) loses to hex "0x1p20f" (7).
        // Rounding to 6 decimal places leaves the value at 2^20, so the
        // hex form (computed from the rounded value) continues to be the
        // shortest candidate.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &dp(6)),
            "0x1p20f"
        );
    }

    #[test]
    fn precision_handles_non_finite_values() {
        // +/-inf / NaN round through unchanged so the emitted token is
        // identical regardless of which precision mode is active.
        // (The output is not valid WGSL for these inputs - naga has no
        // `inf` / `nan` literal - but the formatter must remain
        // deterministic so a future fix can address both code paths in
        // one place.)
        for v in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(
                literal_to_wgsl(naga::Literal::F32(v), &dp(6)),
                literal_to_wgsl(naga::Literal::F32(v), &full()),
            );
        }
    }

    #[test]
    fn precision_hex_form_uses_rounded_value() {
        // Invariant: every candidate passed to `pick_shortest` must
        // encode the same numeric value.  Before the rounding was
        // hoisted to the dispatch arm, `hex_float_f32` received the
        // original `v` while `compact_float_literal_token` received a
        // truncated decimal - the picker could then emit a precise
        // hex of a value the user asked to round.
        //
        // F32(1048575.9) rounds up to 1048576 = 2^20; hex of the
        // rounded value is the short "0x1p20f" form, while hex of the
        // original would carry mantissa bits and be much longer.  Both
        // routes pick a token, but only the rounded one is semantically
        // honest about the truncation the user opted into.
        let s = literal_to_wgsl(naga::Literal::F32(1048575.9), &dp(0));
        assert_eq!(s, "0x1p20f");
        // Same value re-emerges via the bare path.
        let s = literal_to_wgsl_bare(naga::Literal::F32(1048575.9), &dp(0));
        assert_eq!(s, "0x1p20");
    }

    #[test]
    fn scientific_form_chosen_when_shorter() {
        // Pure decimal magnitudes win in scientific notation: `1e6f`
        // (4 chars) beats `1000000f` (8) and `0x1.e848p19f` (12).
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e6), &full()), "1e6f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e10), &full()), "1e10f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1e20), &full()), "1e20f");

        // Negative-exponent scientific wins for tiny magnitudes.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1e-30), &full()),
            "1e-30f"
        );

        // Bare form drops the `f` suffix: `1e10` (4) vs `10000000000` (11).
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1e10), &full()),
            "1e10"
        );

        // AbstractFloat - bare and typed paths both use the sci form
        // when shorter, and `ensure_bare_float` is happy because the
        // `e` already marks it as a float.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1e20), &full()),
            "1e20"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1e100), &full()),
            "1e100"
        );

        // F64 bare: `1e15f` mantissa makes scientific potentially long,
        // but for pure powers of 10 it still wins.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(1e15), &full()),
            "1e15"
        );

        // F16 bare path also considers scientific (no suffix conflict).
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(1e4)), &full()),
            "1e4"
        );

        // Small magnitudes stay decimal - `5e-1f` (5) loses to `.5f` (3).
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &full()), ".5f");

        // Powers of 2 still pick hex when shorter than both decimal
        // and scientific: 2^20 -> "0x1p20f" (7) < "1.048576e6f" (11).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &full()),
            "0x1p20f"
        );

        // Zero and -0 short-circuit out of the sci candidate so we
        // never emit the useless `0e0f` form.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &full()), "-0f");
    }

    #[test]
    fn scientific_form_aligns_with_rounding() {
        // A non-`Full` mode rounds the value first, then the rounded
        // value feeds every candidate (decimal, hex, scientific) - so
        // a value that rounds up to a clean power of 10 picks up the
        // short scientific form even when the original wouldn't have.
        // F32(999999.5) rounds away-from-zero to 1e6: `1e6f` (4) beats
        // decimal `1000000f` (8) and hex `0x1.e848p19f` (12).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(999999.5), &dp(0)),
            "1e6f"
        );
    }

    #[test]
    fn significant_figures_round_independent_of_magnitude() {
        // SignificantFigures is the dual of DecimalPlaces: instead of
        // pinning the count after the dot, it pins the total non-zero
        // digit count regardless of where the value sits on the number
        // line.  `1234567.9` with 3 sig figs -> 1230000 (`1.23e6f`).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1234567.9), &sf(3)),
            "1.23e6f"
        );
        // `0.001234` with 3 sig figs -> 0.00123 -> `.00123f` (decimal
        // wins over `1.23e-3f`).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.001234), &sf(3)),
            ".00123f"
        );
        // 1 sig fig: any value collapses to a single significant digit.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(789.0), &sf(1)), "800f");
        // SignificantFigures(0) treated as (1) - zero sig figs would
        // always round to 0, which is rarely useful.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(123.0), &sf(0)), "100f");
        // Zero / non-finite pass through unchanged.
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &sf(4)), "0f");
        // Integer literals unaffected by float-precision mode.
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), &sf(2)), "42i");
    }

    #[test]
    fn significant_figures_covers_f16_and_f64_kinds() {
        // F16 routes through `round_f32` (f16 widens to f32 in emit)
        // with the f16-mode.  Pick a non-special-constant value so the
        // assertion isn't a stand-in for `std::f32::consts`-flavoured
        // expectations.  0.456 in f16 ~= 0.456; sf=2 rounds to 0.46.
        // f16 sf=2 rounding of 0.456 is deterministic - pin it exactly.
        let s = literal_to_wgsl(naga::Literal::F16(f16::from_f32(0.456)), &sf(2));
        assert_eq!(s, ".46h", "got {s:?}");

        // F64 sig-figs uses `round_f64` directly.  Both the typed and the
        // bare path offer the scientific candidate (naga accepts `...e...lf`),
        // so the rounded `1.23e6` value emits in scientific form either way.
        let s = literal_to_wgsl(naga::Literal::F64(1234567.89_f64), &sf(3));
        assert_eq!(s, "1.23e6lf");
        let s = literal_to_wgsl_bare(naga::Literal::F64(1234567.89_f64), &sf(3));
        assert_eq!(s, "1.23e6");
    }

    #[test]
    fn significant_figures_preserves_precision_at_f32_boundaries() {
        // Regression: values near `f32::MIN_POSITIVE` and below were
        // erroneously losing significant digits because `round_f32`
        // clamped its scale exponent at +/-38.  The arithmetic runs in
        // f64, which can comfortably represent the scale, so the clamp
        // now matches f64's range and the sig-figs target is honoured.
        //
        // 1.18e-38 (near MIN_POSITIVE) with sf=2 should keep 2 digits -
        // not collapse to "1e-38" with the old clamp.
        let s = literal_to_wgsl(naga::Literal::F32(1.18e-38), &sf(2));
        assert!(
            s.starts_with("1.2e-38") || s == "1.2e-38f",
            "expected 2 sig figs of 1.18e-38, got {s:?}"
        );
        // Subnormal: 1e-40 with sf=2 should survive.  We accept any
        // representation that parses back near 1e-40 since exact f32
        // subnormal values are not exact rationals.
        let s = literal_to_wgsl(naga::Literal::F32(1e-40), &sf(2));
        assert!(
            !s.starts_with("0") && (s.contains("e-40") || s.contains("e-41")),
            "expected sf=2 of 1e-40 subnormal to be preserved, got {s:?}"
        );
        // f32::MAX with extreme sig-figs count still works (overflow
        // guard prevents the multiplication itself from going to inf).
        let s = literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(255));
        assert!(s.contains("e38f") || s.contains("0x"), "got {s:?}");
    }

    #[test]
    fn per_type_precision_dispatch() {
        // Different float kinds can carry different precision modes.
        // Here f32 gets aggressive `DecimalPlaces(2)` while f64 stays
        // `Full`; both kinds reach the dispatch arm through `literal_to_wgsl`.
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

        // f16 has its own slot: ask for 1 decimal place on f16 but
        // leave f32 at Full.  The f16 path goes through `round_f32`
        // (the IR widens f16 to f32 for emission) with the f16-mode.
        let precision = FloatPrecision {
            f16: PrecisionMode::DecimalPlaces(1),
            f32: PrecisionMode::Full,
            ..Default::default()
        };
        // f16(0.876) at 1 decimal place -> 0.9 -> ".9h".
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(0.876)), &precision,),
            ".9h",
        );
        // f32(0.876) still emits at full precision.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.876), &precision),
            ".876f"
        );

        // AbstractFloat slot is independent: a sig-figs cap on
        // abstract floats does not affect concrete f32 emission.
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
        // u64::MAX: 20 decimal digits -> 16 hex digits + 0x = 18 chars (saves 2)
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(u64::MAX), &full()),
            "0xfffffffffffffffflu"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::U64(u64::MAX), &full()),
            "0xffffffffffffffff"
        );
        // Large u64: 10^19 = 19 decimal digits
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(10_000_000_000_000_000_000), &full()),
            "0x8ac7230489e80000lu"
        );
        // i64::MAX: 19 decimal digits -> hex saves 1
        assert_eq!(
            literal_to_wgsl(naga::Literal::I64(i64::MAX), &full()),
            "0x7fffffffffffffffli"
        );
        // Large AbstractInt: 13 decimal digits -> hex can save 1
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(1_000_000_000_000), &full()),
            "0xe8d4a51000"
        );
        // Small values stay decimal (hex is longer due to 0x prefix)
        assert_eq!(literal_to_wgsl(naga::Literal::U64(255), &full()), "255lu");
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(42), &full()),
            "42"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::I32(-5), &full()), "-5i");
        // u32 values: hex never shorter (max 10 decimal digits = 10 hex chars)
        assert_eq!(
            literal_to_wgsl(naga::Literal::U32(u32::MAX), &full()),
            "4294967295u"
        );
    }

    #[test]
    fn hex_float_used_when_shorter() {
        // 2^20 = 1048576.0: decimal "1048576f" (8) vs hex "0x1p20f" (7)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), &full()),
            "0x1p20f"
        );
        // 2^24 = 16777216.0: decimal "16777216f" (9) vs hex "0x1p24f" (7)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(16777216.0), &full()),
            "0x1p24f"
        );
        // Negative power of 2: -2^20
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(-1048576.0), &full()),
            "-0x1p20f"
        );
        // Small values stay decimal - hex is longer
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), &full()), ".5f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), &full()), "3f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &full()), "1f");
        // Negative exponent: 2^-14 decimal ".000061035156f"(16) vs hex "0x1p-14f"(8)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(2.0_f32.powi(-14)), &full()),
            "0x1p-14f"
        );
        // f32::MAX: decimal is 40 chars, hex "0x1.fffffep127f" (15),
        // scientific "3.4028235e38f" (13).  Scientific wins because it is
        // one of the candidates `pick_shortest` considers.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), &full()),
            "3.4028235e38f"
        );
        // f32::MIN_POSITIVE: decimal is 49 chars, hex "0x1p-126f" (9)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MIN_POSITIVE), &full()),
            "0x1p-126f"
        );
        // Non-power-of-2 with mantissa bits: 3.0 = 0x1.8p1f (8) vs "3f" (2) -> decimal wins
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), &full()), "3f");
        // Bare 2^20: decimal "1048576" (7) vs hex "0x1p20" (6)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), &full()),
            "0x1p20"
        );
        // Bare small value stays decimal
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), &full()), ".5");
        // F64 bare: 2^50 hex wins
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50"
        );
        // F64 typed: naga accepts a hex-float `lf` literal, so 2^50 wins
        // in hex (`0x1p50lf`, 8 chars) over decimal (`1125899906842624.lf`).
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50lf"
        );
        // F16 typed: hex + 'h' is rejected by naga, so a power of two with
        // no shorter scientific form stays decimal.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(1024.0)), &full()),
            "1024h"
        );
        // F16 bare with hex: 2^-14 as bare is shorter in hex
        assert_eq!(
            literal_to_wgsl_bare(
                naga::Literal::F16(f16::from_f32(2.0_f32.powi(-14))),
                &full()
            ),
            "0x1p-14"
        );
        // AbstractFloat 2^50 typed: hex much shorter (no suffix needed)
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1125899906842624.0), &full()),
            "0x1p50"
        );
        // AbstractFloat bare: same as typed (no suffix in either form)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1125899906842624.0), &full()),
            "0x1p50"
        );
        // Zero falls back to decimal (hex_float returns None for zero)
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), &full()), "-0f");
    }

    #[test]
    fn whole_number_float_literals_are_context_aware() {
        // Bare constructor literals stay as short as possible.
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), &full()), "1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), &full()), "0");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-1.0), &full()),
            "-1"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(3.0), &full()), "3");
        // Negative zero keeps a trailing dot - the bare int `-0` would
        // re-parse as the integer 0 and silently drop the sign bit.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(-0.0), &full()),
            "-0."
        );

        // F16 bare whole numbers - also no dot inside constructors
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(1.0)), &full()),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(0.0)), &full()),
            "0"
        );

        // AbstractFloat bare whole numbers are also kept short inside
        // constructor contexts.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), &full()),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(0.0), &full()),
            "0"
        );

        // Standalone AbstractFloat literals must keep a decimal point so WGSL
        // does not parse them as AbstractInt.
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

        // ensure_bare_float standalone helper
        assert_eq!(ensure_bare_float("1".into()), "1.");
        assert_eq!(ensure_bare_float("-1".into()), "-1.");
        assert_eq!(ensure_bare_float("0".into()), "0.");
        assert_eq!(ensure_bare_float(".5".into()), ".5"); // already has dot
        assert_eq!(ensure_bare_float("1.5".into()), "1.5"); // already has dot
        assert_eq!(ensure_bare_float("0x1p20".into()), "0x1p20"); // hex
        assert_eq!(ensure_bare_float("1e5".into()), "1e5"); // exponent

        // Suffixed forms are NOT affected (suffix provides the type)
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), &full()), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), &full()), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-1.0), &full()), "-1f");

        // Fractional values remain unchanged (already have a dot)
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), &full()), ".5");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1.5), &full()),
            "1.5"
        );

        // Hex representations remain unchanged (already have 'x' marker)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), &full()),
            "0x1p20"
        );
    }

    #[test]
    fn significant_figures_never_overflows_finite_input_to_infinity() {
        // Regression: a finite literal near the type maximum could round
        // UP across its leading decade and overflow to +/-inf, emitting the
        // invalid tokens `inff` / `inflf` / `inf`.  A lossy round must
        // never turn a finite value into a non-finite one; the rounding
        // helpers now fall back to the original value when the scale-back
        // (or the f32 narrowing cast) overflows.
        //
        // f32::MAX to 4 sig figs rounds 3.4028235 -> 3.403, scaled back
        // that exceeds f32::MAX, so the cast would saturate to inf.
        for s in 1..=8u8 {
            let typed = literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(s));
            let bare = literal_to_wgsl_bare(naga::Literal::F32(f32::MAX), &sf(s));
            assert!(!typed.contains("inf"), "F32::MAX sf={s} typed -> {typed:?}");
            assert!(!bare.contains("inf"), "F32::MAX sf={s} bare -> {bare:?}");
        }
        // f64::MAX divides back by a sub-unity scale near 1e-308; the
        // division overflows even though the multiplication never does.
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
        // The fallback keeps the original value verbatim, which still emits
        // a valid (full-precision) token.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), &sf(4)),
            "3.4028235e38f"
        );
        // Negative extremes are symmetric: no inf, and the sign survives
        // the fallback (the overflow guard returns the original signed v).
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
        // Regression: f16 emits through round_f32 (f16 widens to f32), and
        // SignificantFigures of f16::MAX (65504) rounds UP to 66000 (sf=2)
        // or 70000 (sf=1) - finite as f32 but past f16::MAX, so the token
        // `66000h` / `70000h` made naga reject the whole output.  The f16
        // path now falls back to the in-range original when a round leaves
        // f16's range.
        let max16 = f16::from_f32(65504.0);
        assert_eq!(literal_to_wgsl(naga::Literal::F16(max16), &sf(1)), "65504h");
        assert_eq!(literal_to_wgsl(naga::Literal::F16(max16), &sf(2)), "65504h");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(max16), &sf(1)),
            "65504"
        );
        // In-range rounding is unaffected: a value whose sig-fig round
        // stays <= 65504 still rounds normally.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(61234.0)), &sf(2)),
            "61000h"
        );
    }

    #[test]
    fn significant_figures_emits_clean_powers_of_ten() {
        // Regression: scaling a power-of-ten value back by dividing by a
        // sub-unity power (e.g. `/ 1e-5`) reintroduced binary noise, so
        // `100000` to one sig fig came out as `99999.99999999999lf` -
        // longer than the input and a violation of the requested figure
        // count.  Picking the cleaner scale-back keeps it exact, and the
        // scientific candidate then trims it further.
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
        // The f32 path was masked by its narrowing cast but is verified
        // here too for coherence with the f64 sibling.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(100000.0), &sf(1)),
            "1e5f"
        );
    }

    #[test]
    fn significant_figures_stays_clean_at_large_magnitudes() {
        // Regression: the multiply-back that fixed small clean powers
        // (1e5) reintroduced noise at LARGE magnitudes on the f64 /
        // AbstractFloat paths - `2.5e25` to one sig fig emitted the
        // 21-char `3.0000000000000005e25` instead of `3e25`.  Picking the
        // cleaner of the two scale-backs keeps both ends clean.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.5e25), &sf(1)),
            "3e25"
        );
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(2.5e25), &sf(1)),
            "3e25"
        );
        // Typed F64 reaches the same clean value (scientific `lf` form).
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

        // Symmetric case at TINY magnitudes (scale_exp > 22): dividing by
        // an inexact power of ten left noise (`9e-25 -> 8.999...e-25`).  The
        // pick-cleaner path now covers scale_exp outside `0..=22` too.
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
        // A bare `-0` re-parses as the integer 0 and drops the sign bit, a
        // real value change; the bare float arms keep `-0.` so the sign
        // survives.  Other whole numbers still collapse to bare ints.
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
        // F64 bare goes through the same `bare_float_decimal` guard.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-0.0), &full()),
            "-0."
        );
        // Positive zero stays the short bare int; non-zero wholes too.
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
        // The F64 bare arm collapses whole numbers to bare ints like its
        // F16/F32/AbstractFloat siblings (the constructor pins the type),
        // matching the doc contract and saving a byte per literal.
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(2.0), &full()), "2");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.0), &full()), "0");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-1.0), &full()),
            "-1"
        );
        // Negative zero still keeps its sign-preserving `-0.` marker.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(-0.0), &full()),
            "-0."
        );
        // Fractions and large/small magnitudes are unchanged (hex/sci win).
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.5), &full()), ".5");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50"
        );
    }

    #[test]
    fn significant_figures_does_not_zero_f64_subnormals() {
        // Regression: a nonzero f64 subnormal (e.g. 1e-310) up-scaled by
        // the +/-308-clamped power rounds its mantissa to 0; a lossy round
        // must never turn a nonzero value into exactly zero, so it falls
        // back to the original.  (f32/f16 subnormals widen to f64 normals
        // and never hit this; the f64/abstract path needs the guard.)
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
        // Naga accepts hex-float and scientific `lf` literals and
        // scientific `h` literals (but NOT hex `h`).  The typed arms offer
        // those shorter forms, matching the bare arms where the suffix
        // allows it.
        // F64: clean power of two -> hex; clean power of ten -> scientific.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.0_f64.powi(50)), &full()),
            "0x1p50lf"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1e15), &full()), "1e15lf");
        // F16: clean power of ten -> scientific; hex `h` stays excluded.
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(10000.0)), &full()),
            "1e4h"
        );
        // Whole numbers keep the float type (no bare-int reparse): the
        // decimal candidate wins because the alternatives are not shorter.
        assert_eq!(literal_to_wgsl(naga::Literal::F64(1.0), &full()), "1.lf");
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(1.0)), &full()),
            "1h"
        );
    }
}
