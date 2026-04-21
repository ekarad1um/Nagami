//! Grammar-aware helpers shared across the generator.
//!
//! Holds four clusters of functionality that must stay in one place
//! because every emit site consults them:
//!
//! * Literal formatting (shortest decimal / hex / scientific form
//!   that still round-trips, plus `max_precision` gating).
//! * Type-to-string rendering for WGSL type constructors, including
//!   alias lookup and `@align`/`@size` attribute printing.
//! * Operator precedence and parenthesisation decisions.
//! * Identifier classification (keywords, builtins, extractable
//!   constants) shared between [`super::literal_extract`] and the
//!   statement / expression emitters.

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

/// Render an `f32` value as a decimal, truncating to at most
/// `max_precision` fractional digits when supplied.  `None` preserves
/// the full value.
fn fmt_f32(v: f32, max_precision: Option<u8>) -> String {
    match max_precision {
        Some(p) => format!("{v:.prec$}", prec = p as usize),
        None => format!("{v}"),
    }
}

/// Decimal sibling of [`fmt_f32`] for `f64` values.
fn fmt_f64(v: f64, max_precision: Option<u8>) -> String {
    match max_precision {
        Some(p) => format!("{v:.prec$}", prec = p as usize),
        None => format!("{v}"),
    }
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

/// `f64` formatter that defers to `Debug` rendering.  Used for F64
/// literal emission where the extra digits matter for round-trip
/// fidelity.
fn fmt_f64_debug(v: f64, max_precision: Option<u8>) -> String {
    match max_precision {
        Some(p) => format!("{v:.prec$?}", prec = p as usize),
        None => format!("{v:?}"),
    }
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

/// Return whichever representation is shorter.
/// Choose the shorter of `decimal` and `hex`.  `decimal` is always
/// valid; `hex` is the opportunistic alternative from
/// [`hex_float_f32`] or [`hex_float_f64`] and is only adopted when
/// strictly shorter than `decimal`.
fn pick_shorter(decimal: String, hex: Option<String>) -> String {
    if let Some(h) = hex
        && h.len() < decimal.len()
    {
        return h;
    }
    decimal
}

/// Emit a literal with a concrete type suffix (e.g. `1.5f`, `42i`, `3u`).
/// Used in contexts where the literal must carry its own type: standalone
/// expressions, `let` bindings, arithmetic operands, etc.
///
/// When `max_precision` is `Some(n)`, float literals are rounded to at most
/// `n` decimal places (lossy).  `None` preserves full precision.
/// Render a fully-typed literal (retains suffixes like `f`, `u`,
/// `lf`).  Safe to emit anywhere a concrete-typed literal is required;
/// [`literal_to_wgsl_bare`] handles the abstract-typed case.
pub(super) fn literal_to_wgsl(literal: naga::Literal, max_precision: Option<u8>) -> String {
    match literal {
        naga::Literal::F16(v) => {
            compact_float_literal_token(format!("{}h", fmt_f32(f32::from(v), max_precision)))
        }
        naga::Literal::F32(v) => {
            let dec = compact_float_literal_token(format!("{}f", fmt_f32(v, max_precision)));
            pick_shorter(dec, hex_float_f32(v, "f"))
        }
        naga::Literal::F64(v) => {
            compact_float_literal_token(format!("{}lf", fmt_f64_debug(v, max_precision)))
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
            let dec = compact_float_literal_token(fmt_f64(v, max_precision));
            ensure_bare_float(pick_shorter(dec, hex_float_f64(v, "")))
        }
    }
}

/// Emit a literal without any type suffix (e.g. `1.5` instead of `1.5f`).
///
/// Invariant - callers must pin the type via enclosing context
///
/// Whole-number concrete floats (`F32(1.0)`, `F64(2.0)`, `F16(3.0)`)
/// collapse to bare integer tokens (`1`, `2`, `3`) because WGSL's
/// `.0` stripping chooses the shortest decimal.  Such tokens parse as
/// `AbstractInt`, not the original float type.  The same holds for
/// non-suffixed integer tokens: `I32(42)` -> `42` parses as
/// `AbstractInt`.
///
/// This is **only** safe when the enclosing context unambiguously pins
/// the value's type.  Approved call sites (in priority of type-safety):
///
/// 1. Inside a type constructor `T(...)` where `T` is concrete - the
///    constructor signature determines every argument's type.  This
///    covers `Compose` and `Splat` in `expr_emit::emit_constructor_arg`
///    and `module_emit::emit_global_expr`'s Compose/Splat arms.
/// 2. As the RHS of an extracted `const NAME = ...;` declaration - the
///    constant takes the literal's (possibly abstract) type, and every
///    use site of `NAME` then re-binds via normal abstract-coercion.
///    This is [`literal_extract_key`]'s decl path.
///
/// All **other** call sites must use [`literal_to_wgsl`] (typed form).
/// In particular: binary operands where either side is itself a literal,
/// overload-resolution arguments (e.g. `atan2(1.0, x)`), and standalone
/// `let`/`var` initializers should NOT receive a bare-form literal -
/// an abstract-coercion surprise could flip overload resolution.
///
/// `max_precision`: `Some(n)` rounds floats to `n` decimal places
/// (lossy); `None` preserves full precision.
/// Render a literal without its type suffix for contexts where an
/// enclosing type constructor (or an extracted `const NAME = ...;`)
/// pins the value's concrete type.  Using this outside those two
/// sanctioned patterns is unsafe: abstract coercion changes the
/// inferred type.  See `lib.rs::e2e_concrete_float_literals_round_trip_after_minification`.
pub(super) fn literal_to_wgsl_bare(literal: naga::Literal, max_precision: Option<u8>) -> String {
    match literal {
        naga::Literal::F16(v) => {
            let dec = compact_float_literal_token(fmt_f32(f32::from(v), max_precision));
            pick_shorter(dec, hex_float_f32(f32::from(v), ""))
        }
        naga::Literal::F32(v) => {
            let dec = compact_float_literal_token(fmt_f32(v, max_precision));
            pick_shorter(dec, hex_float_f32(v, ""))
        }
        naga::Literal::F64(v) => {
            let dec = compact_float_literal_token(fmt_f64_debug(v, max_precision));
            pick_shorter(dec, hex_float_f64(v, ""))
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
            let dec = compact_float_literal_token(fmt_f64(v, max_precision));
            pick_shorter(dec, hex_float_f64(v, ""))
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
    max_precision: Option<u8>,
) -> LiteralExtractKey {
    let expr_text = literal_to_wgsl_bare(literal, max_precision);
    // U64 values >= 2^63 cannot be represented as AbstractInt and need the
    // `lu`-suffixed typed form in the const declaration.  Everything else can
    // use the expr_text directly as an abstract-typed const literal.
    let decl_text = match literal {
        naga::Literal::U64(v) if v > i64::MAX as u64 => literal_to_wgsl(literal, max_precision),
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

/// Return the short-form suffix for scalar types that have WGSL predeclared
/// type aliases (e.g. `f` for f32, `i` for i32, `u` for u32, `h` for f16).
/// Returns `None` for types without short aliases (bool, i64, u64, f64).
/// Return the single-character shorthand suffix (`i`, `u`, `f`, `h`)
/// for the WGSL vector-alias types, or `None` for scalars that have
/// no alias form.
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

/// Map a [`naga::AddressSpace`] to its WGSL keyword.  `Function`
/// returns the empty string because WGSL leaves `var<function>` implicit.
pub(super) fn address_space(space: naga::AddressSpace) -> &'static str {
    match space {
        naga::AddressSpace::Function => "function",
        naga::AddressSpace::Private => "private",
        naga::AddressSpace::WorkGroup => "workgroup",
        naga::AddressSpace::Uniform => "uniform",
        naga::AddressSpace::Storage { .. } => "storage",
        naga::AddressSpace::RayPayload => "ray_payload",
        naga::AddressSpace::IncomingRayPayload => "incoming_ray_payload",
        // naga-internal address spaces with no WGSL equivalent; map to
        // "private" as a safe fallback.
        naga::AddressSpace::Handle
        | naga::AddressSpace::Immediate
        | naga::AddressSpace::TaskPayload => "private",
    }
}

/// Map [`naga::StorageAccess`] flags to the WGSL access-mode keyword
/// (`read`, `read_write`, `write`).
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
        compact_float_literal_token, ensure_bare_float, literal_extract_key, literal_to_wgsl,
        literal_to_wgsl_bare, scalar_zero,
    };
    use half::f16;

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
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.25), None), ".25f");
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(0.5), None),
            ".5"
        );
        assert_eq!(literal_to_wgsl(naga::Literal::F64(0.5), None), ".5lf");
        // Typed: I32 gets 'i' suffix
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), None), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), None), "7u");

        // Bare: all suffixes stripped for use inside Compose
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.25), None), ".25");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F64(0.5), None), ".5");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U32(7), None), "7");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I32(42), None), "42");
    }

    #[test]
    fn literal_i32_min_wraps_in_constructor() {
        assert_eq!(
            literal_to_wgsl(naga::Literal::I32(i32::MIN), None),
            "i32(-2147483648)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::I32(i32::MIN), None),
            "i32(-2147483648)"
        );
    }

    #[test]
    fn literal_i64_typed_and_bare() {
        assert_eq!(literal_to_wgsl(naga::Literal::I64(99), None), "99li");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I64(99), None), "99");
        // The i64::MIN uses overflow-safe constructor
        assert_eq!(
            literal_to_wgsl(naga::Literal::I64(i64::MIN), None),
            "i64(-0x7fffffffffffffff - 1)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::I64(i64::MIN), None),
            "i64(-0x7fffffffffffffff - 1)"
        );
    }

    #[test]
    fn literal_u64_typed_and_bare() {
        assert_eq!(literal_to_wgsl(naga::Literal::U64(100), None), "100lu");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U64(100), None), "100");
    }

    #[test]
    fn literal_bool_and_abstract_int() {
        assert_eq!(literal_to_wgsl(naga::Literal::Bool(true), None), "true");
        assert_eq!(literal_to_wgsl(naga::Literal::Bool(false), None), "false");
        assert_eq!(literal_to_wgsl(naga::Literal::AbstractInt(42), None), "42");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::Bool(true), None),
            "true"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractInt(-7), None),
            "-7"
        );
        // AbstractInt i64::MIN must use overflow-safe subtraction form.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(i64::MIN), None),
            "(-0x7fffffffffffffff - 1)"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractInt(i64::MIN), None),
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
    fn max_precision_limits_decimal_places() {
        // F32 typed: 0.123456 rounded to 3 decimal places -> 0.123f -> .123f
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), Some(3)),
            ".123f"
        );
        // F32 typed: 0.876543 rounded to 2 -> 0.88f -> .88f
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.876543), Some(2)),
            ".88f"
        );
        // F32 bare: same value, no suffix
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(0.876543), Some(2)),
            ".88"
        );
        // AbstractFloat (f64): precision limiting works
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(7.65432198), Some(4)),
            "7.6543"
        );
        // Integer literals are unaffected by max_precision
        assert_eq!(literal_to_wgsl(naga::Literal::I32(42), Some(2)), "42i");
        assert_eq!(literal_to_wgsl(naga::Literal::U32(7), Some(2)), "7u");
        // None preserves full precision (baseline)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(0.123456), None),
            ".123456f"
        );
    }

    #[test]
    fn hex_repr_used_when_shorter() {
        // u64::MAX: 20 decimal digits -> 16 hex digits + 0x = 18 chars (saves 2)
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(u64::MAX), None),
            "0xfffffffffffffffflu"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::U64(u64::MAX), None),
            "0xffffffffffffffff"
        );
        // Large u64: 10^19 = 19 decimal digits
        assert_eq!(
            literal_to_wgsl(naga::Literal::U64(10_000_000_000_000_000_000), None),
            "0x8ac7230489e80000lu"
        );
        // i64::MAX: 19 decimal digits -> hex saves 1
        assert_eq!(
            literal_to_wgsl(naga::Literal::I64(i64::MAX), None),
            "0x7fffffffffffffffli"
        );
        // Large AbstractInt: 13 decimal digits -> hex can save 1
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractInt(1_000_000_000_000), None),
            "0xe8d4a51000"
        );
        // Small values stay decimal (hex is longer due to 0x prefix)
        assert_eq!(literal_to_wgsl(naga::Literal::U64(255), None), "255lu");
        assert_eq!(literal_to_wgsl(naga::Literal::AbstractInt(42), None), "42");
        assert_eq!(literal_to_wgsl(naga::Literal::I32(-5), None), "-5i");
        // u32 values: hex never shorter (max 10 decimal digits = 10 hex chars)
        assert_eq!(
            literal_to_wgsl(naga::Literal::U32(u32::MAX), None),
            "4294967295u"
        );
    }

    #[test]
    fn hex_float_used_when_shorter() {
        // 2^20 = 1048576.0: decimal "1048576f" (8) vs hex "0x1p20f" (7)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(1048576.0), None),
            "0x1p20f"
        );
        // 2^24 = 16777216.0: decimal "16777216f" (9) vs hex "0x1p24f" (7)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(16777216.0), None),
            "0x1p24f"
        );
        // Negative power of 2: -2^20
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(-1048576.0), None),
            "-0x1p20f"
        );
        // Small values stay decimal - hex is longer
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.5), None), ".5f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), None), "3f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), None), "1f");
        // Negative exponent: 2^-14 decimal ".000061035156f"(16) vs hex "0x1p-14f"(8)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(2.0_f32.powi(-14)), None),
            "0x1p-14f"
        );
        // f32::MAX: decimal is 40 chars, hex "0x1.fffffep127f" (15)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MAX), None),
            "0x1.fffffep127f"
        );
        // f32::MIN_POSITIVE: decimal is 49 chars, hex "0x1p-126f" (9)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F32(f32::MIN_POSITIVE), None),
            "0x1p-126f"
        );
        // Non-power-of-2 with mantissa bits: 3.0 = 0x1.8p1f (8) vs "3f" (2) -> decimal wins
        assert_eq!(literal_to_wgsl(naga::Literal::F32(3.0), None), "3f");
        // Bare 2^20: decimal "1048576" (7) vs hex "0x1p20" (6)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), None),
            "0x1p20"
        );
        // Bare small value stays decimal
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), None), ".5");
        // F64 bare: 2^50 hex wins
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F64(2.0_f64.powi(50)), None),
            "0x1p50"
        );
        // F64 typed stays decimal (no hex support for 'lf' suffix)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F64(2.0_f64.powi(50)), None),
            "1125899906842624.lf"
        );
        // F16 typed stays decimal (naga rejects hex + 'h' suffix)
        assert_eq!(
            literal_to_wgsl(naga::Literal::F16(f16::from_f32(1024.0)), None),
            "1024h"
        );
        // F16 bare with hex: 2^-14 as bare is shorter in hex
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(2.0_f32.powi(-14))), None),
            "0x1p-14"
        );
        // AbstractFloat 2^50 typed: hex much shorter (no suffix needed)
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1125899906842624.0), None),
            "0x1p50"
        );
        // AbstractFloat bare: same as typed (no suffix in either form)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1125899906842624.0), None),
            "0x1p50"
        );
        // Zero falls back to decimal (hex_float returns None for zero)
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), None), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-0.0), None), "-0f");
    }

    #[test]
    fn whole_number_float_literals_are_context_aware() {
        // Bare constructor literals stay as short as possible.
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), None), "1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.0), None), "0");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(-1.0), None), "-1");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(3.0), None), "3");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(-0.0), None), "-0");

        // F16 bare whole numbers - also no dot inside constructors
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(1.0)), None),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F16(f16::from_f32(0.0)), None),
            "0"
        );

        // AbstractFloat bare whole numbers are also kept short inside
        // constructor contexts.
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), None),
            "1"
        );
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(0.0), None),
            "0"
        );

        // Standalone AbstractFloat literals must keep a decimal point so WGSL
        // does not parse them as AbstractInt.
        assert_eq!(
            literal_to_wgsl(naga::Literal::AbstractFloat(1.0), None),
            "1."
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.0), None), "1");
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::AbstractFloat(1.0), None),
            "1"
        );
        assert_eq!(literal_to_wgsl_bare(naga::Literal::I32(42), None), "42");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U32(7), None), "7");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::U64(100), None), "100");
        let u64_key = literal_extract_key(naga::Literal::U64(u64::MAX), None);
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
        assert_eq!(literal_to_wgsl(naga::Literal::F32(1.0), None), "1f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(0.0), None), "0f");
        assert_eq!(literal_to_wgsl(naga::Literal::F32(-1.0), None), "-1f");

        // Fractional values remain unchanged (already have a dot)
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(0.5), None), ".5");
        assert_eq!(literal_to_wgsl_bare(naga::Literal::F32(1.5), None), "1.5");

        // Hex representations remain unchanged (already have 'x' marker)
        assert_eq!(
            literal_to_wgsl_bare(naga::Literal::F32(1048576.0), None),
            "0x1p20"
        );
    }
}
