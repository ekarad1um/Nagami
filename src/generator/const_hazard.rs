//! Const-expression hazards. WGSL evaluates every all-constant expression
//! at shader creation, and tint rejects inf/NaN results, f16 overflow, and
//! violated builtin argument rules (`extractBits` ranges, `clamp` bounds,
//! `smoothstep` edges, `ldexp` exponents) that naga's validator lets
//! through, so the self-check passes and Chrome fails. The input dodged the
//! evaluation through a `let` (`let bits = 0x7f800000u;
//! bitcast<f32>(bits)`), which naga's IR erases; `let`-binding the
//! offending operand right before its consumer restores the runtime
//! evaluation.
//!
//! An all-constant `Binary` / `Math` survivor of `const_fold` is
//! re-evaluated here and binds only when the evaluation fails, so the
//! fold's coverage gaps (f16, i64, vector and matrix shapes) cost nothing
//! when the value is benign.

use super::core::FunctionCtx;

/// Shapes whose text is a const-expression; everything else is runtime.
fn is_const_shape(expr: &naga::Expression) -> bool {
    matches!(
        expr,
        naga::Expression::Literal(_)
            | naga::Expression::Constant(_)
            | naga::Expression::Compose { .. }
            | naga::Expression::Splat { .. }
            | naga::Expression::ZeroValue(_)
    )
}

/// Scalar lanes of a constant tree; `None` once anything is runtime, a
/// `let`-bound subtree included. Bool lanes are kept so the tree still
/// reads as constant.
pub(super) fn const_lanes(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> Option<Vec<naga::Literal>> {
    lanes_in(
        module,
        &ctx.func.expressions,
        &|h| ctx.expr_names.contains_key(&h),
        h,
        0,
    )
}

fn lanes_in(
    module: &naga::Module,
    arena: &naga::Arena<naga::Expression>,
    bound: &dyn Fn(naga::Handle<naga::Expression>) -> bool,
    h: naga::Handle<naga::Expression>,
    depth: usize,
) -> Option<Vec<naga::Literal>> {
    if depth > 16 || bound(h) {
        return None;
    }
    match &arena[h] {
        naga::Expression::Literal(l) => Some(vec![*l]),
        naga::Expression::Constant(c) => lanes_in(
            module,
            &module.global_expressions,
            &|_| false,
            module.constants[*c].init,
            depth + 1,
        ),
        naga::Expression::Compose { components, .. } => {
            let mut out = Vec::new();
            for &c in components {
                out.extend(lanes_in(module, arena, bound, c, depth + 1)?);
            }
            Some(out)
        }
        naga::Expression::Splat { size, value } => {
            Some(lanes_in(module, arena, bound, *value, depth + 1)?.repeat(*size as usize))
        }
        naga::Expression::ZeroValue(ty) => match module.types[*ty].inner {
            naga::TypeInner::Scalar(s) => Some(vec![naga::Literal::zero(s)?]),
            naga::TypeInner::Vector { size, scalar } => {
                Some(vec![naga::Literal::zero(scalar)?; size as usize])
            }
            _ => None,
        },
        _ => None,
    }
}

/// Numeric value of a lane; `None` for bool.
fn lane_f64(l: naga::Literal) -> Option<f64> {
    use naga::Literal as L;
    Some(match l {
        L::F64(v) => v,
        L::F32(v) => v as f64,
        L::F16(v) => v.to_f64(),
        L::U16(v) => v as f64,
        L::I16(v) => v as f64,
        L::U32(v) => v as f64,
        L::I32(v) => v as f64,
        L::U64(v) => v as f64,
        L::I64(v) => v as f64,
        L::AbstractInt(v) => v as f64,
        L::AbstractFloat(v) => v,
        L::Bool(_) => return None,
    })
}

/// `(value, bit width, signed)`; `None` for floats and bools.
fn lane_int(l: naga::Literal) -> Option<(i128, u32, bool)> {
    use naga::Literal as L;
    Some(match l {
        L::U16(v) => (v as i128, 16, false),
        L::I16(v) => (v as i128, 16, true),
        L::U32(v) => (v as i128, 32, false),
        L::I32(v) => (v as i128, 32, true),
        L::U64(v) => (v as i128, 64, false),
        L::I64(v) => (v as i128, 64, true),
        L::AbstractInt(v) => (v as i128, 64, true),
        _ => return None,
    })
}

/// `None` for abstract / bool, which are never bitcast.
fn lane_bytes(l: naga::Literal) -> Option<Vec<u8>> {
    use naga::Literal as L;
    Some(match l {
        L::F64(v) => v.to_le_bytes().to_vec(),
        L::F32(v) => v.to_le_bytes().to_vec(),
        L::F16(v) => v.to_bits().to_le_bytes().to_vec(),
        L::U16(v) => v.to_le_bytes().to_vec(),
        L::I16(v) => v.to_le_bytes().to_vec(),
        L::U32(v) => v.to_le_bytes().to_vec(),
        L::I32(v) => v.to_le_bytes().to_vec(),
        L::U64(v) => v.to_le_bytes().to_vec(),
        L::I64(v) => v.to_le_bytes().to_vec(),
        L::Bool(_) | L::AbstractInt(_) | L::AbstractFloat(_) => return None,
    })
}

/// Any all-ones exponent when the byte image of `lanes` is read as
/// `width`-byte floats; lane and target widths may differ (`vec2<f16>` <->
/// `u32`).
fn bitcast_non_finite(lanes: &[naga::Literal], width: u8) -> bool {
    let mut bytes = Vec::new();
    for &l in lanes {
        match lane_bytes(l) {
            Some(b) => bytes.extend(b),
            None => return false,
        }
    }
    bytes.chunks_exact(width as usize).any(|chunk| match width {
        2 => (u16::from_le_bytes([chunk[0], chunk[1]]) >> 10) & 0x1F == 0x1F,
        4 => (u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) >> 23) & 0xFF == 0xFF,
        8 => (u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])) >> 52) & 0x7FF == 0x7FF,
        _ => false,
    })
}

/// Largest finite value of a float type by byte width (abstract = f64).
fn float_max(width: Option<u8>) -> f64 {
    match width {
        Some(2) => 65504.0,
        Some(4) => f32::MAX as f64,
        _ => f64::MAX,
    }
}

/// Byte width of `h`'s float scalar, `None` for non-float results.
fn result_float_width(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> Option<u8> {
    match ctx.info[h].ty.inner_with(&module.types).scalar()? {
        naga::Scalar {
            kind: naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat,
            width,
        } => Some(width),
        _ => None,
    }
}

/// The constant operand whose inlined text makes `h` a const-expression
/// tint rejects; the caller binds it before emitting `h`.
pub(super) fn creation_error_operand(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::Expression as E;
    let exprs = &ctx.func.expressions;
    let lanes = |e: naga::Handle<naga::Expression>| const_lanes(module, ctx, e);
    match &exprs[h] {
        E::As {
            expr,
            kind: naga::ScalarKind::Float,
            convert,
        } if is_const_shape(&exprs[*expr]) => {
            let src = lanes(*expr)?;
            let width = result_float_width(module, ctx, h)?;
            let hazard = match convert {
                None => bitcast_non_finite(&src, width),
                Some(_) => {
                    let max = float_max(Some(width));
                    src.iter()
                        .any(|&l| lane_f64(l).is_some_and(|v| !v.is_finite() || v.abs() > max))
                }
            };
            hazard.then_some(*expr)
        }
        E::Binary { op, left, right }
            if is_const_shape(&exprs[*left]) && is_const_shape(&exprs[*right]) =>
        {
            let (l, r) = (lanes(*left)?, lanes(*right)?);
            binary_hazard(*op, &l, &r, result_float_width(module, ctx, h)).then_some(*left)
        }
        E::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            let width = result_float_width(module, ctx, h);
            let values = |e: Option<naga::Handle<naga::Expression>>| -> Option<Vec<f64>> {
                let e = e.filter(|&e| is_const_shape(&exprs[e]))?;
                lanes(e)?.into_iter().map(lane_f64).collect()
            };
            // Argument rules first: they name the operand tint checks, and
            // an all-constant call must bind that one
            // (`extractBits(a,40,1)` stays rejected).
            if let Some(operand) =
                argument_rule_hazard(*fun, *arg, *arg1, *arg2, *arg3, width, &values)
            {
                return Some(operand);
            }
            let args: Vec<_> = [Some(*arg), *arg1, *arg2, *arg3]
                .into_iter()
                .flatten()
                .collect();
            let all: Option<Vec<Vec<f64>>> = args.iter().map(|&a| values(Some(a))).collect();
            math_hazard(*fun, &all?, width).then_some(*arg)
        }
        _ => None,
    }
}

/// tint's per-argument rules, enforced even with a runtime value operand:
/// bit ranges (each constant bound against the width, their sum when both
/// are constant), `clamp` bounds, `smoothstep` edges, `ldexp` exponents by
/// float width. Returns the operand to bind.
fn argument_rule_hazard(
    fun: naga::MathFunction,
    arg: naga::Handle<naga::Expression>,
    arg1: Option<naga::Handle<naga::Expression>>,
    arg2: Option<naga::Handle<naga::Expression>>,
    arg3: Option<naga::Handle<naga::Expression>>,
    width: Option<u8>,
    values: &dyn Fn(Option<naga::Handle<naga::Expression>>) -> Option<Vec<f64>>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::MathFunction as M;
    match fun {
        M::ExtractBits | M::InsertBits => {
            let (offset, count) = if fun == M::ExtractBits {
                (arg1, arg2)
            } else {
                (arg2, arg3)
            };
            let (o, c) = (values(offset), values(count));
            let over =
                |v: &Option<Vec<f64>>| v.as_ref().is_some_and(|v| v.iter().any(|&x| x > 32.0));
            if over(&o) {
                return offset;
            }
            if over(&c) {
                return count;
            }
            match (o, c) {
                (Some(o), Some(c)) if o.iter().zip(&c).any(|(a, b)| a + b > 32.0) => offset,
                _ => None,
            }
        }
        M::Clamp => {
            let (lo, hi) = (values(arg1)?, values(arg2)?);
            lo.iter().zip(&hi).any(|(a, b)| a > b).then_some(arg1?)
        }
        M::SmoothStep => {
            let (lo, hi) = (values(Some(arg))?, values(arg1)?);
            lo.iter().zip(&hi).any(|(a, b)| a == b).then_some(arg)
        }
        M::Ldexp => {
            let limit = match width {
                Some(2) => 16.0,
                Some(4) => 128.0,
                _ => 1024.0,
            };
            values(arg1)?.iter().any(|&e| e > limit).then_some(arg1?)
        }
        _ => None,
    }
}

fn is_float_lane(l: naga::Literal) -> bool {
    matches!(
        l,
        naga::Literal::F16(_)
            | naga::Literal::F32(_)
            | naga::Literal::F64(_)
            | naga::Literal::AbstractFloat(_)
    )
}

/// Lane pairs with scalar broadcast; `None` when the shapes do not line up
/// (matrix operands).
fn pair_lanes(
    l: &[naga::Literal],
    r: &[naga::Literal],
) -> Option<Vec<(naga::Literal, naga::Literal)>> {
    if l.len() == r.len() {
        Some(l.iter().copied().zip(r.iter().copied()).collect())
    } else if l.len() == 1 {
        Some(r.iter().map(|&b| (l[0], b)).collect())
    } else if r.len() == 1 {
        Some(l.iter().map(|&a| (a, r[0])).collect())
    } else {
        None
    }
}

/// Products of up to four such lanes (matrix product, determinant) stay
/// finite.
fn moderate(lanes: &[naga::Literal], max: f64) -> bool {
    let bound = max.powf(0.25);
    lanes
        .iter()
        .all(|&l| lane_f64(l).is_none_or(|v| v.is_finite() && v.abs() <= bound))
}

/// Float results outside the finite range, integer division by zero or MIN
/// / -1, shifts by the width or more, left shifts losing bits or the sign;
/// wrapping add/sub/mul and comparisons never fail.
fn binary_hazard(
    op: naga::BinaryOperator,
    l: &[naga::Literal],
    r: &[naga::Literal],
    width: Option<u8>,
) -> bool {
    use naga::BinaryOperator as B;
    let max = float_max(width);
    let Some(pairs) = pair_lanes(l, r) else {
        return matches!(op, B::Multiply | B::Add | B::Subtract)
            && !(moderate(l, max) && moderate(r, max));
    };
    pairs.into_iter().any(|(a, b)| {
        if is_float_lane(a) || is_float_lane(b) {
            let (Some(x), Some(y)) = (lane_f64(a), lane_f64(b)) else {
                return false;
            };
            let res = match op {
                B::Add => x + y,
                B::Subtract => x - y,
                B::Multiply => x * y,
                B::Divide => x / y,
                B::Modulo => x % y,
                _ => return false,
            };
            return !res.is_finite() || res.abs() > max;
        }
        let (Some((x, bits, signed)), Some((y, _, _))) = (lane_int(a), lane_int(b)) else {
            return false;
        };
        let min = -(1i128 << (bits - 1));
        match op {
            B::Divide | B::Modulo => y == 0 || (signed && y == -1 && x == min),
            B::ShiftRight => y < 0 || y >= bits as i128,
            B::ShiftLeft => {
                if y < 0 || y >= bits as i128 {
                    return true;
                }
                let shifted = x << y;
                if signed {
                    shifted < min || shifted > -min - 1
                } else {
                    shifted >> bits != 0
                }
            }
            _ => false,
        }
    })
}

/// Domain errors (`sqrt(-1)`, `log(0)`, `acos(2)`, `normalize(vec(0))`),
/// result overflow, packing / quantizing range; functions absent here are
/// total on finite input.
fn math_hazard(fun: naga::MathFunction, args: &[Vec<f64>], width: Option<u8>) -> bool {
    use naga::MathFunction as M;
    let max = float_max(width);
    let a = |i: usize| args.get(i).map(Vec::as_slice).unwrap_or(&[]);
    let any = |v: &[f64], f: &dyn Fn(f64) -> bool| v.iter().any(|&x| f(x));
    let bound = max.powf(0.25);
    let big = |v: &[f64]| any(v, &|x| !x.is_finite() || x.abs() > bound);
    match fun {
        M::Sqrt => any(a(0), &|x| x < 0.0),
        M::InverseSqrt | M::Log | M::Log2 => any(a(0), &|x| x <= 0.0),
        M::Acos | M::Asin => any(a(0), &|x| x.abs() > 1.0),
        M::Acosh => any(a(0), &|x| x < 1.0),
        M::Atanh => any(a(0), &|x| x.abs() >= 1.0),
        M::Exp => any(a(0), &|x| x.exp() > max),
        M::Exp2 => any(a(0), &|x| x.exp2() > max),
        M::Sinh | M::Cosh => any(a(0), &|x| x.cosh() > max),
        M::Pow => a(0).iter().zip(a(1).iter().cycle()).any(|(&b, &e)| {
            let r = b.powf(e);
            b < 0.0 || (b == 0.0 && e <= 0.0) || !r.is_finite() || r.abs() > max
        }),
        M::Atan2 => a(0).iter().zip(a(1)).any(|(&y, &x)| y == 0.0 && x == 0.0),
        M::Normalize => a(0).iter().all(|&x| x == 0.0),
        M::Refract => true,
        M::Dot
        | M::Cross
        | M::Length
        | M::Distance
        | M::Fma
        | M::Outer
        | M::Reflect
        | M::Degrees
        | M::Radians
        | M::Determinant => args.iter().any(|v| big(v)),
        M::QuantizeToF16 | M::Pack2x16float => {
            any(a(0), &|x| !x.is_finite() || x.abs() > float_max(Some(2)))
        }
        M::Unpack2x16float => any(a(0), &|x| {
            let bits = x as u32;
            [bits & 0xFFFF, bits >> 16]
                .iter()
                .any(|h| (h >> 10) & 0x1F == 0x1F)
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::BinaryOperator as B;
    use naga::Literal as L;

    #[test]
    fn bitcast_lanes_detect_inf_and_nan_across_widths() {
        assert!(bitcast_non_finite(&[L::U32(0x7f80_0000)], 4));
        assert!(bitcast_non_finite(&[L::I32(0x7fc0_0000)], 4));
        assert!(!bitcast_non_finite(&[L::U32(0x3f80_0000)], 4));
        // Two f16 halves packed in one u32: the high half is inf.
        assert!(bitcast_non_finite(&[L::U32(0x7c00_3c00)], 2));
        assert!(!bitcast_non_finite(&[L::U32(0x3c00_3c00)], 2));
        // vec2<u32> -> vec2<f32>: only the second lane is NaN.
        assert!(bitcast_non_finite(&[L::U32(1), L::U32(0xffc0_0000)], 4));
    }

    #[test]
    fn integer_binary_rules_are_exact() {
        assert!(binary_hazard(B::Divide, &[L::U32(5)], &[L::U32(0)], None));
        assert!(binary_hazard(
            B::Divide,
            &[L::I32(i32::MIN)],
            &[L::I32(-1)],
            None
        ));
        assert!(!binary_hazard(
            B::Divide,
            &[L::I32(i32::MIN)],
            &[L::I32(1)],
            None
        ));
        assert!(binary_hazard(
            B::ShiftLeft,
            &[L::U32(1)],
            &[L::U32(32)],
            None
        ));
        assert!(binary_hazard(
            B::ShiftLeft,
            &[L::U32(0xF000_0000)],
            &[L::U32(4)],
            None
        ));
        assert!(!binary_hazard(
            B::ShiftLeft,
            &[L::U32(0x0F00_0000)],
            &[L::U32(4)],
            None
        ));
        assert!(binary_hazard(
            B::ShiftLeft,
            &[L::I32(1)],
            &[L::U32(31)],
            None
        ));
        assert!(!binary_hazard(
            B::ShiftLeft,
            &[L::I32(-1)],
            &[L::U32(31)],
            None
        ));
        assert!(!binary_hazard(
            B::ShiftRight,
            &[L::U32(u32::MAX)],
            &[L::U32(31)],
            None
        ));
        // Wrapping arithmetic is never a creation error.
        assert!(!binary_hazard(
            B::Multiply,
            &[L::I32(i32::MAX)],
            &[L::I32(2)],
            None
        ));
        // i64 is exact past f64's 53 bits.
        assert!(!binary_hazard(
            B::ShiftLeft,
            &[L::I64(1 << 52 | 1)],
            &[L::U32(10)],
            None
        ));
        assert!(binary_hazard(
            B::ShiftLeft,
            &[L::U64(1 << 63 | 1)],
            &[L::U32(1)],
            None
        ));
    }

    #[test]
    fn float_binary_rules_track_the_result_width() {
        assert!(binary_hazard(
            B::Multiply,
            &[L::F32(1e38)],
            &[L::F32(10.0)],
            Some(4)
        ));
        assert!(!binary_hazard(
            B::Multiply,
            &[L::F32(1e37)],
            &[L::F32(10.0)],
            Some(4)
        ));
        assert!(binary_hazard(
            B::Divide,
            &[L::F32(1.0)],
            &[L::F32(0.0)],
            Some(4)
        ));
        // Broadcast scalar against a vector's lanes; f16 range applies.
        assert!(binary_hazard(
            B::Multiply,
            &[L::F32(1000.0), L::F32(1.0)],
            &[L::F32(100.0)],
            Some(2)
        ));
        assert!(!binary_hazard(
            B::Multiply,
            &[L::F32(2.0), L::F32(3.0)],
            &[L::F32(4.0)],
            Some(2)
        ));
    }
}
