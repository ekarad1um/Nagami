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
use crate::analysis::ExprClass;
use crate::config::FloatPrecision;

/// `Some` when `h` renders as one const-expression: literal / `const` leaves
/// under `@const` operators and builtins, nothing bound to a name (a `let` is
/// runtime in WGSL).  The payload is `true` when the tree is opaque: it holds a
/// `bitcast` or `unpack*`, which neither naga's evaluator nor `const_fold`
/// computes, so the caller can only bind on the operator's ability to fail.
/// Any other constant tree the folder left alone (a matrix product, say) was
/// evaluated by tint in the input too, so an unknown value there binds
/// nothing; binding those as well would pay bytes for a hazard no shader
/// writes (`var v = m * u; v * 1e38`), the accepted residual.
/// An `override` leaf is opaque too: its value is the host's, and an
/// override-expression is evaluated at pipeline creation with the same rules
/// (Dawn substitutes the values and re-resolves), so `let n = N; if (n > 0u)
/// { x / n }` emitted as `x / N` fails the pipeline of a host that sets `N`
/// to zero, where the input's runtime division was guarded.  The direct
/// form `x / N` binds for nothing, the price of the erased `let`.
/// A tree deeper than the walk follows reads as opaque, never as runtime:
/// the single-call splice puts no size cap on the operand it manufactures
/// (`out[1] / (unpack4xU8(v).x + 1u + ... - 18u)` with `v` substituted), and
/// an unknown operand of a failable operator must bind, not pass.
fn const_tree(
    exprs: &naga::Arena<naga::Expression>,
    bound: &dyn Fn(naga::Handle<naga::Expression>) -> bool,
    h: naga::Handle<naga::Expression>,
    depth: usize,
) -> Option<bool> {
    use naga::Expression as E;
    use naga::MathFunction as M;
    if bound(h) {
        return None;
    }
    if depth > 16 {
        return Some(true);
    }
    let class = ExprClass::node(&exprs[h]);
    if class.any(ExprClass::CONST_LEAF) {
        return Some(class.any(ExprClass::OVERRIDE));
    }
    if !class.any(ExprClass::PURE_OP) {
        return None;
    }
    // A bitcast reinterprets, an unpack decodes: both opaque to tint's
    // const-evaluator diagnostics.
    let opaque = match &exprs[h] {
        E::As { convert, .. } => convert.is_none(),
        E::Math { fun, .. } => matches!(
            fun,
            M::Unpack4x8snorm
                | M::Unpack4x8unorm
                | M::Unpack2x16snorm
                | M::Unpack2x16unorm
                | M::Unpack2x16float
                | M::Unpack4xI8
                | M::Unpack4xU8
        ),
        _ => false,
    };
    let mut acc = Some(opaque);
    crate::ir::visit::visit_expression_children(&exprs[h], |operand| {
        acc = acc.and_then(|seen| Some(seen | const_tree(exprs, bound, operand, depth + 1)?));
    });
    acc
}

/// Scalar lanes of a constant tree; `None` once anything is runtime, a
/// `let`-bound subtree included. Bool lanes are kept so the tree still
/// reads as constant.
pub(super) fn const_lanes(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> Option<Vec<naga::Literal>> {
    lanes_in(module, ctx.exprs, &|h| ctx.expr_names.contains_key(h), h, 0)
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

/// Numeric value of a lane as the emitter will PRINT it: `precision`
/// rounding is applied here because the rules below judge the const
/// expression the consumer parses, not the one the IR holds.  Rounding two
/// distinct `smoothstep` edges together is a shader-creation error that
/// exists only in the output.  `None` for bool.
fn lane_f64(l: naga::Literal, precision: &FloatPrecision) -> Option<f64> {
    use naga::Literal as L;
    Some(match l {
        L::F64(v) => super::syntax::round_f64(v, precision.f64),
        L::F32(v) => super::syntax::round_f32(v, precision.f32) as f64,
        L::F16(v) => super::syntax::round_f16(f32::from(v), precision.f16) as f64,
        L::U16(v) => v as f64,
        L::I16(v) => v as f64,
        L::U32(v) => v as f64,
        L::I32(v) => v as f64,
        L::U64(v) => v as f64,
        L::I64(v) => v as f64,
        L::AbstractInt(v) => v as f64,
        L::AbstractFloat(v) => super::syntax::round_f64(v, precision.abstract_float),
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
    match ctx.ty(h).inner_with(&module.types).scalar()? {
        naga::Scalar {
            kind: naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat,
            width,
        } => Some(width),
        _ => None,
    }
}

/// `true` when `e` prints as `Type(<one operand>)`: a conversion or a splat,
/// both of which Dawn's MSL writer renders as a type name applied to one
/// argument.  A named expression prints as that name instead, ending the
/// chain.
fn msl_type_constructor(
    exprs: &naga::Arena<naga::Expression>,
    bound: &dyn Fn(naga::Handle<naga::Expression>) -> bool,
    e: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    if bound(e) {
        return None;
    }
    match exprs[e] {
        naga::Expression::As {
            expr,
            convert: Some(_),
            ..
        }
        | naga::Expression::Splat { value: expr, .. } => Some(expr),
        _ => None,
    }
}

/// The second constructor of a unary operator's `T1(T2(..(x)))` operand,
/// which the caller binds so Dawn's MSL never prints `~(uint(int(v)))` or
/// `-(float4(int4(v)))`: Metal's C++ front-end reads the innermost
/// `T(identifier)` as a parameter declaration, making the parenthesized
/// operand a function type and the operator that follows a cast (`& 3u`
/// becomes an address-of).  Binding the second constructor leaves
/// `T1(name)`, a single level, which no declarator can match.  Multi-operand
/// constructors (`int2(v, v)`) are no parameter list and need nothing.
pub(super) fn msl_cast_ambiguity_operand(
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    msl_cast_ambiguity_operand_in(ctx.exprs, &|e| ctx.expr_names.contains_key(e), h)
}

/// [`msl_cast_ambiguity_operand`] over a bare arena.  The `for(...)` header
/// renders before any name is bound, so it asks with `bound` always false.
pub(super) fn msl_cast_ambiguity_operand_in(
    exprs: &naga::Arena<naga::Expression>,
    bound: &dyn Fn(naga::Handle<naga::Expression>) -> bool,
    h: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::Expression as E;
    let E::Unary { expr: outer, .. } = exprs[h] else {
        return None;
    };
    let inner = msl_type_constructor(exprs, bound, outer)?;
    let mut leaf = msl_type_constructor(exprs, bound, inner)?;
    while let Some(next) = msl_type_constructor(exprs, bound, leaf) {
        leaf = next;
    }
    let identifier = bound(leaf)
        || match exprs[leaf] {
            E::FunctionArgument(_) => true,
            E::Load { pointer } => {
                matches!(exprs[pointer], E::LocalVariable(_) | E::GlobalVariable(_))
            }
            _ => false,
        };
    identifier.then_some(inner)
}

/// The constant operand whose inlined text makes `h` a const-expression tint
/// rejects; the caller binds it before emitting `h`.  An opaque tree binds
/// whenever the operator can fail at all.
pub(super) fn creation_error_operand(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
    precision: &FloatPrecision,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::Expression as E;
    let exprs = &ctx.exprs;
    let bound = |e: naga::Handle<naga::Expression>| ctx.expr_names.contains_key(e);
    let tree = |e: naga::Handle<naga::Expression>| const_tree(exprs, &bound, e, 0);
    let lanes = |e: naga::Handle<naga::Expression>| const_lanes(module, ctx, e);
    match &exprs[h] {
        E::As {
            expr,
            kind: naga::ScalarKind::Float,
            convert,
        } if let Some(opaque) = tree(*expr) => {
            let width = result_float_width(module, ctx, h)?;
            let Some(src) = lanes(*expr) else {
                // Unknown bits may read as inf / NaN and an unknown float may
                // overflow a narrower target; an integer overflows only f16
                // (`f16(70000i)` is rejected, f32 spans every integer type).
                let hazard = opaque
                    && (convert.is_none() || width == 2 || scalar_is_float(module, ctx, *expr));
                return hazard.then_some(*expr);
            };
            let hazard = match convert {
                None => bitcast_non_finite(&src, width),
                Some(_) => {
                    let max = float_max(Some(width));
                    src.iter().any(|&l| {
                        lane_f64(l, precision).is_some_and(|v| !v.is_finite() || v.abs() > max)
                    })
                }
            };
            hazard.then_some(*expr)
        }
        E::Binary { op, left, right } => {
            let width = result_float_width(module, ctx, h);
            let (lo, ro) = (tree(*left), tree(*right));
            // Integer `/` `%` `<<` `>>` fail on the RIGHT operand alone
            // (`divisor_hazard`), whatever the left is: the right one binds
            // (binding the left clears nothing) and a value out of reach binds
            // like an opaque tree.  Rules needing both operands const (MIN /
            // -1, a left shift losing bits) fall through to the pair below.
            if ro.is_some() && width.is_none() && divisor_keyed(*op) {
                let bits = scalar_bits(module, ctx, *left);
                let hazard = lanes(*right).is_none_or(|r| {
                    r.iter()
                        .any(|&l| lane_int(l).is_some_and(|(y, _, _)| divisor_hazard(*op, y, bits)))
                });
                if hazard {
                    return Some(*right);
                }
            }
            let (Some(lo), Some(ro)) = (lo, ro) else {
                return None;
            };
            // A value `const_lanes` cannot compute (an unmodeled builtin in
            // the tree) binds like an opaque one for the integer operators:
            // `-100i << (reverseBits(u32(-100i)) & 31u)` overflows at
            // const-evaluation.  Float add / sub / mul keep the residual.
            let hazard = match (lanes(*left), lanes(*right)) {
                (Some(l), Some(r)) => binary_hazard(*op, &l, &r, width, precision),
                _ => {
                    let integer_op = matches!(
                        op,
                        naga::BinaryOperator::Divide
                            | naga::BinaryOperator::Modulo
                            | naga::BinaryOperator::ShiftLeft
                            | naga::BinaryOperator::ShiftRight
                    );
                    (lo || ro || integer_op) && binary_can_fail(*op, width)
                }
            };
            hazard.then_some(*left)
        }
        // A const-expression index outside the base's static length is a
        // shader-creation error.  A value in reach was judged before emission
        // (naga's validator for the literal, the driver's slot check for the
        // const-expression), so what arrives here is what neither could see:
        // a tree the driver's evaluator cannot compute (`unpack4xU8(..).w`)
        // is opaque and binds, as under any failable operator; a
        // runtime-sized base bounds only the sign, an override-sized one
        // everything but index 0.
        E::Access { base, index } if tree(*index).is_some() => {
            let bound = crate::passes::expr_util::static_index_bound(
                ctx.ty(*base).inner_with(&module.types),
                module,
            )?
            .with_index(ctx.ty(*index).inner_with(&module.types));
            let hazard = match lanes(*index) {
                Some(l) => l
                    .iter()
                    .any(|lit| crate::passes::expr_util::index_is_static_error(bound.len, lit)),
                None => crate::passes::const_fold::operand_is_static_error(
                    &module.types,
                    &crate::passes::const_fold::constant_literals(module),
                    exprs,
                    crate::passes::const_fold::Role::index(bound),
                    *index,
                ),
            };
            hazard.then_some(*index)
        }
        E::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            let width = result_float_width(module, ctx, h);
            let values = |e: Option<naga::Handle<naga::Expression>>| -> ConstArg {
                let Some(e) = e.filter(|&e| tree(e).is_some()) else {
                    return ConstArg::Runtime;
                };
                let known: Option<Vec<f64>> = lanes(e)
                    .and_then(|ls| ls.into_iter().map(|l| lane_f64(l, precision)).collect());
                known.map_or(ConstArg::Opaque, ConstArg::Known)
            };
            // Argument rules first: they name the operand tint checks, and an
            // all-constant call must bind that one (`extractBits(a,40,1)` stays
            // rejected).
            if let Some(operand) =
                argument_rule_hazard(*fun, *arg, *arg1, *arg2, *arg3, width, &values)
            {
                return Some(operand);
            }
            let args: Vec<_> = [Some(*arg), *arg1, *arg2, *arg3]
                .into_iter()
                .flatten()
                .collect();
            let opaque = args
                .iter()
                .try_fold(false, |acc, &a| Some(acc | tree(a)?))?;
            let all: Option<Vec<Vec<f64>>> = args
                .iter()
                .map(|&a| match values(Some(a)) {
                    ConstArg::Known(v) => Some(v),
                    ConstArg::Runtime | ConstArg::Opaque => None,
                })
                .collect();
            let hazard = match all {
                Some(all) => math_hazard(*fun, &all, width),
                None => opaque && math_can_fail(*fun),
            };
            hazard.then_some(*arg)
        }
        _ => None,
    }
}

/// `true` when `h` is a float scalar or vector (abstract included).
fn scalar_is_float(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        ctx.ty(h).inner_with(&module.types).scalar(),
        Some(naga::Scalar {
            kind: naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat,
            ..
        })
    )
}

/// A builtin operand as tint's argument rules see it.
enum ConstArg {
    /// Not a const-expression: no rule reads it.
    Runtime,
    /// A const-expression tint evaluates and nagami cannot: it binds in any
    /// position a rule would read.
    Opaque,
    Known(Vec<f64>),
}

/// tint's per-argument rules, enforced even with a runtime value operand:
/// bit ranges (each constant bound against the width, their sum when both
/// are constant), `clamp` bounds, `smoothstep` edges, `ldexp` exponents by
/// float width.  Returns the operand to bind: an offending known value, or
/// an opaque one where the rule would read it.
fn argument_rule_hazard(
    fun: naga::MathFunction,
    arg: naga::Handle<naga::Expression>,
    arg1: Option<naga::Handle<naga::Expression>>,
    arg2: Option<naga::Handle<naga::Expression>>,
    arg3: Option<naga::Handle<naga::Expression>>,
    width: Option<u8>,
    values: &dyn Fn(Option<naga::Handle<naga::Expression>>) -> ConstArg,
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
            let over = |v: &ConstArg| match v {
                ConstArg::Runtime => false,
                ConstArg::Opaque => true,
                ConstArg::Known(v) => v.iter().any(|&x| x > 32.0),
            };
            if over(&o) {
                return offset;
            }
            if over(&c) {
                return count;
            }
            match (o, c) {
                (ConstArg::Known(o), ConstArg::Known(c))
                    if o.iter().zip(&c).any(|(a, b)| a + b > 32.0) =>
                {
                    offset
                }
                _ => None,
            }
        }
        M::Clamp => edge_rule(arg1?, arg2?, values, |lo, hi| lo > hi),
        M::SmoothStep => edge_rule(arg, arg1?, values, |lo, hi| lo == hi),
        M::Ldexp => {
            let limit = match width {
                Some(2) => 16.0,
                Some(4) => 128.0,
                _ => 1024.0,
            };
            match values(arg1) {
                ConstArg::Runtime => None,
                ConstArg::Opaque => arg1,
                ConstArg::Known(e) => e.iter().any(|&e| e > limit).then_some(arg1?),
            }
        }
        _ => None,
    }
}

/// A rule over two edges that tint checks only when both are const: an
/// opaque edge binds, a known pair binds the low edge on `violates`.
fn edge_rule(
    lo: naga::Handle<naga::Expression>,
    hi: naga::Handle<naga::Expression>,
    values: &dyn Fn(Option<naga::Handle<naga::Expression>>) -> ConstArg,
    violates: impl Fn(f64, f64) -> bool,
) -> Option<naga::Handle<naga::Expression>> {
    match (values(Some(lo)), values(Some(hi))) {
        (ConstArg::Runtime, _) | (_, ConstArg::Runtime) => None,
        (ConstArg::Opaque, _) => Some(lo),
        (_, ConstArg::Opaque) => Some(hi),
        (ConstArg::Known(l), ConstArg::Known(h)) => l
            .iter()
            .zip(&h)
            .any(|(&a, &b)| violates(a, b))
            .then_some(lo),
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
fn moderate(lanes: &[naga::Literal], max: f64, precision: &FloatPrecision) -> bool {
    let bound = max.powf(0.25);
    lanes
        .iter()
        .all(|&l| lane_f64(l, precision).is_none_or(|v| v.is_finite() && v.abs() <= bound))
}

/// Operators [`binary_hazard`] can ever reject, for an operand whose value
/// is out of reach: division and shifts for any width, add / sub / mul only
/// where a float result can leave the finite range.
fn binary_can_fail(op: naga::BinaryOperator, width: Option<u8>) -> bool {
    use naga::BinaryOperator as B;
    match op {
        B::Divide | B::Modulo | B::ShiftLeft | B::ShiftRight => true,
        B::Add | B::Subtract | B::Multiply => width.is_some(),
        _ => false,
    }
}

/// Float results outside the finite range, integer division by zero or MIN
/// / -1, shifts by the width or more, left shifts losing bits or the sign;
/// wrapping add/sub/mul and comparisons never fail.
fn binary_hazard(
    op: naga::BinaryOperator,
    l: &[naga::Literal],
    r: &[naga::Literal],
    width: Option<u8>,
    precision: &FloatPrecision,
) -> bool {
    use naga::BinaryOperator as B;
    let max = float_max(width);
    let Some(pairs) = pair_lanes(l, r) else {
        return matches!(op, B::Multiply | B::Add | B::Subtract)
            && !(moderate(l, max, precision) && moderate(r, max, precision));
    };
    pairs.into_iter().any(|(a, b)| {
        if is_float_lane(a) || is_float_lane(b) {
            let (Some(x), Some(y)) = (lane_f64(a, precision), lane_f64(b, precision)) else {
                return false;
            };
            let res = match op {
                B::Add => x + y,
                B::Subtract => x - y,
                B::Multiply => x * y,
                B::Divide => x / y,
                // Const `%` is exact `fmod`; the GPU's stepwise f32
                // `x - y * trunc(x / y)` is a whole divisor off when the rounded
                // quotient crosses an integer (`0x1p25 % 3`: 2 vs 0), and the
                // input's slot was runtime (naga folds a const pair first), so
                // the pair always binds.  `/` is correctly rounded either way.
                B::Modulo => return true,
                _ => return false,
            };
            return !res.is_finite() || res.abs() > max;
        }
        let (Some((x, bits, signed)), Some((y, _, _))) = (lane_int(a), lane_int(b)) else {
            return false;
        };
        if divisor_hazard(op, y, bits) {
            return true;
        }
        let min = -(1i128 << (bits - 1));
        match op {
            B::Divide | B::Modulo => signed && y == -1 && x == min,
            B::ShiftLeft => {
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

/// The four operators with a rule on the right operand alone.
fn divisor_keyed(op: naga::BinaryOperator) -> bool {
    use naga::BinaryOperator as B;
    matches!(op, B::Divide | B::Modulo | B::ShiftLeft | B::ShiftRight)
}

/// The right-operand rule of the [`divisor_keyed`] operators: a zero divisor,
/// a shift amount at the shifted operand's bit width or beyond.
fn divisor_hazard(op: naga::BinaryOperator, y: i128, bits: u32) -> bool {
    use naga::BinaryOperator as B;
    match op {
        B::Divide | B::Modulo => y == 0,
        B::ShiftLeft | B::ShiftRight => y < 0 || y >= i128::from(bits),
        _ => false,
    }
}

/// Bit width of `h`'s scalar; 32, WGSL's default, for anything without one.
fn scalar_bits(
    module: &naga::Module,
    ctx: &FunctionCtx<'_, '_>,
    h: naga::Handle<naga::Expression>,
) -> u32 {
    ctx.ty(h)
        .inner_with(&module.types)
        .scalar()
        .map_or(32, |s| u32::from(s.width) * 8)
}

/// Builtins with a [`math_hazard`] arm, for arguments whose value is out of
/// reach; the two lists must move together.
fn math_can_fail(fun: naga::MathFunction) -> bool {
    use naga::MathFunction as M;
    matches!(
        fun,
        M::Sqrt
            | M::InverseSqrt
            | M::Log
            | M::Log2
            | M::Acos
            | M::Asin
            | M::Acosh
            | M::Atanh
            | M::Exp
            | M::Exp2
            | M::Sinh
            | M::Cosh
            | M::Ldexp
            | M::Pow
            | M::Atan2
            | M::Normalize
            | M::Refract
            | M::Dot
            | M::Cross
            | M::Length
            | M::Distance
            | M::Fma
            | M::Outer
            | M::Reflect
            | M::Degrees
            | M::Radians
            | M::Determinant
            | M::Mix
            | M::FaceForward
            | M::QuantizeToF16
            | M::Pack2x16float
            | M::Unpack2x16float
    )
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
        // `argument_rule_hazard` bounds e2 alone; the RESULT overflows well
        // inside it (`ldexp(1f, 128)`).
        M::Ldexp => a(0).iter().zip(a(1).iter().cycle()).any(|(&m, &e)| {
            let r = m * e.exp2();
            !r.is_finite() || r.abs() > max
        }),
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
        | M::Determinant
        | M::Mix
        | M::FaceForward => args.iter().any(|v| big(v)),
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
    /// The rules are precision-independent; these fixtures exercise them at
    /// full precision, so one shim keeps the call sites free of the argument.
    fn prec() -> FloatPrecision {
        FloatPrecision::default()
    }
    fn bh(
        op: naga::BinaryOperator,
        l: &[naga::Literal],
        r: &[naga::Literal],
        width: Option<u8>,
    ) -> bool {
        binary_hazard(op, l, r, width, &prec())
    }

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
        assert!(bh(B::Divide, &[L::U32(5)], &[L::U32(0)], None));
        assert!(bh(B::Divide, &[L::I32(i32::MIN)], &[L::I32(-1)], None));
        assert!(!bh(B::Divide, &[L::I32(i32::MIN)], &[L::I32(1)], None));
        assert!(bh(B::ShiftLeft, &[L::U32(1)], &[L::U32(32)], None));
        assert!(bh(B::ShiftLeft, &[L::U32(0xF000_0000)], &[L::U32(4)], None));
        assert!(!bh(
            B::ShiftLeft,
            &[L::U32(0x0F00_0000)],
            &[L::U32(4)],
            None
        ));
        assert!(bh(B::ShiftLeft, &[L::I32(1)], &[L::U32(31)], None));
        assert!(!bh(B::ShiftLeft, &[L::I32(-1)], &[L::U32(31)], None));
        assert!(!bh(B::ShiftRight, &[L::U32(u32::MAX)], &[L::U32(31)], None));
        // Wrapping arithmetic is never a creation error.
        assert!(!bh(B::Multiply, &[L::I32(i32::MAX)], &[L::I32(2)], None));
        // i64 is exact past f64's 53 bits.
        assert!(!bh(
            B::ShiftLeft,
            &[L::I64(1 << 52 | 1)],
            &[L::U32(10)],
            None
        ));
        assert!(bh(B::ShiftLeft, &[L::U64(1 << 63 | 1)], &[L::U32(1)], None));
    }

    #[test]
    fn float_binary_rules_track_the_result_width() {
        assert!(bh(B::Multiply, &[L::F32(1e38)], &[L::F32(10.0)], Some(4)));
        assert!(!bh(B::Multiply, &[L::F32(1e37)], &[L::F32(10.0)], Some(4)));
        assert!(bh(B::Divide, &[L::F32(1.0)], &[L::F32(0.0)], Some(4)));
        // Broadcast scalar against a vector's lanes; f16 range applies.
        assert!(bh(
            B::Multiply,
            &[L::F32(1000.0), L::F32(1.0)],
            &[L::F32(100.0)],
            Some(2)
        ));
        assert!(!bh(
            B::Multiply,
            &[L::F32(2.0), L::F32(3.0)],
            &[L::F32(4.0)],
            Some(2)
        ));
    }

    #[test]
    fn interpolation_builtins_can_overflow() {
        use naga::MathFunction as M;
        // tint rejects `mix(3e38, -3e38, 2.0)` and a `faceForward` whose dot
        // product leaves the f32 range; moderate operands stay inline.
        assert!(math_hazard(
            M::Mix,
            &[vec![3e38], vec![-3e38], vec![2.0]],
            Some(4)
        ));
        assert!(!math_hazard(
            M::Mix,
            &[vec![1.0], vec![2.0], vec![0.5]],
            Some(4)
        ));
        assert!(math_hazard(
            M::FaceForward,
            &[vec![1e38, 1e38], vec![1e38, 1e38], vec![1e38, 1e38]],
            Some(4)
        ));
        assert!(math_can_fail(M::Mix) && math_can_fail(M::FaceForward));
        assert!(!math_can_fail(M::Step) && !math_can_fail(M::SmoothStep));
    }
}
