//! Classifiers for [`naga::Expression`] and [`naga::Statement`] shapes and
//! literal values, shared by the passes and the generator; the walks over
//! them live in `crate::ir`.  Every match here is exhaustive with no `_`
//! arm, so a new naga variant fails the build at this single point of truth
//! instead of drifting silently through consumers' private deny-lists.
//! (`coalescing` keeps its own exhaustive statement walker.)

use crate::analysis::ExprClass;
use crate::ir::visit::{
    all_functions, for_each_statement, visit_expression_children, visit_statement_operands,
};

/// A library module: no entry points.  naga's `compact` then keeps every
/// declaration (`KeepUnused::Yes`) because the host that splices the
/// fragment may reference any of them, so whole-program rewrites (inlining,
/// pointer-parameter specialization, dropping unreferenced declarations)
/// are off and only lexical and function-local rewrites apply.
pub(crate) fn is_library_module(module: &naga::Module) -> bool {
    module.entry_points.is_empty()
}

/// An expression tint's uniformity analysis constrains
/// ([`ExprClass::CONVERGENT`]): the emitter pins these at their `Emit`,
/// coalescing declines lane reuse that would change their gating variable's
/// uniformity.
pub(crate) fn is_uniformity_constrained_expr(expr: &naga::Expression) -> bool {
    ExprClass::node(expr).any(ExprClass::CONVERGENT)
}

/// An expression whose evaluation is the cost a guard exists to save
/// ([`ExprClass::EXPENSIVE`]): the passes keep its count, control
/// dependence and loop depth, and every gate that asks
/// (`merge_trailing_returns`, the short-circuit re-sugar, the emitter's loop
/// pin) reads this one bit.  Calls and atomics are statements, never here.
pub(crate) fn is_expensive_expr(expr: &naga::Expression) -> bool {
    ExprClass::node(expr).any(ExprClass::EXPENSIVE)
}

/// The module holds something tint's uniformity analysis can reject (a
/// barrier, a workgroup-uniform load, a subgroup operation, a
/// [`is_uniformity_constrained_expr`]): one walk gates a transformation whose
/// only risk is that analysis.
pub(crate) fn module_has_uniformity_constraint(module: &naga::Module) -> bool {
    all_functions(module).any(|f| {
        f.expressions
            .iter()
            .any(|(_, e)| is_uniformity_constrained_expr(e))
            || {
                let mut found = false;
                for_each_statement(&f.body, &mut |s| {
                    found |= matches!(
                        s,
                        naga::Statement::ControlBarrier(_)
                            | naga::Statement::MemoryBarrier(_)
                            | naga::Statement::WorkGroupUniformLoad { .. }
                            | naga::Statement::SubgroupBallot { .. }
                            | naga::Statement::SubgroupGather { .. }
                            | naga::Statement::SubgroupCollectiveOperation { .. }
                    );
                });
                found
            }
    })
}

// MARK: Expression classifiers

/// Whether `expression` must sit inside an `Emit` range: declarative
/// references and statement-attached results (`ExprClass::PRE_EMIT`) are
/// produced implicitly, every computation is not.
pub fn expression_needs_emit(expression: &naga::Expression) -> bool {
    !ExprClass::node(expression).any(ExprClass::PRE_EMIT)
}

/// Whether `expression` cannot be cloned into a caller during inlining:
/// `LocalVariable` names a function-scoped slot, and
/// `ExprClass::NOT_RELOCATABLE` exists only at its statement or depends
/// on a cursor / lane state.  `GlobalVariable` remaps 1-to-1 and
/// `FunctionArgument` is substituted from the call site, so both are allowed.
pub fn is_disallowed_inline_expression(expression: &naga::Expression) -> bool {
    ExprClass::node(expression).any(ExprClass::NOT_RELOCATABLE | ExprClass::LOCAL_REF)
}

/// The operand of `expression` evaluated CONDITIONALLY: the right side of a
/// short-circuit `&&` / `||`, skipped whenever the left side decides.  Any
/// relocation that needs its landing position evaluated exactly once per
/// evaluation of the parent (a single-use call sunk into its consumer, a
/// barrier preload inlined into a `for` header) must refuse it; `Select` and
/// every other operator evaluate all operands.
pub fn short_circuit_rhs(expression: &naga::Expression) -> Option<naga::Handle<naga::Expression>> {
    match expression {
        naga::Expression::Binary {
            op: naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr,
            right,
            ..
        } => Some(*right),
        _ => None,
    }
}

/// The expression `stmt` binds, if any: a statement-attached result
/// (`CallResult`, `AtomicResult`, ...) has no text of its own and can only be
/// referenced by name AFTER its statement.
pub fn statement_result(stmt: &naga::Statement) -> Option<naga::Handle<naga::Expression>> {
    use naga::Statement as S;
    match stmt {
        S::Call { result, .. } | S::Atomic { result, .. } => *result,
        S::WorkGroupUniformLoad { result, .. }
        | S::SubgroupBallot { result, .. }
        | S::SubgroupGather { result, .. }
        | S::SubgroupCollectiveOperation { result, .. } => Some(*result),
        S::RayQuery {
            fun: naga::RayQueryFunction::Proceed { result },
            ..
        } => Some(*result),
        S::RayQuery { .. }
        | S::RayPipelineFunction(_)
        | S::Emit(_)
        | S::Block(_)
        | S::If { .. }
        | S::Switch { .. }
        | S::Loop { .. }
        | S::Break
        | S::Continue
        | S::Return { .. }
        | S::Kill
        | S::ControlBarrier(_)
        | S::MemoryBarrier(_)
        | S::Store { .. }
        | S::ImageStore { .. }
        | S::ImageAtomic { .. }
        | S::CooperativeStore { .. } => None,
    }
}

// MARK: Literal predicates

/// Convert a width-8 literal (`F64` / `U64` / `I64`) to `target` (f32 / f16 /
/// i32 / u32 / bool), `None` otherwise - f64 / i64 / u64 targets included,
/// since naga accepts those forms.  naga's frontend refuses to const-fold these casts
/// (f64 / u64 / i64 are non-standard WGSL), so the `As` node survives and an
/// emitted `f32(<F64 literal>)` is rejected on re-parse.  Semantics mirror
/// `naga::proc::ConstantEvaluator::cast`: a float source rounds to nearest
/// (declined when non-finite) and CLAMPS to integer range; an integer source
/// WRAPS (`i64(-1) -> u32` is `4294967295u`); `-> bool` is `v != 0`.
///
/// Shared because both sides of the round-trip need it: `const_fold` folds
/// the scalar `As`, and the generator converts the components of a vector
/// narrowing `materialize_vector` cannot, needing each converted literal as
/// an arena handle.
pub(crate) fn cast_width8_to(src: naga::Literal, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    if let L::F64(v) = src {
        return match (target.kind, target.width) {
            (K::Float, 4) => {
                let r = v as f32;
                r.is_finite().then_some(L::F32(r))
            }
            (K::Float, 2) => {
                let r = half::f16::from_f64(v);
                r.is_finite().then_some(L::F16(r))
            }
            (K::Sint, 4) => Some(L::I32(v.clamp(i32::MIN as f64, i32::MAX as f64) as i32)),
            (K::Uint, 4) => Some(L::U32(v.clamp(u32::MIN as f64, u32::MAX as f64) as u32)),
            (K::Bool, _) => Some(L::Bool(v != 0.0)),
            _ => None,
        };
    }
    let v: i128 = match src {
        L::U64(v) => v as i128,
        L::I64(v) => v as i128,
        _ => return None,
    };
    match (target.kind, target.width) {
        (K::Float, 4) => Some(L::F32(v as f32)),
        (K::Float, 2) => narrow_to_f16(v as f32),
        (K::Sint, 4) => Some(L::I32(v as i32)),
        (K::Uint, 4) => Some(L::U32(v as u32)),
        (K::Bool, _) => Some(L::Bool(v != 0)),
        _ => None,
    }
}

pub(crate) fn is_bool_true(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

pub(crate) fn is_bool_false(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Const-ness of `expr` itself as a WGSL const-expression: `Some` decides,
/// `None` defers to the operands ([`ExprClass::const_leaf`]).  Children
/// precede parents in a naga arena, so one forward pass makes it per-handle.
/// `load_dedup` declines only where this says const and `const_fold` only
/// where it says runtime, so either guess stops a decline that matters.
pub(crate) fn const_expression_leaf(expr: &naga::Expression) -> Option<bool> {
    ExprClass::node(expr).const_leaf()
}

/// Per handle, whether the cone reads as a WGSL const-expression, in one
/// forward pass (children precede parents in a naga arena): a leaf by
/// [`const_expression_leaf`], a node by every child, either overridden by
/// `leaf` (a read a pending rewrite turns const, a forward's target), which
/// sees the entries already filled.
pub(crate) fn const_cones(
    arena: &naga::Arena<naga::Expression>,
    mut leaf: impl FnMut(naga::Handle<naga::Expression>, &naga::Expression, &[bool]) -> Option<bool>,
) -> Vec<bool> {
    let mut is_const = vec![false; arena.len()];
    for (handle, expr) in arena.iter() {
        is_const[handle.index()] =
            match leaf(handle, expr, &is_const).or_else(|| const_expression_leaf(expr)) {
                Some(known) => known,
                None => {
                    let mut all = true;
                    visit_expression_children(expr, |child| all &= is_const[child.index()]);
                    all
                }
            };
    }
    is_const
}

/// `true` when a `ZeroValue` of `ty` is a FLOAT zero - the only kind a
/// `-x` / `x * y` / `x / y` can re-sign.  An integer `vec4i()` must answer
/// false or the untyped sign-sensitive roots stop being inert and the guard
/// declines integer slots that have no signed zero at all.  Anything that is
/// not a numeric scalar / vector / matrix cannot be such an operand, so the
/// conservative answer costs nothing.
pub(crate) fn zero_value_is_float(
    types: &naga::UniqueArena<naga::Type>,
    ty: naga::Handle<naga::Type>,
) -> bool {
    match types[ty].inner {
        naga::TypeInner::Scalar(scalar) | naga::TypeInner::Vector { scalar, .. } => matches!(
            scalar.kind,
            naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat
        ),
        naga::TypeInner::Matrix { .. } => true,
        _ => true,
    }
}

/// A float zero of either sign: what an enclosing `-x` / `x * y` / `x / y`
/// is free to re-sign, whatever this literal's own sign is.
pub(crate) fn is_float_zero_literal(lit: &naga::Literal) -> bool {
    match *lit {
        naga::Literal::F32(v) => v == 0.0,
        naga::Literal::F64(v) | naga::Literal::AbstractFloat(v) => v == 0.0,
        naga::Literal::F16(v) => v.to_f32() == 0.0,
        _ => false,
    }
}

/// Any float literal, of any width.
pub(crate) fn is_float_literal(lit: &naga::Literal) -> bool {
    matches!(
        *lit,
        naga::Literal::F32(_)
            | naga::Literal::F64(_)
            | naga::Literal::AbstractFloat(_)
            | naga::Literal::F16(_)
    )
}

/// A literal `-0.0` of any float width.  WGSL leaves a zero's sign to the
/// implementation, and Dawn on Metal flushes a LITERAL `-0.0` to `+0.0` while
/// keeping the sign of a runtime negation: the asymmetry every `-0.0` guard
/// in the passes exists to respect.
pub(crate) fn is_negative_zero_literal(lit: &naga::Literal) -> bool {
    match *lit {
        naga::Literal::F32(v) => v == 0.0 && v.is_sign_negative(),
        naga::Literal::F64(v) | naga::Literal::AbstractFloat(v) => v == 0.0 && v.is_sign_negative(),
        naga::Literal::F16(v) => v.to_bits() == 0x8000,
        _ => false,
    }
}

/// `true` when `h` may carry a `-0.0` only a const-evaluator sees: a literal,
/// a splat / compose of one, or a module-scope value this arena cannot
/// resolve.  Substituting one for a runtime read (forwarding, inlining) flips
/// the sign the input shipped.  `Constant` / `Override` resolve through
/// `module.global_expressions`, which the mutating passes do not hold, and an
/// override has no fixed value at all - both decline.
pub fn has_negative_zero_leaf(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    has_negative_zero_leaf_through(arena, h, &|h| h)
}

/// [`has_negative_zero_leaf`] over the arena as a pending rewrite will leave
/// it: every handle is read through `resolve` (an earlier splice's
/// replacement, a forward), so a value that arrives by substitution counts.
pub(crate) fn has_negative_zero_leaf_through(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
    resolve: &dyn Fn(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) -> bool {
    match &arena[resolve(h)] {
        naga::Expression::Literal(lit) => is_negative_zero_literal(lit),
        naga::Expression::Constant(_) | naga::Expression::Override(_) => true,
        naga::Expression::Splat { value, .. } => {
            has_negative_zero_leaf_through(arena, *value, resolve)
        }
        naga::Expression::Compose { components, .. } => components
            .iter()
            .any(|&c| has_negative_zero_leaf_through(arena, c, resolve)),
        _ => false,
    }
}

/// The operators whose value depends on being a const-expression: float
/// `-x`, `x * y`, `x / y`, whose zero result takes its sign from the operands
/// (Dawn on Metal flushes a const `-0.0`), and float `x % y`, exact `fmod`
/// as a const-expression but the stepwise `x - y * trunc(x / y)` at run
/// time, a full divisor apart when the rounded quotient crosses an integer
/// the exact one does not.  `+` / `-` are absent: their zero has one
/// determined sign either way.  Rooted at the OPERATOR: it is const only once
/// every operand is, so a guard walking down from it sees a zero in the
/// SIBLING of the operand that crosses.  Untyped on purpose: an integer slot
/// carries no float leaf, so [`float_leaf_bits`] leaves the extra roots
/// inert (a `Constant` / `Override` there costs a decline, never a
/// miscompile).  [`ExprClass::SIGN_SENSITIVE_OP`].
pub(crate) fn is_sign_sensitive_op(expr: &naga::Expression) -> bool {
    ExprClass::node(expr).any(ExprClass::SIGN_SENSITIVE_OP)
}

/// Leaf classes a [`float_leaf_bits`] cone can carry.
pub(crate) const LEAF_FLOAT_ZERO: u8 = 1;
pub(crate) const LEAF_FLOAT: u8 = 2;

/// The leaves an [`is_sign_sensitive_op`] slot must reach for its const-ness
/// to change its value: a zero of either sign for `-x` / `x * y` / `x / y`,
/// any float at all for `x % y`.
pub(crate) fn sensitive_leaves(expr: &naga::Expression) -> u8 {
    match expr {
        naga::Expression::Binary {
            op: naga::BinaryOperator::Modulo,
            ..
        } => LEAF_FLOAT,
        _ => LEAF_FLOAT_ZERO,
    }
}

/// Whether an [`is_sign_sensitive_op`] `op` whose cone has just turned const
/// reads differently, from the cone's [`float_leaf_bits`] `leaves` and,
/// where they do not decide, `cone`: the cone evaluated as the rewrite
/// leaves it, to "holds a `-0.0` lane" (`None`: beyond the evaluator).  A
/// [`sensitive_leaves`] hit decides and an integer cone has no zero sign to
/// lose; a float cone without a zero leaf is evaluated, a leaf being the
/// cheap sign of a negative zero but not its only source (`-(p - 1.0)` with
/// `p = 1.0`, `-f32(p)` with `p = 0u`), and one beyond the evaluator is
/// taken as changing, since tint evaluates it whatever this one models.
pub(crate) fn const_sign_changes(
    op: &naga::Expression,
    leaves: u8,
    cone: impl FnOnce() -> Option<bool>,
) -> bool {
    if leaves & sensitive_leaves(op) != 0 {
        true
    } else if leaves & LEAF_FLOAT == 0 {
        false
    } else {
        cone().unwrap_or(true)
    }
}

/// The leaf bits of `expr`: none for a non-leaf, so a cone's bits are the OR
/// over its nodes.  A `ZeroValue` IS a zero and a `Constant` / `Override`
/// hides its value from the passes, so both carry both bits, as
/// [`has_negative_zero_leaf`] treats them.  A conversion or unpacking that
/// makes a float of integers (`f32(0u & 0xFFFFu)`, `unpack4x8snorm(p).x`)
/// carries [`LEAF_FLOAT`] as the cone's float-ness, so a zero it computes
/// is judged by evaluating the cone rather than missed.
pub(crate) fn float_leaf_bits(
    expr: &naga::Expression,
    types: &naga::UniqueArena<naga::Type>,
) -> u8 {
    use naga::MathFunction as M;
    match expr {
        naga::Expression::Literal(lit) if is_float_zero_literal(lit) => {
            LEAF_FLOAT_ZERO | LEAF_FLOAT
        }
        naga::Expression::Literal(lit) if is_float_literal(lit) => LEAF_FLOAT,
        naga::Expression::ZeroValue(ty) if zero_value_is_float(types, *ty) => {
            LEAF_FLOAT_ZERO | LEAF_FLOAT
        }
        naga::Expression::Constant(_) | naga::Expression::Override(_) => {
            LEAF_FLOAT_ZERO | LEAF_FLOAT
        }
        naga::Expression::As {
            kind: naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat,
            ..
        } => LEAF_FLOAT,
        naga::Expression::Math {
            fun:
                M::Unpack4x8snorm
                | M::Unpack4x8unorm
                | M::Unpack2x16snorm
                | M::Unpack2x16unorm
                | M::Unpack2x16float,
            ..
        } => LEAF_FLOAT,
        _ => 0,
    }
}

/// A literal's identity as a hash key: its kind and its bits, so `-0.0`
/// and `+0.0` - and NaNs - never collide, and so a fold that treats two
/// literals as one value never merges IEEE values a shader can tell apart.
pub(crate) type KeyToken = (u8, u64);

pub(crate) fn lit_key(l: naga::Literal) -> KeyToken {
    use naga::Literal as L;
    match l {
        L::F32(v) => (0, v.to_bits() as u64),
        L::F64(v) => (1, v.to_bits()),
        L::F16(v) => (2, v.to_bits() as u64),
        L::I32(v) => (3, v as u32 as u64),
        L::U32(v) => (4, v as u64),
        L::I64(v) => (5, v as u64),
        L::U64(v) => (6, v),
        L::Bool(v) => (7, v as u64),
        L::AbstractInt(v) => (8, v as u64),
        L::AbstractFloat(v) => (9, v.to_bits()),
        L::I16(v) => (10, v as u16 as u64),
        L::U16(v) => (11, v as u64),
    }
}

/// `lit_key` equality.
pub fn literal_bit_eq(a: &naga::Literal, b: &naga::Literal) -> bool {
    lit_key(*a) == lit_key(*b)
}

/// Live consumers of an expression: `u32`, since no shader refers to one
/// value four billion times and the vectors are half the size.
pub(crate) type RefCount = u32;

/// Per-handle reference counts plus the liveness bitmap, in one walk.  Live
/// is "materialised by an `Emit` range", so dead code's children score zero
/// and never claim a short name.  A statement RESULT is not a use - it stays
/// at 0 until something consumes it, which single-use call inlining and
/// dead-`let` elision key on - nor is an `Emit` handle, emission being
/// sequencing.
///
/// The generator prices its output by these counts and `rename` spends short
/// names by them; the two MUST agree, or a name the generator inlines away
/// takes the shortest identifier with it.
pub fn live_expression_ref_counts(function: &naga::Function) -> (Vec<RefCount>, Vec<bool>) {
    let len = function.expressions.len();

    let mut live = vec![false; len];
    for_each_statement(&function.body, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                live[h.index()] = true;
            }
        }
    });

    let mut counts = vec![0 as RefCount; len];
    for (h, expr) in function.expressions.iter() {
        if live[h.index()] {
            visit_expression_children(expr, |child| counts[child.index()] += 1);
        }
    }
    for_each_statement(&function.body, &mut |stmt| {
        visit_statement_operands(
            stmt,
            /*include_emit_handles=*/ false,
            &mut |h| counts[h.index()] += 1,
        );
    });

    (counts, live)
}

/// A literal that turns a legal RUNTIME `/` `%` into a WGSL shader-creation
/// error once it lands in the divisor.  Forwarding and folding both decline
/// on it: either would build IR naga rejects, costing the driver's rollback
/// a whole pass run.  Only INTEGER division qualifies - float `x / 0.0` is a
/// defined `inf` / `nan`.
pub(crate) fn is_integer_zero_literal(lit: &naga::Literal) -> bool {
    matches!(
        lit,
        naga::Literal::I16(0)
            | naga::Literal::U16(0)
            | naga::Literal::I32(0)
            | naga::Literal::U32(0)
            | naga::Literal::I64(0)
            | naga::Literal::U64(0)
            | naga::Literal::AbstractInt(0)
    )
}

/// [`cast_width8_to`]'s counterpart for the 32-bit and `f16` sources: what
/// naga's front-end folds at parse and a pass re-creates by putting a
/// literal under a runtime conversion.  Only exactly-rounded arms: a float
/// truncates into an integer when finite and in range (out of range or
/// NaN is declined, as tint and naga disagree with each other about it), an
/// integer rounds to nearest even into `f32` as `as` and both front-ends
/// do, `i32` / `u32` reinterpret each other, `f16` widens exactly, an `f16`
/// target rounds to nearest even as both front-ends do (an overflow into
/// infinity declines: naga's validator lets a non-finite `f16` through,
/// tint does not), and an identity conversion is the literal itself.
pub(crate) fn cast_width4_to(src: naga::Literal, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    let float = match src {
        L::F32(v) => Some(v),
        L::F16(v) => Some(v.to_f32()),
        _ => None,
    };
    if let Some(v) = float {
        let wide = f64::from(v);
        return match (target.kind, target.width) {
            (K::Float, 4) => Some(L::F32(v)),
            (K::Float, 2) => narrow_to_f16(v),
            (K::Sint, 4) if wide >= f64::from(i32::MIN) && wide < -f64::from(i32::MIN) => {
                Some(L::I32(v as i32))
            }
            (K::Uint, 4) if wide >= 0.0 && wide < f64::from(u32::MAX) + 1.0 => {
                Some(L::U32(v as u32))
            }
            (K::Bool, _) if v.is_finite() => Some(L::Bool(v != 0.0)),
            _ => None,
        };
    }
    let int: i64 = match src {
        L::I32(v) => i64::from(v),
        L::U32(v) => i64::from(v),
        _ => return None,
    };
    match (target.kind, target.width) {
        (K::Float, 4) => Some(L::F32(int as f32)),
        (K::Float, 2) => narrow_to_f16(int as f32),
        (K::Sint, 4) => Some(L::I32(int as i32)),
        (K::Uint, 4) => Some(L::U32(int as u32)),
        (K::Bool, _) => Some(L::Bool(int != 0)),
        _ => None,
    }
}

/// `f16(v)` as both front-ends fold it (`half::f16::from_f32`, ties to
/// even); a value past `f16::MAX` rounds to infinity and declines.
fn narrow_to_f16(v: f32) -> Option<naga::Literal> {
    let r = half::f16::from_f32(v);
    r.is_finite().then_some(naga::Literal::F16(r))
}

/// [`is_integer_zero_literal`]'s counterpart: a shift by `>= 32` overruns the
/// ubiquitous 32-bit operand.  The operand width is out of reach here, so a
/// 16-bit one shifted by `[16, 32)` slips through to the whole-module
/// rollback and a 64-bit one is over-declined; declining only ever costs a
/// missed optimization.
pub(crate) fn shift_amount_is_static_error(lit: &naga::Literal) -> bool {
    let amount: i128 = match lit {
        naga::Literal::U32(v) => i128::from(*v),
        naga::Literal::U16(v) => i128::from(*v),
        naga::Literal::I32(v) => i128::from(*v),
        naga::Literal::I16(v) => i128::from(*v),
        naga::Literal::U64(v) => i128::from(*v),
        naga::Literal::I64(v) => i128::from(*v),
        naga::Literal::AbstractInt(v) => i128::from(*v),
        _ => return false,
    };
    amount >= 32
}

/// [`is_integer_zero_literal`]'s and [`shift_amount_is_static_error`]'s
/// counterpart for the index slot: a const-expression index below zero, or
/// at or past the base's static length (`len == 0`: a runtime-sized base,
/// where only a negative index errors; `1`: an override-sized one, where
/// only index 0 is in bounds for every value a host may set - see
/// [`static_index_bound`]).  tint and naga's WGSL front-end reject it; naga's
/// IR validator sees only a literal or `const` index (`get_const_val_from`
/// folds no arithmetic), so a manufactured `a[4 + 1]` validates and fails
/// only at the text round-trip unless asked here first.
pub(crate) fn index_is_static_error(len: u32, lit: &naga::Literal) -> bool {
    let index: i128 = match lit {
        naga::Literal::U32(v) => i128::from(*v),
        naga::Literal::U16(v) => i128::from(*v),
        naga::Literal::I32(v) => i128::from(*v),
        naga::Literal::I16(v) => i128::from(*v),
        naga::Literal::U64(v) => i128::from(*v),
        naga::Literal::I64(v) => i128::from(*v),
        naga::Literal::AbstractInt(v) => i128::from(*v),
        _ => return false,
    };
    index < 0 || (len != 0 && index >= i128::from(len))
}

/// The bound a const-expression index into `inner` is judged against: the
/// element count of a vector, matrix or fixed-size array (through a pointer
/// too); `0` for a runtime-sized array, which clamps at run time; `1` for
/// an override-sized one ([`IndexBound::pending`]); `None` for a type an
/// index cannot reach.
pub(crate) fn static_index_bound(
    inner: &naga::TypeInner,
    module: &naga::Module,
) -> Option<IndexBound> {
    match inner.indexable_length(module) {
        Ok(naga::proc::IndexableLength::Known(n)) => Some(IndexBound {
            len: n,
            pending: false,
            unsigned_index: false,
        }),
        Ok(naga::proc::IndexableLength::Dynamic) => Some(IndexBound {
            len: 0,
            pending: false,
            unsigned_index: false,
        }),
        Err(naga::proc::IndexableLengthError::Pending(_)) => Some(IndexBound {
            len: 1,
            pending: true,
            unsigned_index: false,
        }),
        Err(_) => None,
    }
}

/// [`static_index_bound`]'s answer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct IndexBound {
    /// As [`index_is_static_error`] judges it.
    pub(crate) len: u32,
    /// The base is an override-sized array, whose count Dawn substitutes at
    /// pipeline creation, rejecting a const index at or past it: `len` is
    /// 1 (only index 0 is in bounds for every count), the writers decline
    /// the slot and the generator binds it, but the driver's post-pass
    /// check must not report it - naga's front-end folds `let i = 8u;
    /// wg[i]` into the shape at parse, so the input itself carries it.
    pub(crate) pending: bool,
    /// The index is unsigned: into a runtime-sized base it has no error to
    /// reach, so a value beyond the evaluator passes.
    pub(crate) unsigned_index: bool,
}

impl IndexBound {
    /// The bound with the index's scalar kind recorded.
    pub(crate) fn with_index(self, index: &naga::TypeInner) -> IndexBound {
        IndexBound {
            unsigned_index: matches!(
                index.scalar(),
                Some(naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    ..
                })
            ),
            ..self
        }
    }
}

/// Per `Access` expression of `function`, [`static_index_bound`] of its base
/// (`None` for any other handle); empty when the arena holds no `Access`.
/// Sized by naga's typifier, so the base's declared type answers, whether it
/// is a parameter, a local, a member chain or a value.
pub(crate) fn access_static_lengths(
    function: &naga::Function,
    module: &naga::Module,
) -> Vec<Option<IndexBound>> {
    let arena = &function.expressions;
    if !arena
        .iter()
        .any(|(_, e)| matches!(e, naga::Expression::Access { .. }))
    {
        return Vec::new();
    }
    let mut typifier = naga::front::Typifier::new();
    let resolve = naga::proc::ResolveContext::with_locals(
        module,
        &function.local_variables,
        &function.arguments,
    );
    let mut lens = vec![None; arena.len()];
    for (h, expr) in arena.iter() {
        let naga::Expression::Access { base, index } = *expr else {
            continue;
        };
        if typifier.grow(base, arena, &resolve).is_err()
            || typifier.grow(index, arena, &resolve).is_err()
        {
            continue;
        }
        lens[h.index()] = static_index_bound(typifier[base].inner_with(&module.types), module)
            .map(|bound| bound.with_index(typifier[index].inner_with(&module.types)));
    }
    lens
}

/// Root local of a pointer expression: the `LocalVariable` an
/// `Access` / `AccessIndex` chain bottoms out in, `None` for any other root
/// (a global, a function-argument pointer).  The one definition: liveness,
/// dead-store elimination and the generator's deferred-variable analysis
/// must agree on which local a pointer names, or one drops a store the
/// other still reads.
pub fn root_local_var(
    pointer: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match &expressions[pointer] {
        naga::Expression::LocalVariable(local) => Some(*local),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            root_local_var(*base, expressions)
        }
        _ => None,
    }
}

/// The value of an integer `Literal` index; a `u64` past `i64::MAX`
/// saturates, which is out of bounds for any composite.
pub fn const_index_value(
    handle: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<i64> {
    match arena[handle] {
        naga::Expression::Literal(naga::Literal::I32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::U32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::I64(v)) => Some(v),
        naga::Expression::Literal(naga::Literal::U64(v)) => {
            Some(i64::try_from(v).unwrap_or(i64::MAX))
        }
        naga::Expression::Literal(naga::Literal::AbstractInt(v)) => Some(v),
        _ => None,
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A real handle for variants whose fields the classifiers never inspect.
    fn dummy_handles() -> (
        naga::Arena<naga::Expression>,
        naga::Handle<naga::Expression>,
    ) {
        let mut arena = naga::Arena::<naga::Expression>::new();
        let h = arena.append(
            naga::Expression::Literal(naga::Literal::I32(0)),
            naga::Span::UNDEFINED,
        );
        (arena, h)
    }

    fn dummy_type_handle() -> naga::Handle<naga::Type> {
        let mut u = naga::UniqueArena::<naga::Type>::new();
        u.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::I32),
            },
            naga::Span::UNDEFINED,
        )
    }

    fn dummy_const_handle() -> naga::Handle<naga::Constant> {
        let mut a = naga::Arena::<naga::Constant>::new();
        let ty = dummy_type_handle();
        a.append(
            naga::Constant {
                name: None,
                ty,
                init: {
                    let (_, h) = dummy_handles();
                    h
                },
            },
            naga::Span::UNDEFINED,
        )
    }

    /// The key is the kind and the bits: `-0.0` and `+0.0` differ, NaN
    /// payloads differ, one value under two kinds differs.
    #[test]
    fn a_literals_key_is_its_kind_and_bits() {
        use naga::Literal as L;
        assert_ne!(lit_key(L::F32(0.0)), lit_key(L::F32(-0.0)));
        let (nan_a, nan_b) = (f32::from_bits(0x7FC0_0000), f32::from_bits(0x7FC0_0001));
        assert_ne!(lit_key(L::F32(nan_a)), lit_key(L::F32(nan_b)));
        assert!(literal_bit_eq(&L::F32(nan_a), &L::F32(nan_a)));
        assert_ne!(lit_key(L::I32(1)), lit_key(L::U32(1)));
        assert_eq!(lit_key(L::AbstractInt(-1)), lit_key(L::AbstractInt(-1)));
    }

    #[test]
    fn needs_emit_declarative_false() {
        assert!(!expression_needs_emit(&naga::Expression::Literal(
            naga::Literal::I32(0)
        )));
        assert!(!expression_needs_emit(&naga::Expression::Constant(
            dummy_const_handle()
        )));
        assert!(!expression_needs_emit(&naga::Expression::ZeroValue(
            dummy_type_handle()
        )));
        assert!(!expression_needs_emit(&naga::Expression::FunctionArgument(
            0
        )));
    }

    #[test]
    fn needs_emit_statement_results_false() {
        let (_, h) = dummy_handles();
        assert!(!expression_needs_emit(&naga::Expression::CallResult(
            naga::Arena::<naga::Function>::new()
                .append(naga::Function::default(), naga::Span::UNDEFINED)
        )));
        let _ = h;
        assert!(!expression_needs_emit(
            &naga::Expression::RayQueryProceedResult
        ));
        assert!(!expression_needs_emit(
            &naga::Expression::SubgroupBallotResult
        ));
    }

    #[test]
    fn needs_emit_computational_true() {
        let (_, h) = dummy_handles();
        assert!(expression_needs_emit(&naga::Expression::AccessIndex {
            base: h,
            index: 0
        }));
        assert!(expression_needs_emit(&naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: h
        }));
        assert!(expression_needs_emit(&naga::Expression::Load {
            pointer: h
        }));
        assert!(expression_needs_emit(&naga::Expression::ArrayLength(h)));
    }

    #[test]
    fn disallowed_inline_statement_results_true() {
        let mut locals = naga::Arena::<naga::LocalVariable>::new();
        let lv = locals.append(
            naga::LocalVariable {
                name: None,
                ty: dummy_type_handle(),
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        assert!(is_disallowed_inline_expression(
            &naga::Expression::LocalVariable(lv)
        ));
        assert!(is_disallowed_inline_expression(
            &naga::Expression::RayQueryProceedResult
        ));
        assert!(is_disallowed_inline_expression(
            &naga::Expression::SubgroupBallotResult
        ));
    }

    #[test]
    fn disallowed_inline_declarative_false() {
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::Literal(naga::Literal::I32(0))
        ));
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::FunctionArgument(0)
        ));

        let mut globals = naga::Arena::<naga::GlobalVariable>::new();
        let gv = globals.append(
            naga::GlobalVariable {
                name: None,
                space: naga::AddressSpace::Private,
                binding: None,
                ty: dummy_type_handle(),
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::GlobalVariable(gv)
        ));
    }

    #[test]
    fn disallowed_inline_computational_false() {
        let (_, h) = dummy_handles();
        assert!(!is_disallowed_inline_expression(&naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: h,
        }));
        assert!(!is_disallowed_inline_expression(&naga::Expression::Load {
            pointer: h,
        }));
    }
}
