use super::*;
use crate::config::Config;

/// Test shim for [`fold_local_expressions`].  Unit tests build bare
/// expression arenas with no `Function` body, so there are no real
/// `Emit` ranges to feed the store-aware relocation guard.  Mapping
/// every handle to a single shared range models the safe "all
/// co-located, no intervening statement" case, so the guard is a
/// no-op and these tests keep exercising the folding logic they
/// target.  Tests that specifically need a cross-range (hazardous)
/// layout call [`fold_local_expressions`] directly with a custom map.
fn fold_local(
    arena: &mut naga::Arena<naga::Expression>,
    refcounts: &[u32],
    const_literals: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    types: &naga::UniqueArena<naga::Type>,
    vector_type_cache: &HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> (HashSet<naga::Handle<naga::Expression>>, usize) {
    let ranges: HashMap<naga::Handle<naga::Expression>, usize> =
        arena.iter().map(|(h, _)| (h, 0usize)).collect();
    fold_local_expressions(
        arena,
        refcounts,
        &ranges,
        const_literals,
        types,
        vector_type_cache,
    )
}

#[test]
fn cast_width8_to_matches_wgsl_value_conversion() {
    use naga::Literal as L;
    let f16 = naga::Scalar {
        kind: naga::ScalarKind::Float,
        width: 2,
    };
    let f64t = naga::Scalar {
        kind: naga::ScalarKind::Float,
        width: 8,
    };
    let i64t = naga::Scalar {
        kind: naga::ScalarKind::Sint,
        width: 8,
    };

    // f64 source: round-to-nearest (f32), CLAMP-then-truncate (int)
    assert_eq!(
        cast_width8_to(L::F64(0.5), naga::Scalar::F32),
        Some(L::F32(0.5))
    );
    assert_eq!(
        cast_width8_to(L::F64(2.5), naga::Scalar::I32),
        Some(L::I32(2))
    );
    assert_eq!(
        cast_width8_to(L::F64(-2.9), naga::Scalar::I32),
        Some(L::I32(-2))
    );
    assert_eq!(
        cast_width8_to(L::F64(3.9), naga::Scalar::U32),
        Some(L::U32(3))
    );
    assert_eq!(
        cast_width8_to(L::F64(-5.0), naga::Scalar::U32),
        Some(L::U32(0))
    );
    assert_eq!(
        cast_width8_to(L::F64(1e30), naga::Scalar::I32),
        Some(L::I32(i32::MAX))
    );
    assert_eq!(
        cast_width8_to(L::F64(0.0), naga::Scalar::BOOL),
        Some(L::Bool(false))
    );
    assert_eq!(
        cast_width8_to(L::F64(1.5), naga::Scalar::BOOL),
        Some(L::Bool(true))
    );
    // f64 declines: non-finite f32 result, and f16/f64 targets.
    assert_eq!(cast_width8_to(L::F64(1e308), naga::Scalar::F32), None);
    assert_eq!(cast_width8_to(L::F64(0.5), f16), None);
    assert_eq!(cast_width8_to(L::F64(0.5), f64t), None);

    // u64 source: WRAP (low 32 bits / mod 2^32), NOT clamp
    assert_eq!(
        cast_width8_to(L::U64(107), naga::Scalar::U32),
        Some(L::U32(107))
    );
    assert_eq!(
        cast_width8_to(L::U64(1 << 32), naga::Scalar::U32),
        Some(L::U32(0))
    );
    assert_eq!(
        cast_width8_to(L::U64(1 << 32), naga::Scalar::I32),
        Some(L::I32(0))
    );
    assert_eq!(
        cast_width8_to(L::U64(5), naga::Scalar::F32),
        Some(L::F32(5.0))
    );
    assert_eq!(
        cast_width8_to(L::U64(0), naga::Scalar::BOOL),
        Some(L::Bool(false))
    );
    assert_eq!(
        cast_width8_to(L::U64(3), naga::Scalar::BOOL),
        Some(L::Bool(true))
    );

    // i64 source: WRAP; `i64(-1) -> u32` is 4294967295, not 0
    assert_eq!(
        cast_width8_to(L::I64(-1), naga::Scalar::U32),
        Some(L::U32(u32::MAX))
    );
    assert_eq!(
        cast_width8_to(L::I64(-1), naga::Scalar::I32),
        Some(L::I32(-1))
    );
    assert_eq!(
        cast_width8_to(L::I64((1 << 33) + 7), naga::Scalar::I32),
        Some(L::I32(7))
    );
    assert_eq!(
        cast_width8_to(L::I64(-5), naga::Scalar::F32),
        Some(L::F32(-5.0))
    );
    assert_eq!(
        cast_width8_to(L::I64(0), naga::Scalar::BOOL),
        Some(L::Bool(false))
    );
    // int -> f16 / i64 declined (naga accepts those forms).
    assert_eq!(cast_width8_to(L::U64(5), f16), None);
    assert_eq!(cast_width8_to(L::I64(5), i64t), None);

    // Non-width-8 sources are never folded here.
    assert_eq!(cast_width8_to(L::I32(5), naga::Scalar::U32), None);
}

fn run_pass(module: &mut naga::Module) -> bool {
    let mut pass = ConstFoldPass;
    let config = Config::default();
    let ctx = PassContext {
        config: &config,
        trace_run_dir: None,
    };

    pass.run(module, &ctx).expect("const fold pass should run")
}

fn assert_f32_literal(
    arena: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    expected: f32,
) {
    match arena[handle] {
        naga::Expression::Literal(naga::Literal::F32(v)) => {
            assert!((v - expected).abs() < f32::EPSILON)
        }
        ref other => panic!("expected f32 literal {expected}, got {other:?}"),
    }
}

#[test]
fn folds_binary_add_in_local_expression_arena() {
    let mut arena = naga::Arena::new();
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let two = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: one,
            right: two,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        changed.len(),
        1,
        "one binary add expression should be folded"
    );
    assert_f32_literal(&arena, add, 3.0);
}

#[test]
fn folds_nested_binary_expressions() {
    let mut arena = naga::Arena::new();
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let two = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let three = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: one,
            right: two,
        },
        Default::default(),
    );
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: add,
            right: three,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(changed.len(), 2, "both nested operations should be folded");
    assert_f32_literal(&arena, add, 3.0);
    assert_f32_literal(&arena, mul, 9.0);
}

#[test]
fn de_morgan_negated_equality_folds_in_place() {
    use naga::{BinaryOperator as B, UnaryOperator as U};
    // `!(a == b)` -> `a != b` and `!(a != b)` -> `a == b`: the `!` node
    // is rewritten into the flipped comparison reusing the operands in
    // place, and the now-dead comparison's Emit is dropped (added to the
    // folded set).  Operands are FunctionArguments so const-folding
    // cannot pre-empt the rewrite by collapsing the comparison.
    for (cmp_op, flipped) in [(B::Equal, B::NotEqual), (B::NotEqual, B::Equal)] {
        let mut arena = naga::Arena::new();
        let a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
        let cmp = arena.append(
            naga::Expression::Binary {
                op: cmp_op,
                left: a,
                right: b,
            },
            Default::default(),
        );
        let not = arena.append(
            naga::Expression::Unary {
                op: U::LogicalNot,
                expr: cmp,
            },
            Default::default(),
        );
        // Comparison (handle index 2) is uniquely owned by the `!`.
        let refcounts = vec![0u32, 0, 1, 0];
        let (folded, _) = fold_local(
            &mut arena,
            &refcounts,
            &HashMap::new(),
            &naga::UniqueArena::new(),
            &HashMap::new(),
        );
        assert!(
            matches!(
                arena[not],
                naga::Expression::Binary { op, left, right }
                    if op == flipped && left == a && right == b
            ),
            "`!({cmp_op:?})` should rewrite in place to `{flipped:?}`, got {:?}",
            arena[not]
        );
        assert!(
            folded.contains(&cmp),
            "the dead comparison's Emit must be dropped"
        );
    }
}

#[test]
fn de_morgan_shared_equality_is_not_folded() {
    use naga::{BinaryOperator as B, UnaryOperator as U};
    // A comparison referenced by more than the `!` (refcount > 1) is
    // left alone: rewriting it in place would corrupt the other
    // consumer, so the emitter keeps `!name`.
    let mut arena = naga::Arena::new();
    let a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
    let cmp = arena.append(
        naga::Expression::Binary {
            op: B::Equal,
            left: a,
            right: b,
        },
        Default::default(),
    );
    let not = arena.append(
        naga::Expression::Unary {
            op: U::LogicalNot,
            expr: cmp,
        },
        Default::default(),
    );
    // refcount 2 on the comparison -> shared, must NOT fold.
    let refcounts = vec![0u32, 0, 2, 0];
    let (folded, _) = fold_local(
        &mut arena,
        &refcounts,
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        matches!(
            arena[not],
            naga::Expression::Unary {
                op: U::LogicalNot,
                ..
            }
        ),
        "a shared comparison must stay `!(==)`, not fold to `!=`"
    );
    assert!(
        !folded.contains(&cmp),
        "a shared comparison's Emit must NOT be dropped"
    );
}

#[test]
fn folds_select_expression_with_literal_condition() {
    let mut arena = naga::Arena::new();
    let cond = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let accept = arena.append(
        naga::Expression::Literal(naga::Literal::F32(4.0)),
        Default::default(),
    );
    let reject = arena.append(
        naga::Expression::Literal(naga::Literal::F32(8.0)),
        Default::default(),
    );
    let select = arena.append(
        naga::Expression::Select {
            condition: cond,
            accept,
            reject,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        changed.len(),
        1,
        "select expression should fold to its accepted literal"
    );
    assert_f32_literal(&arena, select, 4.0);
}

#[test]
fn folds_local_constant_reference_using_cache() {
    let module = naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
    let (constant_handle, _) = module
        .constants
        .iter()
        .next()
        .expect("expected one constant in parsed module");

    let mut arena = naga::Arena::new();
    let c = arena.append(
        naga::Expression::Constant(constant_handle),
        Default::default(),
    );
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: c,
            right: one,
        },
        Default::default(),
    );

    let mut const_literals = HashMap::new();
    const_literals.insert(constant_handle, naga::Literal::F32(41.0));

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &const_literals,
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        changed.len(),
        2,
        "constant reference and dependent add should fold"
    );
    assert_f32_literal(&arena, c, 41.0);
    assert_f32_literal(&arena, add, 42.0);
}

#[test]
fn does_not_fold_divide_by_zero() {
    let mut arena = naga::Arena::new();
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let div = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Divide,
            left: one,
            right: zero,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(changed.len(), 0, "division by zero should not be folded");

    match arena[div] {
        naga::Expression::Binary {
            op: naga::BinaryOperator::Divide,
            left,
            right,
        } => {
            assert_eq!(left, one);
            assert_eq!(right, zero);
        }
        ref other => panic!("expected divide expression to remain, got {other:?}"),
    }
}

#[test]
fn unary_negate_rejects_non_finite_result() {
    // Naga's IR validator rejects `Literal::F32`/`F64` with NaN
    // or infinity values (`check_literal_value` returns
    // `LiteralError::NonFinite`).  The Negate fold therefore
    // refuses both NaN -> NaN and +/-Inf -> -/+Inf, even though
    // the latter is a valid IEEE operation - emitting a non-
    // finite literal would produce IR the validator rejects.
    // Tests both +Inf and NaN inputs.
    assert_eq!(
        eval_unary(
            naga::UnaryOperator::Negate,
            naga::Literal::F32(f32::INFINITY),
        ),
        None,
        "+Inf must not be folded by Negate (-Inf is non-finite per naga IR contract)"
    );
    assert_eq!(
        eval_unary(naga::UnaryOperator::Negate, naga::Literal::F32(f32::NAN),),
        None,
        "NaN must not be folded by Negate"
    );
    // Negative-finite still folds.
    assert_eq!(
        eval_unary(naga::UnaryOperator::Negate, naga::Literal::F32(-2.5),),
        Some(naga::Literal::F32(2.5))
    );
}

#[test]
fn abstract_int_divide_min_by_neg1_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::Divide,
        naga::Literal::AbstractInt(i64::MIN),
        naga::Literal::AbstractInt(-1),
    );
    assert_eq!(
        result, None,
        "i64::MIN / -1 overflows AbstractInt and should not be folded"
    );
}

#[test]
fn abstract_int_modulo_min_by_neg1_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::Modulo,
        naga::Literal::AbstractInt(i64::MIN),
        naga::Literal::AbstractInt(-1),
    );
    assert_eq!(
        result, None,
        "i64::MIN %% -1 overflows AbstractInt and should not be folded"
    );
}

// MARK: Math intrinsics WGSL-edge-case regressions
//
// Each of these pins a fold that was unsound vs. the WGSL spec:
// `min/max/clamp` must propagate NaN (Rust's `min`/`max` treat
// NaN as "missing"); `atan2(0,0)` is implementation-defined;
// `pow(0, 0)` and `pow(0, negative)` are undefined; `sign(NaN)`
// would have introduced a NaN-valued literal into the output.

#[test]
fn min_propagates_nan_does_not_fold() {
    let r = eval_math_scalar(
        naga::MathFunction::Min,
        naga::Literal::F32(f32::NAN),
        Some(naga::Literal::F32(1.0)),
        None,
    );
    assert_eq!(r, None, "min(NaN, x) must not fold to x");
}

#[test]
fn max_propagates_nan_does_not_fold() {
    let r = eval_math_scalar(
        naga::MathFunction::Max,
        naga::Literal::F32(1.0),
        Some(naga::Literal::F32(f32::NAN)),
        None,
    );
    assert_eq!(r, None, "max(x, NaN) must not fold to x");
}

#[test]
fn clamp_propagates_nan_does_not_fold() {
    // NaN in any of v / lo / hi - all three must refuse fold so
    // we never emit a NaN literal and never invoke
    // `f32::clamp(NaN, ...)` which panics.
    for (v, lo, hi) in [
        (f32::NAN, 0.0, 1.0),
        (0.5, f32::NAN, 1.0),
        (0.5, 0.0, f32::NAN),
    ] {
        let r = eval_math_scalar(
            naga::MathFunction::Clamp,
            naga::Literal::F32(v),
            Some(naga::Literal::F32(lo)),
            Some(naga::Literal::F32(hi)),
        );
        assert_eq!(r, None, "clamp({v}, {lo}, {hi}) must not fold");
    }
}

#[test]
fn sign_rejects_nan_does_not_fold() {
    let r = eval_math_scalar(
        naga::MathFunction::Sign,
        naga::Literal::F32(f32::NAN),
        None,
        None,
    );
    assert_eq!(r, None, "sign(NaN) must not fold (would emit NaN literal)");
}

#[test]
fn atan2_zero_zero_does_not_fold() {
    // WGSL leaves atan2(0, 0) implementation-defined; GPUs may
    // return 0, +/-pi/2, or pi.  Rust returns 0.0, which would
    // disagree with some runtimes.
    let r = eval_math_scalar(
        naga::MathFunction::Atan2,
        naga::Literal::F32(0.0),
        Some(naga::Literal::F32(0.0)),
        None,
    );
    assert_eq!(
        r, None,
        "atan2(0, 0) is implementation-defined; must not fold"
    );
}

#[test]
fn atan2_with_one_nonzero_arg_still_folds() {
    let r = eval_math_scalar(
        naga::MathFunction::Atan2,
        naga::Literal::F32(1.0),
        Some(naga::Literal::F32(0.0)),
        None,
    );
    assert!(matches!(r, Some(naga::Literal::F32(_))));
}

#[test]
fn pow_zero_zero_does_not_fold() {
    // pow(0, 0) is implementation-defined in WGSL.
    let r = eval_math_scalar(
        naga::MathFunction::Pow,
        naga::Literal::F32(0.0),
        Some(naga::Literal::F32(0.0)),
        None,
    );
    assert_eq!(
        r, None,
        "pow(0, 0) is implementation-defined; must not fold"
    );
}

#[test]
fn pow_zero_negative_does_not_fold() {
    let r = eval_math_scalar(
        naga::MathFunction::Pow,
        naga::Literal::F32(0.0),
        Some(naga::Literal::F32(-1.0)),
        None,
    );
    assert_eq!(r, None, "pow(0, b<=0) is undefined; must not fold");
}

#[test]
fn pow_positive_base_still_folds() {
    let r = eval_math_scalar(
        naga::MathFunction::Pow,
        naga::Literal::F32(2.0),
        Some(naga::Literal::F32(3.0)),
        None,
    );
    assert_eq!(r, Some(naga::Literal::F32(8.0)));
}

#[test]
fn pow_zero_positive_exp_still_folds() {
    // pow(0, b > 0) = 0 - well-defined in WGSL.
    let r = eval_math_scalar(
        naga::MathFunction::Pow,
        naga::Literal::F32(0.0),
        Some(naga::Literal::F32(2.0)),
        None,
    );
    assert_eq!(r, Some(naga::Literal::F32(0.0)));
}

#[test]
fn abstract_int_add_overflow_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::Add,
        naga::Literal::AbstractInt(i64::MAX),
        naga::Literal::AbstractInt(1),
    );
    assert_eq!(
        result, None,
        "i64::MAX + 1 overflows and should not be folded"
    );
}

#[test]
fn abstract_int_mul_overflow_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::Multiply,
        naga::Literal::AbstractInt(i64::MAX),
        naga::Literal::AbstractInt(2),
    );
    assert_eq!(
        result, None,
        "i64::MAX * 2 overflows and should not be folded"
    );
}

#[test]
fn i32_add_overflow_not_folded() {
    let r = eval_binary(
        naga::BinaryOperator::Add,
        naga::Literal::I32(i32::MAX),
        naga::Literal::I32(1),
    );
    assert_eq!(r, None, "i32::MAX + 1 overflows and should not be folded");
}

#[test]
fn i32_sub_overflow_not_folded() {
    let r = eval_binary(
        naga::BinaryOperator::Subtract,
        naga::Literal::I32(i32::MIN),
        naga::Literal::I32(1),
    );
    assert_eq!(r, None, "i32::MIN - 1 overflows and should not be folded");
}

#[test]
fn i32_mul_overflow_not_folded() {
    let r = eval_binary(
        naga::BinaryOperator::Multiply,
        naga::Literal::I32(i32::MAX),
        naga::Literal::I32(2),
    );
    assert_eq!(r, None, "i32::MAX * 2 overflows and should not be folded");
}

#[test]
fn i32_divide_min_by_neg1_folds_to_defined_value() {
    // WGSL defines runtime `e1 / -1` at MIN as e1; declined, the literal
    // pair (only manufactured by nagami's own transforms) fails naga's text
    // const-eval on re-parse and the emission dies with no fallback.
    let r = eval_binary(
        naga::BinaryOperator::Divide,
        naga::Literal::I32(i32::MIN),
        naga::Literal::I32(-1),
    );
    assert_eq!(r, Some(naga::Literal::I32(i32::MIN)));
}

#[test]
fn i32_modulo_min_by_neg1_folds_to_defined_value() {
    // WGSL defines runtime `MIN % -1` as 0 (see the divide twin above).
    let r = eval_binary(
        naga::BinaryOperator::Modulo,
        naga::Literal::I32(i32::MIN),
        naga::Literal::I32(-1),
    );
    assert_eq!(r, Some(naga::Literal::I32(0)));
}

#[test]
fn i64_add_overflow_not_folded() {
    let r = eval_binary(
        naga::BinaryOperator::Add,
        naga::Literal::I64(i64::MAX),
        naga::Literal::I64(1),
    );
    assert_eq!(r, None, "i64::MAX + 1 overflows and should not be folded");
}

#[test]
fn i64_divide_min_by_neg1_folds_to_defined_value() {
    // Mirrors the i32 rule: WGSL defines `MIN / -1` as MIN.
    let r = eval_binary(
        naga::BinaryOperator::Divide,
        naga::Literal::I64(i64::MIN),
        naga::Literal::I64(-1),
    );
    assert_eq!(r, Some(naga::Literal::I64(i64::MIN)));
}

#[test]
fn i64_modulo_min_by_neg1_folds_to_defined_value() {
    // Mirrors the i32 rule: WGSL defines `MIN % -1` as 0.
    let r = eval_binary(
        naga::BinaryOperator::Modulo,
        naga::Literal::I64(i64::MIN),
        naga::Literal::I64(-1),
    );
    assert_eq!(r, Some(naga::Literal::I64(0)));
}

#[test]
fn abs_i32_min_not_folded() {
    // abs(i32::MIN) overflows (no positive representation); must not fold.
    let r = eval_math_scalar(
        naga::MathFunction::Abs,
        naga::Literal::I32(i32::MIN),
        None,
        None,
    );
    assert_eq!(r, None);
}

#[test]
fn abs_i32_negative_folds() {
    let r = eval_math_scalar(naga::MathFunction::Abs, naga::Literal::I32(-5), None, None);
    assert_eq!(r, Some(naga::Literal::I32(5)));
}

#[test]
fn i32_add_normal_folds() {
    let r = eval_binary(
        naga::BinaryOperator::Add,
        naga::Literal::I32(100),
        naga::Literal::I32(200),
    );
    assert_eq!(r, Some(naga::Literal::I32(300)));
}

#[test]
fn abstract_int_add_normal_folds() {
    let result = eval_binary(
        naga::BinaryOperator::Add,
        naga::Literal::AbstractInt(100),
        naga::Literal::AbstractInt(200),
    );
    assert_eq!(result, Some(naga::Literal::AbstractInt(300)));
}

#[test]
fn shift_left_u32_in_range_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::U32(1),
        naga::Literal::U32(4),
    );
    assert_eq!(result, Some(naga::Literal::U32(16)));
}

#[test]
fn shift_left_u32_out_of_range_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::U32(1),
        naga::Literal::U32(32),
    );
    assert_eq!(result, None, "shift >= bit_width should not be folded");
}

#[test]
fn shift_right_i32_out_of_range_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftRight,
        naga::Literal::I32(1),
        naga::Literal::U32(32),
    );
    assert_eq!(result, None, "shift >= bit_width should not be folded");
}

// Note: the shift amount is always `u32` for WGSL-sourced IR (naga
// concretises the right operand of `<<`/`>>` to u32), so 64-bit-base
// shifts present as `U64/I64 << U32`, never `<< U64`.

#[test]
fn shift_left_u64_in_range_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::U64(1),
        naga::Literal::U32(40),
    );
    assert_eq!(result, Some(naga::Literal::U64(1u64 << 40)));
}

#[test]
fn shift_right_u64_in_range_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftRight,
        naga::Literal::U64(1u64 << 40),
        naga::Literal::U32(8),
    );
    assert_eq!(result, Some(naga::Literal::U64(1u64 << 32)));
}

#[test]
fn shift_left_u64_out_of_range_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::U64(1),
        naga::Literal::U32(64),
    );
    assert_eq!(result, None, "shift >= bit_width should not be folded");
}

#[test]
fn shift_left_i64_in_range_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::I64(1),
        naga::Literal::U32(40),
    );
    assert_eq!(result, Some(naga::Literal::I64(1i64 << 40)));
}

#[test]
fn shift_left_i64_overflow_folds_bit_pattern() {
    // `1i64 << 63` flips the sign bit; a concrete literal pair here sits in
    // a runtime expression (const contexts died at naga's front-end), where
    // WGSL defines the plain bit-pattern shift.  Declining instead poisons
    // emission: the pair fails naga's text const-eval on re-parse.
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::I64(1),
        naga::Literal::U32(63),
    );
    assert_eq!(result, Some(naga::Literal::I64(i64::MIN)));
}

#[test]
fn shift_left_i64_out_of_range_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::I64(1),
        naga::Literal::U32(64),
    );
    assert_eq!(result, None, "shift >= bit_width should not be folded");
}

#[test]
fn shift_abstract_int_negative_amount_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::AbstractInt(1),
        naga::Literal::AbstractInt(-1),
    );
    assert_eq!(result, None, "negative shift amount should not be folded");
}

#[test]
fn shift_abstract_int_out_of_range_not_folded() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::AbstractInt(1),
        naga::Literal::AbstractInt(64),
    );
    assert_eq!(
        result, None,
        "shift >= 64 should not be folded for AbstractInt"
    );
}

#[test]
fn shift_abstract_int_in_range_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::AbstractInt(1),
        naga::Literal::AbstractInt(10),
    );
    assert_eq!(result, Some(naga::Literal::AbstractInt(1024)));
}

// WGSL's sign-changing `<<` shader-creation error applies to CONST
// contexts, which naga's front-end already rejected at ingest; concrete
// literal pairs reaching the fold are runtime expressions manufactured by
// nagami's own transforms, where the spec defines the bit-pattern result.
// Folding it keeps the emission textable (a declined pair fails naga's
// re-parse const-eval and kills the whole emission).
#[test]
fn shift_left_i32_sign_bit_overflow_folds_bit_pattern() {
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I32(1),
            naga::Literal::U32(31),
        ),
        Some(naga::Literal::I32(i32::MIN)),
    );
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I32(2),
            naga::Literal::U32(30),
        ),
        Some(naga::Literal::I32(i32::MIN)),
    );
}

#[test]
fn shift_left_i32_in_range_still_folds() {
    // Boundary case: `1 << 30 = 0x40000000` keeps the sign bit at 0,
    // and `-1 << 31 = i32::MIN` keeps the sign bit at 1 - both are
    // valid and must continue folding.
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I32(1),
            naga::Literal::U32(30),
        ),
        Some(naga::Literal::I32(1 << 30)),
    );
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I32(-1),
            naga::Literal::U32(31),
        ),
        Some(naga::Literal::I32(i32::MIN)),
    );
}

#[test]
fn shift_left_i64_sign_bit_overflow_folds_bit_pattern() {
    // RHS must be `U32` to reach the `(ShiftLeft, I64, U32)` arm: a WGSL
    // shift amount is always `u32`.  (A `U64` RHS would match no arm and
    // return `None` for the wrong reason - vacuously passing this test.)
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I64(2),
            naga::Literal::U32(62),
        ),
        Some(naga::Literal::I64(i64::MIN)),
    );
}

#[test]
fn shift_left_abstract_int_sign_bit_overflow_not_folded() {
    // AbstractInt is i64-backed in naga; the same WGSL rule applies.
    assert_eq!(
        eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::AbstractInt(1),
            naga::Literal::AbstractInt(63),
        ),
        None,
        "AbstractInt 1 << 63 must not fold (sign-bit overflow)"
    );
}

#[test]
fn run_folds_globals_functions_and_entry_points() {
    let source = r#"
const C: f32 = 0.0;

fn helper() -> f32 {
    return 0.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
"#;

    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");

    let (const_handle, const_init) = module
        .constants
        .iter()
        .next()
        .map(|(h, c)| (h, c.init))
        .expect("expected one constant in module");

    let g_one = module.global_expressions.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let g_two = module.global_expressions.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    module.global_expressions[const_init] = naga::Expression::Binary {
        op: naga::BinaryOperator::Add,
        left: g_one,
        right: g_two,
    };

    let helper_handle = module
        .functions
        .iter()
        .map(|(h, _)| h)
        .next()
        .expect("expected helper function");

    let helper_add = {
        let helper = &mut module.functions[helper_handle];
        let c = helper
            .expressions
            .append(naga::Expression::Constant(const_handle), Default::default());
        let four = helper.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        helper.expressions.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: c,
                right: four,
            },
            Default::default(),
        )
    };

    let entry_add = {
        let entry = &mut module.entry_points[0];
        let c = entry
            .function
            .expressions
            .append(naga::Expression::Constant(const_handle), Default::default());
        let five = entry.function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(5.0)),
            Default::default(),
        );
        entry.function.expressions.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: c,
                right: five,
            },
            Default::default(),
        )
    };

    let changed = run_pass(&mut module);
    assert!(
        changed,
        "run should report changes when foldable expressions are present"
    );

    match module.global_expressions[const_init] {
        naga::Expression::Literal(naga::Literal::F32(v)) => {
            assert!((v - 3.0).abs() < f32::EPSILON)
        }
        ref other => panic!("expected global constant initializer to fold, got {other:?}"),
    }

    let helper = &module.functions[helper_handle];
    assert_f32_literal(&helper.expressions, helper_add, 7.0);
    assert_f32_literal(&module.entry_points[0].function.expressions, entry_add, 8.0);
}

#[test]
fn run_returns_false_when_nothing_is_foldable() {
    let source = r#"
fn helper(x: f32) -> f32 {
    return x;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, y, y, 1.0);
}
"#;

    let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
    let changed = run_pass(&mut module);
    assert!(
        !changed,
        "run should report no changes when expressions are already non-foldable"
    );
}

#[test]
fn abstract_int_shift_left_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftLeft,
        naga::Literal::AbstractInt(1),
        naga::Literal::AbstractInt(3),
    );
    assert_eq!(
        result,
        Some(naga::Literal::AbstractInt(8)),
        "1 << 3 should fold to 8"
    );
}

#[test]
fn abstract_int_shift_right_folds() {
    let result = eval_binary(
        naga::BinaryOperator::ShiftRight,
        naga::Literal::AbstractInt(16),
        naga::Literal::AbstractInt(2),
    );
    assert_eq!(
        result,
        Some(naga::Literal::AbstractInt(4)),
        "16 >> 2 should fold to 4"
    );
}

struct IdentityArena {
    arena: naga::Arena<naga::Expression>,
    zero_f32: naga::Handle<naga::Expression>,
    one_f32: naga::Handle<naga::Expression>,
    zero_i32: naga::Handle<naga::Expression>,
    one_i32: naga::Handle<naga::Expression>,
    param: naga::Handle<naga::Expression>,
}

fn make_identity_arena() -> IdentityArena {
    let mut arena = naga::Arena::new();
    let zero_f32 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let one_f32 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let zero_i32 = arena.append(
        naga::Expression::Literal(naga::Literal::I32(0)),
        Default::default(),
    );
    let one_i32 = arena.append(
        naga::Expression::Literal(naga::Literal::I32(1)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    IdentityArena {
        arena,
        zero_f32,
        one_f32,
        zero_i32,
        one_i32,
        param,
    }
}

// Float Add/Subtract identity is intentionally NOT folded; see
// `is_additive_identity_zero` and its IEEE-754 case analysis.
// Integer zero identity remains safe.

#[test]
fn identity_add_zero_left_integer() {
    let a = make_identity_arena();
    let result = check_identity_operand(naga::BinaryOperator::Add, a.zero_i32, a.param, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_add_zero_right_integer() {
    let a = make_identity_arena();
    let result = check_identity_operand(naga::BinaryOperator::Add, a.param, a.zero_i32, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_sub_zero_right_integer() {
    let a = make_identity_arena();
    let result = check_identity_operand(
        naga::BinaryOperator::Subtract,
        a.param,
        a.zero_i32,
        &a.arena,
    );
    assert_eq!(result, Some(a.param));
}

/// Regression for signed-zero correctness: `x + (+0.0) -> x`
/// would be wrong when `x == -0.0` because IEEE 754 says
/// `(-0.0) + (+0.0) = +0.0`.  The identity helper must refuse to
/// fold either side of a float Add/Subtract identity, regardless
/// of which zero is the literal.
#[test]
fn identity_does_not_fold_float_zero_in_add_or_subtract() {
    let a = make_identity_arena();
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Add, a.zero_f32, a.param, &a.arena),
        None,
        "(+0.0) + x must not fold (mis-signs x = -0.0)"
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Add, a.param, a.zero_f32, &a.arena),
        None,
        "x + (+0.0) must not fold (mis-signs x = -0.0)"
    );
    assert_eq!(
        check_identity_operand(
            naga::BinaryOperator::Subtract,
            a.param,
            a.zero_f32,
            &a.arena
        ),
        None,
        "x - (+0.0) must not fold (mis-signs x = -0.0)"
    );
}

#[test]
fn identity_sub_zero_left_not_eliminated() {
    let a = make_identity_arena();
    // 0 - param != param (it's negation)
    let result = check_identity_operand(
        naga::BinaryOperator::Subtract,
        a.zero_f32,
        a.param,
        &a.arena,
    );
    assert_eq!(result, None);
}

#[test]
fn identity_mul_one_left() {
    let a = make_identity_arena();
    // 1 * param -> param
    let result =
        check_identity_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_mul_one_right() {
    let a = make_identity_arena();
    // param * 1 -> param
    let result =
        check_identity_operand(naga::BinaryOperator::Multiply, a.param, a.one_f32, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_div_one_right() {
    let a = make_identity_arena();
    // param / 1 -> param
    let result = check_identity_operand(naga::BinaryOperator::Divide, a.param, a.one_f32, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_div_one_left_not_eliminated() {
    let a = make_identity_arena();
    // 1 / param != param
    let result = check_identity_operand(naga::BinaryOperator::Divide, a.one_f32, a.param, &a.arena);
    assert_eq!(result, None);
}

#[test]
fn identity_integer_types() {
    let a = make_identity_arena();
    // 0i + param -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Add, a.zero_i32, a.param, &a.arena),
        Some(a.param)
    );
    // 1i * param -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, a.one_i32, a.param, &a.arena),
        Some(a.param)
    );
}

#[test]
fn identity_no_false_positive_for_other_ops() {
    let a = make_identity_arena();
    // Modulo with 1 is not an identity (it gives remainder),
    // And with zero (not all-ones) is not an identity,
    // ShiftLeft with float zero is not an identity (wrong type).
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Modulo, a.param, a.one_f32, &a.arena),
        None
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::And, a.param, a.zero_f32, &a.arena),
        None
    );
    assert_eq!(
        check_identity_operand(
            naga::BinaryOperator::ShiftLeft,
            a.param,
            a.zero_f32,
            &a.arena
        ),
        None
    );
}

#[test]
fn identity_fires_on_f16_one_for_multiply_and_divide() {
    // F16 multiplicative identity (`x * 1h`, `1h * x`, `x / 1h`)
    // is safe and continues to fold.  The additive F16 identities
    // (`x + 0h`, `x - 0h`) are intentionally NOT folded; see
    // `identity_does_not_fold_float_zero_in_add_or_subtract` for
    // the signed-zero rationale.
    use half::f16;
    let mut arena = naga::Arena::new();
    let one_f16 = arena.append(
        naga::Expression::Literal(naga::Literal::F16(f16::from_f32(1.0))),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());

    // x * 1h -> x
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, param, one_f16, &arena),
        Some(param)
    );
    // 1h * x -> x  (commutative)
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, one_f16, param, &arena),
        Some(param)
    );
    // x / 1h -> x
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Divide, param, one_f16, &arena),
        Some(param)
    );
}

// Identity elimination: integration via fold_local_expressions

#[test]
fn identity_fold_non_emittable_added_to_folded() {
    // 0 + FunctionArgument -> FunctionArgument is non-emittable,
    // so it must be in the folded set for Emit removal.  Use
    // integer zero because float-zero additive identity is
    // intentionally refused under IEEE signed-zero rules.
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::I32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: zero,
            right: param,
        },
        Default::default(),
    );

    let (folded, identity) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        identity > 0,
        "0 + param should trigger identity elimination"
    );
    assert!(
        folded.contains(&add),
        "FunctionArgument replacement must be in folded set for Emit removal"
    );
    assert!(
        matches!(arena[add], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[add]
    );
}

#[test]
fn identity_fold_emittable_not_in_folded() {
    // 0 + Binary(Mul, a, b) -> Binary(Mul, a, b) is emittable,
    // so it must NOT be in the folded set.  Use integer zero
    // because float-zero additive identity is intentionally
    // refused under IEEE signed-zero rules.
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::I32(0)),
        Default::default(),
    );
    let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: param_a,
            right: param_b,
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: zero,
            right: mul,
        },
        Default::default(),
    );

    let (folded, identity) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        identity > 0,
        "0 + (a*b) should trigger identity elimination"
    );
    // The result is Binary(Mul, ...) - emittable - NOT in folded.
    assert!(
        !folded.contains(&add),
        "emittable replacement must NOT be in folded set"
    );
    // The expression at `add` should now be the same as `mul`.
    assert!(
        matches!(
            arena[add],
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                ..
            }
        ),
        "expected Binary(Multiply), got {:?}",
        arena[add]
    );
}

/// Refcount escape: `0u + Load(p)` with a uniquely-owned Load
/// folds to a Load clone, AND the original Load enters `folded`
/// so the rebuild drops its Emit entry.  Without that drop the
/// let-binding survives and runs a second memory read.
#[test]
fn identity_fold_unique_impure_clone_drops_source() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    // FunctionArgument(0) stands in for a pointer here - the
    // const-fold pass doesn't type-check, it just observes
    // expression shapes.
    let ptr = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let load = arena.append(naga::Expression::Load { pointer: ptr }, Default::default());
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: zero,
            right: load,
        },
        Default::default(),
    );

    // Build refcounts inline because there's no Function to walk.
    // The only intra-arena uses are: Load->ptr, Binary->{zero, load}.
    // So `load` has refcount 1 (only the Binary references it).
    let mut refcounts = vec![0u32; arena.len()];
    for (_, expr) in arena.iter() {
        crate::passes::expr_util::visit_expression_children(expr, |child| {
            refcounts[child.index()] += 1;
        });
    }
    assert_eq!(
        refcounts[load.index()],
        1,
        "test setup invariant: Load should be referenced only by the Binary"
    );

    let (folded, identity) = fold_local(
        &mut arena,
        &refcounts,
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        identity > 0,
        "0u + Load should fold once Load is uniquely owned"
    );
    // The original Load's Emit-range entry must be dropped (folded
    // contains `load`).  Otherwise the generator would emit a
    // dead let-binding that runs a second memory read.
    assert!(
        folded.contains(&load),
        "uniquely-owned impure source must be in `folded` so the \
             rebuild walk drops its Emit-range entry"
    );
    // The `add` slot now carries a Load expression (clone of the
    // original); it's still emittable, so it stays out of `folded`.
    assert!(
        !folded.contains(&add),
        "the cloned Load at `add` is still emittable - must NOT be \
             dropped from its own Emit range"
    );
    assert!(
        matches!(arena[add], naga::Expression::Load { .. }),
        "expected Load after fold, got {:?}",
        arena[add]
    );
}

/// Store-aware guard: a uniquely-owned impure operand whose
/// `Emit` range differs from the folding Binary's MUST NOT be
/// relocated - a statement (here, a hazardous memory write) sits
/// between them, so cloning the Load into the Binary's later slot
/// would move the read past the write (read-after-write reorder).
/// Models `let a = data[0]; data[0] = ...; data[1] = 0u + a;`.
#[test]
fn identity_fold_unique_impure_cross_emit_range_blocked() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let ptr = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let load = arena.append(naga::Expression::Load { pointer: ptr }, Default::default());
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: zero,
            right: load,
        },
        Default::default(),
    );

    let mut refcounts = vec![0u32; arena.len()];
    for (_, expr) in arena.iter() {
        crate::passes::expr_util::visit_expression_children(expr, |child| {
            refcounts[child.index()] += 1;
        });
    }
    assert_eq!(refcounts[load.index()], 1, "Load is uniquely owned");

    // Load lives in Emit range 0, the Binary in range 1: a
    // statement (a store) separates them.  The guard must refuse.
    let ranges: HashMap<naga::Handle<naga::Expression>, usize> =
        HashMap::from([(load, 0usize), (add, 1usize)]);
    let (folded, identity) = fold_local_expressions(
        &mut arena,
        &refcounts,
        &ranges,
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        identity, 0,
        "cross-`Emit`-range impure operand must NOT be relocated by the identity fold"
    );
    assert!(
        !folded.contains(&load),
        "the original Load must stay in its Emit range (not dropped)"
    );
    assert!(
        matches!(arena[add], naga::Expression::Binary { .. }),
        "the Binary must be left untouched, got {:?}",
        arena[add]
    );
}

/// Involution arm of the store-aware guard: the same rule applies to
/// `-(-x)` - a uniquely-owned impure inner operand whose `Emit` range
/// differs from the outer Unary's must NOT be relocated.  Models
/// `let a = data[0]; data[0] = ...; data[1] = -(-a);`.
#[test]
fn involution_fold_unique_impure_cross_emit_range_blocked() {
    let mut arena = naga::Arena::new();
    let ptr = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let load = arena.append(naga::Expression::Load { pointer: ptr }, Default::default());
    let neg1 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: load,
        },
        Default::default(),
    );
    let neg2 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: neg1,
        },
        Default::default(),
    );

    let mut refcounts = vec![0u32; arena.len()];
    for (_, expr) in arena.iter() {
        crate::passes::expr_util::visit_expression_children(expr, |child| {
            refcounts[child.index()] += 1;
        });
    }
    assert_eq!(refcounts[load.index()], 1, "inner Load is uniquely owned");
    assert_eq!(
        refcounts[neg1.index()],
        1,
        "intermediate Unary is uniquely owned"
    );

    // Inner Load in Emit range 0; the outer Unary in range 1 - a
    // statement (a store) separates them, so relocation is unsound.
    let ranges: HashMap<naga::Handle<naga::Expression>, usize> =
        HashMap::from([(load, 0usize), (neg1, 1usize), (neg2, 1usize)]);
    let (folded, identity) = fold_local_expressions(
        &mut arena,
        &refcounts,
        &ranges,
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        identity, 0,
        "cross-`Emit`-range impure inner must NOT be relocated by the involution fold"
    );
    assert!(
        !folded.contains(&load),
        "the inner Load must stay in its Emit range"
    );
    assert!(
        matches!(arena[neg2], naga::Expression::Unary { .. }),
        "the outer Unary must be left untouched, got {:?}",
        arena[neg2]
    );
}

/// Counterpart: a multi-referenced impure operand stays alive
/// even after the rewrite, so cloning would emit a second Load -
/// observable for storage / workgroup vars.  Gate must refuse.
#[test]
fn identity_fold_multi_ref_impure_blocked() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let ptr = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let load = arena.append(naga::Expression::Load { pointer: ptr }, Default::default());
    // Two consumers of `load`: the identity Binary and a sibling
    // Unary.  The sibling keeps `load` alive even if the Binary
    // is rewritten.
    let _sibling = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: load,
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: zero,
            right: load,
        },
        Default::default(),
    );

    let mut refcounts = vec![0u32; arena.len()];
    for (_, expr) in arena.iter() {
        crate::passes::expr_util::visit_expression_children(expr, |child| {
            refcounts[child.index()] += 1;
        });
    }
    assert_eq!(
        refcounts[load.index()],
        2,
        "test setup invariant: Load is referenced by Unary and Binary"
    );

    let (folded, identity) = fold_local(
        &mut arena,
        &refcounts,
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        identity, 0,
        "multi-ref impure operand must NOT trigger identity fold"
    );
    assert!(
        !folded.contains(&load),
        "multi-ref Load must stay in its Emit range"
    );
    assert!(
        matches!(
            arena[add],
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                ..
            }
        ),
        "Binary must be preserved when refcount gate blocks the fold"
    );
}

#[test]
fn identity_fold_constant_added_to_folded() {
    let module = naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
    let (constant_handle, _) = module.constants.iter().next().unwrap();

    let mut arena = naga::Arena::new();
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let const_expr = arena.append(
        naga::Expression::Constant(constant_handle),
        Default::default(),
    );
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: one,
            right: const_expr,
        },
        Default::default(),
    );

    // With const in literal cache -> const_expr folds to Literal(41.0) first,
    // then 1 * 41.0 is identity-eliminated.
    let mut const_literals = HashMap::new();
    const_literals.insert(constant_handle, naga::Literal::F32(41.0));
    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &const_literals,
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(folded.contains(&mul), "result should be in folded set");
    assert_f32_literal(&arena, mul, 41.0);

    // Without const in literal cache -> const_expr stays as Constant,
    // identity elim makes mul = Constant (non-emittable -> in folded).
    let mut arena2 = naga::Arena::new();
    let one2 = arena2.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let const_expr2 = arena2.append(
        naga::Expression::Constant(constant_handle),
        Default::default(),
    );
    let mul2 = arena2.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: one2,
            right: const_expr2,
        },
        Default::default(),
    );
    let (folded2, identity2) = fold_local(
        &mut arena2,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        identity2 > 0,
        "1 * const should trigger identity elimination"
    );
    assert!(
        folded2.contains(&mul2),
        "Constant replacement must be in folded set (non-emittable)"
    );
    assert!(
        matches!(arena2[mul2], naga::Expression::Constant(_)),
        "expected Constant, got {:?}",
        arena2[mul2]
    );
}

// Absorbing operand tests

#[test]
fn absorbing_mul_zero_left() {
    let a = make_identity_arena();
    // Integer `0 * param -> 0` (no signed zero / NaN: always absorbs).
    assert_eq!(
        check_absorbing_operand(
            naga::BinaryOperator::Multiply,
            a.zero_i32,
            a.param,
            &a.arena
        ),
        Some(a.zero_i32)
    );
}

#[test]
fn absorbing_mul_zero_right() {
    let a = make_identity_arena();
    // Integer `param * 0 -> 0`.
    assert_eq!(
        check_absorbing_operand(
            naga::BinaryOperator::Multiply,
            a.param,
            a.zero_i32,
            &a.arena
        ),
        Some(a.zero_i32)
    );
}

/// A FLOAT zero must NOT absorb in a multiply: `x * 0.0` carries the
/// product's IEEE sign (and is NaN for non-finite `x`), which the
/// absorbing arm cannot reconstruct by cloning the matched zero.  Only
/// `eval_binary` (sign-aware, for the both-literal case) may fold it.
#[test]
fn absorbing_mul_float_zero_declined() {
    let a = make_identity_arena();
    assert_eq!(
        check_absorbing_operand(
            naga::BinaryOperator::Multiply,
            a.zero_f32,
            a.param,
            &a.arena
        ),
        None
    );
    assert_eq!(
        check_absorbing_operand(
            naga::BinaryOperator::Multiply,
            a.param,
            a.zero_f32,
            &a.arena
        ),
        None
    );
}

#[test]
fn absorbing_mul_zero_integer() {
    let a = make_identity_arena();
    // 0i * param -> 0i
    assert_eq!(
        check_absorbing_operand(
            naga::BinaryOperator::Multiply,
            a.zero_i32,
            a.param,
            &a.arena
        ),
        Some(a.zero_i32)
    );
}

#[test]
fn absorbing_and_zero() {
    let mut arena = naga::Arena::new();
    let zero_u32 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    // param & 0u -> 0u
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::And, param, zero_u32, &arena),
        Some(zero_u32)
    );
    // 0u & param -> 0u
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::And, zero_u32, param, &arena),
        Some(zero_u32)
    );
}

#[test]
fn absorbing_or_all_ones() {
    let mut arena = naga::Arena::new();
    let all_ones = arena.append(
        naga::Expression::Literal(naga::Literal::U32(u32::MAX)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    // param | 0xFFFFFFFF -> 0xFFFFFFFF
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::InclusiveOr, param, all_ones, &arena),
        Some(all_ones)
    );
    // 0xFFFFFFFF | param -> 0xFFFFFFFF
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::InclusiveOr, all_ones, param, &arena),
        Some(all_ones)
    );
}

#[test]
fn absorbing_logical_and_false() {
    let mut arena = naga::Arena::new();
    let f = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::LogicalAnd, param, f, &arena),
        Some(f)
    );
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::LogicalAnd, f, param, &arena),
        Some(f)
    );
}

#[test]
fn absorbing_logical_or_true() {
    let mut arena = naga::Arena::new();
    let t = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::LogicalOr, param, t, &arena),
        Some(t)
    );
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::LogicalOr, t, param, &arena),
        Some(t)
    );
}

#[test]
fn absorbing_no_false_positive() {
    let a = make_identity_arena();
    // Add with zero is identity, NOT absorbing.
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::Add, a.zero_f32, a.param, &a.arena),
        None
    );
    // Multiply with 1 is identity, NOT absorbing.
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena),
        None
    );
}

// Extended identity tests (bitwise / logical)

#[test]
fn identity_or_zero() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    // param | 0 -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::InclusiveOr, param, zero, &arena),
        Some(param)
    );
    // 0 | param -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::InclusiveOr, zero, param, &arena),
        Some(param)
    );
}

#[test]
fn identity_xor_zero() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    // param ^ 0 -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::ExclusiveOr, param, zero, &arena),
        Some(param)
    );
    // 0 ^ param -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::ExclusiveOr, zero, param, &arena),
        Some(param)
    );
}

#[test]
fn identity_and_all_ones() {
    let mut arena = naga::Arena::new();
    let all_ones = arena.append(
        naga::Expression::Literal(naga::Literal::U32(u32::MAX)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    // param & 0xFFFFFFFF -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::And, param, all_ones, &arena),
        Some(param)
    );
    // 0xFFFFFFFF & param -> param
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::And, all_ones, param, &arena),
        Some(param)
    );
}

#[test]
fn identity_logical_and_true() {
    let mut arena = naga::Arena::new();
    let t = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::LogicalAnd, param, t, &arena),
        Some(param)
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::LogicalAnd, t, param, &arena),
        Some(param)
    );
}

#[test]
fn identity_logical_or_false() {
    let mut arena = naga::Arena::new();
    let f = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::LogicalOr, param, f, &arena),
        Some(param)
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::LogicalOr, f, param, &arena),
        Some(param)
    );
}

#[test]
fn identity_and_all_ones_i32() {
    let mut arena = naga::Arena::new();
    // i32 -1 is all ones (0xFFFFFFFF)
    let all_ones = arena.append(
        naga::Expression::Literal(naga::Literal::I32(-1)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::And, param, all_ones, &arena),
        Some(param)
    );
}

// Involution tests

#[test]
fn involution_double_negate() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let neg1 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: param,
        },
        Default::default(),
    );
    let neg2 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: neg1,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(count > 0, "-(-x) should be simplified");
    // neg2 should now be FunctionArgument(0) (non-emittable -> in folded)
    assert!(
        matches!(arena[neg2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[neg2]
    );
    assert!(folded.contains(&neg2));
}

#[test]
fn involution_double_logical_not() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let not1 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::LogicalNot,
            expr: param,
        },
        Default::default(),
    );
    let not2 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::LogicalNot,
            expr: not1,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(count > 0, "!(!x) should be simplified");
    assert!(
        matches!(arena[not2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[not2]
    );
    assert!(folded.contains(&not2));
}

#[test]
fn involution_double_bitwise_not() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let not1 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::BitwiseNot,
            expr: param,
        },
        Default::default(),
    );
    let not2 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::BitwiseNot,
            expr: not1,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(count > 0, "~(~x) should be simplified");
    assert!(
        matches!(arena[not2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[not2]
    );
    assert!(folded.contains(&not2));
}

#[test]
fn involution_different_ops_not_simplified() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let neg = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: param,
        },
        Default::default(),
    );
    let not = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::BitwiseNot,
            expr: neg,
        },
        Default::default(),
    );

    let (_, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(count, 0, "different unary ops should not be simplified");
    assert!(
        matches!(
            arena[not],
            naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                ..
            }
        ),
        "expression should remain unchanged"
    );
}

#[test]
fn involution_emittable_inner_not_in_folded() {
    // -(-Binary(Mul, a, b)) -> Binary(Mul, a, b) which is emittable,
    // so it must NOT be in the folded set.
    let mut arena = naga::Arena::new();
    let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: param_a,
            right: param_b,
        },
        Default::default(),
    );
    let neg1 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: mul,
        },
        Default::default(),
    );
    let neg2 = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: neg1,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(count > 0, "-(-Binary(Mul)) should be simplified");
    assert!(
        matches!(
            arena[neg2],
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                ..
            }
        ),
        "expected Binary(Multiply), got {:?}",
        arena[neg2]
    );
    // Binary is emittable -> must NOT be in folded set
    assert!(
        !folded.contains(&neg2),
        "emittable involution result must NOT be in folded set"
    );
}

// Select simplification tests

#[test]
fn select_same_arms_simplified() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let cond = arena.append(naga::Expression::FunctionArgument(1), Default::default());
    let sel = arena.append(
        naga::Expression::Select {
            condition: cond,
            accept: param,
            reject: param,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(count > 0, "select(x, x, cond) should be simplified");
    assert!(
        matches!(arena[sel], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[sel]
    );
    assert!(folded.contains(&sel));
}

#[test]
fn select_different_arms_not_simplified() {
    let mut arena = naga::Arena::new();
    let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
    let cond = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );
    let sel = arena.append(
        naga::Expression::Select {
            condition: cond,
            accept: param_a,
            reject: param_b,
        },
        Default::default(),
    );

    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    // With constant condition, resolve_literal will fold this to param_b's value.
    // But param_b is FunctionArgument(1), not a literal, so resolve_literal returns None.
    // Select with different arms and non-foldable result stays as-is.
    // Actually: resolve_literal for Select needs both arms to be literals too.
    // So this should remain a Select.
    assert!(
        matches!(arena[sel], naga::Expression::Select { .. }),
        "select with different arms should not be simplified by the simplify loop, got {:?}",
        arena[sel]
    );
    assert!(!folded.contains(&sel));
}

// Absorbing integration test (fold_local_expressions)

#[test]
fn absorbing_fold_mul_zero_param_not_rewritten() {
    // `param * 0.0` (param = FunctionArgument) must NOT be rewritten to the
    // scalar literal `0.0` regardless of param's type: that would produce
    // invalid IR whenever param is a vector or matrix.
    //
    // The new gate requires BOTH operands be scalar Literal for the
    // simplify-loop absorbing rewrite to fire.  A non-literal operand
    // (FunctionArgument here) carries unknown type, so absorbing is
    // safely declined.  The Binary stays as a Binary.
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: param,
            right: zero,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        count, 0,
        "param * 0 with unknown-type param must not fire absorbing"
    );
    assert!(
        matches!(arena[mul], naga::Expression::Binary { .. }),
        "Binary must be preserved, got {:?}",
        arena[mul]
    );
    assert!(!folded.contains(&mul));
}

#[test]
fn absorbing_fold_and_zero_u32_param_not_rewritten() {
    // Same absorbing case for integer `&`.
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let and = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::And,
            left: param,
            right: zero,
        },
        Default::default(),
    );

    let (folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert_eq!(
        count, 0,
        "param & 0u with unknown-type param must not fire absorbing"
    );
    assert!(
        matches!(arena[and], naga::Expression::Binary { .. }),
        "Binary must be preserved, got {:?}",
        arena[and]
    );
    assert!(!folded.contains(&and));
}

#[test]
fn absorbing_fold_mul_zero_both_literal_produces_literal() {
    // When BOTH operands are scalar Literals, absorbing is type-safe
    // (the Binary's result type equals both operand types).  This is
    // the only case the simplify-loop absorbing rewrite fires.
    // (In practice, the preceding `resolve_const_value` stage usually
    // folds it first, but the safety net is still exercised here.)
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let two = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: two,
            right: zero,
        },
        Default::default(),
    );

    let (_folded, _count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    // Either eval_binary or absorbing must have collapsed this to a literal 0.
    assert_f32_literal(&arena, mul, 0.0);
}

#[test]
fn absorbing_fold_logical_and_false_with_non_literal_rhs_rewritten() {
    // `LogicalAnd`/`LogicalOr` are strictly scalar-bool in valid WGSL IR
    // (no vector broadcasting possible),
    // so absorbing is type-safe even when the other operand is NOT a
    // literal.  This is the pattern produced by dead_branch Phase 0 when
    // re-sugaring `var tmp = false && (j < -8)` from its lowered form;
    // without this rewrite the dead loop `for(;0<0&&j<0-8;){}` cannot be
    // collapsed and the minified output grows rather than shrinks.
    let mut arena = naga::Arena::new();
    let false_lit = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );
    // Non-literal RHS (FunctionArgument stands in for `j < -8`, which at
    // the point of absorbing is a Binary, also a non-Literal).
    let rhs = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let and_ = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::LogicalAnd,
            left: false_lit,
            right: rhs,
        },
        Default::default(),
    );

    let (_folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );

    assert!(count > 0, "false && rhs must be absorbed to false");
    assert!(
        matches!(
            arena[and_],
            naga::Expression::Literal(naga::Literal::Bool(false))
        ),
        "expected Literal(Bool(false)), got {:?}",
        arena[and_]
    );
}

#[test]
fn absorbing_fold_logical_or_true_with_non_literal_rhs_rewritten() {
    // Symmetric case: `true || rhs` must collapse to `true` even when
    // `rhs` is non-literal.
    let mut arena = naga::Arena::new();
    let true_lit = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let rhs = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let or_ = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::LogicalOr,
            left: true_lit,
            right: rhs,
        },
        Default::default(),
    );

    let (_folded, count) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );

    assert!(count > 0, "true || rhs must be absorbed to true");
    assert!(
        matches!(
            arena[or_],
            naga::Expression::Literal(naga::Literal::Bool(true))
        ),
        "expected Literal(Bool(true)), got {:?}",
        arena[or_]
    );
}

/// End-to-end regression for the `0<0 && j<-8` pattern inside a dead
/// `for` loop.  If the absorbing rewrite declines to fold `false && (j < -8)`,
/// the dead loop survives into the emitter as
/// `loop { var a = false && A<-8; if !(a) { break; } }` and the minified
/// output GROWS (66 -> 87 bytes); folding collapses the loop so the output
/// stays at or below input size.
#[test]
fn e2e_dead_for_loop_with_short_circuit_condition_does_not_grow() {
    let source = "@compute @workgroup_size(1) fn d(){var j:i32;for(;0<0&&j<0-8;){}}\n";
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
    assert!(
        out.source.len() <= source.len(),
        "dead for-loop minification must not grow: input={} bytes, output={} bytes\n  in:  {:?}\n  out: {:?}",
        source.len(),
        out.source.len(),
        source,
        out.source,
    );
}

// MARK: Vector constant folding tests

/// Helper: insert a vec type into the type arena and return its handle.
fn make_vec_type(
    types: &mut naga::UniqueArena<naga::Type>,
    size: naga::VectorSize,
    scalar: naga::Scalar,
) -> naga::Handle<naga::Type> {
    types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector { size, scalar },
        },
        Default::default(),
    )
}

/// Helper: assert that expression is a Compose of literal scalars.
fn assert_compose_of_f32(
    arena: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    expected: &[f32],
) {
    match &arena[handle] {
        naga::Expression::Compose { components, .. } => {
            assert_eq!(
                components.len(),
                expected.len(),
                "compose component count mismatch"
            );
            for (i, &c) in components.iter().enumerate() {
                assert_f32_literal(arena, c, expected[i]);
            }
        }
        other => panic!("expected Compose, got {other:?}"),
    }
}

#[test]
fn vector_splat_folds_to_compose() {
    let mut types = naga::UniqueArena::new();
    let _vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let one = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let splat = arena.append(
        naga::Expression::Splat {
            size: naga::VectorSize::Tri,
            value: one,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // Splat of literal 1.0 -> Compose(vec3f, [1.0, 1.0, 1.0])
    assert_compose_of_f32(&arena, splat, &[1.0, 1.0, 1.0]);
}

#[test]
fn vector_compose_binary_add_folds() {
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let a1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let a2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let a3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let b1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(10.0)),
        Default::default(),
    );
    let b2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(20.0)),
        Default::default(),
    );
    let b3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(30.0)),
        Default::default(),
    );
    // Pre-place result literals for materialization
    let _r1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(11.0)),
        Default::default(),
    );
    let _r2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(22.0)),
        Default::default(),
    );
    let _r3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(33.0)),
        Default::default(),
    );

    let va = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![a1, a2, a3],
        },
        Default::default(),
    );
    let vb = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![b1, b2, b3],
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: va,
            right: vb,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // vec3(1,2,3) + vec3(10,20,30) = vec3(11,22,33)
    assert_compose_of_f32(&arena, add, &[11.0, 22.0, 33.0]);
}

#[test]
fn vector_negate_folds() {
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(-2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.5)),
        Default::default(),
    );
    // Pre-place result literals for materialization
    let _rn1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(-1.0)),
        Default::default(),
    );
    let _rn2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let _rn3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(-3.5)),
        Default::default(),
    );
    let v = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c1, c2, c3],
        },
        Default::default(),
    );
    let neg = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: v,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // -vec3(1, -2, 3.5) = vec3(-1, 2, -3.5)
    assert_compose_of_f32(&arena, neg, &[-1.0, 2.0, -3.5]);
}

#[test]
fn vector_access_index_folds_to_scalar() {
    let mut types = naga::UniqueArena::new();
    let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(10.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(20.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(30.0)),
        Default::default(),
    );
    let c4 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(40.0)),
        Default::default(),
    );
    let v = arena.append(
        naga::Expression::Compose {
            ty: vec4f_ty,
            components: vec![c1, c2, c3, c4],
        },
        Default::default(),
    );
    // .y == index 1
    let access = arena.append(
        naga::Expression::AccessIndex { base: v, index: 1 },
        Default::default(),
    );

    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // vec4f(10,20,30,40).y -> 20.0
    assert_f32_literal(&arena, access, 20.0);
    assert!(
        folded.contains(&access),
        "scalar result should be in folded set"
    );
}

#[test]
fn vector_swizzle_folds() {
    let mut types = naga::UniqueArena::new();
    let _vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);
    let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let c4 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(4.0)),
        Default::default(),
    );
    let v = arena.append(
        naga::Expression::Compose {
            ty: vec4f_ty,
            components: vec![c1, c2, c3, c4],
        },
        Default::default(),
    );
    // .zw swizzle -> vec2(3.0, 4.0)
    let swiz = arena.append(
        naga::Expression::Swizzle {
            size: naga::VectorSize::Bi,
            vector: v,
            pattern: [
                naga::SwizzleComponent::Z,
                naga::SwizzleComponent::W,
                naga::SwizzleComponent::X, // unused
                naga::SwizzleComponent::X, // unused
            ],
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, swiz, &[3.0, 4.0]);
}

#[test]
fn vector_scalar_broadcast_mul() {
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(4.0)),
        Default::default(),
    );
    // Pre-place result literals for materialization
    let _r1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(20.0)),
        Default::default(),
    );
    let _r2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(30.0)),
        Default::default(),
    );
    let _r3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(40.0)),
        Default::default(),
    );
    let v = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c1, c2, c3],
        },
        Default::default(),
    );
    let s = arena.append(
        naga::Expression::Literal(naga::Literal::F32(10.0)),
        Default::default(),
    );
    // vec3(2,3,4) * 10.0
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: v,
            right: s,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, mul, &[20.0, 30.0, 40.0]);
}

#[test]
fn vector_zero_value_stays_non_emittable() {
    // ZeroValue is non-emittable, so it must NOT be replaced with
    // an emittable Compose - that would produce invalid IR.
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let _z = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let zero = arena.append(naga::Expression::ZeroValue(vec3f_ty), Default::default());

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert!(
        matches!(arena[zero], naga::Expression::ZeroValue(_)),
        "ZeroValue must stay as ZeroValue, got {:?}",
        arena[zero]
    );
}

#[test]
fn vector_add_zero_value_folds() {
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let v = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c1, c2, c3],
        },
        Default::default(),
    );
    let zero = arena.append(naga::Expression::ZeroValue(vec3f_ty), Default::default());
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: v,
            right: zero,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // vec3(1,2,3) + vec3(0,0,0) = vec3(1,2,3)
    assert_compose_of_f32(&arena, add, &[1.0, 2.0, 3.0]);
}

#[test]
fn vector_splat_binary_scalar_add() {
    // Splat(2.0) + Splat(3.0) -> Compose([5.0, 5.0])
    let mut types = naga::UniqueArena::new();
    let _vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let two = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let three = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    // Need a literal 5.0 in the arena for materialization
    let _five = arena.append(
        naga::Expression::Literal(naga::Literal::F32(5.0)),
        Default::default(),
    );

    let sp1 = arena.append(
        naga::Expression::Splat {
            size: naga::VectorSize::Bi,
            value: two,
        },
        Default::default(),
    );
    let sp2 = arena.append(
        naga::Expression::Splat {
            size: naga::VectorSize::Bi,
            value: three,
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: sp1,
            right: sp2,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, add, &[5.0, 5.0]);
}

#[test]
fn vector_nested_chain_folds() {
    // Test: -negate(compose(1, 2, 3)) + compose(10, 20, 30) == compose(9, 18, 27)
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let c10 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(10.0)),
        Default::default(),
    );
    let c20 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(20.0)),
        Default::default(),
    );
    let c30 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(30.0)),
        Default::default(),
    );
    // Pre-place result literals for materialisation
    let _c9 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(9.0)),
        Default::default(),
    );
    let _c18 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(18.0)),
        Default::default(),
    );
    let _c27 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(27.0)),
        Default::default(),
    );

    let va = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c1, c2, c3],
        },
        Default::default(),
    );
    let neg = arena.append(
        naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: va,
        },
        Default::default(),
    );
    let vb = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c10, c20, c30],
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: neg,
            right: vb,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // -vec3(1,2,3) + vec3(10,20,30) = vec3(9, 18, 27)
    assert_compose_of_f32(&arena, add, &[9.0, 18.0, 27.0]);
}

#[test]
fn vector_integer_types_fold() {
    let mut types = naga::UniqueArena::new();
    let vec2i_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::I32);

    let mut arena = naga::Arena::new();
    let a = arena.append(
        naga::Expression::Literal(naga::Literal::I32(10)),
        Default::default(),
    );
    let b = arena.append(
        naga::Expression::Literal(naga::Literal::I32(20)),
        Default::default(),
    );
    let c = arena.append(
        naga::Expression::Literal(naga::Literal::I32(3)),
        Default::default(),
    );
    let d = arena.append(
        naga::Expression::Literal(naga::Literal::I32(7)),
        Default::default(),
    );
    // Results
    let _r1 = arena.append(
        naga::Expression::Literal(naga::Literal::I32(13)),
        Default::default(),
    );
    let _r2 = arena.append(
        naga::Expression::Literal(naga::Literal::I32(27)),
        Default::default(),
    );

    let va = arena.append(
        naga::Expression::Compose {
            ty: vec2i_ty,
            components: vec![a, b],
        },
        Default::default(),
    );
    let vb = arena.append(
        naga::Expression::Compose {
            ty: vec2i_ty,
            components: vec![c, d],
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: va,
            right: vb,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // vec2i(10,20) + vec2i(3,7) = vec2i(13,27)
    match &arena[add] {
        naga::Expression::Compose { components, .. } => {
            assert_eq!(components.len(), 2);
            assert!(matches!(
                arena[components[0]],
                naga::Expression::Literal(naga::Literal::I32(13))
            ));
            assert!(matches!(
                arena[components[1]],
                naga::Expression::Literal(naga::Literal::I32(27))
            ));
        }
        other => panic!("expected Compose, got {other:?}"),
    }
}

#[test]
fn vector_no_matching_literal_skips_materialization() {
    // If the result literal doesn't exist in the arena before the target,
    // materialization is skipped and the expression remains unchanged.
    let mut types = naga::UniqueArena::new();
    let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(100.0)),
        Default::default(),
    );
    let c4 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(200.0)),
        Default::default(),
    );
    // Intentionally do NOT add literals 101.0 and 202.0

    let va = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![c1, c2],
        },
        Default::default(),
    );
    let vb = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![c3, c4],
        },
        Default::default(),
    );
    let add = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: va,
            right: vb,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // Result 101.0 and 202.0 are not in the arena, so add stays as Binary
    assert!(
        matches!(arena[add], naga::Expression::Binary { .. }),
        "should remain Binary when materialization fails, got {:?}",
        arena[add]
    );
}

#[test]
fn vector_compose_mixed_scalar_and_vector() {
    // Compose(vec4f, [scalar, vec3f]) should flatten to 4 components.
    let mut types = naga::UniqueArena::new();
    let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);
    let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let c4 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(4.0)),
        Default::default(),
    );
    let v3 = arena.append(
        naga::Expression::Compose {
            ty: vec3f_ty,
            components: vec![c2, c3, c4],
        },
        Default::default(),
    );
    // Compose(vec4f, [scalar(1.0), vec3(2.0, 3.0, 4.0)])
    let v4 = arena.append(
        naga::Expression::Compose {
            ty: vec4f_ty,
            components: vec![c1, v3],
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, v4, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn vector_relational_op_produces_bool_vector() {
    // vec2(1.0, 3.0) < vec2(2.0, 2.0) -> vec2<bool>(true, false)
    let mut types = naga::UniqueArena::new();
    let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);
    let _vec2b_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::BOOL);

    let mut arena = naga::Arena::new();
    let a1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let a2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    let b1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let b2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    // Result literals for materialization
    let _t = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let _f = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );

    let va = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![a1, a2],
        },
        Default::default(),
    );
    let vb = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![b1, b2],
        },
        Default::default(),
    );
    let lt = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Less,
            left: va,
            right: vb,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // 1.0 < 2.0 -> true,  3.0 < 2.0 -> false
    match &arena[lt] {
        naga::Expression::Compose { components, .. } => {
            assert_eq!(components.len(), 2);
            assert!(matches!(
                arena[components[0]],
                naga::Expression::Literal(naga::Literal::Bool(true))
            ));
            assert!(matches!(
                arena[components[1]],
                naga::Expression::Literal(naga::Literal::Bool(false))
            ));
        }
        other => panic!("expected Compose(vec2<bool>), got {other:?}"),
    }
}

#[test]
fn vector_select_constant_true() {
    // select(accept_vec, reject_vec, true) -> accept_vec
    let mut types = naga::UniqueArena::new();
    let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

    let mut arena = naga::Arena::new();
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let c3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(9.0)),
        Default::default(),
    );
    let c4 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(8.0)),
        Default::default(),
    );
    let accept = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![c1, c2],
        },
        Default::default(),
    );
    let reject = arena.append(
        naga::Expression::Compose {
            ty: vec2f_ty,
            components: vec![c3, c4],
        },
        Default::default(),
    );
    let cond = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(true)),
        Default::default(),
    );
    let sel = arena.append(
        naga::Expression::Select {
            condition: cond,
            accept,
            reject,
        },
        Default::default(),
    );

    let (_, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    // select with true condition -> accept = vec2(1.0, 2.0)
    assert_compose_of_f32(&arena, sel, &[1.0, 2.0]);
}

#[test]
fn scalar_zero_value_resolves() {
    // ZeroValue(f32) should fold to Literal(F32(0.0))
    let mut types = naga::UniqueArena::new();
    let f32_ty = types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::F32),
        },
        Default::default(),
    );

    let mut arena = naga::Arena::new();
    let zv = arena.append(naga::Expression::ZeroValue(f32_ty), Default::default());

    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert!(
        matches!(arena[zv], naga::Expression::Literal(naga::Literal::F32(v)) if v == 0.0),
        "ZeroValue(f32) should fold to Literal(0.0), got {:?}",
        arena[zv]
    );
    assert!(
        folded.contains(&zv),
        "scalar ZeroValue should be in folded set"
    );
}

#[test]
fn math_abs_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Abs,
            naga::Literal::F32(-3.0),
            None,
            None
        ),
        Some(naga::Literal::F32(3.0))
    );
}

#[test]
fn math_abs_i32() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Abs, naga::Literal::I32(-5), None, None),
        Some(naga::Literal::I32(5))
    );
}

#[test]
fn math_abs_u32_identity() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Abs, naga::Literal::U32(7), None, None),
        Some(naga::Literal::U32(7))
    );
}

#[test]
fn math_abs_i32_min_not_folded() {
    // WGSL: abs(i32::MIN) has no positive representation; folding to
    // wrapping i32::MIN would silently change observable semantics
    // (naga would otherwise produce an execution error), so the folder
    // must decline.
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Abs,
            naga::Literal::I32(i32::MIN),
            None,
            None
        ),
        None
    );
}

#[test]
fn math_min_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Min,
            naga::Literal::F32(3.0),
            Some(naga::Literal::F32(1.0)),
            None
        ),
        Some(naga::Literal::F32(1.0))
    );
}

#[test]
fn math_max_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Max,
            naga::Literal::F32(3.0),
            Some(naga::Literal::F32(7.0)),
            None
        ),
        Some(naga::Literal::F32(7.0))
    );
}

#[test]
fn math_min_i32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Min,
            naga::Literal::I32(10),
            Some(naga::Literal::I32(-2)),
            None
        ),
        Some(naga::Literal::I32(-2))
    );
}

#[test]
fn math_clamp_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Clamp,
            naga::Literal::F32(5.0),
            Some(naga::Literal::F32(0.0)),
            Some(naga::Literal::F32(1.0)),
        ),
        Some(naga::Literal::F32(1.0))
    );
}

#[test]
fn math_clamp_rejects_inverted_range() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Clamp,
            naga::Literal::F32(0.5),
            Some(naga::Literal::F32(1.0)),
            Some(naga::Literal::F32(0.0)),
        ),
        None,
        "clamp with low > high should not fold"
    );
}

#[test]
fn math_saturate_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Saturate,
            naga::Literal::F32(2.0),
            None,
            None
        ),
        Some(naga::Literal::F32(1.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Saturate,
            naga::Literal::F32(-0.5),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_sign_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sign,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sign,
            naga::Literal::F32(-5.0),
            None,
            None
        ),
        Some(naga::Literal::F32(-1.0))
    );
}

#[test]
fn math_floor_ceil_trunc_round() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Floor,
            naga::Literal::F32(1.7),
            None,
            None
        ),
        Some(naga::Literal::F32(1.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Ceil,
            naga::Literal::F32(1.2),
            None,
            None
        ),
        Some(naga::Literal::F32(2.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Trunc,
            naga::Literal::F32(-1.9),
            None,
            None
        ),
        Some(naga::Literal::F32(-1.0))
    );
    // Round uses ties-to-even: 0.5 -> 0.0, 1.5 -> 2.0
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Round,
            naga::Literal::F32(0.5),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Round,
            naga::Literal::F32(1.5),
            None,
            None
        ),
        Some(naga::Literal::F32(2.0))
    );
}

#[test]
fn math_fract_f32() {
    // WGSL fract(e) = e - floor(e)
    let result = eval_math_scalar(
        naga::MathFunction::Fract,
        naga::Literal::F32(1.75),
        None,
        None,
    );
    assert_eq!(result, Some(naga::Literal::F32(0.75)));
    // Negative: fract(-0.25) = -0.25 - floor(-0.25) = -0.25 - (-1.0) = 0.75
    let result = eval_math_scalar(
        naga::MathFunction::Fract,
        naga::Literal::F32(-0.25),
        None,
        None,
    );
    assert_eq!(result, Some(naga::Literal::F32(0.75)));
}

#[test]
fn math_step_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Step,
            naga::Literal::F32(0.5),
            Some(naga::Literal::F32(1.0)),
            None,
        ),
        Some(naga::Literal::F32(1.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Step,
            naga::Literal::F32(0.5),
            Some(naga::Literal::F32(0.3)),
            None,
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_sqrt_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sqrt,
            naga::Literal::F32(4.0),
            None,
            None
        ),
        Some(naga::Literal::F32(2.0))
    );
}

#[test]
fn math_fract_huge_value_does_not_emit_non_finite() {
    // Regression: `v - v.floor()` for very-large finite `v` can
    // produce NaN/Inf when `v.floor()` saturates near the float
    // range boundary or when the subtraction underflows precision.
    // The fold must refuse such cases via the `finite_*` guard so
    // the emitted literal is always representable WGSL.
    for v in [f32::MAX, f32::MIN, -f32::MAX, 1.0e38_f32] {
        let result = eval_math_scalar(naga::MathFunction::Fract, naga::Literal::F32(v), None, None);
        match result {
            None => {}
            Some(naga::Literal::F32(out)) => assert!(
                out.is_finite(),
                "fract({v}) folded to non-finite {out}; finite_f32 guard must reject"
            ),
            Some(other) => panic!("expected None or F32, got {other:?}"),
        }
    }
}

#[test]
fn math_sqrt_negative_not_folded() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sqrt,
            naga::Literal::F32(-1.0),
            None,
            None
        ),
        None,
        "sqrt of negative should not fold"
    );
}

#[test]
fn math_inverse_sqrt_f32() {
    let result = eval_math_scalar(
        naga::MathFunction::InverseSqrt,
        naga::Literal::F32(4.0),
        None,
        None,
    );
    assert_eq!(result, Some(naga::Literal::F32(0.5)));
}

#[test]
fn math_inverse_sqrt_zero_not_folded() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::InverseSqrt,
            naga::Literal::F32(0.0),
            None,
            None,
        ),
        None,
        "inverseSqrt(0) should not fold"
    );
}

#[test]
fn math_fma_f32() {
    // fma(2.0, 3.0, 1.0) = 2.0 * 3.0 + 1.0 = 7.0
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Fma,
            naga::Literal::F32(2.0),
            Some(naga::Literal::F32(3.0)),
            Some(naga::Literal::F32(1.0)),
        ),
        Some(naga::Literal::F32(7.0))
    );
}

#[test]
fn math_sin_cos_zero() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Sin, naga::Literal::F32(0.0), None, None),
        Some(naga::Literal::F32(0.0))
    );
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Cos, naga::Literal::F32(0.0), None, None),
        Some(naga::Literal::F32(1.0))
    );
}

#[test]
fn math_exp_log() {
    // exp(0) = 1
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Exp, naga::Literal::F32(0.0), None, None),
        Some(naga::Literal::F32(1.0))
    );
    // log(1) = 0
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Log, naga::Literal::F32(1.0), None, None),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_log_zero_not_folded() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Log, naga::Literal::F32(0.0), None, None),
        None,
        "log(0) should not fold"
    );
}

#[test]
fn math_log_negative_not_folded() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Log,
            naga::Literal::F32(-1.0),
            None,
            None
        ),
        None,
        "log(negative) should not fold"
    );
}

#[test]
fn math_exp_overflow_not_folded() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Exp,
            naga::Literal::F32(1000.0),
            None,
            None
        ),
        None,
        "exp(1000) overflows and should not fold"
    );
}

#[test]
fn math_pow_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Pow,
            naga::Literal::F32(2.0),
            Some(naga::Literal::F32(3.0)),
            None,
        ),
        Some(naga::Literal::F32(8.0))
    );
}

#[test]
fn math_log2_exp2() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Log2,
            naga::Literal::F32(8.0),
            None,
            None
        ),
        Some(naga::Literal::F32(3.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Exp2,
            naga::Literal::F32(3.0),
            None,
            None
        ),
        Some(naga::Literal::F32(8.0))
    );
}

#[test]
fn math_acos_domain_violation() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Acos,
            naga::Literal::F32(2.0),
            None,
            None
        ),
        None,
        "acos(2.0) is outside [-1, 1] and should not fold"
    );
}

#[test]
fn math_acosh_domain_violation() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Acosh,
            naga::Literal::F32(0.5),
            None,
            None
        ),
        None,
        "acosh(0.5) is below 1.0 and should not fold"
    );
}

#[test]
fn math_atanh_domain_violation() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Atanh,
            naga::Literal::F32(1.0),
            None,
            None
        ),
        None,
        "atanh(1.0) is at boundary and should not fold"
    );
}

#[test]
fn math_radians_degrees_roundtrip() {
    let rad = eval_math_scalar(
        naga::MathFunction::Radians,
        naga::Literal::F32(180.0),
        None,
        None,
    );
    match rad {
        Some(naga::Literal::F32(v)) => {
            assert!((v - std::f32::consts::PI).abs() < 1e-5);
        }
        other => panic!("expected F32, got {other:?}"),
    }
}

#[test]
fn math_count_trailing_zeros_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::CountTrailingZeros,
            naga::Literal::U32(8),
            None,
            None
        ),
        Some(naga::Literal::U32(3))
    );
}

#[test]
fn math_count_leading_zeros_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::CountLeadingZeros,
            naga::Literal::U32(1),
            None,
            None
        ),
        Some(naga::Literal::U32(31))
    );
}

#[test]
fn math_count_one_bits_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::CountOneBits,
            naga::Literal::U32(0b1011),
            None,
            None
        ),
        Some(naga::Literal::U32(3))
    );
}

#[test]
fn math_reverse_bits_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::ReverseBits,
            naga::Literal::U32(1),
            None,
            None
        ),
        Some(naga::Literal::U32(1u32 << 31))
    );
}

#[test]
fn math_first_trailing_bit_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstTrailingBit,
            naga::Literal::U32(0),
            None,
            None
        ),
        Some(naga::Literal::U32(u32::MAX))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstTrailingBit,
            naga::Literal::U32(12),
            None,
            None
        ),
        Some(naga::Literal::U32(2))
    );
}

#[test]
fn math_first_leading_bit_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::U32(0),
            None,
            None
        ),
        Some(naga::Literal::U32(u32::MAX))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::U32(8),
            None,
            None
        ),
        Some(naga::Literal::U32(3))
    );
}

#[test]
fn math_first_leading_bit_i32() {
    // 0 and -1 both return -1
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(0),
            None,
            None
        ),
        Some(naga::Literal::I32(-1))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(-1),
            None,
            None
        ),
        Some(naga::Literal::I32(-1))
    );
    // Positive: firstLeadingBit(8) = 3
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(8),
            None,
            None
        ),
        Some(naga::Literal::I32(3))
    );
}

#[test]
fn math_unsupported_returns_none() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Dot, naga::Literal::F32(1.0), None, None),
        None,
        "Dot is Tier 4 and should not fold at scalar level"
    );
}

#[test]
fn math_abstract_float_sqrt() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sqrt,
            naga::Literal::AbstractFloat(9.0),
            None,
            None
        ),
        Some(naga::Literal::AbstractFloat(3.0))
    );
}

#[test]
fn math_pow_negative_base_not_folded() {
    // WGSL precondition: e1 >= 0.0; negative base is undefined.
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Pow,
            naga::Literal::F32(-1.0),
            Some(naga::Literal::F32(2.0)),
            None,
        ),
        None,
        "pow with negative base should not fold"
    );
}

#[test]
fn math_pow_zero_base_negative_exp_not_folded() {
    // pow(0, -1) -> inf -> not finite -> None
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Pow,
            naga::Literal::F32(0.0),
            Some(naga::Literal::F32(-1.0)),
            None,
        ),
        None,
        "pow(0, negative) overflows and should not fold"
    );
}

#[test]
fn math_sign_negative_zero() {
    // WGSL: sign(-0.0) should return 0.0 (or -0.0, impl-defined)
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sign,
            naga::Literal::F32(-0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_sign_i32() {
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(42), None, None),
        Some(naga::Literal::I32(1))
    );
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(0), None, None),
        Some(naga::Literal::I32(0))
    );
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(-7), None, None),
        Some(naga::Literal::I32(-1))
    );
}

#[test]
fn math_tan_f32() {
    let result = eval_math_scalar(naga::MathFunction::Tan, naga::Literal::F32(0.0), None, None);
    assert_eq!(result, Some(naga::Literal::F32(0.0)));
}

#[test]
fn math_atan2_f32() {
    let result = eval_math_scalar(
        naga::MathFunction::Atan2,
        naga::Literal::F32(1.0),
        Some(naga::Literal::F32(1.0)),
        None,
    );
    match result {
        Some(naga::Literal::F32(v)) => {
            assert!((v - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        }
        other => panic!("expected F32, got {other:?}"),
    }
}

#[test]
fn math_tanh_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Tanh,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_cosh_sinh_f32() {
    // cosh(0) = 1, sinh(0) = 0
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Cosh,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(1.0))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Sinh,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_cosh_overflow_not_folded() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Cosh,
            naga::Literal::F32(1000.0),
            None,
            None
        ),
        None,
        "cosh(1000) overflows and should not fold"
    );
}

#[test]
fn math_asinh_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Asinh,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_asin_f32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Asin,
            naga::Literal::F32(0.0),
            None,
            None
        ),
        Some(naga::Literal::F32(0.0))
    );
}

#[test]
fn math_asin_domain_violation() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Asin,
            naga::Literal::F32(2.0),
            None,
            None
        ),
        None,
        "asin(2.0) is outside [-1, 1] and should not fold"
    );
}

#[test]
fn math_first_leading_bit_negative_i32_edge_cases() {
    // -2 = 0xFFFFFFFE: first differing bit from sign is at position 0
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(-2),
            None,
            None
        ),
        Some(naga::Literal::I32(0))
    );
    // i32::MIN = 0x80000000: first differing bit from sign is at position 30
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(i32::MIN),
            None,
            None
        ),
        Some(naga::Literal::I32(30))
    );
}

#[test]
fn math_first_trailing_bit_i32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstTrailingBit,
            naga::Literal::I32(0),
            None,
            None
        ),
        Some(naga::Literal::I32(-1))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstTrailingBit,
            naga::Literal::I32(12),
            None,
            None
        ),
        Some(naga::Literal::I32(2))
    );
}

#[test]
fn math_clamp_i32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Clamp,
            naga::Literal::I32(10),
            Some(naga::Literal::I32(0)),
            Some(naga::Literal::I32(5)),
        ),
        Some(naga::Literal::I32(5))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Clamp,
            naga::Literal::I32(-3),
            Some(naga::Literal::I32(0)),
            Some(naga::Literal::I32(5)),
        ),
        Some(naga::Literal::I32(0))
    );
}

#[test]
fn math_min_max_u32() {
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Min,
            naga::Literal::U32(10),
            Some(naga::Literal::U32(3)),
            None
        ),
        Some(naga::Literal::U32(3))
    );
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::Max,
            naga::Literal::U32(10),
            Some(naga::Literal::U32(3)),
            None
        ),
        Some(naga::Literal::U32(10))
    );
}

#[test]
fn math_abs_vector_component_wise() {
    let v = ConstValue::Vector {
        components: vec![
            naga::Literal::F32(-1.0),
            naga::Literal::F32(2.0),
            naga::Literal::F32(-3.0),
        ],
        size: naga::VectorSize::Tri,
        scalar: naga::Scalar::F32,
    };
    let result = eval_const_math(naga::MathFunction::Abs, v, None, None);
    match result {
        Some(ConstValue::Vector { components, .. }) => {
            assert_eq!(components[0], naga::Literal::F32(1.0));
            assert_eq!(components[1], naga::Literal::F32(2.0));
            assert_eq!(components[2], naga::Literal::F32(3.0));
        }
        other => panic!("expected vector, got {other:?}"),
    }
}

#[test]
fn math_min_vector_component_wise() {
    let a = ConstValue::Vector {
        components: vec![naga::Literal::F32(3.0), naga::Literal::F32(1.0)],
        size: naga::VectorSize::Bi,
        scalar: naga::Scalar::F32,
    };
    let b = ConstValue::Vector {
        components: vec![naga::Literal::F32(1.0), naga::Literal::F32(5.0)],
        size: naga::VectorSize::Bi,
        scalar: naga::Scalar::F32,
    };
    let result = eval_const_math(naga::MathFunction::Min, a, Some(b), None);
    match result {
        Some(ConstValue::Vector { components, .. }) => {
            assert_eq!(components[0], naga::Literal::F32(1.0));
            assert_eq!(components[1], naga::Literal::F32(1.0));
        }
        other => panic!("expected vector, got {other:?}"),
    }
}

#[test]
fn math_vector_size_mismatch_returns_none() {
    let a = ConstValue::Vector {
        components: vec![naga::Literal::F32(1.0), naga::Literal::F32(2.0)],
        size: naga::VectorSize::Bi,
        scalar: naga::Scalar::F32,
    };
    let b = ConstValue::Vector {
        components: vec![
            naga::Literal::F32(1.0),
            naga::Literal::F32(2.0),
            naga::Literal::F32(3.0),
        ],
        size: naga::VectorSize::Tri,
        scalar: naga::Scalar::F32,
    };
    assert_eq!(
        eval_const_math(naga::MathFunction::Min, a, Some(b), None),
        None,
        "mismatched vector sizes should not fold"
    );
}

#[test]
fn folds_math_sqrt_in_local_arena() {
    let mut arena = naga::Arena::new();
    let four = arena.append(
        naga::Expression::Literal(naga::Literal::F32(4.0)),
        Default::default(),
    );
    let sqrt_expr = arena.append(
        naga::Expression::Math {
            fun: naga::MathFunction::Sqrt,
            arg: four,
            arg1: None,
            arg2: None,
            arg3: None,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        !changed.is_empty(),
        "sqrt(4.0) should be folded to a literal"
    );
    assert_f32_literal(&arena, sqrt_expr, 2.0);
}

#[test]
fn folds_math_max_in_local_arena() {
    let mut arena = naga::Arena::new();
    let a = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let b = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let max_expr = arena.append(
        naga::Expression::Math {
            fun: naga::MathFunction::Max,
            arg: a,
            arg1: Some(b),
            arg2: None,
            arg3: None,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        !changed.is_empty(),
        "max(1.0, 2.0) should be folded to a literal"
    );
    assert_f32_literal(&arena, max_expr, 2.0);
}

#[test]
fn folds_math_clamp_in_local_arena() {
    let mut arena = naga::Arena::new();
    let val = arena.append(
        naga::Expression::Literal(naga::Literal::F32(5.0)),
        Default::default(),
    );
    let lo = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    let hi = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let clamp_expr = arena.append(
        naga::Expression::Math {
            fun: naga::MathFunction::Clamp,
            arg: val,
            arg1: Some(lo),
            arg2: Some(hi),
            arg3: None,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(!changed.is_empty(), "clamp(5.0, 0.0, 1.0) should be folded");
    assert_f32_literal(&arena, clamp_expr, 1.0);
}

#[test]
fn does_not_fold_math_with_non_constant_arg() {
    let mut arena = naga::Arena::new();
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let sqrt_expr = arena.append(
        naga::Expression::Math {
            fun: naga::MathFunction::Sqrt,
            arg: param,
            arg1: None,
            arg2: None,
            arg3: None,
        },
        Default::default(),
    );

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );
    assert!(
        changed.is_empty(),
        "sqrt(param) should not fold - arg is not constant"
    );
    assert!(
        matches!(arena[sqrt_expr], naga::Expression::Math { .. }),
        "expression should remain as Math"
    );
}

#[test]
fn e2e_math_sqrt_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let x = sqrt(4.0);
    return vec4f(x, x, x, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    // After folding, sqrt(4.0) should become 2.0 (literal).
    // The output should NOT contain "sqrt".
    assert!(
        !out.source.contains("sqrt"),
        "sqrt(4.0) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_max_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let m = max(1.0, 2.0);
    return vec4f(m, m, m, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("max"),
        "max(1.0, 2.0) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_abs_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let a = abs(-3.0);
    return vec4f(a, a, a, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("abs"),
        "abs(-3.0) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_sin_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let s = sin(0.0);
    return vec4f(s, s, s, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("sin"),
        "sin(0.0) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_floor_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let f = floor(1.7);
    return vec4f(f, f, f, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("floor"),
        "floor(1.7) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_clamp_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let c = clamp(5.0, 0.0, 1.0);
    return vec4f(c, c, c, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("clamp"),
        "clamp(5.0, 0.0, 1.0) should be folded away: {}",
        out.source
    );
}

#[test]
fn e2e_math_saturate_folds_through_pipeline() {
    let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let s = saturate(2.0);
    return vec4f(s, s, s, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    assert!(
        !out.source.contains("saturate"),
        "saturate(2.0) should be folded away: {}",
        out.source
    );
}

// Regression: absorbing rewrite must not corrupt vec <op> scalar types.

/// `vec3<f32> * 0.0` must stay well-typed end-to-end.  Prior to the
/// "both operands must be scalar Literal" gate, the Binary was rewritten
/// to the scalar literal `0.0`, corrupting every downstream use.  The
/// pipeline rolled back each sweep, wasting the sweep budget; with the
/// gate, either the Binary is left alone or a later stage folds it to a
/// correctly-typed vec3 zero.  Either outcome keeps the output valid.
#[test]
fn e2e_vec_times_scalar_zero_stays_valid() {
    let source = r#"
@fragment
fn main(@location(0) v: vec3<f32>) -> @location(0) vec4<f32> {
    let z = v * 0.0;
    return vec4<f32>(z, 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    // Output must parse back cleanly.
    crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
}

#[test]
fn e2e_vec_and_zero_stays_valid() {
    // Integer-typed shader I/O must carry `@interpolate(flat)` (integers
    // cannot be interpolated); naga 30 enforces this at validation.
    let source = r#"
@fragment
fn main(@location(0) @interpolate(flat) v: vec4<u32>) -> @location(0) @interpolate(flat) vec4<u32> {
    return v & vec4<u32>(0u);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
}

#[test]
fn e2e_bvec_or_true_stays_valid() {
    let source = r#"
@fragment
fn main(@location(0) v: vec3<f32>) -> @location(0) vec4<f32> {
    let b = vec3<bool>(v.x > 0.0, v.y > 0.0, v.z > 0.0);
    let t = b | vec3<bool>(true);
    return vec4<f32>(f32(t.x), f32(t.y), f32(t.z), 1.0);
}
"#;
    let out = crate::run(source, &crate::config::Config::default()).expect("source should compile");
    crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
}

/// Unit-level regression: even when called with a vec operand, the
/// simplify loop must NOT rewrite the Binary into a scalar literal.
/// The type-safe gate requires both operands be scalar `Literal`; a
/// `FunctionArgument` (typed as vec3) fails that gate.
#[test]
fn simplify_vec_times_zero_does_not_rewrite_to_scalar() {
    let mut arena = naga::Arena::new();
    let zero_f32 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    // FunctionArgument(0) is *typed* as a vector at the module level,
    // but within `fold_local_expressions` we have no type info for it.
    // The gate must therefore err on the side of not rewriting.
    let vec_param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let mul = arena.append(
        naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: vec_param,
            right: zero_f32,
        },
        Default::default(),
    );

    let (_folded, simplified) = fold_local(
        &mut arena,
        &[],
        &HashMap::new(),
        &naga::UniqueArena::new(),
        &HashMap::new(),
    );

    // The Binary must NOT be replaced by the scalar literal.
    assert!(
        matches!(arena[mul], naga::Expression::Binary { .. }),
        "Binary(vec * scalar_0) must remain Binary, got {:?}",
        arena[mul]
    );
    assert_eq!(
        simplified, 0,
        "no simplification should fire for vec * scalar_0"
    );
}

// Regression: literal cache correctness (NaN / -0.0 / smallest-handle invariant).
#[test]
fn literal_key_distinguishes_negative_zero_from_positive_zero() {
    // `f32::to_bits` gives -0.0 and +0.0 different bit patterns, so the
    // cache must not collapse them.
    assert_ne!(
        literal_key(naga::Literal::F32(0.0)),
        literal_key(naga::Literal::F32(-0.0)),
        "-0.0 and +0.0 must have distinct cache keys"
    );
}

#[test]
fn literal_key_distinguishes_distinct_nans() {
    // Two NaN bit patterns must be distinguishable; the cache must not
    // treat them as equal even though `f32::nan() != f32::nan()`.
    let nan_a = f32::from_bits(0x7FC00000); // quiet NaN
    let nan_b = f32::from_bits(0x7FC00001); // different payload
    assert_ne!(
        literal_key(naga::Literal::F32(nan_a)),
        literal_key(naga::Literal::F32(nan_b)),
        "distinct NaN payloads must have distinct cache keys"
    );
}

#[test]
fn literal_cache_smallest_handle_wins() {
    // Two Literal(1.0f32) appended at different handles; the cache
    // must point to the one with the smaller index so materialize_vector
    // can satisfy its topological-order check more often.
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let h_early = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    // Insert an unrelated expression between the two literals.
    let _spacer = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let _h_late = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );

    let cache = build_literal_cache(&arena);
    let got = cache
        .get(&literal_key(naga::Literal::F32(1.0)))
        .copied()
        .expect("cache must contain 1.0");
    assert_eq!(
        got.index(),
        h_early.index(),
        "cache must point to the smallest-index literal"
    );
}

#[test]
fn materialize_vector_uses_cache_for_component_lookup() {
    // Verify the hash-cache path still produces a Compose with the right
    // component handles (integration test for the API change).
    let mut types: naga::UniqueArena<naga::Type> = naga::UniqueArena::new();
    let vec3f_ty = types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar::F32,
            },
        },
        Default::default(),
    );
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let h1 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
    let h2 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(2.0)),
        Default::default(),
    );
    let h3 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(3.0)),
        Default::default(),
    );
    // `target` must be greater than all component handles.
    let target = arena.append(naga::Expression::FunctionArgument(0), Default::default());

    let lit_cache = build_literal_cache(&arena);
    let type_cache = build_vector_type_cache(&types);
    let out = materialize_vector(
        target,
        &[
            naga::Literal::F32(1.0),
            naga::Literal::F32(2.0),
            naga::Literal::F32(3.0),
        ],
        naga::VectorSize::Tri,
        naga::Scalar::F32,
        &lit_cache,
        &type_cache,
    )
    .expect("materialize_vector must succeed");

    match out {
        naga::Expression::Compose { ty, components } => {
            assert_eq!(ty, vec3f_ty);
            assert_eq!(components, vec![h1, h2, h3]);
        }
        other => panic!("expected Compose, got {other:?}"),
    }
}

#[test]
fn materialize_vector_rejects_component_at_or_after_target() {
    // When the only matching literal is at or after `target`, the
    // topological-safety check must reject the materialization.
    let mut types: naga::UniqueArena<naga::Type> = naga::UniqueArena::new();
    types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Vector {
                size: naga::VectorSize::Bi,
                scalar: naga::Scalar::F32,
            },
        },
        Default::default(),
    );
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    // target first, then the literal.
    let target = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );

    let lit_cache = build_literal_cache(&arena);
    let type_cache = build_vector_type_cache(&types);
    let out = materialize_vector(
        target,
        &[naga::Literal::F32(1.0), naga::Literal::F32(1.0)],
        naga::VectorSize::Bi,
        naga::Scalar::F32,
        &lit_cache,
        &type_cache,
    );
    assert!(
        out.is_none(),
        "component handle >= target must reject materialization"
    );
}

// MARK: Clone-purity gate regressions

/// `select(load, load, cond)` must not be folded to `load`, because
/// the rewrite clones the `Load` expression into a second arena slot
/// whose own `Emit` would re-execute the memory read at runtime - a
/// second observable load that can return a different value under
/// concurrent writes.  Pre-fix this passed the gate and produced
/// IR that load_dedup or downstream consumers may have miscompiled.
#[test]
fn select_collapse_skips_impure_load() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
fn helper(c: bool) -> u32 {
    let v = buf[0];
    return select(v, v, c);
}
@compute @workgroup_size(1) fn main() { _ = helper(true); }
"#;
    let mut module = naga::front::wgsl::parse_str(src).expect("parses");
    // Pre-pass: locate the Select.
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("helper"))
        .map(|(h, _)| h)
        .expect("helper exists");
    let select_handle = module.functions[f]
        .expressions
        .iter()
        .find_map(|(h, e)| matches!(e, naga::Expression::Select { .. }).then_some(h))
        .expect("Select expression present pre-fold");

    let _ = run_pass(&mut module);
    crate::io::validate_module(&module).expect("post-fold module valid");

    // The Select must NOT have been replaced with the Load operand.
    assert!(
        !matches!(
            module.functions[f].expressions[select_handle],
            naga::Expression::Load { .. }
        ),
        "select(load, load, cond) must not collapse to Load - the duplicate Emit \
             would re-execute the storage read and observe a second (potentially \
             distinct) value: got {:?}",
        module.functions[f].expressions[select_handle]
    );
}

/// `-(- Load)` involution must not be folded to `Load` for the same
/// reason: the inner Load gets cloned into the outer slot, doubling
/// the runtime read.  Pure operands (Literal, Constant, Splat of a
/// literal, etc.) ARE safe and should still fold; only impure
/// operands are gated.
#[test]
fn involution_skips_impure_load() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<i32>;
fn helper() -> i32 {
    let v = buf[0];
    return -(-v);
}
@compute @workgroup_size(1) fn main() { _ = helper(); }
"#;
    let mut module = naga::front::wgsl::parse_str(src).expect("parses");
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("helper"))
        .map(|(h, _)| h)
        .expect("helper exists");
    // The outermost Unary is the `-(-v)` we are testing.
    let outer_unary = module.functions[f]
        .expressions
        .iter()
        .find_map(|(h, e)| match e {
            naga::Expression::Unary { expr: inner, .. } => {
                match module.functions[f].expressions[*inner] {
                    naga::Expression::Unary { .. } => Some(h),
                    _ => None,
                }
            }
            _ => None,
        })
        .expect("outer Unary present pre-fold");

    let _ = run_pass(&mut module);
    crate::io::validate_module(&module).expect("post-fold module valid");

    assert!(
        !matches!(
            module.functions[f].expressions[outer_unary],
            naga::Expression::Load { .. }
        ),
        "-(- Load) must not collapse to Load: got {:?}",
        module.functions[f].expressions[outer_unary]
    );
}

/// `x + 0` where `x = Load` must not be folded to `Load` either:
/// the identity rewrite clones the non-literal operand into the
/// Binary's slot, and a `Load` there is unsound.
#[test]
fn identity_skips_impure_load() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<i32>;
fn helper() -> i32 {
    let v = buf[0];
    return v + 0;
}
@compute @workgroup_size(1) fn main() { _ = helper(); }
"#;
    let mut module = naga::front::wgsl::parse_str(src).expect("parses");
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("helper"))
        .map(|(h, _)| h)
        .expect("helper exists");
    let binary_handle = module.functions[f]
        .expressions
        .iter()
        .find_map(|(h, e)| matches!(e, naga::Expression::Binary { .. }).then_some(h))
        .expect("Binary present pre-fold");

    let _ = run_pass(&mut module);
    crate::io::validate_module(&module).expect("post-fold module valid");

    assert!(
        !matches!(
            module.functions[f].expressions[binary_handle],
            naga::Expression::Load { .. }
        ),
        "Load + 0 identity must not collapse to Load: got {:?}",
        module.functions[f].expressions[binary_handle]
    );
}

/// Caching an `AbstractInt`/`AbstractFloat` constant would let the
/// fold loop emit a function-arena `Literal(AbstractInt)`, which
/// naga's validator rejects - the whole pass would roll back every
/// sweep.  The shape is only reachable by hand-built IR because
/// naga's WGSL frontend concretises constants before any pass runs.
#[test]
fn build_constant_literal_cache_skips_abstract_literals() {
    let mut module = naga::Module::default();

    let i32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::I32),
        },
        naga::Span::UNDEFINED,
    );
    let abstract_int_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::ABSTRACT_INT),
        },
        naga::Span::UNDEFINED,
    );

    let concrete_init = module.global_expressions.append(
        naga::Expression::Literal(naga::Literal::I32(7)),
        naga::Span::UNDEFINED,
    );
    let abstract_init = module.global_expressions.append(
        naga::Expression::Literal(naga::Literal::AbstractInt(7)),
        naga::Span::UNDEFINED,
    );

    let concrete = module.constants.append(
        naga::Constant {
            name: Some("C".into()),
            ty: i32_ty,
            init: concrete_init,
        },
        naga::Span::UNDEFINED,
    );
    let abstract_int = module.constants.append(
        naga::Constant {
            name: Some("A".into()),
            ty: abstract_int_ty,
            init: abstract_init,
        },
        naga::Span::UNDEFINED,
    );

    let cache = build_constant_literal_cache(&module);

    assert_eq!(
        cache.get(&concrete),
        Some(&naga::Literal::I32(7)),
        "concrete-typed constant must land in the cache",
    );
    assert!(
        !cache.contains_key(&abstract_int),
        "abstract literal must be filtered out of the cache; caching one \
             would silently roll the whole pass back every sweep",
    );
}

/// `Access` whose index folds to a constant, into a syntactic array
/// `Compose`, must fold to the picked element.  naga materialises a
/// dynamically-indexed function-scope `const` array as a full composite at
/// the use site and load_dedup forwards the index variable's stored
/// literal, so without this fold the emitter ships the entire array inline
/// (`array<u32,2310>(...)[0]`) and only the NEXT minification round
/// collapses it - the signature idempotence gap.
#[test]
fn access_with_const_index_into_array_compose_folds_to_element() {
    let mut types = naga::UniqueArena::new();
    let u32_ty = types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::U32),
        },
        naga::Span::UNDEFINED,
    );
    let arr_ty = types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Array {
                base: u32_ty,
                size: naga::ArraySize::Constant(std::num::NonZeroU32::new(3).unwrap()),
                stride: 4,
            },
        },
        naga::Span::UNDEFINED,
    );

    let mut arena = naga::Arena::new();
    let c0 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(10)),
        naga::Span::UNDEFINED,
    );
    let c1 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(20)),
        naga::Span::UNDEFINED,
    );
    let c2 = arena.append(
        naga::Expression::Literal(naga::Literal::U32(30)),
        naga::Span::UNDEFINED,
    );
    let compose = arena.append(
        naga::Expression::Compose {
            ty: arr_ty,
            components: vec![c0, c1, c2],
        },
        naga::Span::UNDEFINED,
    );
    let idx = arena.append(
        naga::Expression::Literal(naga::Literal::I32(1)),
        naga::Span::UNDEFINED,
    );
    let access = arena.append(
        naga::Expression::Access {
            base: compose,
            index: idx,
        },
        naga::Span::UNDEFINED,
    );
    let oob_idx = arena.append(
        naga::Expression::Literal(naga::Literal::I32(7)),
        naga::Span::UNDEFINED,
    );
    let oob = arena.append(
        naga::Expression::Access {
            base: compose,
            index: oob_idx,
        },
        naga::Span::UNDEFINED,
    );

    let refcounts = vec![1u32; arena.len()];
    let (folded, _) = fold_local(
        &mut arena,
        &refcounts,
        &HashMap::new(),
        &types,
        &HashMap::new(),
    );

    assert!(
        matches!(
            arena[access],
            naga::Expression::Literal(naga::Literal::U32(20))
        ),
        "in-bounds pick must fold to the element literal, got {:?}",
        arena[access]
    );
    assert!(
        folded.contains(&access),
        "folded pick leaves its Emit range"
    );
    assert!(
        matches!(arena[oob], naga::Expression::Access { .. }),
        "out-of-bounds pick must decline, got {:?}",
        arena[oob]
    );
}

/// Float `%` folds only while `trunc(a/b)` is exactly representable in the
/// operand type: WGSL lowers `a % b` to `a - b*trunc(a/b)` in that
/// precision, so beyond the 2^24 (f32) / 2^53 (f64) quotient limit the
/// runtime value diverges from Rust's exact fmod and the fold must
/// decline.
#[test]
fn float_modulo_beyond_trunc_precision_declines() {
    use naga::BinaryOperator as B;
    use naga::Literal as L;
    // Control: small quotient folds exactly.
    assert_eq!(
        eval_binary(B::Modulo, L::F32(7.5), L::F32(2.0)),
        Some(L::F32(1.5))
    );
    // Quotient 2^25 >= 2^24: trunc(a/b) is not exactly representable.
    assert_eq!(
        eval_binary(B::Modulo, L::F32(67_108_864.0), L::F32(2.0)),
        None
    );
    // f64 mirror: quotient 2^54 >= 2^53 declines, small folds.
    assert_eq!(
        eval_binary(B::Modulo, L::F64(7.5), L::F64(2.0)),
        Some(L::F64(1.5))
    );
    assert_eq!(
        eval_binary(B::Modulo, L::F64(36_028_797_018_963_968.0), L::F64(2.0)),
        None
    );
}
