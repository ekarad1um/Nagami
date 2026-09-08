use super::*;
use crate::config::Config;
use crate::handle_set::{HandleMap, HandleSet};
use crate::passes::expr_util::cast_width8_to;

/// Shim for [`fold_local_expressions`]: bare arenas have no `Emit` ranges,
/// so mapping every handle to one shared range models the safe co-located
/// case and the store-aware relocation guard is a no-op; they have no
/// locals either, so nothing reads as a zero initialiser.  Cross-range tests
/// call [`fold_local_expressions`] with their own map.
fn fold_local(
    arena: &mut naga::Arena<naga::Expression>,
    refcounts: &[u32],
    const_literals: &HandleMap<naga::Constant, naga::Literal>,
    types: &naga::UniqueArena<naga::Type>,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> (HandleSet<naga::Expression>, usize) {
    let ranges = vec![0u32; arena.len()];
    fold_local_expressions(
        arena,
        refcounts,
        &ranges,
        const_literals,
        /*zero_locals=*/ &Default::default(),
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
        name_log: None,
    };

    pass.run(module, &ctx).expect("const fold pass should run")
}

fn constant_refs(function: &naga::Function) -> usize {
    function
        .expressions
        .iter()
        .filter(|(_, e)| matches!(e, naga::Expression::Constant(_)))
        .count()
}

#[test]
fn library_module_keeps_named_constant_references() {
    // Without an entry point `PI` survives compaction whatever the fold does,
    // so its runtime references stay; an entry-point module folds them.
    // (`2.0 * PI` never reaches the pass: naga's front-end evaluates it.)
    let lib = r#"
const PI: f32 = 3.14159;
fn a(x: f32) -> f32 { return x * PI; }
fn b(x: f32) -> f32 { return x / PI; }
"#;
    let mut module = naga::front::wgsl::parse_str(lib).expect("source should parse");
    run_pass(&mut module);
    let refs: usize = module.functions.iter().map(|(_, f)| constant_refs(f)).sum();
    assert_eq!(refs, 2, "library module must keep both `PI` references");

    let entry = format!(
        "{lib}@fragment fn fs(@location(0) x: f32) -> @location(0) vec4f {{ return vec4f(a(x) + b(x)); }}"
    );
    let mut module = naga::front::wgsl::parse_str(&entry).expect("source should parse");
    run_pass(&mut module);
    let refs: usize = module.functions.iter().map(|(_, f)| constant_refs(f)).sum();
    assert_eq!(refs, 0, "entry-point module folds `PI` into literals");

    // Unmangled, the name `PI` stays as written and the literal is the
    // cheaper spelling: the library gate follows the mangle setting.
    let mut module = naga::front::wgsl::parse_str(lib).expect("source should parse");
    let config = Config {
        mangle: Some(false),
        ..Config::default()
    };
    let ctx = PassContext {
        config: &config,
        name_log: None,
    };
    ConstFoldPass
        .run(&mut module, &ctx)
        .expect("const fold pass should run");
    let refs: usize = module.functions.iter().map(|(_, f)| constant_refs(f)).sum();
    assert_eq!(
        refs, 0,
        "an unmangled library module folds `PI` into literals"
    );
}

#[test]
fn bitcast_literal_reinterprets_bits_and_declines_unspellable_floats() {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    assert_eq!(bitcast_literal(L::I32(-1), K::Uint), Some(L::U32(u32::MAX)));
    assert_eq!(bitcast_literal(L::U32(2), K::Sint), Some(L::I32(2)));
    assert_eq!(
        bitcast_literal(L::F32(1.0), K::Uint),
        Some(L::U32(0x3f80_0000))
    );
    assert_eq!(
        bitcast_literal(L::U32(0x3f80_0000), K::Float),
        Some(L::F32(1.0))
    );
    // inf, NaN, a subnormal and both zeros keep the runtime bitcast, on
    // either side: the sign of a zero is the platform's call.
    assert_eq!(bitcast_literal(L::U32(0x7f80_0000), K::Float), None);
    assert_eq!(bitcast_literal(L::U32(0x7fc0_0000), K::Float), None);
    assert_eq!(bitcast_literal(L::U32(1), K::Float), None);
    assert_eq!(bitcast_literal(L::U32(0), K::Float), None);
    assert_eq!(bitcast_literal(L::U32(0x8000_0000), K::Float), None);
    assert_eq!(bitcast_literal(L::F32(0.0), K::Uint), None);
    assert_eq!(bitcast_literal(L::F32(-0.0), K::Uint), None);
    assert_eq!(bitcast_literal(L::F32(1e-40), K::Uint), None);
    assert_eq!(
        bitcast_literal(L::F16(half::f16::from_f32(1.0)), K::Uint),
        None
    );
    assert_eq!(bitcast_literal(L::U16(0x3c00), K::Float), None);
    assert_eq!(bitcast_literal(L::AbstractInt(1), K::Uint), None);
    assert_eq!(bitcast_literal(L::Bool(true), K::Uint), None);
}

#[test]
fn folds_scalar_bitcast_of_literal() {
    // `bitcast<i32>(2u)` is a const-expression neither naga's evaluator nor
    // tint leaves alone; folding it lets the tree above collapse.
    let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(1) fn main() { out[0] = clamp(sign(bitcast<i32>(2u)), -100i, 100i); }
"#;
    let mut module = naga::front::wgsl::parse_str(src).expect("source should parse");
    run_pass(&mut module);
    let func = &module.entry_points[0].function;
    assert!(
        !func
            .expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::As { convert: None, .. })),
        "the literal bitcast must fold away"
    );
    assert!(
        func.expressions
            .iter()
            .any(|(_, e)| matches!(e, naga::Expression::Literal(naga::Literal::I32(1)))),
        "sign / clamp above the bitcast must fold to 1"
    );
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert_eq!(changed.len(), 2, "both nested operations should be folded");
    assert_f32_literal(&arena, add, 3.0);
    assert_f32_literal(&arena, mul, 9.0);
}

#[test]
fn de_morgan_negated_equality_folds_in_place() {
    use naga::{BinaryOperator as B, UnaryOperator as U};
    // The `!` node is rewritten in place into the flipped comparison and the
    // dead comparison's Emit is dropped; FunctionArgument operands keep
    // const-folding from pre-empting the rewrite.
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
            &Default::default(),
            &naga::UniqueArena::new(),
            &Default::default(),
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
            folded.contains(cmp),
            "the dead comparison's Emit must be dropped"
        );
    }
}

#[test]
fn de_morgan_shared_equality_is_not_folded() {
    use naga::{BinaryOperator as B, UnaryOperator as U};
    // A comparison with another consumer (refcount > 1) must not be rewritten
    // in place; the emitter keeps `!name`.
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
    let refcounts = vec![0u32, 0, 2, 0];
    let (folded, _) = fold_local(
        &mut arena,
        &refcounts,
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        !folded.contains(cmp),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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

    let mut const_literals = HandleMap::default();
    const_literals.insert(constant_handle, naga::Literal::F32(41.0));

    let (changed, _) = fold_local(
        &mut arena,
        &[],
        &const_literals,
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
    // naga's validator rejects non-finite `Literal::F32`/`F64`
    // (`LiteralError::NonFinite`), so Negate refuses NaN and +/-Inf even
    // though negating Inf is valid IEEE.
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
// Each pins a fold unsound vs the WGSL spec: `min/max/clamp` must propagate
// NaN (Rust's treat it as missing), `atan2(0,0)` is implementation-defined,
// `pow(0, 0)` and `pow(0, negative)` are undefined, `sign(NaN)` would emit a
// NaN literal.

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
    // Also guards the `f32::clamp` panic on a NaN bound.
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
    // GPUs may return 0, +/-pi/2, or pi; Rust returns 0.0.
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
    // WGSL defines runtime `MIN / -1` as MIN; declined, the literal pair
    // (only manufactured by nagami's transforms) fails naga's re-parse
    // const-eval with no fallback.
    let r = eval_binary(
        naga::BinaryOperator::Divide,
        naga::Literal::I32(i32::MIN),
        naga::Literal::I32(-1),
    );
    assert_eq!(r, Some(naga::Literal::I32(i32::MIN)));
}

#[test]
fn i32_modulo_min_by_neg1_folds_to_defined_value() {
    // WGSL defines runtime `MIN % -1` as 0.
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
    // WGSL defines `MIN / -1` as MIN.
    let r = eval_binary(
        naga::BinaryOperator::Divide,
        naga::Literal::I64(i64::MIN),
        naga::Literal::I64(-1),
    );
    assert_eq!(r, Some(naga::Literal::I64(i64::MIN)));
}

#[test]
fn i64_modulo_min_by_neg1_folds_to_defined_value() {
    // WGSL defines `MIN % -1` as 0.
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

// naga concretises a shift amount to u32, so 64-bit-base shifts present as
// `U64/I64 << U32`, never `<< U64`.

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
    // A concrete literal pair here is a runtime expression (const contexts
    // died at naga's front-end), where WGSL defines the plain bit-pattern
    // shift; declining poisons emission (the pair fails re-parse const-eval).
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

// WGSL's sign-changing `<<` shader-creation error applies to const contexts,
// which naga's front-end rejected at ingest; a concrete pair reaching the
// fold is a runtime expression with a defined bit-pattern result, and
// declining kills the emission on re-parse.
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
    // Boundary: the sign bit is unchanged in both.
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
    // A WGSL shift amount is always `u32`; a `U64` RHS would match no arm and
    // return `None` for the wrong reason.
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

// Float Add/Subtract identity is not folded (`is_additive_identity_zero`,
// IEEE-754 signed zero); integer zero identity is safe.

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

/// IEEE 754 gives `(-0.0) + (+0.0) = +0.0`, so `x + 0.0 -> x` mis-signs
/// `x = -0.0`; neither side of a float Add/Subtract identity folds.
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
    let result =
        check_identity_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_mul_one_right() {
    let a = make_identity_arena();
    let result =
        check_identity_operand(naga::BinaryOperator::Multiply, a.param, a.one_f32, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_div_one_right() {
    let a = make_identity_arena();
    let result = check_identity_operand(naga::BinaryOperator::Divide, a.param, a.one_f32, &a.arena);
    assert_eq!(result, Some(a.param));
}

#[test]
fn identity_div_one_left_not_eliminated() {
    let a = make_identity_arena();
    let result = check_identity_operand(naga::BinaryOperator::Divide, a.one_f32, a.param, &a.arena);
    assert_eq!(result, None);
}

#[test]
fn identity_integer_types() {
    let a = make_identity_arena();
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Add, a.zero_i32, a.param, &a.arena),
        Some(a.param)
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, a.one_i32, a.param, &a.arena),
        Some(a.param)
    );
}

#[test]
fn identity_no_false_positive_for_other_ops() {
    let a = make_identity_arena();
    // `% 1` is a remainder, `& 0` is absorbing not identity, `<< 0.0` is the
    // wrong type.
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
    // The F16 multiplicative identity folds; the additive F16 identities are
    // declined for the same signed-zero reason as f32.
    use half::f16;
    let mut arena = naga::Arena::new();
    let one_f16 = arena.append(
        naga::Expression::Literal(naga::Literal::F16(f16::from_f32(1.0))),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());

    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, param, one_f16, &arena),
        Some(param)
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Multiply, one_f16, param, &arena),
        Some(param)
    );
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::Divide, param, one_f16, &arena),
        Some(param)
    );
}

#[test]
fn identity_fold_non_emittable_added_to_folded() {
    // The replacement `FunctionArgument` is non-emittable, so it must enter
    // `folded` for Emit removal; integer zero because the float identity is
    // refused.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(
        identity > 0,
        "0 + param should trigger identity elimination"
    );
    assert!(
        folded.contains(add),
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
    // The replacement `Binary(Mul)` is emittable, so it must stay out of
    // `folded`; integer zero because the float identity is refused.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(
        identity > 0,
        "0 + (a*b) should trigger identity elimination"
    );
    assert!(
        !folded.contains(add),
        "emittable replacement must NOT be in folded set"
    );
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

/// `0u + Load(p)` with a uniquely-owned Load folds to a Load clone, and the
/// original Load enters `folded` so the rebuild drops its Emit entry;
/// otherwise the let-binding survives and runs a second read.
#[test]
fn identity_fold_unique_impure_clone_drops_source() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    // `FunctionArgument(0)` stands in for a pointer: the fold observes
    // shapes, not types.
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

    // No Function to walk, so count refs inline.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(
        identity > 0,
        "0u + Load should fold once Load is uniquely owned"
    );
    assert!(
        folded.contains(load),
        "uniquely-owned impure source must be in `folded` so the \
             rebuild walk drops its Emit-range entry"
    );
    assert!(
        !folded.contains(add),
        "the cloned Load at `add` is still emittable - must NOT be \
             dropped from its own Emit range"
    );
    assert!(
        matches!(arena[add], naga::Expression::Load { .. }),
        "expected Load after fold, got {:?}",
        arena[add]
    );
}

/// A uniquely-owned impure operand in a different `Emit` range than the
/// folding Binary must not be relocated: a statement (a hazardous write) sits
/// between them, so cloning the Load would move the read past the write.
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

    // Range 0 vs range 1: a statement separates them.
    let mut ranges = vec![NO_EMIT; arena.len()];
    ranges[load.index()] = 0;
    ranges[add.index()] = 1;
    let (folded, identity) = fold_local_expressions(
        &mut arena,
        &refcounts,
        &ranges,
        /*const_literals=*/ &Default::default(),
        /*zero_locals=*/ &Default::default(),
        &naga::UniqueArena::new(),
        /*vector_type_cache=*/ &Default::default(),
    );
    assert_eq!(
        identity, 0,
        "cross-`Emit`-range impure operand must NOT be relocated by the identity fold"
    );
    assert!(
        !folded.contains(load),
        "the original Load must stay in its Emit range (not dropped)"
    );
    assert!(
        matches!(arena[add], naga::Expression::Binary { .. }),
        "the Binary must be left untouched, got {:?}",
        arena[add]
    );
}

/// The involution arm of the store-aware guard: `-(-x)` with the inner Load
/// in another `Emit` range.  Models
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

    let mut ranges = vec![NO_EMIT; arena.len()];
    ranges[load.index()] = 0;
    ranges[neg1.index()] = 1;
    ranges[neg2.index()] = 1;
    let (folded, identity) = fold_local_expressions(
        &mut arena,
        &refcounts,
        &ranges,
        /*const_literals=*/ &Default::default(),
        /*zero_locals=*/ &Default::default(),
        &naga::UniqueArena::new(),
        /*vector_type_cache=*/ &Default::default(),
    );
    assert_eq!(
        identity, 0,
        "cross-`Emit`-range impure inner must NOT be relocated by the involution fold"
    );
    assert!(
        !folded.contains(load),
        "the inner Load must stay in its Emit range"
    );
    assert!(
        matches!(arena[neg2], naga::Expression::Unary { .. }),
        "the outer Unary must be left untouched, got {:?}",
        arena[neg2]
    );
}

/// A multi-referenced impure operand stays alive after the rewrite, so
/// cloning would emit a second observable Load.
#[test]
fn identity_fold_multi_ref_impure_blocked() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let ptr = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    let load = arena.append(naga::Expression::Load { pointer: ptr }, Default::default());
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert_eq!(
        identity, 0,
        "multi-ref impure operand must NOT trigger identity fold"
    );
    assert!(
        !folded.contains(load),
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

    // Cached: the Constant folds to a literal first, then `1 * 41.0`
    // identity-folds.
    let mut const_literals = HandleMap::default();
    const_literals.insert(constant_handle, naga::Literal::F32(41.0));
    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &const_literals,
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(folded.contains(mul), "result should be in folded set");
    assert_f32_literal(&arena, mul, 41.0);

    // Uncached: the Constant stays and identity makes `mul` a non-emittable
    // Constant.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(
        identity2 > 0,
        "1 * const should trigger identity elimination"
    );
    assert!(
        folded2.contains(mul2),
        "Constant replacement must be in folded set (non-emittable)"
    );
    assert!(
        matches!(arena2[mul2], naga::Expression::Constant(_)),
        "expected Constant, got {:?}",
        arena2[mul2]
    );
}

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

/// `x * 0.0` carries the product's IEEE sign (NaN for non-finite `x`), which
/// cloning the matched zero cannot reproduce; only the sign-aware
/// `eval_binary` may fold the both-literal case.
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
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::And, param, zero_u32, &arena),
        Some(zero_u32)
    );
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
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::InclusiveOr, param, all_ones, &arena),
        Some(all_ones)
    );
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
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::Add, a.zero_f32, a.param, &a.arena),
        None
    );
    assert_eq!(
        check_absorbing_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena),
        None
    );
}

#[test]
fn identity_or_zero() {
    let mut arena = naga::Arena::new();
    let zero = arena.append(
        naga::Expression::Literal(naga::Literal::U32(0)),
        Default::default(),
    );
    let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::InclusiveOr, param, zero, &arena),
        Some(param)
    );
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
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::ExclusiveOr, param, zero, &arena),
        Some(param)
    );
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
    assert_eq!(
        check_identity_operand(naga::BinaryOperator::And, param, all_ones, &arena),
        Some(param)
    );
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
    // i32 -1 is all ones.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(count > 0, "-(-x) should be simplified");
    assert!(
        matches!(arena[neg2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[neg2]
    );
    assert!(folded.contains(neg2));
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(count > 0, "!(!x) should be simplified");
    assert!(
        matches!(arena[not2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[not2]
    );
    assert!(folded.contains(not2));
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(count > 0, "~(~x) should be simplified");
    assert!(
        matches!(arena[not2], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[not2]
    );
    assert!(folded.contains(not2));
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
    assert!(
        !folded.contains(neg2),
        "emittable involution result must NOT be in folded set"
    );
}

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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert!(count > 0, "select(x, x, cond) should be simplified");
    assert!(
        matches!(arena[sel], naga::Expression::FunctionArgument(0)),
        "expected FunctionArgument(0), got {:?}",
        arena[sel]
    );
    assert!(folded.contains(sel));
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    // `resolve_const_value` needs literal arms, so the Select stays.
    assert!(
        matches!(arena[sel], naga::Expression::Select { .. }),
        "select with different arms should not be simplified by the simplify loop, got {:?}",
        arena[sel]
    );
    assert!(!folded.contains(sel));
}

#[test]
fn absorbing_fold_mul_zero_param_not_rewritten() {
    // Rewriting `param * 0.0` to the scalar `0.0` is invalid IR whenever
    // `param` is a vector or matrix; the simplify-loop absorbing rewrite fires
    // only when both operands are scalar Literals, and a FunctionArgument has
    // unknown type.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
    assert!(!folded.contains(mul));
}

#[test]
fn absorbing_fold_and_zero_u32_param_not_rewritten() {
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
    assert!(!folded.contains(and));
}

#[test]
fn absorbing_fold_mul_zero_both_literal_produces_literal() {
    // Both operands scalar Literals is the one type-safe absorbing case;
    // `resolve_const_value` usually folds it first, so this exercises the
    // safety net.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );
    assert_f32_literal(&arena, mul, 0.0);
}

#[test]
fn absorbing_fold_logical_and_false_with_non_literal_rhs_rewritten() {
    // `LogicalAnd`/`LogicalOr` are strictly scalar-bool, so absorbing is
    // type-safe with a non-literal operand; dead_branch's re-sugar of
    // `var tmp = false && (j < -8)` produces this shape, and without the fold
    // the dead loop `for(;0<0&&j<0-8;){}` cannot collapse and the output
    // grows.
    let mut arena = naga::Arena::new();
    let false_lit = arena.append(
        naga::Expression::Literal(naga::Literal::Bool(false)),
        Default::default(),
    );
    // The FunctionArgument stands in for the non-literal `j < -8`.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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

/// If the absorbing rewrite declines `false && (j < -8)`, the dead loop
/// reaches the emitter as `loop { var a = false && A<-8; if !(a) { break; } }`
/// and the output grows (66 -> 87 bytes).
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
    let access = arena.append(
        naga::Expression::AccessIndex { base: v, index: 1 },
        Default::default(),
    );

    let (folded, _) = fold_local(
        &mut arena,
        &[],
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_f32_literal(&arena, access, 20.0);
    assert!(
        folded.contains(access),
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
        &Default::default(),
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, mul, &[20.0, 30.0, 40.0]);
}

#[test]
fn vector_zero_value_stays_non_emittable() {
    // ZeroValue is non-emittable; replacing it with an emittable Compose is
    // invalid IR.
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
        &Default::default(),
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, add, &[1.0, 2.0, 3.0]);
}

#[test]
fn vector_splat_binary_scalar_add() {
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
    // Result literal pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, add, &[5.0, 5.0]);
}

#[test]
fn vector_nested_chain_folds() {
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
    // Materialization needs the result literals to precede the target.
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
    // No 101.0 / 202.0 literals on purpose.

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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert!(
        matches!(arena[add], naga::Expression::Binary { .. }),
        "should remain Binary when materialization fails, got {:?}",
        arena[add]
    );
}

#[test]
fn vector_compose_mixed_scalar_and_vector() {
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, v4, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn vector_relational_op_produces_bool_vector() {
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
    // Result literals pre-placed for materialization.
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert_compose_of_f32(&arena, sel, &[1.0, 2.0]);
}

#[test]
fn scalar_zero_value_resolves() {
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
        &Default::default(),
        &types,
        &build_vector_type_cache(&types),
    );
    assert!(
        matches!(arena[zv], naga::Expression::Literal(naga::Literal::F32(v)) if v == 0.0),
        "ZeroValue(f32) should fold to Literal(0.0), got {:?}",
        arena[zv]
    );
    assert!(
        folded.contains(zv),
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
    // abs(i32::MIN) has no positive representation; naga would raise an
    // execution error.
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
    // Round is ties-to-even.
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
    // `v - v.floor()` near the float range boundary can produce NaN/Inf; the
    // `finite_*` guard must refuse so the literal stays representable.
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
    assert_eq!(
        eval_math_scalar(naga::MathFunction::Exp, naga::Literal::F32(0.0), None, None),
        Some(naga::Literal::F32(1.0))
    );
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
    // 0 and -1 both return -1.
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
    // WGSL precondition `e1 >= 0.0`: a negative base is undefined.
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
    // WGSL: sign(-0.0) is 0.0 (or -0.0, implementation-defined).
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
    // -2 = 0xFFFFFFFE: the first bit differing from the sign is at 0.
    assert_eq!(
        eval_math_scalar(
            naga::MathFunction::FirstLeadingBit,
            naga::Literal::I32(-2),
            None,
            None
        ),
        Some(naga::Literal::I32(0))
    );
    // i32::MIN = 0x80000000: the first bit differing from the sign is at 30.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
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

/// Rewriting `vec3<f32> * 0.0` to the scalar `0.0` corrupts every downstream
/// use and rolls the pipeline back each sweep; with the both-scalar-Literal
/// gate the Binary is left alone or later folded to a typed vec3 zero.
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
    crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
}

#[test]
fn e2e_vec_and_zero_stays_valid() {
    // Integer shader I/O must carry `@interpolate(flat)`.
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

/// The simplify loop must not rewrite `vec * 0.0` to a scalar literal: the
/// gate requires both operands to be scalar `Literal`s, which a
/// `FunctionArgument` fails.
#[test]
fn simplify_vec_times_zero_does_not_rewrite_to_scalar() {
    let mut arena = naga::Arena::new();
    let zero_f32 = arena.append(
        naga::Expression::Literal(naga::Literal::F32(0.0)),
        Default::default(),
    );
    // `fold_local_expressions` has no type info for the argument.
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
        &Default::default(),
        &naga::UniqueArena::new(),
        &Default::default(),
    );

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

#[test]
fn literal_key_distinguishes_negative_zero_from_positive_zero() {
    // The key is `f32::to_bits`, which separates them.
    assert_ne!(
        literal_key(naga::Literal::F32(0.0)),
        literal_key(naga::Literal::F32(-0.0)),
        "-0.0 and +0.0 must have distinct cache keys"
    );
}

#[test]
fn literal_key_distinguishes_distinct_nans() {
    // Distinct NaN payloads must not merge even though `NaN != NaN`.
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
    // The smallest index wins so `materialize_vector`'s topological-order
    // check succeeds more often.
    let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
    let h_early = arena.append(
        naga::Expression::Literal(naga::Literal::F32(1.0)),
        Default::default(),
    );
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
    // The target precedes the literal.
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

/// Folding `select(load, load, cond)` to `load` clones the `Load` into a
/// second arena slot whose own `Emit` re-executes the read: a second
/// observable load that can differ under concurrent writes.
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

/// `-(- Load)` would clone the inner Load into the outer slot, doubling the
/// read; pure operands (Literal, Constant, Splat of a literal) still fold.
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

/// The identity rewrite clones the non-literal operand into the Binary's
/// slot; a `Load` there doubles the read.
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

/// Caching an `AbstractInt`/`AbstractFloat` constant lets the fold loop emit
/// a function-arena `Literal(AbstractInt)` naga's validator rejects, rolling
/// the pass back every sweep; only hand-built IR reaches this since the
/// frontend concretises constants.
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
        cache.get(concrete),
        Some(&naga::Literal::I32(7)),
        "concrete-typed constant must land in the cache",
    );
    assert!(
        !cache.contains_key(abstract_int),
        "abstract literal must be filtered out of the cache; caching one \
             would silently roll the whole pass back every sweep",
    );
}

/// `build_constant_literal_cache` reads literals straight out of the arena,
/// which is complete only because `fold_global_expressions` ran first and
/// collapsed initializer chains.  naga's frontend concretises every constant,
/// so only hand-built IR (via `run_module`) or a pass that rewrites
/// `global_expressions` reaches an unfolded chain - and reordering the two
/// would silently stop inlining those constants.
#[test]
fn build_constant_literal_cache_needs_global_folding_first() {
    let mut module = naga::Module::default();
    let i32_ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar::I32),
        },
        naga::Span::UNDEFINED,
    );
    let konst = |module: &mut naga::Module, name: &str, init| {
        let init = module
            .global_expressions
            .append(init, naga::Span::UNDEFINED);
        module.constants.append(
            naga::Constant {
                name: Some(name.into()),
                ty: i32_ty,
                init,
            },
            naga::Span::UNDEFINED,
        )
    };
    // const A = 7;  const B = A;  const C = B + 1;
    let a = konst(
        &mut module,
        "A",
        naga::Expression::Literal(naga::Literal::I32(7)),
    );
    let b = konst(&mut module, "B", naga::Expression::Constant(a));
    let b_init = module.constants[b].init;
    let one = module.global_expressions.append(
        naga::Expression::Literal(naga::Literal::I32(1)),
        naga::Span::UNDEFINED,
    );
    let c = konst(
        &mut module,
        "C",
        naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: b_init,
            right: one,
        },
    );

    let unfolded = build_constant_literal_cache(&module);
    assert_eq!(unfolded.get(a), Some(&naga::Literal::I32(7)));
    assert!(
        !unfolded.contains_key(b) && !unfolded.contains_key(c),
        "without the global fold a chained initializer is not yet a Literal",
    );

    let vector_type_cache = build_vector_type_cache(&module.types);
    fold_global_expressions(&mut module, &vector_type_cache);
    let folded = build_constant_literal_cache(&module);
    assert_eq!(folded.get(b), Some(&naga::Literal::I32(7)));
    assert_eq!(folded.get(c), Some(&naga::Literal::I32(8)));
}

/// naga materialises a dynamically-indexed function-scope `const` array as a
/// full composite at the use site and load_dedup forwards the index literal;
/// without this fold the emitter ships the whole array inline and only the
/// next round collapses it (idempotence gap).
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
        &Default::default(),
        &types,
        &Default::default(),
    );

    assert!(
        matches!(
            arena[access],
            naga::Expression::Literal(naga::Literal::U32(20))
        ),
        "in-bounds pick must fold to the element literal, got {:?}",
        arena[access]
    );
    assert!(folded.contains(access), "folded pick leaves its Emit range");
    assert!(
        matches!(arena[oob], naga::Expression::Access { .. }),
        "out-of-bounds pick must decline, got {:?}",
        arena[oob]
    );
}

/// Float `%` folds only while `trunc(a/b)` is exactly representable in the
/// operand type: WGSL lowers `a % b` to `a - b*trunc(a/b)` in that precision,
/// so past the 2^24 (f32) / 2^53 (f64) quotient the runtime value diverges
/// from exact fmod.
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

/// Below the `|a/b| < 2^24` cap the exact fmod can still differ from the
/// hardware's `a - b*trunc(a/b)` by a full divisor when the rounded quotient
/// crosses an integer: `33554432f % 3f` is fmod 2.0 but 0.0 on every
/// round-to-nearest device (`f32(a/b)` rounds 11184810.67 to 11184811).  The
/// stepwise-agreement guard declines it; agreeing cases still fold.
#[test]
fn float_modulo_boundary_crossing_within_cap_declines() {
    use naga::BinaryOperator as B;
    use naga::Literal as L;
    // Diverges (exact fmod 2.0 vs stepwise 0.0) -> decline.
    assert_eq!(
        eval_binary(B::Modulo, L::F32(33_554_432.0), L::F32(3.0)),
        None
    );
    // Exact multiple within the cap: stepwise agrees, still folds.
    assert_eq!(
        eval_binary(B::Modulo, L::F32(30.0), L::F32(3.0)),
        Some(L::F32(0.0))
    );
    // Non-boundary quotient: stepwise agrees, still folds.
    assert_eq!(
        eval_binary(B::Modulo, L::F32(7.0), L::F32(2.0)),
        Some(L::F32(1.0))
    );
}

/// Folding a divisor to a literal zero manufactures IR naga's validator
/// rejects, and the driver's rollback then discards every OTHER fold the
/// same run made - permanently, since the pass is deterministic and re-runs
/// to the same failure.  The guard declines that one fold, so the rest of
/// the module still folds.
#[test]
fn a_foldable_zero_divisor_does_not_cost_the_whole_fold() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<i32>;\n\
        @compute @workgroup_size(1) fn f() {\n\
          var a = 1;\n  var b = 0;\n\
          out[0] = 2 + 3 * 4 - 1;\n  out[1] = (7 * 8) / 2;\n  out[2] = a / (b + b);\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    let rolled_back: Vec<_> = output
        .report
        .pass_reports
        .iter()
        .filter(|p| p.pass_name == "constant_folding" && p.rolled_back)
        .collect();
    assert!(
        rolled_back.is_empty(),
        "constant_folding must not manufacture a static division by zero: \
         {} rollback(s) in {}",
        rolled_back.len(),
        output.source
    );
}

/// The same guard for shift amounts: `x << (n + n)` with `n` forwarded to 20
/// is a shift past the bit width, which naga rejects the same way.
#[test]
fn a_foldable_out_of_range_shift_amount_does_not_roll_back() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: i32;\n\
        @compute @workgroup_size(1) fn f() {\n\
          var a = 1i;\n  var n = 20u;\n  out = a << (n + n);\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    assert!(
        !output
            .report
            .pass_reports
            .iter()
            .any(|p| p.pass_name == "constant_folding" && p.rolled_back),
        "constant_folding must not manufacture an out-of-range shift: {}",
        output.source
    );
}

/// The guard must cover every arm that narrows an expression to one of its
/// operands, not just the literal fold: `select(z, z, c)` with both branches
/// the same handle clones that operand over the `Select`, which put a literal
/// zero straight into the divisor slot the fold had just declined.
#[test]
fn narrowing_a_select_does_not_reopen_the_zero_divisor() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<i32>;\n\
        @compute @workgroup_size(1) fn f() {\n\
          var a = 7;\n  var c = true;\n  let z = 0;\n\
          out[0] = 2 + 3 * 4 - 1;\n  out[1] = a / select(z, z, c);\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    assert!(
        !output
            .report
            .pass_reports
            .iter()
            .any(|p| p.pass_name == "constant_folding" && p.rolled_back),
        "narrowing `select(z, z, c)` must not manufacture a zero divisor: {}",
        output.source
    );
}

/// naga lowers `a && b` through a local whose stores `dead_branch` then
/// deletes as redundant, so the join survives only if a never-written local
/// reads as its zero init.  Not merely a size bug: GPU fuzz seed 960009
/// shipped `-select(-0.0, 2.0, d)` as `-0.0` where the input's
/// const-expression gave `+0.0`, the sign of zero being lost once a
/// const-expression is demoted to a runtime one under fast math.
#[test]
fn a_never_written_local_reads_as_its_zero_initialiser() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
        @compute @workgroup_size(1) fn f() {\n\
          out[0] = bitcast<u32>(-(select(-0.0, floor(2.0), (false && false))));\n\
          if (false && false) { out[1] = 1u; }\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    assert!(
        !output.source.contains("select("),
        "the short-circuit join must fold to a constant, not a runtime select: {}",
        output.source
    );
    assert!(
        !output.source.contains("out[1]") && !output.source.contains("if "),
        "a provably false condition must not survive as a branch: {}",
        output.source
    );
}

/// Neither write is a whole-variable `Store`, which is why the census
/// demands that every reference to the local BE one of its loads; relaxing
/// that ships `out[0] = 0` for both.
#[test]
fn a_local_written_through_a_pointer_does_not_read_as_zero() {
    for body in [
        // Written through a pointer argument; `out[0]` is 9.
        "var v: u32;\n  w(&v);\n  out[0] = v;",
        // Written through an element pointer, then read whole; `out[0]` is 7.
        "var a: array<u32, 2>;\n  a[inp[0] & 1u] = 7u;\n  let b = a;\n  \
         out[0] = b[0] + b[1];",
    ] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @group(0) @binding(1) var<storage, read> inp: array<u32>;\n\
             fn w(p: ptr<function, u32>) {{ *p = 9u; }}\n\
             @compute @workgroup_size(1) fn f() {{\n  {body}\n}}"
        );
        let output = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            !output.source.contains("]=0;"),
            "a local something else can write must not fold to its initialiser: \
             {body} -> {}",
            output.source
        );
    }
}

/// A read before the only store is the zero init, a read after it is the
/// stored value; one `=0` is the former, two would be the miscompile.
#[test]
fn folding_a_pre_store_read_leaves_the_post_store_read_alone() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
        @group(0) @binding(1) var<storage, read> inp: array<u32>;\n\
        @compute @workgroup_size(1) fn f() {\n\
          var v: u32;\n  out[0] = v;\n  v = inp[0];\n  out[1] = v;\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    assert_eq!(
        output.source.matches("=0;").count(),
        1,
        "only the pre-store read is zero: {}",
        output.source
    );
}

/// The fold only pays if the evaluator can finish the const-expression it
/// creates.  `u32(false)` is one naga's frontend would have folded and this
/// one would not, so it was extracted to a `const` and each use paid more
/// than the variable it replaced: six of them GREW the output by 20 bytes.
#[test]
fn folding_a_bool_local_does_not_grow_its_casts() {
    let uses = (0..6)
        .map(|i| format!("out[{i}u] = u32(b);"))
        .collect::<Vec<_>>()
        .join("");
    let src = format!(
        "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
         @compute @workgroup_size(1) fn f() {{\n  var b: bool;\n  {uses}\n}}"
    );
    let output = crate::run(&src, &Config::default()).expect("must minify");
    assert!(
        !output.source.contains("u32("),
        "a cast of a folded bool must fold too: {}",
        output.source
    );
}

/// The fold is where const-ness ENTERS an expression, so the guard has to
/// cover the whole failable slot, not the operand the literal lands in:
/// `5u / u32(b)` is legal until `b` folds to `false`, and the error lands on
/// the cast.  Missing it cost the whole module - emit fails and the run
/// ships the lexically compacted INPUT.
#[test]
fn folding_a_local_under_a_failable_operator_does_not_lose_the_module() {
    for body in [
        // Scalar, through a cast: the divisor becomes a const zero.
        "var b: bool;\n  out[0] = 5u / u32(b);\n  out[1] = 2u + 3u;",
        // Vector, through the componentwise arm: lane 0 becomes a const zero.
        "out[8] = 0u;\n  var v: vec2u;\n  \
         out[0] = dot(vec2u(4u, 6u) / (v + vec2u(0u, 1u)), vec2u(1u));\n  \
         out[1] = 2u + 3u;",
        // Vector, shift amount past the bit width.
        "out[8] = 0u;\n  var v: vec2u;\n  \
         out[0] = dot(vec2u(4u, 6u) >> (v + vec2u(40u, 1u)), vec2u(1u));\n  \
         out[1] = 2u + 3u;",
    ] {
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @compute @workgroup_size(1) fn f() {{\n  {body}\n}}"
        );
        let output = crate::run(&src, &Config::default()).expect("must minify");
        assert!(
            output.report.bailout.is_none(),
            "the slot must stay a runtime read: {body} -> {:?}",
            output.report.bailout
        );
        // The rest of the module still folds, which is what a bailout costs.
        assert!(
            output.source.contains("=5;"),
            "unrelated folding must survive: {body} -> {}",
            output.source
        );
    }
}

/// The guard is about const-ness reaching the slot, not about refusing every
/// fold under one: a divisor that stays safe keeps folding.
#[test]
fn a_safe_divisor_still_folds_through_the_guard() {
    let src = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
        @compute @workgroup_size(1) fn f() {\n\
          var b: bool;\n  out[0] = 5u / (u32(b) | 1u);\n  out[1] = 5u << u32(b);\n}";
    let output = crate::run(src, &Config::default()).expect("must minify");
    assert!(
        !output.source.contains("var "),
        "a divisor forced non-zero, and a shift amount, must still fold: {}",
        output.source
    );
}
