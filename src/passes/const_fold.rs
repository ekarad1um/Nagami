//! Constant folding: expressions over statically known operands become
//! `Literal` / `Compose` nodes, in `module.global_expressions` and in every
//! function arena.  A folded literal is declarative, so it must leave its
//! `Emit` range; each function fold returns that handle set and the body's
//! ranges are rebuilt around it.
//!
//! Exactly-rounded operations fold bit-exactly; accuracy-tolerant
//! transcendentals (sin/cos/exp/log/pow/...) and float divide / modulo fold
//! to a value inside WGSL's permitted error envelope, not a bit-identical
//! one.  Overflow-, NaN-, and signed-zero-sensitive cases decline.

use rustc_hash::FxHashMap;

use naga::Handle;

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::passes::expr_util::{
    is_bool_false, is_bool_true, is_integer_zero_literal, rebuild_emit_ranges_after_removal,
    shift_amount_is_static_error,
};
use crate::pipeline::{Pass, PassContext};

/// Whether a clone of `expression` in another arena slot reproduces the same
/// value: true for declarative leaves and structural / arithmetic wrappers;
/// false for memory reads, derivatives, and statement-attached or
/// cursor-dependent results, whose duplicate `Emit` would re-execute against
/// possibly-different shared state or land disconnected from its producing
/// statement.  Exhaustive on purpose: `_ => true` would silently corrupt,
/// `_ => false` silently lose folds.
fn is_pure_to_clone(expression: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expression {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Compose { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. } => true,
        E::Load { .. }
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::Derivative { .. }
        | E::ArrayLength(_) => false,
        E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::RayQueryVertexPositions { .. }
        | E::RayQueryGetIntersection { .. }
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => false,
    }
}

/// Constant folding across globals, functions, and entry points.
#[derive(Debug, Default)]
pub struct ConstFoldPass;

impl Pass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // A pure function of `module.types`, which no fold mutates.
        let vector_type_cache = build_vector_type_cache(&module.types);

        changed += fold_global_expressions(module, &vector_type_cache);
        // A library module (no entry points) keeps every `const`, so once
        // rename shortens its name a reference never loses to the literal it
        // names, and folding through one only trades the name for a longer
        // literal beside a now dead declaration.  Named constants stay opaque
        // here (`dead_branch` resolves `if FLAG` itself).  Without mangling
        // the name stays as written and the literal wins, hence the gate.
        let const_literals = if module.entry_points.is_empty() && ctx.config.mangle() {
            Default::default()
        } else {
            build_constant_literal_cache(module)
        };

        for (_, function) in module.functions.iter_mut() {
            // The identity gate keys on the pre-fold graph, not on mid-loop
            // partial rewrites.
            let refcounts = count_handle_refs(function);
            let emit_ranges = build_emit_range_map(&function.body, function.expressions.len());
            let (folded, simplified) = fold_local_expressions(
                &mut function.expressions,
                &refcounts,
                &emit_ranges,
                &const_literals,
                &module.types,
                &vector_type_cache,
            );
            changed += simplified;
            if !folded.is_empty() {
                changed += folded.len();
                rebuild_emit_ranges_after_removal(&mut function.body, &folded);
            }
        }
        for entry in module.entry_points.iter_mut() {
            let refcounts = count_handle_refs(&entry.function);
            let emit_ranges =
                build_emit_range_map(&entry.function.body, entry.function.expressions.len());
            let (folded, simplified) = fold_local_expressions(
                &mut entry.function.expressions,
                &refcounts,
                &emit_ranges,
                &const_literals,
                &module.types,
                &vector_type_cache,
            );
            changed += simplified;
            if !folded.is_empty() {
                changed += folded.len();
                rebuild_emit_ranges_after_removal(&mut entry.function.body, &folded);
            }
        }

        Ok(changed > 0)
    }
}

// MARK: Global-expression folding

/// Fold `module.global_expressions` into the most compact emittable form.
fn fold_global_expressions(
    module: &mut naga::Module,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> usize {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HandleMap<_, _>>();

    let handles = module
        .global_expressions
        .iter()
        .map(|(h, _)| h)
        .collect::<Vec<_>>();
    let mut changed = 0usize;

    let mut literal_cache = build_literal_cache(&module.global_expressions);

    let mut visiting = HandleSet::default();
    let mut memo: ConstValueMemo = vec![None; module.global_expressions.len()];
    for handle in handles {
        visiting.clear();
        let value = {
            let ctx = ConstFoldContext {
                arena: &module.global_expressions,
                types: &module.types,
                constants: ConstSource::Inits(&const_inits),
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        if let Some(ConstValue::Scalar(literal)) = value {
            if !matches!(module.global_expressions[handle], naga::Expression::Literal(existing) if existing == literal)
            {
                module.global_expressions[handle] = naga::Expression::Literal(literal);
                note_literal_in_cache(&mut literal_cache, handle, literal);
                changed += 1;
            }
            continue;
        }

        if let Some(ConstValue::Vector {
            ref components,
            size,
            scalar,
        }) = value
            && let Some(new_expr) = materialize_vector(
                handle,
                components,
                size,
                scalar,
                &literal_cache,
                vector_type_cache,
            )
            && module.global_expressions[handle] != new_expr
        {
            module.global_expressions[handle] = new_expr;
            changed += 1;
        }
    }

    changed
}

/// `Constant -> Literal` for every constant whose initializer resolves.
///
/// MUST run after [`fold_global_expressions`], which is what makes this a
/// plain arena read: that pass resolves every global expression and rewrites
/// each scalar-valued one to its `Literal`, chains through
/// `Expression::Constant` included, so "the init resolves to a scalar" and
/// "the init IS a literal" are already the same predicate.
///
/// Abstract literals are skipped: naga's validator rejects `AbstractInt` /
/// `AbstractFloat` in a function arena, and a cached one would roll the whole
/// pass back every sweep, whereas skipping merely leaves the constant
/// un-inlined.  (naga's frontend concretises `const X = 1` early, so the
/// guard is defensive.)
fn build_constant_literal_cache(module: &naga::Module) -> HandleMap<naga::Constant, naga::Literal> {
    module
        .constants
        .iter()
        .filter_map(|(ch, c)| match module.global_expressions[c.init] {
            naga::Expression::Literal(
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_),
            ) => None,
            naga::Expression::Literal(lit) => Some((ch, lit)),
            _ => None,
        })
        .collect()
}

// MARK: Per-function folding

/// Data-flow reference count per handle: expression-as-child uses, statement
/// operands and results, `named_expressions`, and local initializers.  `Emit`
/// ranges are NOT counted: they fix execution order, not consumption, and an
/// expression whose only "use" is its Emit entry is dead.  The identity gate
/// tests `== 1` for impure operands whose Emit entry can go after cloning,
/// so `saturating_add` is indistinguishable from exact.
///
/// Counting statement RESULTS is load-bearing: statement-attached expressions
/// (`CallResult`, `AtomicResult`, ...) are uncloneable, and their producer
/// bump pushes any consumed one to `>= 2` so the `== 1` escape never fires.
fn count_handle_refs(function: &naga::Function) -> Vec<u32> {
    let mut counts = vec![0u32; function.expressions.len()];

    fn bump(counts: &mut [u32], h: naga::Handle<naga::Expression>) {
        let i = h.index();
        if i < counts.len() {
            counts[i] = counts[i].saturating_add(1);
        }
    }

    for (_, expr) in function.expressions.iter() {
        super::expr_util::visit_expression_children(expr, |child| bump(&mut counts, child));
    }

    super::expr_util::visit_block_expression_handles(
        &function.body,
        /*include_emit_handles=*/ false,
        &mut |h| bump(&mut counts, h),
    );

    // naga restricts a local init to override-expressions (never impure), so
    // counting it only guards against a relaxation letting the gate drop an
    // impure init's Emit entry.
    for &handle in function.named_expressions.keys() {
        bump(&mut counts, handle);
    }
    for (_, lvar) in function.local_variables.iter() {
        if let Some(init) = lvar.init {
            bump(&mut counts, init);
        }
    }

    counts
}

const NO_EMIT: u32 = u32::MAX;

/// Per-handle id of the `Emit` range that materialises it (`NO_EMIT` when
/// none).  Two handles share an id IFF one `Emit` statement covers both, i.e.
/// no statement of any kind sits between them.  Every non-`Emit` statement is
/// a memory write, a barrier, or a control-flow edge, each of which makes
/// moving a read across it unsound, so "same `Emit` range" is exactly
/// "provably no intervening write": the store-aware guard for relocating an
/// impure operand (a `Load`) onto its consumer's slot, without re-walking
/// control flow per fold.  Nested blocks continue the counter, so ids never
/// collide across blocks; the map is read-only until the ranges are rebuilt
/// after the fold.
fn build_emit_range_map(body: &naga::Block, expression_count: usize) -> Vec<u32> {
    let mut map = vec![NO_EMIT; expression_count];
    let mut next_id = 0u32;
    crate::passes::expr_util::for_each_statement(body, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            let id = next_id;
            next_id += 1;
            for h in range.clone() {
                map[h.index()] = id;
            }
        }
    });
    map
}

/// Roles that make a folded literal a WGSL shader-creation error.
const ROLE_DIVISOR: u8 = 1;
const ROLE_SHIFT_AMOUNT: u8 = 2;

/// Per-handle roles: the right operand of an integer `/` `%` or of a shift,
/// reached through `Splat` / `Compose` since any offending lane condemns a
/// componentwise op.  Folding an offender into one of these cannot survive
/// post-pass validation, and the rollback discards every OTHER fold of the
/// same run - permanently, the pass being deterministic - so declining is
/// free.
fn static_error_roles(arena: &naga::Arena<naga::Expression>) -> Vec<u8> {
    let mut stack: Vec<(Handle<naga::Expression>, u8)> = arena
        .iter()
        .filter_map(|(_, expr)| match expr {
            naga::Expression::Binary { op, right, .. } => match op {
                naga::BinaryOperator::Divide | naga::BinaryOperator::Modulo => {
                    Some((*right, ROLE_DIVISOR))
                }
                naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight => {
                    Some((*right, ROLE_SHIFT_AMOUNT))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    // Most arenas hold no `/` `%` `<<` `>>`; an empty map costs no allocation
    // and reads as no role.
    if stack.is_empty() {
        return Vec::new();
    }
    let mut roles = vec![0u8; arena.len()];
    while let Some((h, role)) = stack.pop() {
        if roles[h.index()] & role != 0 {
            continue;
        }
        roles[h.index()] |= role;
        match &arena[h] {
            naga::Expression::Splat { value, .. } => stack.push((*value, role)),
            naga::Expression::Compose { components, .. } => {
                stack.extend(components.iter().map(|&c| (c, role)));
            }
            _ => {}
        }
    }
    roles
}

/// Role bits for `h`; an empty map is "no failable operator in the arena".
fn role_of(roles: &[u8], h: Handle<naga::Expression>) -> u8 {
    roles.get(h.index()).copied().unwrap_or(0)
}

/// `literal` in a slot with `roles` is a shader-creation error.
fn literal_is_static_error(roles: u8, literal: naga::Literal) -> bool {
    (roles & ROLE_DIVISOR != 0 && is_integer_zero_literal(&literal))
        || (roles & ROLE_SHIFT_AMOUNT != 0 && shift_amount_is_static_error(&literal))
}

/// Clone `source` over `target`, declining (and writing nothing) when that
/// would leave an offender where a runtime operation stood.  Every arm that
/// narrows an expression to one of its operands writes through here, so the
/// guard is structural rather than a rule each new arm must remember.
fn clone_over(
    arena: &mut naga::Arena<naga::Expression>,
    roles: &[u8],
    target: Handle<naga::Expression>,
    source: Handle<naga::Expression>,
) -> bool {
    if matches!(arena[source], naga::Expression::Literal(lit)
        if literal_is_static_error(role_of(roles, target), lit))
    {
        return false;
    }
    arena[target] = arena[source].clone();
    true
}

/// Fold `arena` in place, returning the handles that must leave their `Emit`
/// ranges and the number of simplifications.  `refcounts` and `emit_ranges`
/// together gate cloning an impure operand in the identity / involution
/// arms: only when the folding expression is its sole consumer AND shares
/// its `Emit` range, so the operand dies (its Emit entry is dropped, no
/// double execution) and the relocated read crosses no statement.
fn fold_local_expressions(
    arena: &mut naga::Arena<naga::Expression>,
    refcounts: &[u32],
    emit_ranges: &[u32],
    const_literals: &HandleMap<naga::Constant, naga::Literal>,
    types: &naga::UniqueArena<naga::Type>,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> (HandleSet<naga::Expression>, usize) {
    // Absent ids (`NO_EMIT`, or a handle past the map) read as not
    // co-located, which only suppresses a relocation.
    let same_emit_range =
        |a: naga::Handle<naga::Expression>, b: naga::Handle<naga::Expression>| match (
            emit_ranges.get(a.index()),
            emit_ranges.get(b.index()),
        ) {
            (Some(&x), Some(&y)) => x != NO_EMIT && x == y,
            _ => false,
        };
    let mut handles = Vec::with_capacity(arena.len());
    handles.extend(arena.iter().map(|(h, _)| h));
    // Sound to compute once: the loops below replace whole `arena[handle]`
    // values, never a `Binary`'s operand slots, so a role can retire but
    // never appear.
    let roles = static_error_roles(arena);
    let mut folded = HandleSet::default();

    let mut literal_cache = build_literal_cache(arena);

    let mut visiting = HandleSet::default();
    let mut memo: ConstValueMemo = vec![None; arena.len()];
    for handle in handles.iter().copied() {
        visiting.clear();
        let value = {
            let ctx = ConstFoldContext {
                arena: &*arena,
                types,
                constants: ConstSource::Literals(const_literals),
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        let role = role_of(&roles, handle);
        match value {
            Some(ConstValue::Scalar(literal)) => {
                // An abstract result (both operands abstract) trips naga's
                // `WidthError::Abstract` in a function arena and would roll
                // the whole pass back.
                if matches!(
                    literal,
                    naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                ) {
                    continue;
                }
                // Same rollback, different cause; see `static_error_roles`.
                if literal_is_static_error(role, literal) {
                    continue;
                }
                if !matches!(arena[handle], naga::Expression::Literal(existing) if existing == literal)
                {
                    arena[handle] = naga::Expression::Literal(literal);
                    note_literal_in_cache(&mut literal_cache, handle, literal);
                    folded.insert(handle);
                }
            }
            Some(ConstValue::Vector {
                ref components,
                size,
                scalar,
            }) => {
                // One offending lane condemns the whole componentwise op.
                if components
                    .iter()
                    .any(|&lane| literal_is_static_error(role, lane))
                {
                    continue;
                }
                // A `Compose` needs an `Emit` range, which only an
                // already-emittable original sits in.
                if crate::passes::expr_util::expression_needs_emit(&arena[handle])
                    && let Some(new_expr) = materialize_vector(
                        handle,
                        components,
                        size,
                        scalar,
                        &literal_cache,
                        vector_type_cache,
                    )
                    && arena[handle] != new_expr
                {
                    arena[handle] = new_expr;
                }
            }
            None => {}
        }
    }

    // Identity (`x * 1 -> x`), absorbing (`x * 0 -> 0`), involution
    // (`-(-x) -> x`), and `select(x, x, c) -> x`.
    let mut simplify_count = 0usize;
    for handle in handles {
        match arena[handle] {
            naga::Expression::Binary { op, left, right } => {
                // Absorbing clones the matched zero / all-ones operand over
                // the Binary, whose type follows naga broadcasting
                // (`vec3<f32> * 0.0` is a vec3), so a scalar clone onto a
                // vector slot mis-types.  Safe only for `&&` / `||` (pinned
                // to `bool x bool -> bool`) or when both operands are
                // literals (scalar result); the latter admits what
                // `eval_binary` leaves unfolded, notably F16, where
                // `check_absorbing_operand` declines anything sign-sensitive.
                let both_literal = matches!(arena[left], naga::Expression::Literal(_))
                    && matches!(arena[right], naga::Expression::Literal(_));
                let is_logical_op = matches!(
                    op,
                    naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
                );
                if (is_logical_op || both_literal)
                    && let Some(absorb) = check_absorbing_operand(op, left, right, arena)
                    && clone_over(arena, &roles, handle, absorb)
                {
                    simplify_count += 1;
                    folded.insert(handle); // result is a Literal, declarative.
                    continue;
                }
                // Identity is type-safe by construction: the matched literal
                // is the neutral element, so `other` already has the
                // broadcast result type.  Cloning an impure `other` would
                // re-execute it on the duplicate `Emit`; the escape is sole
                // ownership by this Binary, which leaves `other` dead with
                // its Emit entry dropped.  Sole ownership does not fix the
                // evaluation POSITION, so the relocated read must also share
                // this Binary's `Emit` range, else a statement separates them.
                if let Some(other) = check_identity_operand(op, left, right, arena) {
                    let other_pure = is_pure_to_clone(&arena[other]);
                    let other_uniquely_owned = !other_pure
                        && refcounts.get(other.index()).copied() == Some(1)
                        && same_emit_range(other, handle);
                    if (other_pure || other_uniquely_owned)
                        && clone_over(arena, &roles, handle, other)
                    {
                        simplify_count += 1;
                        if !crate::passes::expr_util::expression_needs_emit(&arena[handle]) {
                            folded.insert(handle);
                        }
                        if other_uniquely_owned {
                            folded.insert(other);
                        }
                        continue;
                    }
                }
            }
            naga::Expression::Unary { op, expr } => {
                // Involution: the identity escape, plus the intermediate
                // Unary at `expr` is hoisted past.  It leaves Emit only when
                // solely owned; `inner` additionally needs `expr` solely
                // owned, since `expr`'s residual slot still references it
                // until compact runs.
                if let naga::Expression::Unary {
                    op: inner_op,
                    expr: inner,
                } = arena[expr]
                    && op == inner_op
                {
                    let inner_pure = is_pure_to_clone(&arena[inner]);
                    let intermediate_uniquely_owned =
                        refcounts.get(expr.index()).copied() == Some(1);
                    // `expr` sits topologically between `inner` and `handle`,
                    // so co-location of that pair covers it.
                    let inner_uniquely_owned = !inner_pure
                        && intermediate_uniquely_owned
                        && refcounts.get(inner.index()).copied() == Some(1)
                        && same_emit_range(inner, handle);
                    if (inner_pure || inner_uniquely_owned)
                        && clone_over(arena, &roles, handle, inner)
                    {
                        simplify_count += 1;
                        if !crate::passes::expr_util::expression_needs_emit(&arena[handle]) {
                            folded.insert(handle);
                        }
                        if inner_uniquely_owned {
                            folded.insert(inner);
                        }
                        if intermediate_uniquely_owned {
                            folded.insert(expr);
                        }
                        continue;
                    }
                }

                // `!(a == b)` -> `a != b`: equality negation is exact for
                // every type, NaN included, unlike ordered relations.  Done
                // in IR so the `Binary` emit path parenthesises correctly
                // (an emit-time fold mis-parenthesises a nested comparison),
                // and only when this `!` solely owns the comparison; a shared
                // one emits as `!name`.
                if op == naga::UnaryOperator::LogicalNot
                    && let naga::Expression::Binary {
                        op: cmp,
                        left,
                        right,
                    } = arena[expr]
                    && let Some(flipped) = flip_equality(cmp)
                    && refcounts.get(expr.index()).copied() == Some(1)
                {
                    arena[handle] = naga::Expression::Binary {
                        op: flipped,
                        left,
                        right,
                    };
                    simplify_count += 1;
                    folded.insert(expr);
                    continue;
                }
            }
            // `x` has two consumers by construction, so only the pure-clone
            // gate applies.
            naga::Expression::Select { accept, reject, .. }
                if accept == reject && is_pure_to_clone(&arena[accept]) =>
            {
                if clone_over(arena, &roles, handle, accept) {
                    simplify_count += 1;
                    if !crate::passes::expr_util::expression_needs_emit(&arena[handle]) {
                        folded.insert(handle);
                    }
                }
                continue;
            }
            _ => {}
        }
    }

    (folded, simplify_count)
}

/// `==` <-> `!=`; ordered relations are excluded because their negation is
/// NaN-unsafe for floats and this pass does not resolve operand types.
fn flip_equality(op: naga::BinaryOperator) -> Option<naga::BinaryOperator> {
    match op {
        naga::BinaryOperator::Equal => Some(naga::BinaryOperator::NotEqual),
        naga::BinaryOperator::NotEqual => Some(naga::BinaryOperator::Equal),
        _ => None,
    }
}

// MARK: Constant value resolution

// Matrices are deliberately absent: rare in constant contexts, not worth the
// case analysis.
#[derive(Debug, Clone, PartialEq)]
enum ConstValue {
    Scalar(naga::Literal),
    Vector {
        components: Vec<naga::Literal>,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    },
}

impl ConstValue {
    fn as_scalar(&self) -> Option<naga::Literal> {
        match self {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }
}

/// Memo for [`resolve_const_value`], indexed by handle: outer `None` = not
/// computed, inner = the full result including "not a constant".  Failures
/// are cached too, since the dominant cost is re-walking long runtime chains
/// that fail at every ancestor (quadratic on the deep chains our own passes
/// build).  It stays valid across the folding loops' rewrites because those
/// only substitute a `Literal` / `Compose` of the value already resolved, and
/// it is safe even on cyclic IR: a cycle-guard hit returns `None` before any
/// store and no arm turns a child failure into `Some`, so a wrong `Some` is
/// never cached.
type ConstValueMemo = Vec<Option<Option<ConstValue>>>;

/// Where `Expression::Constant` resolves.  One enum rather than a trait with
/// two impls: the resolver below is a seven-function mutual recursion, and a
/// generic context monomorphised it twice for a single field's worth of
/// difference.
enum ConstSource<'a> {
    /// Module scope: a constant resolves through its initializer, which lives
    /// in the SAME arena, so the recursion continues and shares the memo.
    Inits(&'a HandleMap<naga::Constant, naga::Handle<naga::Expression>>),
    /// Function scope: constants were pre-resolved to literals by
    /// [`build_constant_literal_cache`], so the lookup is terminal.
    Literals(&'a HandleMap<naga::Constant, naga::Literal>),
}

/// Everything [`resolve_const_value`] reads: the arena it walks, the type
/// arena `Compose` / `Splat` need for component type and vector size, and
/// where a constant handle resolves.
struct ConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    constants: ConstSource<'a>,
}

impl ConstFoldContext<'_> {
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HandleSet<naga::Expression>,
        memo: &mut ConstValueMemo,
    ) -> Option<ConstValue> {
        match self.constants {
            ConstSource::Inits(inits) => {
                resolve_const_value(*inits.get(handle)?, self, visiting, memo)
            }
            ConstSource::Literals(lits) => lits.get(handle).copied().map(ConstValue::Scalar),
        }
    }
}

// MARK: Resolver entry points

/// Convert a width-8 literal (`F64` / `U64` / `I64`) to `target` (f32 / i32 /
/// u32 / bool), or `None` for any other pair.  naga's frontend refuses to
/// const-fold these casts (f64 / u64 / i64 are non-standard WGSL), so the
/// `As` node survives and the emitter's `f32(<F64 literal>)` is rejected on
/// re-parse; folding fixes the round-trip.  Semantics mirror
/// `naga::proc::ConstantEvaluator::cast`: a float source rounds to nearest
/// (declined when non-finite) and CLAMPS to integer range; an integer source
/// WRAPS (`i64(-1) -> u32` is `4294967295u`); `-> bool` is `v != 0`.  Other
/// targets (f16 / f64 / i64 / u64) return `None`; naga accepts those forms.
/// `generator::expr_emit::cast_width8_to_literal` covers the vector form:
/// keep the two conversions in sync (the round-trip tests catch divergence).
fn cast_width8_to(src: naga::Literal, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    if let L::F64(v) = src {
        return match (target.kind, target.width) {
            (K::Float, 4) => {
                let r = v as f32;
                r.is_finite().then_some(L::F32(r))
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
        (K::Sint, 4) => Some(L::I32(v as i32)),
        (K::Uint, 4) => Some(L::U32(v as u32)),
        (K::Bool, _) => Some(L::Bool(v != 0)),
        _ => None,
    }
}

/// Resolve `handle` to a [`ConstValue`], cycle-guarded by `visiting` and
/// memoised; composites resolve componentwise.
fn resolve_const_value(
    handle: Handle<naga::Expression>,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    if let Some(Some(cached)) = memo.get(handle.index()) {
        return cached.clone();
    }
    if !visiting.insert(handle) {
        return None;
    }
    let out = resolve_const_value_uncached(handle, ctx, visiting, memo);
    visiting.remove(handle);
    if let Some(slot) = memo.get_mut(handle.index()) {
        *slot = Some(out.clone());
    }
    out
}

/// `?` exits are safe here only because the wrapper owns `visiting.remove`
/// and the memo store on every return path.
fn resolve_const_value_uncached(
    handle: Handle<naga::Expression>,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let expr = &ctx.arena[handle];
    match expr {
        naga::Expression::Literal(lit) => Some(ConstValue::Scalar(*lit)),
        naga::Expression::Constant(ch) => ctx.resolve_constant_value(*ch, visiting, memo),

        naga::Expression::ZeroValue(ty) => resolve_zero_value(*ty, ctx),

        naga::Expression::Splat { size, value } => {
            let inner = resolve_const_value(*value, ctx, visiting, memo)?;
            let lit = inner.as_scalar()?;
            Some(ConstValue::Vector {
                scalar: lit.scalar(),
                size: *size,
                components: vec![lit; *size as usize],
            })
        }

        naga::Expression::Compose { ty, components } => {
            resolve_compose(*ty, components, ctx, visiting, memo)
        }

        naga::Expression::AccessIndex { base, index } => {
            resolve_composite_element(*base, *index as usize, ctx, visiting, memo)
        }

        // A dynamic `Access` whose index folds is a static pick: naga
        // materialises a dynamically-indexed function-scope `const` array as
        // a full `Compose` at the use site and load_dedup forwards the index
        // literal, so without this the whole composite ships inline
        // (`array<u32,2310>(...)[0]`) and only the NEXT round collapses it
        // via naga's const-eval.  A pointer-typed base is a variable /
        // pointer chain that never resolves to a value, so it cannot fold.
        naga::Expression::Access { base, index } => {
            let idx = match resolve_const_value(*index, ctx, visiting, memo)? {
                ConstValue::Scalar(l) => literal_index(l)?,
                ConstValue::Vector { .. } => return None,
            };
            resolve_composite_element(*base, idx, ctx, visiting, memo)
        }

        naga::Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let vec_val = resolve_const_value(*vector, ctx, visiting, memo)?;
            match vec_val {
                ConstValue::Vector {
                    ref components,
                    scalar,
                    ..
                } => {
                    let n = *size as usize;
                    let mut out = Vec::with_capacity(n);
                    for &sw in &pattern[..n] {
                        let idx = sw as usize;
                        out.push(*components.get(idx)?);
                    }
                    Some(ConstValue::Vector {
                        components: out,
                        size: *size,
                        scalar,
                    })
                }
                _ => None,
            }
        }

        naga::Expression::Unary { op, expr } => {
            let inner = resolve_const_value(*expr, ctx, visiting, memo)?;
            eval_const_unary(*op, inner)
        }

        naga::Expression::Binary { op, left, right } => {
            let l = resolve_const_value(*left, ctx, visiting, memo)?;
            let r = resolve_const_value(*right, ctx, visiting, memo)?;
            eval_const_binary(*op, l, r)
        }

        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_const_value(*arg, ctx, visiting, memo)?;
            let b = match arg1 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting, memo)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting, memo)?),
                None => None,
            };
            eval_const_math(*fun, a, b, c)
        }

        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_const_value(*condition, ctx, visiting, memo)?;
            match cond {
                ConstValue::Scalar(naga::Literal::Bool(true)) => {
                    resolve_const_value(*accept, ctx, visiting, memo)
                }
                ConstValue::Scalar(naga::Literal::Bool(false)) => {
                    resolve_const_value(*reject, ctx, visiting, memo)
                }
                _ => None,
            }
        }

        // Converting casts fold width-8 scalars only (naga already folds
        // every other narrowing cast); a VECTOR operand falls through, its
        // converted component literals do not exist in the arena, and the
        // generator's vector path handles it.
        naga::Expression::As {
            expr: operand,
            kind,
            convert,
        } => {
            let ConstValue::Scalar(lit) = resolve_const_value(*operand, ctx, visiting, memo)?
            else {
                return None;
            };
            match convert {
                Some(width) => match lit {
                    naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_) => {
                        let target = naga::Scalar {
                            kind: *kind,
                            width: *width,
                        };
                        cast_width8_to(lit, target).map(ConstValue::Scalar)
                    }
                    _ => None,
                },
                None => bitcast_literal(lit, *kind).map(ConstValue::Scalar),
            }
        }

        _ => None,
    }
}

/// Reinterpret a scalar literal's bits as `kind` at the same width, as
/// `bitcast` does at runtime; naga's evaluator declines bitcast, so without
/// this the tree above a literal bitcast survives every fold and tint
/// evaluates it at shader creation.  A float on either side must be NORMAL:
/// inf / NaN have no literal spelling, a subnormal may flush to zero on the
/// GPU, and a zero's sign is not kept by every platform (Dawn on Metal reads
/// `bitcast<u32>(-0f)` as 0), so those keep the runtime bitcast and the
/// platform decides, as it did for the input.  `f16` is declined too: its
/// literal needs the test-only `half` crate, and its zero has the same
/// problem.
fn bitcast_literal(lit: naga::Literal, kind: naga::ScalarKind) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    let (bits, width) = match lit {
        L::F32(v) if v.is_normal() => (u64::from(v.to_bits()), 4),
        L::F64(v) if v.is_normal() => (v.to_bits(), 8),
        L::U32(v) => (u64::from(v), 4),
        L::I32(v) => (u64::from(v as u32), 4),
        L::U16(v) => (u64::from(v), 2),
        L::I16(v) => (u64::from(v as u16), 2),
        L::U64(v) => (v, 8),
        L::I64(v) => (v as u64, 8),
        _ => return None,
    };
    Some(match (kind, width) {
        (K::Uint, 4) => L::U32(bits as u32),
        (K::Sint, 4) => L::I32(bits as u32 as i32),
        (K::Float, 4) => {
            let f = f32::from_bits(bits as u32);
            if !f.is_normal() {
                return None;
            }
            L::F32(f)
        }
        (K::Uint, 2) => L::U16(bits as u16),
        (K::Sint, 2) => L::I16(bits as u16 as i16),
        (K::Uint, 8) => L::U64(bits),
        (K::Sint, 8) => L::I64(bits as i64),
        (K::Float, 8) => {
            let f = f64::from_bits(bits);
            if !f.is_normal() {
                return None;
            }
            L::F64(f)
        }
        _ => return None,
    })
}

/// Non-negative value of an integer index literal; anything else declines
/// the fold.
fn literal_index(lit: naga::Literal) -> Option<usize> {
    use naga::Literal as L;
    match lit {
        L::I32(v) => usize::try_from(v).ok(),
        L::U32(v) => Some(v as usize),
        L::I64(v) => usize::try_from(v).ok(),
        L::U64(v) => usize::try_from(v).ok(),
        L::AbstractInt(v) => usize::try_from(v).ok(),
        _ => None,
    }
}

/// Element `idx` of the composite VALUE at `base`.  Array / matrix bases are
/// read STRUCTURALLY (one `Compose` / `ZeroValue` component per element) so
/// [`ConstValue`] never models whole-array shapes; everything else resolves
/// the base fully, which covers vectors (whose `Compose` components flatten,
/// so cannot be picked positionally) and vector-valued chains like
/// `arr[1][2]`.  Nested array-of-array picks decline.
fn resolve_composite_element(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match &ctx.arena[base] {
        naga::Expression::Compose { ty, components } => match &ctx.types[*ty].inner {
            naga::TypeInner::Array { .. } | naga::TypeInner::Matrix { .. } => {
                resolve_const_value(*components.get(idx)?, ctx, visiting, memo)
            }
            _ => resolve_vector_component(base, idx, ctx, visiting, memo),
        },
        naga::Expression::ZeroValue(ty) => match &ctx.types[*ty].inner {
            naga::TypeInner::Array {
                base: elem,
                size: naga::ArraySize::Constant(n),
                ..
            } => (idx < n.get() as usize)
                .then(|| resolve_zero_value(*elem, ctx))
                .flatten(),
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let zero = naga::Literal::zero(*scalar)?;
                (idx < *columns as usize).then(|| ConstValue::Vector {
                    components: vec![zero; *rows as usize],
                    size: *rows,
                    scalar: *scalar,
                })
            }
            _ => resolve_vector_component(base, idx, ctx, visiting, memo),
        },
        _ => resolve_vector_component(base, idx, ctx, visiting, memo),
    }
}

/// Component `idx` of `base` once it resolves to a [`ConstValue::Vector`].
fn resolve_vector_component(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match resolve_const_value(base, ctx, visiting, memo)? {
        ConstValue::Vector { components, .. } => {
            components.get(idx).copied().map(ConstValue::Scalar)
        }
        ConstValue::Scalar(_) => None,
    }
}

/// `ZeroValue(ty)` for a scalar or vector type; matrices, structs, and
/// arrays return `None`.
fn resolve_zero_value(ty: Handle<naga::Type>, ctx: &ConstFoldContext<'_>) -> Option<ConstValue> {
    match ctx.types[ty].inner {
        naga::TypeInner::Scalar(s) => naga::Literal::zero(s).map(ConstValue::Scalar),
        naga::TypeInner::Vector { size, scalar } => {
            let z = naga::Literal::zero(scalar)?;
            Some(ConstValue::Vector {
                components: vec![z; size as usize],
                size,
                scalar,
            })
        }
        _ => None,
    }
}

/// `Compose { ty, components }` as a [`ConstValue::Vector`] when every
/// component (scalar or flattened vector) resolves to `ty`'s scalar.  The
/// per-component scalar check keeps `materialize_vector` from building a
/// `Compose<vec4<f32>>` over `Literal::I32` handles, which naga's validator
/// rejects; naga's frontend concretises first, so it fires only on
/// hand-built IR.
fn resolve_compose(
    ty: Handle<naga::Type>,
    components: &[Handle<naga::Expression>],
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let inner = &ctx.types[ty].inner;
    match inner {
        naga::TypeInner::Vector { size, scalar } => {
            let expected = *size as usize;
            let target_scalar = *scalar;
            let mut out = Vec::with_capacity(expected);
            for &c in components {
                let val = resolve_const_value(c, ctx, visiting, memo)?;
                match val {
                    ConstValue::Scalar(l) => {
                        if l.scalar() != target_scalar {
                            return None;
                        }
                        out.push(l);
                    }
                    ConstValue::Vector {
                        components: v,
                        scalar: inner_scalar,
                        ..
                    } => {
                        if inner_scalar != target_scalar {
                            return None;
                        }
                        out.extend(v);
                    }
                }
            }
            if out.len() != expected {
                return None;
            }
            Some(ConstValue::Vector {
                components: out,
                size: *size,
                scalar: target_scalar,
            })
        }
        _ => None,
    }
}

/// [`eval_unary`] broadcast over vector components.
fn eval_const_unary(op: naga::UnaryOperator, val: ConstValue) -> Option<ConstValue> {
    match val {
        ConstValue::Scalar(lit) => eval_unary(op, lit).map(ConstValue::Scalar),
        ConstValue::Vector {
            components,
            size,
            scalar,
        } => {
            let folded: Option<Vec<_>> =
                components.into_iter().map(|l| eval_unary(op, l)).collect();
            Some(ConstValue::Vector {
                components: folded?,
                size,
                scalar,
            })
        }
    }
}

/// Operators whose result scalar is `bool` regardless of operand type.
fn is_relational_op(op: naga::BinaryOperator) -> bool {
    matches!(
        op,
        naga::BinaryOperator::Equal
            | naga::BinaryOperator::NotEqual
            | naga::BinaryOperator::Less
            | naga::BinaryOperator::LessEqual
            | naga::BinaryOperator::Greater
            | naga::BinaryOperator::GreaterEqual
    )
}

/// [`eval_binary`] over scalar / same-size vector / broadcast scalar-vector
/// operand pairs.
fn eval_const_binary(
    op: naga::BinaryOperator,
    lhs: ConstValue,
    rhs: ConstValue,
) -> Option<ConstValue> {
    match (lhs, rhs) {
        (ConstValue::Scalar(l), ConstValue::Scalar(r)) => {
            eval_binary(op, l, r).map(ConstValue::Scalar)
        }
        (
            ConstValue::Vector {
                components: lc,
                size: ls,
                scalar: lscalar,
            },
            ConstValue::Vector {
                components: rc,
                size: rs,
                ..
            },
        ) if ls == rs => {
            let folded: Option<Vec<_>> = lc
                .into_iter()
                .zip(rc)
                .map(|(l, r)| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                lscalar
            };
            Some(ConstValue::Vector {
                components,
                size: ls,
                scalar: out_scalar,
            })
        }
        (
            ConstValue::Scalar(l),
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
        ) => {
            let folded: Option<Vec<_>> = components
                .into_iter()
                .map(|r| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                scalar
            };
            Some(ConstValue::Vector {
                components,
                size,
                scalar: out_scalar,
            })
        }
        (
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
            ConstValue::Scalar(r),
        ) => {
            let folded: Option<Vec<_>> = components
                .into_iter()
                .map(|l| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                scalar
            };
            Some(ConstValue::Vector {
                components,
                size,
                scalar: out_scalar,
            })
        }
        _ => None,
    }
}

// MARK: Materialisation helpers

/// A `Compose` for a folded vector built from `Literal` handles already in
/// the arena, or `None` unless every component's handle precedes `target`
/// (the Compose must stay topologically valid).  `literal_cache` maps each
/// literal to its SMALLEST handle, which maximises those hits.
fn materialize_vector(
    target: Handle<naga::Expression>,
    literals: &[naga::Literal],
    size: naga::VectorSize,
    scalar: naga::Scalar,
    literal_cache: &FxHashMap<LiteralKey, Handle<naga::Expression>>,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> Option<naga::Expression> {
    let ty = *vector_type_cache.get(&(size, scalar))?;

    let mut handles = Vec::with_capacity(literals.len());
    for lit in literals {
        let h = *literal_cache.get(&literal_key(*lit))?;
        if h.index() >= target.index() {
            return None;
        }
        handles.push(h);
    }

    Some(naga::Expression::Compose {
        ty,
        components: handles,
    })
}

/// Hash key for a `naga::Literal`: floats by `to_bits`, so NaN payloads
/// survive and `-0.0` differs from `+0.0`.
#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
enum LiteralKey {
    F32(u32),
    F64(u64),
    F16(u16),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    AbstractInt(i64),
    AbstractFloat(u64),
}

fn literal_key(lit: naga::Literal) -> LiteralKey {
    match lit {
        naga::Literal::F32(v) => LiteralKey::F32(v.to_bits()),
        naga::Literal::F64(v) => LiteralKey::F64(v.to_bits()),
        naga::Literal::F16(v) => LiteralKey::F16(v.to_bits()),
        naga::Literal::U16(v) => LiteralKey::U16(v),
        naga::Literal::I16(v) => LiteralKey::I16(v),
        naga::Literal::U32(v) => LiteralKey::U32(v),
        naga::Literal::I32(v) => LiteralKey::I32(v),
        naga::Literal::U64(v) => LiteralKey::U64(v),
        naga::Literal::I64(v) => LiteralKey::I64(v),
        naga::Literal::Bool(v) => LiteralKey::Bool(v),
        naga::Literal::AbstractInt(v) => LiteralKey::AbstractInt(v),
        naga::Literal::AbstractFloat(v) => LiteralKey::AbstractFloat(v.to_bits()),
    }
}

/// Each scalar `Literal` in `arena` -> the SMALLEST handle carrying it;
/// [`note_literal_in_cache`] keeps the invariant as the fold writes new
/// literals.
fn build_literal_cache(
    arena: &naga::Arena<naga::Expression>,
) -> FxHashMap<LiteralKey, Handle<naga::Expression>> {
    let mut cache: FxHashMap<LiteralKey, Handle<naga::Expression>> = Default::default();
    for (h, expr) in arena.iter() {
        if let naga::Expression::Literal(lit) = expr {
            cache
                .entry(literal_key(*lit))
                .and_modify(|cur| {
                    if h.index() < cur.index() {
                        *cur = h;
                    }
                })
                .or_insert(h);
        }
    }
    cache
}

/// `(size, scalar)` -> vector `Type` handle, so [`materialize_vector`] never
/// scans the type arena.
fn build_vector_type_cache(
    types: &naga::UniqueArena<naga::Type>,
) -> FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>> {
    let mut cache = FxHashMap::default();
    for (h, t) in types.iter() {
        if let naga::TypeInner::Vector { size, scalar } = t.inner {
            cache.entry((size, scalar)).or_insert(h);
        }
    }
    cache
}

/// Record `handle` as a carrier of `literal`, keeping the smallest-handle
/// invariant.
fn note_literal_in_cache(
    cache: &mut FxHashMap<LiteralKey, Handle<naga::Expression>>,
    handle: Handle<naga::Expression>,
    literal: naga::Literal,
) {
    cache
        .entry(literal_key(literal))
        .and_modify(|cur| {
            if handle.index() < cur.index() {
                *cur = handle;
            }
        })
        .or_insert(handle);
}

/// Per-scalar unary evaluator; declines operator / type pairs whose fold
/// would change observable behaviour.
fn eval_unary(op: naga::UnaryOperator, rhs: naga::Literal) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::UnaryOperator as U;

    match (op, rhs) {
        // naga's validator rejects non-finite `F32` / `F64` literals
        // (`LiteralError::NonFinite`); `is_finite()` rather than `!is_nan()`
        // keeps a negated infinity out of the IR whatever upstream injects.
        (U::Negate, L::F32(v)) if (-v).is_finite() => Some(L::F32(-v)),
        (U::Negate, L::F64(v)) if (-v).is_finite() => Some(L::F64(-v)),
        (U::Negate, L::I32(v)) => v.checked_neg().map(L::I32),
        (U::Negate, L::I64(v)) => v.checked_neg().map(L::I64),
        (U::Negate, L::AbstractInt(v)) => v.checked_neg().map(L::AbstractInt),
        (U::Negate, L::AbstractFloat(v)) if (-v).is_finite() => Some(L::AbstractFloat(-v)),
        (U::LogicalNot, L::Bool(v)) => Some(L::Bool(!v)),
        (U::BitwiseNot, L::U32(v)) => Some(L::U32(!v)),
        (U::BitwiseNot, L::U64(v)) => Some(L::U64(!v)),
        (U::BitwiseNot, L::I32(v)) => Some(L::I32(!v)),
        (U::BitwiseNot, L::I64(v)) => Some(L::I64(!v)),
        (U::BitwiseNot, L::AbstractInt(v)) => Some(L::AbstractInt(!v)),
        _ => None,
    }
}

/// Per-scalar binary evaluator; NaN, overflow, and divide-by-zero cases
/// decline so folding never changes observable output.
fn eval_binary(
    op: naga::BinaryOperator,
    lhs: naga::Literal,
    rhs: naga::Literal,
) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    use naga::Literal as L;

    match (op, lhs, rhs) {
        (B::Add, L::F32(a), L::F32(b)) if (a + b).is_finite() => Some(L::F32(a + b)),
        (B::Subtract, L::F32(a), L::F32(b)) if (a - b).is_finite() => Some(L::F32(a - b)),
        (B::Multiply, L::F32(a), L::F32(b)) if (a * b).is_finite() => Some(L::F32(a * b)),
        (B::Divide, L::F32(a), L::F32(b)) if b != 0.0 && (a / b).is_finite() => Some(L::F32(a / b)),
        // WGSL lowers float `a % b` to `a - b*trunc(a/b)` in OPERAND
        // precision, which diverges from exact fmod by a FULL divisor
        // whenever the rounded quotient crosses an integer the exact one
        // does not (`33554432f % 3f` is fmod 2.0 but 0.0 on every
        // round-to-nearest GPU).  Fold only when both agree; the `|a/b|` cap
        // keeps a quotient whose trunc is unrepresentable out of the
        // comparison.
        (B::Modulo, L::F32(a), L::F32(b))
            if b != 0.0
                && (a % b).is_finite()
                && (a / b).abs() < 16_777_216.0
                && a % b == a - b * (a / b).trunc() =>
        {
            Some(L::F32(a % b))
        }

        (B::Add, L::F64(a), L::F64(b)) if (a + b).is_finite() => Some(L::F64(a + b)),
        (B::Subtract, L::F64(a), L::F64(b)) if (a - b).is_finite() => Some(L::F64(a - b)),
        (B::Multiply, L::F64(a), L::F64(b)) if (a * b).is_finite() => Some(L::F64(a * b)),
        (B::Divide, L::F64(a), L::F64(b)) if b != 0.0 && (a / b).is_finite() => Some(L::F64(a / b)),
        // Same stepwise guard; WGSL has no runtime f64, so this sees only
        // non-WGSL-frontend IR.
        (B::Modulo, L::F64(a), L::F64(b))
            if b != 0.0
                && (a % b).is_finite()
                && (a / b).abs() < 9_007_199_254_740_992.0
                && a % b == a - b * (a / b).trunc() =>
        {
            Some(L::F64(a % b))
        }

        // Signed overflow declines rather than wraps: the surviving
        // expression still ships (both validators accept `2147483647i+1i`
        // in runtime position), and a const-context original never reaches
        // the passes.
        (B::Add, L::I32(a), L::I32(b)) => a.checked_add(b).map(L::I32),
        (B::Subtract, L::I32(a), L::I32(b)) => a.checked_sub(b).map(L::I32),
        (B::Multiply, L::I32(a), L::I32(b)) => a.checked_mul(b).map(L::I32),
        // `MIN / -1` and `MIN % -1` MUST fold: the pair only arises from
        // nagami's own literal substitution into runtime expressions, where
        // WGSL defines the results as e1 and 0
        // (https://www.w3.org/TR/WGSL/#arithmetic-expr); declined, it
        // round-trips into naga's text const-eval, which rejects it and kills
        // the emission.
        (B::Divide, L::I32(a), L::I32(b)) if b != 0 => Some(L::I32(a.checked_div(b).unwrap_or(a))),
        (B::Modulo, L::I32(a), L::I32(b)) if b != 0 => Some(L::I32(a.checked_rem(b).unwrap_or(0))),

        (B::Add, L::I64(a), L::I64(b)) => a.checked_add(b).map(L::I64),
        (B::Subtract, L::I64(a), L::I64(b)) => a.checked_sub(b).map(L::I64),
        (B::Multiply, L::I64(a), L::I64(b)) => a.checked_mul(b).map(L::I64),
        (B::Divide, L::I64(a), L::I64(b)) if b != 0 => Some(L::I64(a.checked_div(b).unwrap_or(a))),
        (B::Modulo, L::I64(a), L::I64(b)) if b != 0 => Some(L::I64(a.checked_rem(b).unwrap_or(0))),

        (B::Add, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_add(b))),
        (B::Subtract, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_sub(b))),
        (B::Multiply, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_mul(b))),
        (B::Divide, L::U32(a), L::U32(b)) if b != 0 => Some(L::U32(a / b)),
        (B::Modulo, L::U32(a), L::U32(b)) if b != 0 => Some(L::U32(a % b)),

        (B::Add, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_add(b))),
        (B::Subtract, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_sub(b))),
        (B::Multiply, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_mul(b))),
        (B::Divide, L::U64(a), L::U64(b)) if b != 0 => Some(L::U64(a / b)),
        (B::Modulo, L::U64(a), L::U64(b)) if b != 0 => Some(L::U64(a % b)),

        (B::Add, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_add(b).map(L::AbstractInt),
        (B::Subtract, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_sub(b).map(L::AbstractInt),
        (B::Multiply, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_mul(b).map(L::AbstractInt),
        (B::Divide, L::AbstractInt(a), L::AbstractInt(b)) if b != 0 => {
            a.checked_div(b).map(L::AbstractInt)
        }
        (B::Modulo, L::AbstractInt(a), L::AbstractInt(b)) if b != 0 => {
            a.checked_rem(b).map(L::AbstractInt)
        }

        (B::Add, L::AbstractFloat(a), L::AbstractFloat(b)) if (a + b).is_finite() => {
            Some(L::AbstractFloat(a + b))
        }
        (B::Subtract, L::AbstractFloat(a), L::AbstractFloat(b)) if (a - b).is_finite() => {
            Some(L::AbstractFloat(a - b))
        }
        (B::Multiply, L::AbstractFloat(a), L::AbstractFloat(b)) if (a * b).is_finite() => {
            Some(L::AbstractFloat(a * b))
        }
        (B::Divide, L::AbstractFloat(a), L::AbstractFloat(b))
            if b != 0.0 && (a / b).is_finite() =>
        {
            Some(L::AbstractFloat(a / b))
        }
        (B::Modulo, L::AbstractFloat(a), L::AbstractFloat(b))
            if b != 0.0 && (a % b).is_finite() =>
        {
            Some(L::AbstractFloat(a % b))
        }

        (B::Equal, a, b) => Some(L::Bool(a == b)),
        (B::NotEqual, a, b) => Some(L::Bool(a != b)),

        (B::Less, L::F32(a), L::F32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::F32(a), L::F32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::F32(a), L::F32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::F32(a), L::F32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::F64(a), L::F64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::F64(a), L::F64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::F64(a), L::F64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::F64(a), L::F64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::I32(a), L::I32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::I32(a), L::I32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::I32(a), L::I32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::I32(a), L::I32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::I64(a), L::I64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::I64(a), L::I64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::I64(a), L::I64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::I64(a), L::I64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::U32(a), L::U32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::U32(a), L::U32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::U32(a), L::U32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::U32(a), L::U32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::U64(a), L::U64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::U64(a), L::U64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::U64(a), L::U64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::U64(a), L::U64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a >= b)),

        (B::LogicalAnd, L::Bool(a), L::Bool(b)) => Some(L::Bool(a && b)),
        (B::LogicalOr, L::Bool(a), L::Bool(b)) => Some(L::Bool(a || b)),

        (B::And, L::U32(a), L::U32(b)) => Some(L::U32(a & b)),
        (B::ExclusiveOr, L::U32(a), L::U32(b)) => Some(L::U32(a ^ b)),
        (B::InclusiveOr, L::U32(a), L::U32(b)) => Some(L::U32(a | b)),
        (B::ShiftLeft, L::U32(a), L::U32(b)) if b < 32 => Some(L::U32(a.wrapping_shl(b))),
        (B::ShiftRight, L::U32(a), L::U32(b)) if b < 32 => Some(L::U32(a.wrapping_shr(b))),

        (B::And, L::U64(a), L::U64(b)) => Some(L::U64(a & b)),
        (B::ExclusiveOr, L::U64(a), L::U64(b)) => Some(L::U64(a ^ b)),
        (B::InclusiveOr, L::U64(a), L::U64(b)) => Some(L::U64(a | b)),
        // naga's WGSL frontend concretises every shift amount to `u32`
        // whatever the left operand's width; a `U64` right pattern could
        // never match.
        (B::ShiftLeft, L::U64(a), L::U32(b)) if b < 64 => Some(L::U64(a.wrapping_shl(b))),
        (B::ShiftRight, L::U64(a), L::U32(b)) if b < 64 => Some(L::U64(a.wrapping_shr(b))),

        (B::And, L::I32(a), L::I32(b)) => Some(L::I32(a & b)),
        (B::ExclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a ^ b)),
        (B::InclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a | b)),
        // A sign-changing `e1 << e2` is a shader-creation error only in CONST
        // contexts, which naga rejected at ingest; a literal pair here is a
        // runtime expression manufactured by nagami's own transforms, where
        // WGSL defines the plain bit-pattern result
        // (https://www.w3.org/TR/WGSL/#bit-expr).  Declined, the pair fails
        // naga's text const-eval on re-parse and kills the emission.
        (B::ShiftLeft, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shl(b))),
        (B::ShiftRight, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shr(b))),

        (B::And, L::I64(a), L::I64(b)) => Some(L::I64(a & b)),
        (B::ExclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a ^ b)),
        (B::InclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a | b)),
        (B::ShiftLeft, L::I64(a), L::U32(b)) if b < 64 => Some(L::I64(a.wrapping_shl(b))),
        (B::ShiftRight, L::I64(a), L::U32(b)) if b < 64 => Some(L::I64(a.wrapping_shr(b))),

        (B::And, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a & b)),
        (B::ExclusiveOr, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a ^ b)),
        (B::InclusiveOr, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a | b)),
        (B::ShiftLeft, L::AbstractInt(a), L::AbstractInt(b)) if (0..64).contains(&b) => {
            let wide = (a as i128).wrapping_shl(b as u32);
            let narrowed = wide as i64;
            (narrowed as i128 == wide).then_some(L::AbstractInt(narrowed))
        }
        (B::ShiftRight, L::AbstractInt(a), L::AbstractInt(b)) if (0..64).contains(&b) => {
            Some(L::AbstractInt(a.wrapping_shr(b as u32)))
        }

        _ => None,
    }
}

/// Fold a math built-in over scalar literals.  Comparison, decomposition,
/// and integer-bit functions are bit-exact; the trigonometric / exponential
/// family has a WGSL-defined error envelope, so those folds substitute a
/// conformant value, not a bit-identical one.  Unsupported functions, type
/// mismatches, NaN-sensitive cases, and domain errors decline.
fn eval_math_scalar(
    fun: naga::MathFunction,
    arg: naga::Literal,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::MathFunction as M;

    // naga's validator rejects non-finite literals, so every float result
    // routes through `finite_*`.
    fn finite_f32(v: f32) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::F32(v))
    }
    fn finite_f64(v: f64) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::F64(v))
    }
    fn finite_af(v: f64) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::AbstractFloat(v))
    }

    match fun {
        M::Abs => match arg {
            L::F32(v) => finite_f32(v.abs()),
            L::F64(v) => finite_f64(v.abs()),
            L::AbstractFloat(v) => finite_af(v.abs()),
            L::I32(v) => v.checked_abs().map(L::I32),
            L::I64(v) => v.checked_abs().map(L::I64),
            L::AbstractInt(v) => v.checked_abs().map(L::AbstractInt),
            L::U32(v) => Some(L::U32(v)),
            L::U64(v) => Some(L::U64(v)),
            _ => None,
        },
        M::Min => match (arg, arg1?) {
            // WGSL propagates NaN through min / max / clamp; Rust's return
            // the other operand (and `clamp` panics on NaN bounds).
            (L::F32(a), L::F32(b)) if !a.is_nan() && !b.is_nan() => Some(L::F32(a.min(b))),
            (L::F64(a), L::F64(b)) if !a.is_nan() && !b.is_nan() => Some(L::F64(a.min(b))),
            (L::AbstractFloat(a), L::AbstractFloat(b)) if !a.is_nan() && !b.is_nan() => {
                Some(L::AbstractFloat(a.min(b)))
            }
            (L::I32(a), L::I32(b)) => Some(L::I32(a.min(b))),
            (L::I64(a), L::I64(b)) => Some(L::I64(a.min(b))),
            (L::U32(a), L::U32(b)) => Some(L::U32(a.min(b))),
            (L::U64(a), L::U64(b)) => Some(L::U64(a.min(b))),
            (L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a.min(b))),
            _ => None,
        },
        M::Max => match (arg, arg1?) {
            (L::F32(a), L::F32(b)) if !a.is_nan() && !b.is_nan() => Some(L::F32(a.max(b))),
            (L::F64(a), L::F64(b)) if !a.is_nan() && !b.is_nan() => Some(L::F64(a.max(b))),
            (L::AbstractFloat(a), L::AbstractFloat(b)) if !a.is_nan() && !b.is_nan() => {
                Some(L::AbstractFloat(a.max(b)))
            }
            (L::I32(a), L::I32(b)) => Some(L::I32(a.max(b))),
            (L::I64(a), L::I64(b)) => Some(L::I64(a.max(b))),
            (L::U32(a), L::U32(b)) => Some(L::U32(a.max(b))),
            (L::U64(a), L::U64(b)) => Some(L::U64(a.max(b))),
            (L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a.max(b))),
            _ => None,
        },
        M::Clamp => {
            let lo = arg1?;
            let hi = arg2?;
            match (arg, lo, hi) {
                (L::F32(v), L::F32(lo), L::F32(hi))
                    if lo <= hi && !v.is_nan() && !lo.is_nan() && !hi.is_nan() =>
                {
                    Some(L::F32(v.clamp(lo, hi)))
                }
                (L::F64(v), L::F64(lo), L::F64(hi))
                    if lo <= hi && !v.is_nan() && !lo.is_nan() && !hi.is_nan() =>
                {
                    Some(L::F64(v.clamp(lo, hi)))
                }
                (L::AbstractFloat(v), L::AbstractFloat(lo), L::AbstractFloat(hi))
                    if lo <= hi && !v.is_nan() && !lo.is_nan() && !hi.is_nan() =>
                {
                    Some(L::AbstractFloat(v.clamp(lo, hi)))
                }
                (L::I32(v), L::I32(lo), L::I32(hi)) if lo <= hi => Some(L::I32(v.clamp(lo, hi))),
                (L::I64(v), L::I64(lo), L::I64(hi)) if lo <= hi => Some(L::I64(v.clamp(lo, hi))),
                (L::U32(v), L::U32(lo), L::U32(hi)) if lo <= hi => Some(L::U32(v.clamp(lo, hi))),
                (L::U64(v), L::U64(lo), L::U64(hi)) if lo <= hi => Some(L::U64(v.clamp(lo, hi))),
                (L::AbstractInt(v), L::AbstractInt(lo), L::AbstractInt(hi)) if lo <= hi => {
                    Some(L::AbstractInt(v.clamp(lo, hi)))
                }
                _ => None,
            }
        }
        M::Saturate => match arg {
            L::F32(v) => finite_f32(v.clamp(0.0, 1.0)),
            L::F64(v) => finite_f64(v.clamp(0.0, 1.0)),
            L::AbstractFloat(v) => finite_af(v.clamp(0.0, 1.0)),
            _ => None,
        },

        M::Sign => match arg {
            // Rust's `signum(0.0)` is 1.0; WGSL `sign(0)` is 0.
            L::F32(v) if !v.is_nan() => Some(L::F32(if v == 0.0 { 0.0 } else { v.signum() })),
            L::F64(v) if !v.is_nan() => Some(L::F64(if v == 0.0 { 0.0 } else { v.signum() })),
            L::AbstractFloat(v) if !v.is_nan() => {
                Some(L::AbstractFloat(if v == 0.0 { 0.0 } else { v.signum() }))
            }
            L::I32(v) => Some(L::I32(v.signum())),
            L::I64(v) => Some(L::I64(v.signum())),
            L::AbstractInt(v) => Some(L::AbstractInt(v.signum())),
            _ => None,
        },

        M::Floor => match arg {
            L::F32(v) => finite_f32(v.floor()),
            L::F64(v) => finite_f64(v.floor()),
            L::AbstractFloat(v) => finite_af(v.floor()),
            _ => None,
        },
        M::Ceil => match arg {
            L::F32(v) => finite_f32(v.ceil()),
            L::F64(v) => finite_f64(v.ceil()),
            L::AbstractFloat(v) => finite_af(v.ceil()),
            _ => None,
        },
        M::Round => match arg {
            // WGSL `round` is ties-to-even.
            L::F32(v) => finite_f32(v.round_ties_even()),
            L::F64(v) => finite_f64(v.round_ties_even()),
            L::AbstractFloat(v) => finite_af(v.round_ties_even()),
            _ => None,
        },
        M::Trunc => match arg {
            L::F32(v) => finite_f32(v.trunc()),
            L::F64(v) => finite_f64(v.trunc()),
            L::AbstractFloat(v) => finite_af(v.trunc()),
            _ => None,
        },
        M::Fract => match arg {
            // WGSL `fract(e)` is `e - floor(e)`, not Rust's `e - trunc(e)`.
            L::F32(v) => finite_f32(v - v.floor()),
            L::F64(v) => finite_f64(v - v.floor()),
            L::AbstractFloat(v) => finite_af(v - v.floor()),
            _ => None,
        },

        M::Step => match (arg, arg1?) {
            // A NaN operand compares false and would fold to a wrong 0.0;
            // WGSL propagates it.
            (L::F32(edge), L::F32(x)) if !edge.is_nan() && !x.is_nan() => {
                Some(L::F32(if edge <= x { 1.0 } else { 0.0 }))
            }
            (L::F64(edge), L::F64(x)) if !edge.is_nan() && !x.is_nan() => {
                Some(L::F64(if edge <= x { 1.0 } else { 0.0 }))
            }
            (L::AbstractFloat(edge), L::AbstractFloat(x)) if !edge.is_nan() && !x.is_nan() => {
                Some(L::AbstractFloat(if edge <= x { 1.0 } else { 0.0 }))
            }
            _ => None,
        },
        M::Sqrt => match arg {
            L::F32(v) if v >= 0.0 => finite_f32(v.sqrt()),
            L::F64(v) if v >= 0.0 => finite_f64(v.sqrt()),
            L::AbstractFloat(v) if v >= 0.0 => finite_af(v.sqrt()),
            _ => None,
        },
        M::InverseSqrt => match arg {
            L::F32(v) if v > 0.0 => finite_f32(1.0 / v.sqrt()),
            L::F64(v) if v > 0.0 => finite_f64(1.0 / v.sqrt()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(1.0 / v.sqrt()),
            _ => None,
        },
        M::Fma => {
            let b = arg1?;
            let c = arg2?;
            match (arg, b, c) {
                (L::F32(a), L::F32(b), L::F32(c)) => finite_f32(a.mul_add(b, c)),
                (L::F64(a), L::F64(b), L::F64(c)) => finite_f64(a.mul_add(b, c)),
                (L::AbstractFloat(a), L::AbstractFloat(b), L::AbstractFloat(c)) => {
                    finite_af(a.mul_add(b, c))
                }
                _ => None,
            }
        }

        M::Cos => match arg {
            L::F32(v) => finite_f32(v.cos()),
            L::F64(v) => finite_f64(v.cos()),
            L::AbstractFloat(v) => finite_af(v.cos()),
            _ => None,
        },
        M::Sin => match arg {
            L::F32(v) => finite_f32(v.sin()),
            L::F64(v) => finite_f64(v.sin()),
            L::AbstractFloat(v) => finite_af(v.sin()),
            _ => None,
        },
        M::Tan => match arg {
            L::F32(v) => finite_f32(v.tan()),
            L::F64(v) => finite_f64(v.tan()),
            L::AbstractFloat(v) => finite_af(v.tan()),
            _ => None,
        },
        M::Cosh => match arg {
            L::F32(v) => finite_f32(v.cosh()),
            L::F64(v) => finite_f64(v.cosh()),
            L::AbstractFloat(v) => finite_af(v.cosh()),
            _ => None,
        },
        M::Sinh => match arg {
            L::F32(v) => finite_f32(v.sinh()),
            L::F64(v) => finite_f64(v.sinh()),
            L::AbstractFloat(v) => finite_af(v.sinh()),
            _ => None,
        },
        M::Tanh => match arg {
            L::F32(v) => finite_f32(v.tanh()),
            L::F64(v) => finite_f64(v.tanh()),
            L::AbstractFloat(v) => finite_af(v.tanh()),
            _ => None,
        },
        M::Acos => match arg {
            L::F32(v) if v.abs() <= 1.0 => finite_f32(v.acos()),
            L::F64(v) if v.abs() <= 1.0 => finite_f64(v.acos()),
            L::AbstractFloat(v) if v.abs() <= 1.0 => finite_af(v.acos()),
            _ => None,
        },
        M::Asin => match arg {
            L::F32(v) if v.abs() <= 1.0 => finite_f32(v.asin()),
            L::F64(v) if v.abs() <= 1.0 => finite_f64(v.asin()),
            L::AbstractFloat(v) if v.abs() <= 1.0 => finite_af(v.asin()),
            _ => None,
        },
        M::Atan => match arg {
            L::F32(v) => finite_f32(v.atan()),
            L::F64(v) => finite_f64(v.atan()),
            L::AbstractFloat(v) => finite_af(v.atan()),
            _ => None,
        },
        M::Atan2 => match (arg, arg1?) {
            // WGSL leaves `atan2(0, 0)` implementation-defined: a GPU may
            // return any of {0, +/-pi/2, pi}.
            (L::F32(y), L::F32(x)) if y != 0.0 || x != 0.0 => finite_f32(y.atan2(x)),
            (L::F64(y), L::F64(x)) if y != 0.0 || x != 0.0 => finite_f64(y.atan2(x)),
            (L::AbstractFloat(y), L::AbstractFloat(x)) if y != 0.0 || x != 0.0 => {
                finite_af(y.atan2(x))
            }
            _ => None,
        },
        M::Asinh => match arg {
            L::F32(v) => finite_f32(v.asinh()),
            L::F64(v) => finite_f64(v.asinh()),
            L::AbstractFloat(v) => finite_af(v.asinh()),
            _ => None,
        },
        M::Acosh => match arg {
            L::F32(v) if v >= 1.0 => finite_f32(v.acosh()),
            L::F64(v) if v >= 1.0 => finite_f64(v.acosh()),
            L::AbstractFloat(v) if v >= 1.0 => finite_af(v.acosh()),
            _ => None,
        },
        M::Atanh => match arg {
            L::F32(v) if v.abs() < 1.0 => finite_f32(v.atanh()),
            L::F64(v) if v.abs() < 1.0 => finite_f64(v.atanh()),
            L::AbstractFloat(v) if v.abs() < 1.0 => finite_af(v.atanh()),
            _ => None,
        },
        M::Radians => match arg {
            L::F32(v) => finite_f32(v.to_radians()),
            L::F64(v) => finite_f64(v.to_radians()),
            L::AbstractFloat(v) => finite_af(v.to_radians()),
            _ => None,
        },
        M::Degrees => match arg {
            L::F32(v) => finite_f32(v.to_degrees()),
            L::F64(v) => finite_f64(v.to_degrees()),
            L::AbstractFloat(v) => finite_af(v.to_degrees()),
            _ => None,
        },

        M::Exp => match arg {
            L::F32(v) => finite_f32(v.exp()),
            L::F64(v) => finite_f64(v.exp()),
            L::AbstractFloat(v) => finite_af(v.exp()),
            _ => None,
        },
        M::Exp2 => match arg {
            L::F32(v) => finite_f32(v.exp2()),
            L::F64(v) => finite_f64(v.exp2()),
            L::AbstractFloat(v) => finite_af(v.exp2()),
            _ => None,
        },
        M::Log => match arg {
            L::F32(v) if v > 0.0 => finite_f32(v.ln()),
            L::F64(v) if v > 0.0 => finite_f64(v.ln()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(v.ln()),
            _ => None,
        },
        M::Log2 => match arg {
            L::F32(v) if v > 0.0 => finite_f32(v.log2()),
            L::F64(v) if v > 0.0 => finite_f64(v.log2()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(v.log2()),
            _ => None,
        },
        M::Pow => match (arg, arg1?) {
            // WGSL requires `e1 >= 0`, and `pow(0, b)` with `b <= 0` is
            // implementation-defined (Rust says 1.0 for 0^0; a GPU may say
            // NaN or 0).
            (L::F32(a), L::F32(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => finite_f32(a.powf(b)),
            (L::F64(a), L::F64(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => finite_f64(a.powf(b)),
            (L::AbstractFloat(a), L::AbstractFloat(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => {
                finite_af(a.powf(b))
            }
            _ => None,
        },

        M::CountTrailingZeros => match arg {
            L::U32(v) => Some(L::U32(v.trailing_zeros())),
            L::I32(v) => Some(L::I32(v.trailing_zeros() as i32)),
            L::U64(v) => Some(L::U64(v.trailing_zeros() as u64)),
            L::I64(v) => Some(L::I64(v.trailing_zeros() as i64)),
            _ => None,
        },
        M::CountLeadingZeros => match arg {
            L::U32(v) => Some(L::U32(v.leading_zeros())),
            L::I32(v) => Some(L::I32(v.leading_zeros() as i32)),
            L::U64(v) => Some(L::U64(v.leading_zeros() as u64)),
            L::I64(v) => Some(L::I64(v.leading_zeros() as i64)),
            _ => None,
        },
        M::CountOneBits => match arg {
            L::U32(v) => Some(L::U32(v.count_ones())),
            L::I32(v) => Some(L::I32(v.count_ones() as i32)),
            L::U64(v) => Some(L::U64(v.count_ones() as u64)),
            L::I64(v) => Some(L::I64(v.count_ones() as i64)),
            _ => None,
        },
        M::ReverseBits => match arg {
            L::U32(v) => Some(L::U32(v.reverse_bits())),
            L::I32(v) => Some(L::I32(v.reverse_bits())),
            L::U64(v) => Some(L::U64(v.reverse_bits())),
            L::I64(v) => Some(L::I64(v.reverse_bits())),
            _ => None,
        },
        M::FirstTrailingBit => match arg {
            L::U32(v) => Some(L::U32(if v == 0 { u32::MAX } else { v.trailing_zeros() })),
            L::I32(v) => Some(L::I32(if v == 0 {
                -1
            } else {
                v.trailing_zeros() as i32
            })),
            L::U64(v) => Some(L::U64(if v == 0 {
                u64::MAX
            } else {
                v.trailing_zeros() as u64
            })),
            L::I64(v) => Some(L::I64(if v == 0 {
                -1
            } else {
                v.trailing_zeros() as i64
            })),
            _ => None,
        },
        M::FirstLeadingBit => match arg {
            L::U32(v) => Some(L::U32(if v == 0 {
                u32::MAX
            } else {
                31 - v.leading_zeros()
            })),
            L::I32(v) => Some(L::I32(if v == 0 || v == -1 {
                -1
            } else if v > 0 {
                31 - (v.leading_zeros() as i32)
            } else {
                // Negative: the highest bit differing from the sign bit.
                31 - (v.leading_ones() as i32)
            })),
            L::U64(v) => Some(L::U64(if v == 0 {
                u64::MAX
            } else {
                63 - v.leading_zeros() as u64
            })),
            L::I64(v) => Some(L::I64(if v == 0 || v == -1 {
                -1
            } else if v > 0 {
                63 - (v.leading_zeros() as i64)
            } else {
                63 - (v.leading_ones() as i64)
            })),
            _ => None,
        },

        _ => None,
    }
}

/// [`eval_math_scalar`] broadcast over vector arguments; sizes must match.
fn eval_const_math(
    fun: naga::MathFunction,
    arg: ConstValue,
    arg1: Option<ConstValue>,
    arg2: Option<ConstValue>,
) -> Option<ConstValue> {
    fn as_scalar(v: &ConstValue) -> Option<naga::Literal> {
        match v {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }

    fn as_vector(v: &ConstValue) -> Option<(&[naga::Literal], naga::VectorSize, naga::Scalar)> {
        match v {
            ConstValue::Vector {
                components,
                size,
                scalar,
            } => Some((components, *size, *scalar)),
            _ => None,
        }
    }

    if let Some(a) = as_scalar(&arg) {
        let b = match &arg1 {
            Some(v) => Some(as_scalar(v)?),
            None => None,
        };
        let c = match &arg2 {
            Some(v) => Some(as_scalar(v)?),
            None => None,
        };
        return eval_math_scalar(fun, a, b, c).map(ConstValue::Scalar);
    }

    if let Some((comps, size, scalar)) = as_vector(&arg) {
        let n = comps.len();

        let arg1_comps: Option<Vec<naga::Literal>> = match &arg1 {
            Some(v) => {
                let (c1, s1, _) = as_vector(v)?;
                if s1 != size {
                    return None;
                }
                Some(c1.to_vec())
            }
            None => None,
        };
        let arg2_comps: Option<Vec<naga::Literal>> = match &arg2 {
            Some(v) => {
                let (c2, s2, _) = as_vector(v)?;
                if s2 != size {
                    return None;
                }
                Some(c2.to_vec())
            }
            None => None,
        };

        let folded: Option<Vec<naga::Literal>> = (0..n)
            .map(|i| {
                let a = comps[i];
                let b = arg1_comps.as_ref().map(|c| c[i]);
                let c = arg2_comps.as_ref().map(|c| c[i]);
                eval_math_scalar(fun, a, b, c)
            })
            .collect();

        return Some(ConstValue::Vector {
            components: folded?,
            size,
            scalar,
        });
    }

    None
}

// MARK: Identity / absorbing operand detection

/// Literal zero of any scalar type, both float signs included: right for
/// absorbing rules (`x & 0 -> 0`), wrong for additive identities, where the
/// sign matters ([`is_additive_identity_zero`]).
fn is_zero(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F32(v)) if v == 0.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F64(v)) if v == 0.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F16(v)) if v.to_bits() & 0x7FFF == 0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::I32(0)
                | naga::Literal::U32(0)
                | naga::Literal::I64(0)
                | naga::Literal::U64(0)
                | naga::Literal::AbstractInt(0)
        )
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::AbstractFloat(v)) if v == 0.0
    )
}

/// Literal zeros safe to drop as an additive identity.  Under IEEE 754
/// round-to-nearest `(-0.0) + (+0.0)` and `(-0.0) - (-0.0)` are both
/// `+0.0`, so removing a float zero can flip the sign of `x = -0.0`, and
/// the safe sign differs per operator; integer zeros only, uniformly, at
/// the cost of a few unfolded `float + 0.0`.
fn is_additive_identity_zero(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    is_integer_zero(arena, h)
}

/// Literal integer zero: no signed zero or NaN / Inf to preserve.
fn is_integer_zero(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::I32(0)
                | naga::Literal::U32(0)
                | naga::Literal::I64(0)
                | naga::Literal::U64(0)
                | naga::Literal::AbstractInt(0)
        )
    )
}

fn is_one(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    // `0x3C00` is binary16 `+1.0`.
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F32(v)) if v == 1.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F64(v)) if v == 1.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F16(v)) if v.to_bits() == 0x3C00
    ) || matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::I32(1)
                | naga::Literal::U32(1)
                | naga::Literal::I64(1)
                | naga::Literal::U64(1)
                | naga::Literal::AbstractInt(1)
        )
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::AbstractFloat(v)) if v == 1.0
    )
}

fn is_all_ones(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::U32(u32::MAX)
                | naga::Literal::I32(-1)
                | naga::Literal::U64(u64::MAX)
                | naga::Literal::I64(-1)
                | naga::Literal::AbstractInt(-1)
        )
    )
}

/// The surviving operand of an `x <op> identity` pattern (left-operand forms
/// too where the op commutes):
///
/// ```text
/// x + 0 = x          x - 0 = x          x * 1 = x          x / 1 = x
/// x | 0 = x          x ^ 0 = x          x & all_ones = x
/// x && true = x      x || false = x
/// ```
fn check_identity_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Add => {
            if is_additive_identity_zero(arena, left) {
                Some(right)
            } else if is_additive_identity_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Subtract => {
            // One-sided: `x - 0 = x` but `0 - x = -x`.
            if is_additive_identity_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Multiply => {
            if is_one(arena, left) {
                Some(right)
            } else if is_one(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Divide => {
            if is_one(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::InclusiveOr => {
            if is_zero(arena, left) {
                Some(right)
            } else if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::ExclusiveOr => {
            if is_zero(arena, left) {
                Some(right)
            } else if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::And => {
            if is_all_ones(arena, left) {
                Some(right)
            } else if is_all_ones(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::LogicalAnd => {
            if is_bool_true(arena, left) {
                Some(right)
            } else if is_bool_true(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::LogicalOr => {
            if is_bool_false(arena, left) {
                Some(right)
            } else if is_bool_false(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The operand an `x <op> absorbing` pattern collapses to (left-operand
/// forms too):
///
/// ```text
/// x * 0 = 0          x & 0 = 0          x | all_ones = all_ones
/// x && false = false x || true = true
/// ```
fn check_absorbing_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Multiply => {
            // Integer zeros only: a float product carries the IEEE sign of
            // BOTH operands (`-2.0h * 0.0h` is `-0.0h`) and is NaN for a
            // non-finite `x`, and F16 has no `eval_binary` arm to fold it
            // sign-aware first.  Left as a bare product, naga re-parses it
            // to the correctly-signed zero.
            if is_integer_zero(arena, left) {
                Some(left)
            } else if is_integer_zero(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::And => {
            if is_zero(arena, left) {
                Some(left)
            } else if is_zero(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::InclusiveOr => {
            if is_all_ones(arena, left) {
                Some(left)
            } else if is_all_ones(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::LogicalAnd => {
            if is_bool_false(arena, left) {
                Some(left)
            } else if is_bool_false(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::LogicalOr => {
            if is_bool_true(arena, left) {
                Some(left)
            } else if is_bool_true(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        _ => None,
    }
}

// MARK: Tests

#[cfg(test)]
#[path = "const_fold_tests.rs"]
mod tests;
