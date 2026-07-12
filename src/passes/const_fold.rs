//! Constant folding.
//!
//! Rewrites expressions whose operands are statically known values
//! into `Literal`, `ZeroValue`, or `Compose` nodes that the generator
//! can emit directly.  The pass operates in three arenas:
//!
//! * `module.global_expressions` for `const` initializers and global
//!   splats.
//! * each function's `expressions` arena, folded with a per-function
//!   pass that resolves `Constant` operands through
//!   `build_constant_literal_cache` and tracks which handles must drop
//!   out of their `Emit` range (folded literals are declarative, not
//!   emittable).
//! * the `Emit` ranges themselves, rebuilt in place via
//!   `rebuild_emit_ranges_after_removal` after each function fold.
//!
//! Structural folding covers scalar arithmetic, binary and unary
//! operators, casts, relational built-ins, and a subset of math
//! built-ins; composite folding handles vector construction, swizzle
//! collapse, and splat elision.  Exactly-rounded built-ins fold
//! bit-exactly; the accuracy-tolerant transcendentals (sin/cos/exp/log/
//! pow/...) fold to a value within WGSL's permitted error envelope - the
//! same conformance-tolerance doctrine as the float divide/modulo folds,
//! not bit-identical.  Overflow- and NaN-sensitive cases decline.

use std::collections::{HashMap, HashSet};

use naga::Handle;

use crate::error::Error;
use crate::passes::expr_util::rebuild_emit_ranges_after_removal;
use crate::pipeline::{Pass, PassContext};

/// `true` when [`fold_local_expressions`]'s identity / involution /
/// select-collapse rewrites can safely overwrite `arena[handle]` with
/// a clone of `arena[other]`.  Pure expressions (declarative leaves,
/// structural / arithmetic wrappers) reproduce the same value at any
/// evaluation site; memory reads, derivatives, and statement-attached
/// results do not, and a clone would re-execute on the duplicate
/// `Emit` against possibly-different shared state - or land in an
/// arena slot disconnected from the statement that produces it.
///
/// Match is exhaustive on purpose: a new naga variant must trip the
/// build so the maintainer classifies it explicitly.  A `_ => true`
/// fallback would silently corrupt; `_ => false` would silently lose
/// optimisation.  Keep no catch-all.
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
        // Memory / derivative / array-length reads: re-execution at a
        // different program point observes shared state that may have
        // been written between the original and clone sites.
        E::Load { .. }
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::Derivative { .. }
        | E::ArrayLength(_) => false,
        // Statement-attached / state-coupled: the expression exists
        // only at its originating statement's site (CallResult etc.)
        // or its value depends on a mutable cursor (ray-query state,
        // cooperative-matrix lane state) - both unsafe to relocate.
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

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // The vector-type cache is a pure function of `module.types`,
        // which no fold step mutates.  Built once and threaded through
        // every callee so we don't re-scan the type arena for each of
        // the E + F + 1 (entry-points + functions + globals) folding
        // sites.
        let vector_type_cache = build_vector_type_cache(&module.types);

        changed += fold_global_expressions(module, &vector_type_cache);
        let const_literals = build_constant_literal_cache(module);

        for (_, function) in module.functions.iter_mut() {
            // Snapshot the pre-fold reference graph; the identity gate
            // checks the original count, not whatever it would be
            // mid-loop after partial rewrites.
            let refcounts = count_handle_refs(function);
            let emit_ranges = build_emit_range_map(&function.body);
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
            let emit_ranges = build_emit_range_map(&entry.function.body);
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

/// Fold `module.global_expressions`, converting composite
/// sub-expressions (vectors, swizzles, splats) into the most compact
/// form the generator can emit.  The literal cache is built once per
/// invocation, and the vector-type cache is supplied by the caller so
/// the inner materialisation pass stays `O(1)` per handle.
fn fold_global_expressions(
    module: &mut naga::Module,
    vector_type_cache: &HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> usize {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HashMap<_, _>>();

    let handles = module
        .global_expressions
        .iter()
        .map(|(h, _)| h)
        .collect::<Vec<_>>();
    let mut changed = 0usize;

    // O(N) one-time setup for O(1) materialize_vector lookups.
    let mut literal_cache = build_literal_cache(&module.global_expressions);

    // Shared cycle-tracker reused across every global handle (see the
    // identical optimisation in `fold_local_expressions`).  The memo
    // survives the in-loop rewrites because they only ever replace an
    // expression with a same-value Literal / Compose (see `ConstValueMemo`).
    let mut visiting = HashSet::new();
    let mut memo: ConstValueMemo = vec![None; module.global_expressions.len()];
    for handle in handles {
        visiting.clear();
        // Try composite-aware resolution first (covers vectors, swizzle, etc.).
        let value = {
            let ctx = GlobalConstFoldContext {
                arena: &module.global_expressions,
                types: &module.types,
                const_inits: &const_inits,
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        if let Some(ConstValue::Scalar(literal)) = value {
            if !matches!(module.global_expressions[handle], naga::Expression::Literal(existing) if existing == literal)
            {
                module.global_expressions[handle] = naga::Expression::Literal(literal);
                // Keep the literal cache in sync so later materialize_vector
                // calls can discover the new literal at `handle`.
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

/// Map every `Constant` whose initializer resolves to a `Literal` to
/// that literal.  Local-expression folding consults this cache
/// whenever it needs to treat a `Constant(handle)` operand as a
/// concrete value.
///
/// Abstract literals are filtered out: naga's validator rejects
/// `AbstractInt` / `AbstractFloat` in function-arena contexts, so a
/// cached abstract literal would silently roll the whole pass back
/// every sweep.  Skipping them here just leaves the constant
/// un-inlined - sound, no IR mutation.  Currently unreachable
/// because naga's frontend concretises `const X = 1` early; the
/// guard is defensive against future relaxation.
fn build_constant_literal_cache(
    module: &naga::Module,
) -> HashMap<naga::Handle<naga::Constant>, naga::Literal> {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HashMap<_, _>>();
    let context = GlobalLiteralContext {
        arena: &module.global_expressions,
        const_inits: &const_inits,
    };

    let mut out = HashMap::new();
    let mut visiting = HashSet::new();
    for (ch, c) in module.constants.iter() {
        visiting.clear();
        if let Some(lit) = resolve_literal(c.init, &context, &mut visiting) {
            if matches!(
                lit,
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
            ) {
                continue;
            }
            out.insert(ch, lit);
        }
    }
    out
}

// MARK: Per-function folding

/// Per-handle data-flow reference count: expression-as-child uses
/// plus statement-as-root uses (`If.condition`, `Store.value`, `Call`
/// arguments and result, etc.) plus `named_expressions` and
/// `local_variables[h].init`.  `Emit` ranges are NOT counted - they
/// only fix execution order, and an expression's Emit entry can be
/// dropped when the data-flow says no consumer survives.  The
/// identity gate uses `== 1` to detect impure operands whose Emit
/// entry can be removed after cloning.
///
/// `saturating_add` over `u32`: the gate's only consumer is `== 1`,
/// so saturation is indistinguishable from precision on counts that
/// would already imply a pathological IR.
///
/// # Safety invariant: statement-attached results count their producer
///
/// `CallResult`, `AtomicResult`, `WorkGroupUniformLoadResult`,
/// `RayQueryProceedResult`, `SubgroupBallotResult`, and
/// `SubgroupOperationResult` are uncloneable - their content is tied
/// to the producing statement.  [`is_pure_to_clone`] returns `false`,
/// so the fold gate falls back to this refcount check.  The walker
/// bumps `result` for every producing statement; any expression
/// consumer therefore pushes the count to `>= 2`, and the gate's
/// `== 1` escape never fires.  Removing the producer bumps would
/// silently corrupt IR; preserve them.
fn count_handle_refs(function: &naga::Function) -> Vec<u32> {
    let mut counts = vec![0u32; function.expressions.len()];

    fn bump(counts: &mut [u32], h: naga::Handle<naga::Expression>) {
        let i = h.index();
        if i < counts.len() {
            counts[i] = counts[i].saturating_add(1);
        }
    }

    // Expression-as-child uses.
    for (_, expr) in function.expressions.iter() {
        super::expr_util::visit_expression_children(expr, |child| bump(&mut counts, child));
    }

    // Statement-as-root uses; excluding Emit handles is load-bearing
    // for the unique-owner gate (see fn doc).
    super::expr_util::visit_block_expression_handles(
        &function.body,
        /*include_emit_handles=*/ false,
        &mut |h| bump(&mut counts, h),
    );

    // Named bindings (`let NAME = ...;`) and local-variable
    // initialisers consume their target handles outside the
    // statement walk above.  Currently naga restricts
    // `LocalVariable::init` to pure override-expressions so the
    // refcount escape never applies to them in practice, but
    // counting future-proofs against any relaxation that would let
    // an impure init's Emit entry be wrongly dropped by the gate.
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

/// Map every materialised expression handle to the id of the
/// `Statement::Emit` range that produces it.  Two handles share an id
/// IFF the *same* `Emit` statement materialises both - i.e. no
/// statement of any kind separates them in program order.
///
/// The identity / involution folds consult this to decide when an
/// impure operand (a `Load`) may be relocated to its consumer's `Emit`
/// slot.  Between two distinct `Emit` ranges there is always a non-`Emit`
/// statement, and every non-`Emit` statement is either a memory write,
/// a synchronisation barrier, or a control-flow edge (loop / branch /
/// terminator) - each of which makes moving a read across it unsound
/// (a post-write value, a re-execution, or a conditional execution).
/// So "operand and consumer share one `Emit` range" is exactly
/// "provably no intervening memory write": the precise store-aware
/// guard, expressed without re-walking control flow per fold.
///
/// Nested blocks continue the same id counter so handles in different
/// blocks never collide on an id; the map is read-only for the duration
/// of one fold pass (the body's `Emit` ranges are rebuilt only
/// afterwards by `rebuild_emit_ranges_after_removal`).
fn build_emit_range_map(body: &naga::Block) -> HashMap<naga::Handle<naga::Expression>, usize> {
    fn walk(
        block: &naga::Block,
        map: &mut HashMap<naga::Handle<naga::Expression>, usize>,
        next_id: &mut usize,
    ) {
        for stmt in block.iter() {
            if let naga::Statement::Emit(range) = stmt {
                let id = *next_id;
                *next_id += 1;
                for h in range.clone() {
                    map.insert(h, id);
                }
            }
            for nested in crate::passes::expr_util::nested_blocks(stmt) {
                walk(nested, map, next_id);
            }
        }
    }

    let mut map = HashMap::new();
    let mut next_id = 0usize;
    walk(body, &mut map, &mut next_id);
    map
}

/// Fold `arena` in place.  Returns two outputs: the set of handles
/// that must leave their original `Emit` ranges (folded-to-literal
/// expressions lose their emit requirement) and the number of
/// simplifications performed, so the caller can roll both into the
/// module-wide change counter.
///
/// `refcounts` (from [`count_handle_refs`]) and `emit_ranges` (from
/// [`build_emit_range_map`]) together gate cloning an impure operand (e.g. a
/// `Load`) in the identity / involution arms: clone only when the operand is
/// uniquely referenced by the folding expression AND shares its `Emit` range,
/// so the operand becomes dead (caller drops its `Emit` entry, no double
/// execution) and the relocated read never crosses an intervening statement.
/// The `select`-collapse arm clones pure operands only and consults neither.
fn fold_local_expressions(
    arena: &mut naga::Arena<naga::Expression>,
    refcounts: &[u32],
    emit_ranges: &HashMap<naga::Handle<naga::Expression>, usize>,
    const_literals: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    types: &naga::UniqueArena<naga::Type>,
    vector_type_cache: &HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> (HashSet<naga::Handle<naga::Expression>>, usize) {
    // Two handles are co-located (no statement, hence no memory write,
    // between them) when they belong to the same `Emit` range.  Absent
    // ids default to "not co-located" - sound: it only ever suppresses
    // an impure-operand relocation.
    let same_emit_range = |a: naga::Handle<naga::Expression>, b: naga::Handle<naga::Expression>| {
        emit_ranges
            .get(&a)
            .zip(emit_ranges.get(&b))
            .is_some_and(|(x, y)| x == y)
    };
    let mut handles = Vec::with_capacity(arena.len());
    handles.extend(arena.iter().map(|(h, _)| h));
    let mut folded = HashSet::new();

    // O(N) one-time setup for O(1) materialize_vector lookups.
    // The literal cache maps each scalar literal -> smallest handle carrying
    // it.  We keep it in sync as the scalar-fold branch writes new literals
    // into the arena; see `note_literal_in_cache`.
    let mut literal_cache = build_literal_cache(arena);

    // Reuse a single `visiting` HashSet across every handle.  The cycle
    // tracker only needs to be empty at the start of each
    // `resolve_const_value` call, not freshly allocated - `clear()`
    // keeps the bucket capacity around and avoids N allocations on
    // large function arenas.  The memo persists across handles AND the
    // in-loop rewrites (same-value Literal / Compose only; see
    // `ConstValueMemo`) - without it every handle re-descends its whole
    // operand chain, quadratic on the deep chains load_dedup builds.
    let mut visiting = HashSet::new();
    let mut memo: ConstValueMemo = vec![None; arena.len()];
    for handle in handles.iter().copied() {
        visiting.clear();
        let value = {
            let ctx = LocalConstFoldContext {
                arena: &*arena,
                types,
                const_literals,
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        match value {
            Some(ConstValue::Scalar(literal)) => {
                // Abstract literals trip `WidthError::Abstract` in
                // function-arena contexts; eval_binary / eval_math_scalar
                // preserve abstract types when both operands were
                // abstract, so we must refuse the write rather than
                // let the whole pass roll back.
                if matches!(
                    literal,
                    naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                ) {
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
                // Only replace with Compose if the original expression is
                // already emittable.  Replacing a non-emittable expression
                // (ZeroValue, Constant, etc.) with an emittable Compose would
                // produce invalid IR because the handle isn't in any Emit range.
                if needs_emit(&arena[handle])
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
                    // Compose is emittable - do NOT add to folded set.
                }
            }
            None => {}
        }
    }

    // Identity / absorbing / involution / select elimination.
    //
    // Identity:   `0 + x -> x`, `x * 1 -> x`, `x | 0 -> x`, etc.
    // Absorbing:  `x * 0 -> 0`, `x & 0 -> 0`, `x | all_ones -> all_ones`, etc.
    // Involution: `-(-x) -> x`, `!(!x) -> x`, `~(~x) -> x`.
    // Select:     `select(x, x, cond) -> x` when accept == reject.
    let mut simplify_count = 0usize;
    for handle in handles {
        match arena[handle] {
            naga::Expression::Binary { op, left, right } => {
                // Absorbing first - its result is always a literal.
                //
                // Gate: absorbing replaces the Binary with a clone of
                // the matched zero/all-ones operand, but the Binary's
                // result type follows naga broadcasting (`vec3<f32> *
                // 0.0` is a vec3).  Cloning a scalar onto a vector
                // slot corrupts the type.  Allow only when:
                //   * `LogicalAnd` / `LogicalOr` - WGSL pins these
                //     to `bool x bool -> bool`, so the clone's type
                //     matches by construction.
                //   * Both operands are literals - the result is a scalar,
                //     so cloning the absorbing operand is type-safe (a
                //     non-literal operand could be a vector and mis-type the
                //     result).  This admits the both-literal cases `eval_binary`
                //     above leaves unfolded - notably F16, which has no
                //     `eval_binary` arm - where `check_absorbing_operand` then
                //     declines anything sign-sensitive (only integer zero
                //     absorbs; `x * 0.0h` stays a correctly-signed bare product).
                let both_literal = matches!(arena[left], naga::Expression::Literal(_))
                    && matches!(arena[right], naga::Expression::Literal(_));
                let is_logical_op = matches!(
                    op,
                    naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
                );
                if (is_logical_op || both_literal)
                    && let Some(absorb) = check_absorbing_operand(op, left, right, arena)
                {
                    arena[handle] = arena[absorb].clone();
                    simplify_count += 1;
                    folded.insert(handle); // result is a Literal, declarative.
                    continue;
                }
                // Identity (`0 + x -> x`, `x * 1 -> x`).  Type-safe
                // by definition: the matched literal is the neutral
                // element, so `other` already carries the Binary's
                // broadcast result type.
                //
                // Cloning duplicates content into a new arena slot
                // whose own Emit re-executes at runtime.  For
                // `Load`/`ImageLoad`/`Derivative` etc. (`is_pure_to_clone`
                // = false) that would be a second memory read or a
                // re-evaluation at a different quad context.  Refcount
                // escape: when `other` has exactly one consumer (this
                // Binary), the rewrite leaves `other` dead and we drop
                // its Emit entry - so still one runtime evaluation.
                // Statement-attached results (CallResult, AtomicResult,
                // ...) are protected by `count_handle_refs`'s producer
                // bump pushing their count to `>= 2`; see that fn's doc.
                if let Some(other) = check_identity_operand(op, left, right, arena) {
                    let other_pure = is_pure_to_clone(&arena[other]);
                    // Relocating an impure operand (a `Load`) onto this
                    // Binary's `Emit` slot is sound only when the two share
                    // an `Emit` range: otherwise a statement separates them
                    // and the relocated read could cross a memory write,
                    // barrier, or control-flow edge (read-after-write
                    // reorder).  Refcount uniqueness alone does not protect
                    // the evaluation *position*; see `build_emit_range_map`.
                    let other_uniquely_owned = !other_pure
                        && refcounts.get(other.index()).copied() == Some(1)
                        && same_emit_range(other, handle);
                    if other_pure || other_uniquely_owned {
                        arena[handle] = arena[other].clone();
                        simplify_count += 1;
                        if !needs_emit(&arena[handle]) {
                            folded.insert(handle);
                        }
                        if other_uniquely_owned {
                            // `other` is now dead.  Drop it from its
                            // `Emit` range so the generator does not
                            // emit a second materialisation.
                            folded.insert(other);
                        }
                        continue;
                    }
                }
            }
            naga::Expression::Unary { op, expr } => {
                // Involution: op(op(x)) -> x.  Same refcount escape
                // hatch as identity, plus the intermediate Unary at
                // `expr` is also hoisted past; only mark it dead when
                // it is itself uniquely owned (otherwise an unrelated
                // consumer would observe a slot we removed from Emit).
                // Marking `inner` dead requires both that and inner's
                // own unique-ownership, since intermediate's residual
                // arena slot still references inner until compact runs.
                if let naga::Expression::Unary {
                    op: inner_op,
                    expr: inner,
                } = arena[expr]
                    && op == inner_op
                {
                    let inner_pure = is_pure_to_clone(&arena[inner]);
                    let intermediate_uniquely_owned =
                        refcounts.get(expr.index()).copied() == Some(1);
                    // Same store-aware guard as the identity arm: the
                    // relocated inner read must share an `Emit` range with
                    // this outer Unary (`handle`).  When `inner` and `handle`
                    // co-locate, the intermediate `expr` - topologically
                    // between them - necessarily does too, so gating on the
                    // (inner, handle) pair also covers it.
                    let inner_uniquely_owned = !inner_pure
                        && intermediate_uniquely_owned
                        && refcounts.get(inner.index()).copied() == Some(1)
                        && same_emit_range(inner, handle);
                    if inner_pure || inner_uniquely_owned {
                        arena[handle] = arena[inner].clone();
                        simplify_count += 1;
                        if !needs_emit(&arena[handle]) {
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

                // de Morgan on equality: `!(a == b)` -> `a != b` and
                // `!(a != b)` -> `a == b`, dropping the `!()` wrapper (3
                // bytes).  Equality negation is EXACT for every type -
                // scalars, vectors (component-wise `vecN<bool>`), floats
                // including NaN (`!(x==y)` <-> `x!=y` always) - so no
                // operand-type check is needed, unlike ordered relations.
                // Doing it here (not at emit time) lets the normal `Binary`
                // emit path compute parentheses correctly; a value-position
                // emit-time fold would mis-parenthesise a nested comparison.
                // Gate on the comparison being uniquely owned by this `!`
                // so its operands are reused in place with no duplication;
                // a shared comparison is left as `!name` by the emitter.
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
                    // The comparison node is now dead; drop its Emit so the
                    // generator does not materialise it a second time.
                    folded.insert(expr);
                    continue;
                }
            }
            // select(x, x, c) -> x.  Refcount escape does NOT apply:
            // accept == reject means the shared handle has refcount
            // >= 2 by construction, so the unique-owner check would
            // always block.  Pure-clone gate only.
            naga::Expression::Select { accept, reject, .. }
                if accept == reject && is_pure_to_clone(&arena[accept]) =>
            {
                arena[handle] = arena[accept].clone();
                simplify_count += 1;
                if !needs_emit(&arena[handle]) {
                    folded.insert(handle);
                }
                continue;
            }
            _ => {}
        }
    }

    (folded, simplify_count)
}

/// The negated form of an equality operator (`==` <-> `!=`), or `None`
/// for any other operator.  Only equality is included because its
/// negation is exact for every WGSL type including floats with NaN;
/// ordered relations (`<`, `>=`, ...) would need the operand type to
/// rule out the NaN-unsafe float case, which this pass does not resolve.
fn flip_equality(op: naga::BinaryOperator) -> Option<naga::BinaryOperator> {
    match op {
        naga::BinaryOperator::Equal => Some(naga::BinaryOperator::NotEqual),
        naga::BinaryOperator::NotEqual => Some(naga::BinaryOperator::Equal),
        _ => None,
    }
}

trait LiteralContext {
    fn arena(&self) -> &naga::Arena<naga::Expression>;
    fn resolve_constant(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<naga::Handle<naga::Expression>>,
    ) -> Option<naga::Literal>;
}

struct GlobalLiteralContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    const_inits: &'a HashMap<naga::Handle<naga::Constant>, naga::Handle<naga::Expression>>,
}

impl LiteralContext for GlobalLiteralContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }

    fn resolve_constant(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<naga::Handle<naga::Expression>>,
    ) -> Option<naga::Literal> {
        let init = *self.const_inits.get(&handle)?;
        resolve_literal(init, self, visiting)
    }
}

// MARK: Constant value resolution

// A fully-resolved constant value: either a scalar literal or a vector
// of scalar literals.  Matrices are intentionally excluded; they rarely
// appear in constant contexts and the folding effort would not justify
// the added case analysis.
#[derive(Debug, Clone, PartialEq)]
enum ConstValue {
    Scalar(naga::Literal),
    /// A constant vector whose components are all known scalar literals.
    Vector {
        components: Vec<naga::Literal>,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    },
}

impl ConstValue {
    /// Return `Some(literal)` if this is a scalar.
    fn as_scalar(&self) -> Option<naga::Literal> {
        match self {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }
}

/// Per-arena memo for [`resolve_const_value`], indexed by expression-handle
/// index.  Outer `None` = not yet computed; inner value = the full resolution
/// result (including "not a constant").  Without it the per-handle folding
/// loops re-descend every operand chain from scratch, which is O(N^2) on the
/// deep chains our own passes build and dominated corpus-wide pass time.
///
/// Safety of caching across the in-loop rewrites: the folding loops only ever
/// replace an expression with a `Literal` / `Compose` of the value it already
/// resolved to, so a stored entry never goes stale.  Cached results are
/// entry-point-independent even on cyclic (corrupt) IR: a cycle-guard hit
/// returns `None` BEFORE the memo store, and no resolver arm converts a child
/// failure into `Some`, so a wrong `Some` can never be cached - at worst a
/// cycle-dependent member caches the `None` it deterministically resolves to.
type ConstValueMemo = Vec<Option<Option<ConstValue>>>;

/// Composite-resolution context.  Extends [`LiteralContext`] with
/// access to the type arena because `Compose` / `Splat` rewrites need
/// to discover the component type and vector size.
trait ConstFoldContext {
    fn arena(&self) -> &naga::Arena<naga::Expression>;
    fn types(&self) -> &naga::UniqueArena<naga::Type>;
    /// `memo` is threaded through so an implementation that resolves the
    /// constant's init by descending the SAME arena (the global context)
    /// keeps memoization; implementations backed by a prebuilt map ignore it.
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<Handle<naga::Expression>>,
        memo: &mut ConstValueMemo,
    ) -> Option<ConstValue>;
}

struct GlobalConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    const_inits: &'a HashMap<naga::Handle<naga::Constant>, naga::Handle<naga::Expression>>,
}

impl ConstFoldContext for GlobalConstFoldContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }
    fn types(&self) -> &naga::UniqueArena<naga::Type> {
        self.types
    }
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<Handle<naga::Expression>>,
        memo: &mut ConstValueMemo,
    ) -> Option<ConstValue> {
        let init = *self.const_inits.get(&handle)?;
        resolve_const_value(init, self, visiting, memo)
    }
}

struct LocalConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    const_literals: &'a HashMap<naga::Handle<naga::Constant>, naga::Literal>,
}

impl ConstFoldContext for LocalConstFoldContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }
    fn types(&self) -> &naga::UniqueArena<naga::Type> {
        self.types
    }
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        _visiting: &mut HashSet<Handle<naga::Expression>>,
        _memo: &mut ConstValueMemo,
    ) -> Option<ConstValue> {
        self.const_literals
            .get(&handle)
            .copied()
            .map(ConstValue::Scalar)
    }
}

// MARK: Resolver entry points

/// Convert a constant **width-8** literal (`F64` / `U64` / `I64`) to `target`
/// (one of f32 / i32 / u32 / bool), matching naga's value-conversion
/// semantics, or `None` for any other source/target.
///
/// Used by [`resolve_const_value`]'s `As` arm to fold a narrowing cast of a
/// width-8 literal: naga's own frontend refuses to const-fold these (f64/u64/
/// i64 are non-standard WGSL), so the `As` node otherwise survives into our IR
/// and the emitter produces `f32(<F64 literal>)` / `u32(<U64 literal>)`, which
/// naga then *rejects* on re-parse.  Folding it both fixes the round-trip and
/// (for floats) shortens the output.
///
/// Semantics mirror `naga::proc::ConstantEvaluator::cast` exactly:
/// * `f64 -> f32` rounds to nearest, declined when it overflows to a
///   non-finite value (so the emitter never has to render `inf`).
/// * `f64 -> i32/u32` truncates toward zero and **clamps** to range (the
///   i32/u32 bounds are exactly representable in f64, so clamp-then-`as` is
///   exact) - naga clamps *float* sources.
/// * `u64/i64 -> i32/u32` **wraps** (low 32 bits / mod 2^32) - naga does NOT
///   clamp integer sources; `i64(-1) -> u32` is `4294967295u`, not `0`.
/// * `u64/i64 -> f32` rounds to nearest (always finite: `|v| <= u64::MAX` is
///   far below `f32::MAX`).
/// * `-> bool` is `v != 0`.
///
/// f16 / f64 / i64 / u64 targets return `None` (naga accepts those
/// `T(<width-8 literal>)` forms, so the cast is left as-is).
///
/// Mirrored by `generator::expr_emit::cast_width8_to_literal`, which handles
/// the *vector* form this scalar fold cannot; keep the two value-conversion
/// routines in sync (any divergence is caught by the round-trip tests).
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
    // 64-bit integer sources, carried through i128 so one arm covers both u64
    // and i64; integer narrowing WRAPS (`as`), it does not clamp.
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

/// Resolve `handle` to a fully-constant [`ConstValue`], threading
/// `visiting` to short-circuit cyclic expression graphs and `memo` to
/// resolve every handle at most once per arena walk (see
/// [`ConstValueMemo`]).  This is the composite-aware generalisation of
/// [`resolve_literal`]; it handles `Compose`, `Splat`, `ZeroValue`,
/// `Swizzle`, `AccessIndex`, and component-wise `Binary` / `Unary` on
/// vectors.
fn resolve_const_value<C: ConstFoldContext>(
    handle: Handle<naga::Expression>,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    if let Some(Some(cached)) = memo.get(handle.index()) {
        return cached.clone();
    }
    if !visiting.insert(handle) {
        return None;
    }
    let out = resolve_const_value_uncached(handle, ctx, visiting, memo);
    visiting.remove(&handle);
    // Store failures too: the corpus-dominant cost is re-walking long
    // RUNTIME (non-const) chains that fail resolution at every ancestor, so
    // a success-only memo would leave the quadratic blow-up in place.
    if let Some(slot) = memo.get_mut(handle.index()) {
        *slot = Some(out.clone());
    }
    out
}

/// Arm-by-arm resolver behind [`resolve_const_value`]'s cycle guard and
/// memo.  `?` failure exits are safe here only because the wrapper owns
/// the `visiting.remove` and memo store on every return path.
fn resolve_const_value_uncached<C: ConstFoldContext>(
    handle: Handle<naga::Expression>,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let expr = &ctx.arena()[handle];
    match expr {
        // Scalar leaves
        naga::Expression::Literal(lit) => Some(ConstValue::Scalar(*lit)),
        naga::Expression::Constant(ch) => ctx.resolve_constant_value(*ch, visiting, memo),

        // Composite constructors
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

        // Indexing / swizzle
        naga::Expression::AccessIndex { base, index } => {
            resolve_composite_element(*base, *index as usize, ctx, visiting, memo)
        }

        // A dynamic `Access` whose index folds to a constant is a static
        // pick in disguise.  Our own passes manufacture the shape: naga
        // materialises a dynamically-indexed function-scope `const` array
        // as a full `Compose` at the use site, and load_dedup then forwards
        // the index variable's stored literal.  Without this fold the
        // emitter ships the whole composite inline
        // (`array<u32,2310>(...)[0]`), which only the NEXT minification
        // round collapses via naga's front-end const-eval; folding here
        // makes round one match round two.  A pointer-typed `base` can
        // never fold: it is structurally a variable / pointer chain, not a
        // `Compose`/`ZeroValue`, and those never resolve to a value.
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

        // Scalar ops (delegate to existing evaluators)
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

        // Fold a converting cast (`T(x)`, `convert: Some`) of a constant
        // WIDTH-8 SCALAR (f64/u64/i64) to f32/i32/u32/bool.  naga's frontend
        // refuses to const-fold these (the source types are non-standard
        // WGSL), so the `As` node survives into our IR and the emitter
        // produces `f32(<F64 literal>)` / `u32(<U64 literal>)`, which naga
        // rejects on re-parse.  Folding to the converted literal fixes the
        // round-trip (and, for floats, shortens the output).  Left untouched:
        // bitcasts (`convert: None`), other targets (f16/f64/i64/u64 - naga
        // accepts those forms), non-width-8 operands (naga already const-folds
        // every other narrowing cast), and VECTOR operands - a converted
        // vector can't be materialized here (the converted component literals
        // don't exist in the arena yet), so it falls through and the
        // generator's vector path / run() fallback-revalidation guard handles it.
        naga::Expression::As {
            expr: operand,
            kind,
            convert,
        } => {
            let width = (*convert)?;
            let target = naga::Scalar { kind: *kind, width };
            match resolve_const_value(*operand, ctx, visiting, memo)? {
                ConstValue::Scalar(
                    lit @ (naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_)),
                ) => cast_width8_to(lit, target).map(ConstValue::Scalar),
                _ => None,
            }
        }

        _ => None,
    }
}

/// The non-negative compile-time value of an integer index literal, or
/// `None` for negative / non-integer literals (the fold then declines and
/// the expression ships as written).
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

/// Resolve element `idx` of the composite VALUE produced by `base`.
///
/// Array / matrix bases are read STRUCTURALLY (a syntactic `Compose` /
/// `ZeroValue` node, one component handle per element) so [`ConstValue`]
/// never has to model whole-array shapes; everything else falls through to
/// full base resolution, which covers vectors (whose `Compose` components
/// flatten and so cannot be picked positionally) and vector-valued chains
/// like `arr[1][2]` (the inner `Access` resolves to a `Vector` first).
/// Nested array-of-array picks decline (the middle layer has no
/// `ConstValue` form).
fn resolve_composite_element<C: ConstFoldContext>(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match &ctx.arena()[base] {
        naga::Expression::Compose { ty, components } => match &ctx.types()[*ty].inner {
            naga::TypeInner::Array { .. } | naga::TypeInner::Matrix { .. } => {
                resolve_const_value(*components.get(idx)?, ctx, visiting, memo)
            }
            _ => resolve_vector_component(base, idx, ctx, visiting, memo),
        },
        naga::Expression::ZeroValue(ty) => match &ctx.types()[*ty].inner {
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

/// Component `idx` of `base` once `base` fully resolves to a
/// [`ConstValue::Vector`]; the vector-base arm of
/// [`resolve_composite_element`].
fn resolve_vector_component<C: ConstFoldContext>(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match resolve_const_value(base, ctx, visiting, memo)? {
        ConstValue::Vector { components, .. } => {
            components.get(idx).copied().map(ConstValue::Scalar)
        }
        ConstValue::Scalar(_) => None,
    }
}

/// Materialise `ZeroValue(ty)` as a [`ConstValue`] when the type is a
/// scalar or zeroable vector.  Other types (matrices, structs, arrays)
/// return `None`; the pass does not fold them.
fn resolve_zero_value<C: ConstFoldContext>(ty: Handle<naga::Type>, ctx: &C) -> Option<ConstValue> {
    match ctx.types()[ty].inner {
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

/// Resolve `Compose { ty, components }` to a [`ConstValue::Vector`]
/// when every component folds to a scalar matching `ty`'s scalar
/// kind/width.  Mixed kinds, non-vector composites, and unresolved
/// operands short-circuit to `None`.
///
/// The per-component scalar match defends against a downstream
/// `materialize_vector` building, e.g., `Compose<vec4<f32>>` whose
/// component handles point at `Literal::I32` - naga's validator
/// rejects this on type-mismatch.  Naga's front-end concretises
/// before us in practice, so this fires only on hand-built or
/// fuzzed IR; cheap defense regardless.
fn resolve_compose<C: ConstFoldContext>(
    ty: Handle<naga::Type>,
    components: &[Handle<naga::Expression>],
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let inner = &ctx.types()[ty].inner;
    match inner {
        naga::TypeInner::Vector { size, scalar } => {
            let expected = *size as usize;
            let target_scalar = *scalar;
            // Two forms: N scalar components, or a mix of scalars/vectors
            // that flatten to N scalar components.
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

/// Apply a unary operator to a fully-resolved [`ConstValue`],
/// broadcasting over vector components.  Per-scalar evaluation is
/// delegated to [`eval_unary`].
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

/// `true` for binary operators whose result type is `bool` (or
/// `vec<bool>`) regardless of operand type.  The vector binary folder
/// uses this to pick the result scalar.
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

/// Apply a binary operator pair-wise over two [`ConstValue`]
/// operands.  Handles three shapes:
///
/// - scalar `<op>` scalar, delegating to [`eval_binary`];
/// - vector `<op>` vector, component-wise, same size; and
/// - scalar `<op>` vector / vector `<op>` scalar broadcast for
///   arithmetic ops.
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
        // scalar <op> vector -> broadcast scalar then component-wise
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
        // vector <op> scalar -> broadcast scalar then component-wise
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

/// Build a `Compose` expression for a folded vector by reusing
/// existing `Literal` handles already present in the arena.  Returns
/// `Some(new_expression)` when every component literal resolves to a
/// handle strictly less than `target` (so the resulting Compose is
/// topologically valid), and `None` otherwise.
///
/// The caller supplies prebuilt `O(1)` lookup caches:
/// - `literal_cache`: scalar `Literal` -> smallest handle carrying it
///   (smallest-handle policy maximises topological-safety hits).
/// - `vector_type_cache`: `(VectorSize, Scalar)` -> the matching
///   `TypeInner::Vector` handle.
fn materialize_vector(
    target: Handle<naga::Expression>,
    literals: &[naga::Literal],
    size: naga::VectorSize,
    scalar: naga::Scalar,
    literal_cache: &HashMap<LiteralKey, Handle<naga::Expression>>,
    vector_type_cache: &HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> Option<naga::Expression> {
    let ty = *vector_type_cache.get(&(size, scalar))?;

    // For each component literal, find a matching `Literal` expression at an
    // index strictly less than `target` so that the resulting Compose respects
    // topological ordering.
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

/// Canonical hash key for a `naga::Literal`.
///
/// Float variants use `to_bits` so that `NaN` payloads are preserved and
/// `-0.0` compares distinct from `+0.0`.  Using `f32`/`f64`/`f16` directly as
/// map keys is impossible anyway (they don't implement `Eq`/`Hash`).
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

/// Map each scalar `Literal` in `arena` to the SMALLEST handle
/// carrying it.  The "smallest" policy maximises topological-order
/// hits in [`materialize_vector`]; [`note_literal_in_cache`] preserves
/// the invariant as the fold pass writes new literals into the arena.
fn build_literal_cache(
    arena: &naga::Arena<naga::Expression>,
) -> HashMap<LiteralKey, Handle<naga::Expression>> {
    let mut cache: HashMap<LiteralKey, Handle<naga::Expression>> = HashMap::new();
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

/// Index every vector `Type` by its `(size, scalar)` shape so
/// [`materialize_vector`] can synthesise a `Compose` with the right
/// result type without scanning the type arena each call.
fn build_vector_type_cache(
    types: &naga::UniqueArena<naga::Type>,
) -> HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>> {
    let mut cache = HashMap::new();
    for (h, t) in types.iter() {
        if let naga::TypeInner::Vector { size, scalar } = t.inner {
            cache.entry((size, scalar)).or_insert(h);
        }
    }
    cache
}

/// Record `handle` as the canonical carrier for `literal`, preserving
/// the smallest-handle invariant.  Called whenever the scalar-fold
/// branch writes a new `Literal` into the arena so subsequent
/// [`materialize_vector`] calls can find it.
fn note_literal_in_cache(
    cache: &mut HashMap<LiteralKey, Handle<naga::Expression>>,
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

// MARK: Scalar evaluation

/// Resolve `handle` to a bare scalar literal, cycling through
/// `Constant`, `ZeroValue`, and simple unary/binary/cast nodes whose
/// operands themselves resolve.  Shallower than
/// [`resolve_const_value`]: bails out on any vector or composite.
fn resolve_literal<C: LiteralContext>(
    handle: naga::Handle<naga::Expression>,
    context: &C,
    visiting: &mut HashSet<naga::Handle<naga::Expression>>,
) -> Option<naga::Literal> {
    if !visiting.insert(handle) {
        return None;
    }

    let expr = &context.arena()[handle];
    let out = match expr {
        naga::Expression::Literal(lit) => Some(*lit),
        naga::Expression::Constant(ch) => context.resolve_constant(*ch, visiting),
        naga::Expression::Unary { op, expr } => {
            let rhs = resolve_literal(*expr, context, visiting)?;
            eval_unary(*op, rhs)
        }
        naga::Expression::Binary { op, left, right } => {
            let l = resolve_literal(*left, context, visiting)?;
            let r = resolve_literal(*right, context, visiting)?;
            eval_binary(*op, l, r)
        }
        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_literal(*arg, context, visiting)?;
            let b = match arg1 {
                Some(h) => Some(resolve_literal(*h, context, visiting)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_literal(*h, context, visiting)?),
                None => None,
            };
            eval_math_scalar(*fun, a, b, c)
        }
        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_literal(*condition, context, visiting)?;
            match cond {
                naga::Literal::Bool(true) => resolve_literal(*accept, context, visiting),
                naga::Literal::Bool(false) => resolve_literal(*reject, context, visiting),
                _ => None,
            }
        }
        _ => None,
    };

    visiting.remove(&handle);
    out
}

/// Per-scalar unary evaluator.  Returns `None` for operator/type
/// pairs that would change observable behaviour (e.g. negation of
/// `u32::MIN`); those cases fall through to runtime evaluation.
fn eval_unary(op: naga::UnaryOperator, rhs: naga::Literal) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::UnaryOperator as U;

    match (op, rhs) {
        // Float negation is exact for every non-NaN value (including
        // +/-Inf), but naga's validator rejects `Literal::F32`/`F64`
        // whose value is NaN or infinity (`LiteralError::NonFinite`
        // in `naga::valid::expression::check_literal_value`).  Gating
        // on `is_finite()` keeps the post-fold IR aligned with that
        // contract; a `!is_nan()` relaxation would re-introduce
        // +/-Inf-bearing literals on any upstream that injects them
        // (e.g. a future SPIR-V -> naga round-trip).
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

/// Per-scalar binary evaluator.  Scalar-type specific; NaN, overflow,
/// and divide-by-zero behaviours match the runtime's contract so
/// folding never changes observable output.
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
        // Float `%`: WGSL lowers `a % b` to `a - b*trunc(a/b)` evaluated in
        // OPERAND precision, so a round-to-nearest device diverges from Rust's
        // exact fmod whenever the rounded quotient crosses an integer the exact
        // one does not - by a FULL divisor, not a ULP (e.g. `33554432f % 3f` is
        // fmod 2.0 but 0.0 on every round-to-nearest GPU).  Fold only when the
        // exact result equals the operand-precision stepwise value, so the baked
        // constant is what the hardware computes; the `|a/b|` cap keeps a quotient
        // whose own trunc is unrepresentable off the comparison.  Declining leaves
        // the runtime `%`, which the device evaluates correctly.
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
        // Same stepwise-agreement guard as the f32 arm above (WGSL has no
        // runtime f64, so this covers only non-WGSL-frontend IR).
        (B::Modulo, L::F64(a), L::F64(b))
            if b != 0.0
                && (a % b).is_finite()
                && (a / b).abs() < 9_007_199_254_740_992.0
                && a % b == a - b * (a / b).trunc() =>
        {
            Some(L::F64(a % b))
        }

        // Signed add/sub/mul use checked_* so overflow is declined rather than
        // wrapped: the surviving expression still ships (both validators accept
        // e.g. `2147483647i+1i` in runtime position via the naga fallback), and
        // a const-context original never reaches the passes (naga's front-end
        // const-evaluates and rejects it at ingest).
        (B::Add, L::I32(a), L::I32(b)) => a.checked_add(b).map(L::I32),
        (B::Subtract, L::I32(a), L::I32(b)) => a.checked_sub(b).map(L::I32),
        (B::Multiply, L::I32(a), L::I32(b)) => a.checked_mul(b).map(L::I32),
        // Div/rem MUST fold their sole checked-failure case (MIN / -1): the
        // literal pair only arises from nagami's own transforms substituting
        // literals into runtime expressions (e.g. inlining `f(-2147483648)`
        // into `x / -1`), where WGSL DEFINES the results - e1 for `/`, 0 for
        // `%` (https://www.w3.org/TR/WGSL/#arithmetic-expr).  Declined, the
        // pair round-trips into naga's text const-eval which rejects it, and
        // the whole emission dies (no fallback can express it either).
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
        // The shift amount is ALWAYS `u32`: naga's WGSL front-end concretises
        // the right operand of `<<`/`>>` to `u32` regardless of the left
        // operand's width, so the right literal is `U32`, never `U64`.  (A
        // `U64`-right pattern here would be dead - it can never match.)
        (B::ShiftLeft, L::U64(a), L::U32(b)) if b < 64 => Some(L::U64(a.wrapping_shl(b))),
        (B::ShiftRight, L::U64(a), L::U32(b)) if b < 64 => Some(L::U64(a.wrapping_shr(b))),

        (B::And, L::I32(a), L::I32(b)) => Some(L::I32(a & b)),
        (B::ExclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a ^ b)),
        (B::InclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a | b)),
        // Signed `e1 << e2` discarding sign-changing bits is a shader-creation
        // error only in CONST contexts, which naga's front-end already
        // rejected at ingest; a literal pair here sits in a runtime
        // expression manufactured by nagami's own transforms, where WGSL
        // defines the shift as the plain bit-pattern result
        // (https://www.w3.org/TR/WGSL/#bit-expr).  Fold that value: declined,
        // the pair fails naga's text const-eval on re-parse and the whole
        // emission dies.
        (B::ShiftLeft, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shl(b))),
        (B::ShiftRight, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shr(b))),

        (B::And, L::I64(a), L::I64(b)) => Some(L::I64(a & b)),
        (B::ExclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a ^ b)),
        (B::InclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a | b)),
        // Shift amount is `u32` (see the U64 note above); bit-pattern fold
        // mirrors the i32 path's runtime-context rationale.
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

/// Fold a WGSL math built-in over scalar literal arguments.  Covers
/// Tier 1 (comparison, decomposition, computational), Tier 2
/// (trigonometric, exponential), and Tier 3 (integer bit) functions.
/// Tier 1/3 are bit-exact; Tier 2 has WGSL-defined accuracy (an error
/// envelope, not correct rounding), so those folds substitute a
/// conformant value within that envelope, not a bit-identical one.
/// Returns `None` for unsupported functions, type mismatches,
/// NaN-sensitive branches, and domain errors so the call survives to runtime.
fn eval_math_scalar(
    fun: naga::MathFunction,
    arg: naga::Literal,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::MathFunction as M;

    /// Finite-check wrapper: returns `Some(v)` only if `v` is finite.
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
        // Tier 1: comparison
        M::Abs => match arg {
            // `abs(NaN)=NaN`, `abs(Inf)=Inf` - both rejected by
            // naga's literal validator, so we route through
            // `finite_*` and refuse rather than write invalid IR.
            // Integer arms use `checked_abs` to reject `INT::MIN`.
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
            // WGSL propagates NaN through min/max; Rust's `f32::min`
            // returns the non-NaN operand instead.  Refuse fold when
            // either side is NaN to keep compile-time agreement with
            // runtime.
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
            // Same NaN-propagation rationale as Min.
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
                // Same NaN-propagation rationale as Min/Max - any NaN
                // in v / lo / hi must not be folded, since Rust's
                // `f32::clamp` panics on NaN bounds and the runtime
                // semantics propagate NaN regardless.
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
            // Rust's `clamp(NaN, 0.0, 1.0)` returns NaN (NaN
            // comparisons miss both branches).  `finite_*` refuses
            // NaN inputs; Inf clamps to 0.0/1.0 finitely.
            L::F32(v) => finite_f32(v.clamp(0.0, 1.0)),
            L::F64(v) => finite_f64(v.clamp(0.0, 1.0)),
            L::AbstractFloat(v) => finite_af(v.clamp(0.0, 1.0)),
            _ => None,
        },

        // Tier 1: computational
        M::Sign => match arg {
            // `signum(NaN) = NaN`; reject explicitly so the result
            // doesn't escape as `Literal::F32(NaN)` (invalid IR) or
            // re-fold through a NaN-sensitive Binary.
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

        // Tier 1: decomposition.  Rust's `floor`/`ceil`/`round_ties_even`/
        // `trunc` propagate NaN and pass +/-Inf through verbatim; both
        // tripped by naga's literal validator, so route through
        // `finite_*` and let the runtime evaluate the unfolded form
        // under IEEE rules.
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
            // WGSL specifies ties-to-even; Rust 1.77+ has round_ties_even.
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
            // WGSL `fract(e) = e - floor(e)`, NOT Rust's `fract`.
            // At large |v| the subtraction can lose precision or
            // overflow to non-finite; `finite_*` refuses those.
            L::F32(v) => finite_f32(v - v.floor()),
            L::F64(v) => finite_f64(v - v.floor()),
            L::AbstractFloat(v) => finite_af(v - v.floor()),
            _ => None,
        },

        // Tier 1: computational (continued)
        M::Step => match (arg, arg1?) {
            // WGSL propagates NaN through `step`; `edge <= x` returns
            // false on NaN and would fold to a wrong 0.0.  Refuse NaN
            // inputs explicitly (the 0.0/1.0 output domain is fine).
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

        // Tier 2: trigonometric
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
            // WGSL leaves `atan2(0, 0)` implementation-defined; Rust
            // returns `0.0`, but a GPU may legitimately return any of
            // {0, +/-pi/2, pi}.  Refuse fold when both arguments are
            // zero so runtime semantics are preserved.
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

        // Tier 2: exponential
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
            // WGSL precondition: e1 >= 0.0.  Additionally,
            // `pow(0, b)` for `b <= 0` is implementation-defined
            // (Rust's `f32::powf(0.0, 0.0)` returns 1.0; GPUs may
            // return NaN or 0).  Tighten the guard so 0^0 and
            // 0^(negative) fall through to runtime semantics.
            (L::F32(a), L::F32(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => finite_f32(a.powf(b)),
            (L::F64(a), L::F64(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => finite_f64(a.powf(b)),
            (L::AbstractFloat(a), L::AbstractFloat(b)) if a > 0.0 || (a == 0.0 && b > 0.0) => {
                finite_af(a.powf(b))
            }
            _ => None,
        },

        // Tier 3: integer bit operations
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
                // For negative numbers, find the most significant bit that
                // differs from the sign bit.
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

        // Tier 4+ / unsupported: graceful fallthrough
        _ => None,
    }
}

/// Fold a math built-in over a [`ConstValue`], broadcasting scalar
/// implementations across vector arguments.  Returns `None` when any
/// argument is non-constant, vector sizes mismatch, or the math
/// function has no per-scalar implementation registered.  Mirrors
/// [`eval_const_unary`] / [`eval_const_binary`].
fn eval_const_math(
    fun: naga::MathFunction,
    arg: ConstValue,
    arg1: Option<ConstValue>,
    arg2: Option<ConstValue>,
) -> Option<ConstValue> {
    // Helper: extract the scalar from a ConstValue, or None for vectors.
    fn as_scalar(v: &ConstValue) -> Option<naga::Literal> {
        match v {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }

    // Helper: extract vector components + metadata, or None for scalars.
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

    // All-scalar case
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

    // Vector case: apply per-component
    if let Some((comps, size, scalar)) = as_vector(&arg) {
        let n = comps.len();

        // Resolve optional arg1 components (must be same-size vector or absent).
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
        // Resolve optional arg2 components.
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

/// `true` when `expr` must live inside an `Emit` range.  Inline
/// mirror of [`crate::passes::expr_util::expression_needs_emit`]
/// (the identity loop calls it once per arena entry per sweep, so
/// a cross-module call would compound).
///
/// Exhaustive match - a new naga variant must trip the build at
/// BOTH copies so the maintainer classifies it deliberately rather
/// than letting `!matches!` silently default it to `needs_emit = true`
/// (validator rejection) or `false` (silently wedged in an Emit
/// range).  Update both lists in one commit on naga upgrade.
fn needs_emit(expr: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expr {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => false,
        E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Compose { .. }
        | E::Load { .. }
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Derivative { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. }
        | E::ArrayLength(_)
        | E::RayQueryGetIntersection { .. }
        | E::RayQueryVertexPositions { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => true,
    }
}

/// `true` when `h` is a literal zero of any scalar type.
///
/// Matches both `+0.0` and `-0.0` for floats (via bit-mask on `F16`
/// and `v == 0.0` for wider floats which the IEEE spec defines as
/// equating both signed zeroes).  Used for absorbing rules
/// (`x * 0 -> 0`, `x & 0 -> 0`) where signed zero is irrelevant.
/// For ADDITIVE identity rules where signed zero matters,
/// [`is_additive_identity_zero`] is the right gate.
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

/// `true` when `h` is a literal zero whose removal as an additive
/// identity (`x + Z -> x` or `x - Z -> x`) cannot mis-sign `x` under
/// IEEE 754 round-to-nearest.
///
/// The relevant case is `x = -0.0`:
/// * `(-0.0) + (+0.0) = +0.0` (NOT -0.0)  -> rejecting `+0.0` here
/// * `(-0.0) + (-0.0) = -0.0` (preserves x) -> `-0.0` is safe
/// * `(-0.0) - (+0.0) = -0.0` (preserves x) -> `+0.0` is safe (on the
///   right operand of subtraction the sign-flip neutralises)
/// * `(-0.0) - (-0.0) = +0.0` (NOT -0.0)  -> rejecting `-0.0` here
///
/// To keep the gate symmetric across Add/Subtract and avoid a
/// per-operator distinction, this helper accepts ONLY integer zeros.
/// The cost is a small set of unfolded `float + 0.0` identities; the
/// benefit is a single uniformly-safe gate that cannot mis-sign zero
/// regardless of operator.  Integer arithmetic has no signed-zero,
/// so integer-zero identities are always safe.
fn is_additive_identity_zero(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    is_integer_zero(arena, h)
}

/// `true` when `h` is a literal INTEGER zero.  Integers carry neither a
/// signed zero nor NaN/Inf, so an integer zero can be folded or substituted
/// without preserving an IEEE result sign - unlike a float zero.
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
    // `0x3C00` is the IEEE 754 binary16 bit pattern for `+1.0`.
    // (Negative one would be `0xBC00` and is irrelevant for `is_one`.)
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

fn is_bool_true(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

fn is_bool_false(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Detect `x <op> identity` patterns and return the surviving operand
/// handle.  Only fires for operators whose identity element is
/// well-defined across the scalar type set the emitter supports:
///
/// ```text
/// x + 0 = x          x - 0 = x          x * 1 = x          x / 1 = x
/// x | 0 = x          x ^ 0 = x          x & all_ones = x
/// x && true = x      x || false = x
/// ```
///
/// (and the symmetric left-operand forms where the op is commutative).
fn check_identity_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Add => {
            // Gate via `is_additive_identity_zero` (integer-only) so
            // we do not silently mis-sign `x = -0.0`: see the helper's
            // doc-comment for the IEEE-754 case analysis.
            if is_additive_identity_zero(arena, left) {
                Some(right)
            } else if is_additive_identity_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Subtract => {
            // Integer-only by design.  `is_additive_identity_zero`
            // matches integer zeros only, so we never reach this arm
            // with a float zero; the gate is uniform with the Add arm
            // above for symmetry and future-proofing.  The float
            // hazard that motivates the integer-only policy is
            // specifically `x - (-0.0)` when `x = -0.0`: IEEE-754
            // gives `+0.0` (mis-sign), so a folded `x - 0` rewrite
            // would discard the sign of the zero on the LHS.  Integer
            // subtraction has no signed-zero, so integer-zero
            // identities are always safe.  Subtract is one-sided
            // (`x - 0 = x` but `0 - x = -x`), so only the
            // right-operand check fires.
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

/// Detect `x <op> absorbing` patterns and return the operand handle
/// the expression collapses to:
///
/// ```text
/// x * 0 = 0          x & 0 = 0          x | all_ones = all_ones
/// x && false = false x || true = true
/// ```
///
/// (and the symmetric left-operand forms).  Mirrors
/// [`check_identity_operand`] but for annihilator elements.
fn check_absorbing_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Multiply => {
            // Only INTEGER zeros absorb here.  A float `x * 0.0` must carry
            // the IEEE sign of the product (and is NaN when `x` is
            // non-finite), so cloning the matched zero verbatim is correct
            // only when the sign-aware result was already computed by
            // `eval_binary` - which folds F32/F64/AbstractFloat both-literal
            // products before this arm runs.  F16 has NO `eval_binary` arm,
            // so a both-literal F16 `x * 0.0h` reaches here unfolded; cloning
            // the zero would take its sign, not the product's (e.g.
            // `-2.0h * 0.0h` -> `+0.0h`, but the true value is `-0.0h`).
            // Declining floats keeps that case as a bare product, which naga
            // re-parses to the correctly-signed zero.
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
