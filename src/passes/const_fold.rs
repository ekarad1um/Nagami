//! Constant folding.
//!
//! Rewrites expressions whose operands are statically known values
//! into `Literal`, `ZeroValue`, or `Compose` nodes that the generator
//! can emit directly.  The pass operates in three arenas:
//!
//! * `module.global_expressions` for `const` initializers and global
//!   splats, seeded with every other constant via
//!   `build_constant_literal_cache`.
//! * each function's `expressions` arena, folded with a per-function
//!   pass that tracks which handles must drop out of their `Emit`
//!   range (folded literals are declarative, not emittable).
//! * the `Emit` ranges themselves, rebuilt in place via
//!   `rebuild_emit_ranges_after_removal` after each function fold.
//!
//! Structural folding covers scalar arithmetic, binary and unary
//! operators, casts, relational built-ins, and a subset of math
//! built-ins; composite folding handles vector construction, swizzle
//! collapse, and splat elision.  NaN- and overflow-sensitive math
//! built-ins are intentionally not folded because the emitted WGSL
//! must match the runtime's behaviour bit-for-bit.

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
    // identical optimisation in `fold_local_expressions`).
    let mut visiting = HashSet::new();
    for handle in handles {
        visiting.clear();
        // Try composite-aware resolution first (covers vectors, swizzle, etc.).
        let value = {
            let ctx = GlobalConstFoldContext {
                arena: &module.global_expressions,
                types: &module.types,
                const_inits: &const_inits,
            };
            resolve_const_value(handle, &ctx, &mut visiting)
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
        use naga::Statement as S;
        for stmt in block.iter() {
            match stmt {
                S::Emit(range) => {
                    let id = *next_id;
                    *next_id += 1;
                    for h in range.clone() {
                        map.insert(h, id);
                    }
                }
                S::Block(inner) => walk(inner, map, next_id),
                S::If { accept, reject, .. } => {
                    walk(accept, map, next_id);
                    walk(reject, map, next_id);
                }
                S::Switch { cases, .. } => {
                    for case in cases {
                        walk(&case.body, map, next_id);
                    }
                }
                S::Loop {
                    body, continuing, ..
                } => {
                    walk(body, map, next_id);
                    walk(continuing, map, next_id);
                }
                _ => {}
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
    // large function arenas.
    let mut visiting = HashSet::new();
    for handle in handles.iter().copied() {
        visiting.clear();
        let value = {
            let ctx = LocalConstFoldContext {
                arena: &*arena,
                types,
                const_literals,
            };
            resolve_const_value(handle, &ctx, &mut visiting)
        };

        match value {
            Some(ConstValue::Scalar(literal)) => {
                // Abstract literals trip `WidthError::Abstract` in
                // function-arena contexts; eval_binary / eval_math1
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
                //   * Both operands are literals - then
                //     `resolve_const_value` already handled it above
                //     with full IEEE / wrapping semantics, so this
                //     arm is a redundant safety net; a non-literal
                //     other operand could be a vector and would
                //     mis-type the result.
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

/// Composite-resolution context.  Extends [`LiteralContext`] with
/// access to the type arena because `Compose` / `Splat` rewrites need
/// to discover the component type and vector size.
trait ConstFoldContext {
    fn arena(&self) -> &naga::Arena<naga::Expression>;
    fn types(&self) -> &naga::UniqueArena<naga::Type>;
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<Handle<naga::Expression>>,
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
    ) -> Option<ConstValue> {
        let init = *self.const_inits.get(&handle)?;
        resolve_const_value(init, self, visiting)
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
/// `visiting` to short-circuit cyclic expression graphs.  This is the
/// composite-aware generalisation of [`resolve_literal`]; it handles
/// `Compose`, `Splat`, `ZeroValue`, `Swizzle`, `AccessIndex`, and
/// component-wise `Binary` / `Unary` on vectors.
fn resolve_const_value<C: ConstFoldContext>(
    handle: Handle<naga::Expression>,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
) -> Option<ConstValue> {
    if !visiting.insert(handle) {
        return None;
    }

    let expr = &ctx.arena()[handle];
    let out = match expr {
        // Scalar leaves
        naga::Expression::Literal(lit) => Some(ConstValue::Scalar(*lit)),
        naga::Expression::Constant(ch) => ctx.resolve_constant_value(*ch, visiting),

        // Composite constructors
        naga::Expression::ZeroValue(ty) => resolve_zero_value(*ty, ctx),

        naga::Expression::Splat { size, value } => {
            let inner = resolve_const_value(*value, ctx, visiting)?;
            let lit = inner.as_scalar()?;
            Some(ConstValue::Vector {
                scalar: lit.scalar(),
                size: *size,
                components: vec![lit; *size as usize],
            })
        }

        naga::Expression::Compose { ty, components } => {
            resolve_compose(*ty, components, ctx, visiting)
        }

        // Indexing / swizzle
        naga::Expression::AccessIndex { base, index } => {
            let base_val = resolve_const_value(*base, ctx, visiting)?;
            match base_val {
                ConstValue::Vector { ref components, .. } => components
                    .get(*index as usize)
                    .copied()
                    .map(ConstValue::Scalar),
                _ => None,
            }
        }

        naga::Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let vec_val = resolve_const_value(*vector, ctx, visiting)?;
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
            let inner = resolve_const_value(*expr, ctx, visiting)?;
            eval_const_unary(*op, inner)
        }

        naga::Expression::Binary { op, left, right } => {
            let l = resolve_const_value(*left, ctx, visiting)?;
            let r = resolve_const_value(*right, ctx, visiting)?;
            eval_const_binary(*op, l, r)
        }

        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_const_value(*arg, ctx, visiting)?;
            let b = match arg1 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting)?),
                None => None,
            };
            eval_const_math(*fun, a, b, c)
        }

        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_const_value(*condition, ctx, visiting)?;
            match cond {
                ConstValue::Scalar(naga::Literal::Bool(true)) => {
                    resolve_const_value(*accept, ctx, visiting)
                }
                ConstValue::Scalar(naga::Literal::Bool(false)) => {
                    resolve_const_value(*reject, ctx, visiting)
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
            match resolve_const_value(*operand, ctx, visiting)? {
                ConstValue::Scalar(
                    lit @ (naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_)),
                ) => cast_width8_to(lit, target).map(ConstValue::Scalar),
                _ => None,
            }
        }

        _ => None,
    };

    visiting.remove(&handle);
    out
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
                let val = resolve_const_value(c, ctx, visiting)?;
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
        (B::Modulo, L::F32(a), L::F32(b)) if b != 0.0 && (a % b).is_finite() => Some(L::F32(a % b)),

        (B::Add, L::F64(a), L::F64(b)) if (a + b).is_finite() => Some(L::F64(a + b)),
        (B::Subtract, L::F64(a), L::F64(b)) if (a - b).is_finite() => Some(L::F64(a - b)),
        (B::Multiply, L::F64(a), L::F64(b)) if (a * b).is_finite() => Some(L::F64(a * b)),
        (B::Divide, L::F64(a), L::F64(b)) if b != 0.0 && (a / b).is_finite() => Some(L::F64(a / b)),
        (B::Modulo, L::F64(a), L::F64(b)) if b != 0.0 && (a % b).is_finite() => Some(L::F64(a % b)),

        // Signed integer arithmetic uses checked_* so that overflow (which is a
        // shader-creation/execution error in WGSL) is declined rather than
        // silently wrapped.  Folding `i32::MAX + 1` to `-2147483648` would
        // turn an error-producing program into a defined-value program.
        (B::Add, L::I32(a), L::I32(b)) => a.checked_add(b).map(L::I32),
        (B::Subtract, L::I32(a), L::I32(b)) => a.checked_sub(b).map(L::I32),
        (B::Multiply, L::I32(a), L::I32(b)) => a.checked_mul(b).map(L::I32),
        (B::Divide, L::I32(a), L::I32(b)) if b != 0 => a.checked_div(b).map(L::I32),
        (B::Modulo, L::I32(a), L::I32(b)) if b != 0 => a.checked_rem(b).map(L::I32),

        (B::Add, L::I64(a), L::I64(b)) => a.checked_add(b).map(L::I64),
        (B::Subtract, L::I64(a), L::I64(b)) => a.checked_sub(b).map(L::I64),
        (B::Multiply, L::I64(a), L::I64(b)) => a.checked_mul(b).map(L::I64),
        (B::Divide, L::I64(a), L::I64(b)) if b != 0 => a.checked_div(b).map(L::I64),
        (B::Modulo, L::I64(a), L::I64(b)) if b != 0 => a.checked_rem(b).map(L::I64),

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
        // WGSL (https://www.w3.org/TR/WGSL/#logical-expr):
        // `e1 << e2` with signed `e1` is a shader-creation error
        // when shifted-out bits differ from the resulting sign bit, i.e.
        // when the shift would not round-trip through i32.  Check via i64
        // and decline the fold on overflow so a malformed program stays a
        // compile-time error instead of being silently rewritten.
        (B::ShiftLeft, L::I32(a), L::U32(b)) if b < 32 => {
            let wide = (a as i64).wrapping_shl(b);
            let narrowed = wide as i32;
            (narrowed as i64 == wide).then_some(L::I32(narrowed))
        }
        (B::ShiftRight, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shr(b))),

        (B::And, L::I64(a), L::I64(b)) => Some(L::I64(a & b)),
        (B::ExclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a ^ b)),
        (B::InclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a | b)),
        // Shift amount is `u32` (see the U64 note above); decline the fold on
        // i64 overflow via the i128 round-trip, mirroring the i32 path.
        (B::ShiftLeft, L::I64(a), L::U32(b)) if b < 64 => {
            let wide = (a as i128).wrapping_shl(b);
            let narrowed = wide as i64;
            (narrowed as i128 == wide).then_some(L::I64(narrowed))
        }
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
/// (trigonometric, exponential), and Tier 3 (integer bit) functions
/// whose results are IEEE-stable across implementations.  Returns
/// `None` for unsupported functions, type mismatches, NaN-sensitive
/// branches, and domain errors so the call survives to runtime.
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
            // Permissive on floats by design: the caller's
            // `both_literal` gate routes `x * 0.0` through
            // `eval_binary` (IEEE sign-of-product); without that
            // gate, cloning the matched zero would mis-sign the
            // product.  The caller-side gate is load-bearing.
            if is_zero(arena, left) {
                Some(left)
            } else if is_zero(arena, right) {
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
mod tests {
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

        // --- f64 source: round-to-nearest (f32), CLAMP-then-truncate (int) ---
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

        // --- u64 source: WRAP (low 32 bits / mod 2^32), NOT clamp ---
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

        // --- i64 source: WRAP; `i64(-1) -> u32` is 4294967295, not 0 ---
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
        let module =
            naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
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
    fn i32_divide_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Divide,
            naga::Literal::I32(i32::MIN),
            naga::Literal::I32(-1),
        );
        assert_eq!(r, None, "i32::MIN / -1 overflows and should not be folded");
    }

    #[test]
    fn i32_modulo_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Modulo,
            naga::Literal::I32(i32::MIN),
            naga::Literal::I32(-1),
        );
        assert_eq!(r, None, "i32::MIN %% -1 overflows and should not be folded");
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
    fn i64_divide_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Divide,
            naga::Literal::I64(i64::MIN),
            naga::Literal::I64(-1),
        );
        assert_eq!(r, None, "i64::MIN / -1 overflows and should not be folded");
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
    fn shift_left_i64_overflow_not_folded() {
        // `1i64 << 63` flips the sign bit and does not round-trip through i64,
        // so it must NOT fold (stays a compile-time error in WGSL).
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I64(1),
            naga::Literal::U32(63),
        );
        assert_eq!(result, None, "i64 sign-bit overflow must not be folded");
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

    // Regression: WGSL (https://www.w3.org/TR/WGSL/#logical-expr):
    // declares `e1 << e2` for signed `e1` to be a shader-creation
    // error when shifted-out bits differ from the resulting sign
    // bit (i.e., the operation would overflow the signed range).
    // Folding `1_i32 << 31` to `i32::MIN` would turn an invalid program
    // into a defined-value one; the fold must decline instead.
    #[test]
    fn shift_left_i32_sign_bit_overflow_not_folded() {
        assert_eq!(
            eval_binary(
                naga::BinaryOperator::ShiftLeft,
                naga::Literal::I32(1),
                naga::Literal::U32(31),
            ),
            None,
            "1_i32 << 31 must not fold: shifted-out 0 differs from resulting sign bit 1"
        );
        assert_eq!(
            eval_binary(
                naga::BinaryOperator::ShiftLeft,
                naga::Literal::I32(2),
                naga::Literal::U32(30),
            ),
            None,
            "2_i32 << 30 must not fold for the same reason"
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
    fn shift_left_i64_sign_bit_overflow_not_folded() {
        // RHS must be `U32` to reach the `(ShiftLeft, I64, U32)` arm: a WGSL
        // shift amount is always `u32`, and that arm is where the i128
        // round-trip overflow check declines `1_i64 << 63`.  (A `U64` RHS would
        // match no arm and return `None` for the wrong reason - vacuously
        // passing this test.)
        assert_eq!(
            eval_binary(
                naga::BinaryOperator::ShiftLeft,
                naga::Literal::I64(1),
                naga::Literal::U32(63),
            ),
            None,
            "1_i64 << 63 must not fold (sign-bit overflow)"
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
        let result =
            check_identity_operand(naga::BinaryOperator::Add, a.zero_i32, a.param, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_add_zero_right_integer() {
        let a = make_identity_arena();
        let result =
            check_identity_operand(naga::BinaryOperator::Add, a.param, a.zero_i32, &a.arena);
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
        let result =
            check_identity_operand(naga::BinaryOperator::Divide, a.param, a.one_f32, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_div_one_left_not_eliminated() {
        let a = make_identity_arena();
        // 1 / param != param
        let result =
            check_identity_operand(naga::BinaryOperator::Divide, a.one_f32, a.param, &a.arena);
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
        let module =
            naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
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
        // 0 * param -> 0
        assert_eq!(
            check_absorbing_operand(
                naga::BinaryOperator::Multiply,
                a.zero_f32,
                a.param,
                &a.arena
            ),
            Some(a.zero_f32)
        );
    }

    #[test]
    fn absorbing_mul_zero_right() {
        let a = make_identity_arena();
        // param * 0 -> 0
        assert_eq!(
            check_absorbing_operand(
                naga::BinaryOperator::Multiply,
                a.param,
                a.zero_f32,
                &a.arena
            ),
            Some(a.zero_f32)
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
            let result =
                eval_math_scalar(naga::MathFunction::Fract, naga::Literal::F32(v), None, None);
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        // Output must parse back cleanly.
        crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
    }

    #[test]
    fn e2e_vec_and_zero_stays_valid() {
        let source = r#"
@fragment
fn main(@location(0) v: vec4<u32>) -> @location(0) vec4<u32> {
    return v & vec4<u32>(0u);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
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
}
