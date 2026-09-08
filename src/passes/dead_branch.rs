//! Dead-branch elimination.  Four phases run per function on every sweep,
//! in this order:
//!
//! 1. Short-circuit re-sugaring folds the `if/else store` shapes naga's
//!    WGSL frontend emits for `&&` / `||` back into `Binary` `LogicalAnd` /
//!    `LogicalOr` expressions.  It must run first: the later phases destroy
//!    the untransformed frontend shape it matches.
//! 2. Register promotion (`forward_single_store_locals`) forwards a
//!    same-block single-store / single-load local to its value, which makes
//!    the re-sugar + const-fold collapse idempotent.
//! 3. Redundant else-store elimination drops stores of the literal the
//!    variable already holds on that branch.
//! 4. Constant-condition / structural cleanup (`eliminate_dead_branches`)
//!    strips `if (true)` / `if (false)` arms and, in the same walk, elides
//!    empty If / Block / Switch, all-arms-noop Switch (selector included:
//!    naga expressions carry no effects), `else` after an unconditionally
//!    terminating accept arm (`discard` never counts: demote-to-helper
//!    continues past it), code after a terminator, `loop { .. break if
//!    true; }` when no bare break / continue would mis-target, `break if
//!    false` when the body proves another exit, and single-default-case
//!    Switch splicing.
//!
//! Every fold that could delete - or, by splicing a never-falls-through
//! arm, make unreachable - what tint credits as a loop's only exit is
//! guarded (`contains_return`, `splice_loses_tint_loop_exit`).
//!
//! Branch flipping (`if c {} else { x; }` -> `if !c { x; }`) is an
//! emit-time rewrite in the generator, not a pass here.

use rustc_hash::FxHashMap;

use super::expr_util::{
    has_negative_zero_leaf, is_bool_false, is_bool_true, literal_bit_eq, nested_blocks,
    nested_blocks_mut, root_local_var,
};
use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Match `[Emit..] Store` - a re-sugar accept arm that computes an
/// intermediate (`{ let _e = a < b; d = _e; }`) before storing it - and
/// return the store's `(pointer, value)`.
///
/// The leading `Emit`s get hoisted into the parent block, so after the fold
/// they evaluate UNCONDITIONALLY.  Sound because `Emit` expressions are
/// side-effect-free (every effectful or control-flow construct is a distinct
/// `Statement` kind, rejected here), WGSL bounds-checks out-of-range
/// indexing rather than trapping, the `&&` / `||` discards the hoisted value
/// whenever the guard fails, and lifting a computation OUT of a conditional
/// can only reduce non-uniformity (it never pushes a derivative /
/// implicit-LOD sample into non-uniform control flow).
fn store_with_leading_emits(
    block: &naga::Block,
) -> Option<(
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
)> {
    let mut store = None;
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(_) => {
                if store.is_some() {
                    return None;
                }
            }
            naga::Statement::Store { pointer, value } => {
                if store.is_some() {
                    return None;
                }
                store = Some((*pointer, *value));
            }
            _ => return None,
        }
    }
    store
}

/// The re-sugar skips constant-`bool` guards: `const_fold` runs earlier, so
/// such an `if` is dead-branch-eliminable outright, and re-sugaring it to
/// `d = false && x` would block that elimination and can leave a larger
/// residue (e.g. a dead loop gated on a zero-init bool).
fn is_const_bool(
    cond: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    matches!(
        expressions[cond],
        naga::Expression::Literal(naga::Literal::Bool(_))
    )
}

/// Hoist a matched accept arm's `Emit`s into the parent block, dropping its
/// `Store` (its value is now the operator's right operand).
fn hoist_leading_emits(rebuilt: &mut naga::Block, accept: naga::Block) {
    for (stmt, span) in accept.span_into_iter() {
        if matches!(stmt, naga::Statement::Emit(_)) {
            rebuilt.push(stmt, span);
        }
    }
}

// MARK: Single-store local forwarding (register promotion)

/// Forward a local with exactly one whole-variable store and otherwise only
/// loads (`var t; t = E; use(t)`): every `Load(t)` is redirected to `E` and
/// the store dropped.  This collapses the per-join re-sugar chain
/// `t1 = a && b; t2 = t1 && c; if t2` to `if (a && b && c)` and makes the
/// re-sugar idempotent: re-parsing that output re-lowers to fresh single-use
/// temps that fold straight back to the same shape.
///
/// Soundness: the reference census is exhaustive (`total_refs == 1 + loads`,
/// built from the canonical handle visitors, so any missed reference kind
/// fails the equality and bails); `t` has no initialiser; and the store and
/// the materialising `Emit` of every load are top-level statements of the
/// SAME block, store first, so no load observes a pre-store value.  `E`'s
/// evaluation position is unchanged (its `Emit` stays put), and the
/// generator's `must_bind_loads` still guards load-versus-write hazards when
/// it later inlines `E`.
///
/// The dead `var t` and orphaned `Load(t)` are left for the downstream
/// dead-local / dead-expression cleanup later in the fixpoint; without it a
/// stray `var t;` survives (size, never correctness).
fn forward_single_store_locals(
    function: &mut naga::Function,
    types: &naga::UniqueArena<naga::Type>,
) -> bool {
    let nlocals = function.local_variables.len();
    if nlocals == 0 {
        return false;
    }

    let mut total_refs = vec![0u32; nlocals];
    let mut whole_stores = vec![0u32; nlocals];
    let mut store_value: Vec<Option<naga::Handle<naga::Expression>>> = vec![None; nlocals];
    let mut load_count = vec![0usize; nlocals];
    // Lowest-indexed EXPRESSION consuming each handle: forwarding a load to
    // a value appended behind one of its consumers would create a forward
    // reference naga rejects.  Only expression consumers constrain arena
    // order; statement consumers rely on the store-before-load-`Emit` gate.
    let mut min_consumer = vec![u32::MAX; function.expressions.len()];
    {
        let exprs = &function.expressions;
        let local_index = |h: naga::Handle<naga::Expression>| -> Option<usize> {
            match exprs[h] {
                naga::Expression::LocalVariable(l) => Some(l.index()),
                _ => None,
            }
        };
        for (hc, expr) in exprs.iter() {
            if let naga::Expression::Load { pointer } = expr
                && let Some(t) = local_index(*pointer)
            {
                load_count[t] += 1;
            }
            super::expr_util::visit_expression_children(expr, |child| {
                if let Some(t) = local_index(child) {
                    total_refs[t] += 1;
                }
                let slot = &mut min_consumer[child.index()];
                *slot = (*slot).min(hc.index() as u32);
            });
        }
        super::expr_util::visit_block_expression_handles(&function.body, false, &mut |h| {
            if let Some(t) = local_index(h) {
                total_refs[t] += 1;
            }
        });
        count_whole_stores(&function.body, exprs, &mut whole_stores, &mut store_value);
    }

    let mut candidate = vec![false; nlocals];
    for (lh, lvar) in function.local_variables.iter() {
        let t = lh.index();
        // One load only: forwarding then inlines `E` at a single site and
        // always shrinks; a multi-load forward could duplicate `E`.
        if whole_stores[t] == 1
            && load_count[t] == 1
            && total_refs[t] as usize == 2
            && lvar.init.is_none()
            && !store_value[t].is_some_and(|v| has_negative_zero_leaf(&function.expressions, v))
        {
            candidate[t] = true;
        }
    }
    if !candidate.iter().any(|&c| c) {
        return false;
    }

    let mut redirects: HandleMap<naga::Expression, naga::Handle<naga::Expression>> =
        Default::default();
    let mut remove_store = vec![false; nlocals];
    let census = ForwardCensus {
        candidate: &candidate,
        load_count: &load_count,
        store_value: &store_value,
        min_consumer: &min_consumer,
    };
    collect_forwards(
        &function.body,
        &function.expressions,
        &census,
        &mut redirects,
        &mut remove_store,
        &mut HandleSet::default(),
    );
    // Substituting a const store value for a runtime local read crosses a
    // float `-x` / `x * y` / `x / y` slot exactly as `load_dedup`'s own
    // forwarding does, so it answers to the same test.  A declined load still
    // reads the local, so its store has to stay.
    for handle in super::load_dedup::decline_sign_sensitive_forwards(
        &function.expressions,
        types,
        &mut redirects,
    ) {
        if let naga::Expression::Load { pointer } = function.expressions[handle]
            && let naga::Expression::LocalVariable(l) = function.expressions[pointer]
        {
            remove_store[l.index()] = false;
        }
    }
    if redirects.is_empty() {
        return false;
    }

    for (_, expr) in function.expressions.iter_mut() {
        let _ = super::expr_util::try_map_expression_handles_in_place(expr, &mut |h| {
            Some(*redirects.get(h).unwrap_or(&h))
        });
    }
    remap_block_handles(&mut function.body, &redirects);
    remove_forwarded_stores(&mut function.body, &function.expressions, &remove_store);
    true
}

fn count_whole_stores(
    block: &naga::Block,
    exprs: &naga::Arena<naga::Expression>,
    whole_stores: &mut [u32],
    store_value: &mut [Option<naga::Handle<naga::Expression>>],
) {
    super::expr_util::for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Store { pointer, value } = stmt
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
        {
            whole_stores[l.index()] += 1;
            store_value[l.index()] = Some(*value);
        }
    });
}

struct ForwardCensus<'a> {
    candidate: &'a [bool],
    load_count: &'a [usize],
    store_value: &'a [Option<naga::Handle<naga::Expression>>],
    min_consumer: &'a [u32],
}

/// `value` is a bare load of another candidate (`var t = c;`).  Forwarding
/// both in one sweep would chain `Load(t) -> Load(c) -> E_c`, which the
/// non-transitive apply cannot follow: `c`'s store is removed while `Load(t)`
/// still points at the orphaned `Load(c)`, which then reads zero-init.  Such
/// `t` are deferred to a later sweep.
fn is_candidate_copy(
    value: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    candidate: &[bool],
) -> bool {
    matches!(
        expressions[value],
        naga::Expression::Load { pointer }
            if matches!(
                expressions[pointer],
                naga::Expression::LocalVariable(c) if candidate[c.index()]
            )
    )
}

/// Per block, redirect a candidate's loads to its stored value when the store
/// and the materialising `Emit` of every load are top-level statements of
/// THIS block, store first.
fn collect_forwards(
    block: &naga::Block,
    exprs: &naga::Arena<naga::Expression>,
    census: &ForwardCensus<'_>,
    redirects: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    remove_store: &mut [bool],
    counted: &mut HandleSet<naga::Expression>,
) {
    let mut store_idx: FxHashMap<usize, usize> = Default::default();
    for (i, stmt) in block.iter().enumerate() {
        if let naga::Statement::Store { pointer, .. } = stmt
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
            && census.candidate[l.index()]
        {
            store_idx.insert(l.index(), i);
        }
    }
    // Key on the load's MATERIALISATION (its `Emit` index), never on a
    // consumer: `let snap = t; t = E; use(snap)` emits `Load(t)` BEFORE the
    // store yet consumes it after, and forwarding it to `E` would silently
    // replace the pre-store value.  Scanning only this block's own `Emit`s
    // also keeps forwarding same-block: a load inside a nested loop / if is
    // matched only against a store of that block, never an enclosing one
    // (folding a loop-invariant into a guard can expose an infinite loop tint
    // rejects).
    // One list per local rather than a set: an expression is emitted once, so
    // `counted` (shared with the nested walks, since no handle can be emitted
    // in two blocks) rejects a repeat and the count below stays exact.
    let mut found: FxHashMap<usize, Vec<naga::Handle<naga::Expression>>> = Default::default();
    for (i, stmt) in block.iter().enumerate() {
        let naga::Statement::Emit(range) = stmt else {
            continue;
        };
        for h in range.clone() {
            if let naga::Expression::Load { pointer } = exprs[h]
                && let naga::Expression::LocalVariable(l) = exprs[pointer]
                && let Some(&si) = store_idx.get(&l.index())
                && i > si
                && counted.insert(h)
            {
                found.entry(l.index()).or_default().push(h);
            }
        }
    }
    for &t in store_idx.keys() {
        let Some(loads_here) = found.get(&t) else {
            continue;
        };
        if loads_here.len() == census.load_count[t]
            && census.load_count[t] > 0
            && let Some(e) = census.store_value[t]
            && loads_here
                .iter()
                .all(|&lh| (e.index() as u32) < census.min_consumer[lh.index()])
            && !is_candidate_copy(e, exprs, census.candidate)
        {
            for &lh in loads_here {
                redirects.insert(lh, e);
            }
            remove_store[t] = true;
        }
    }
    for stmt in block.iter() {
        for nested in nested_blocks(stmt) {
            collect_forwards(nested, exprs, census, redirects, remove_store, counted);
        }
    }
}

fn remap_block_handles(
    block: &mut naga::Block,
    redirects: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    for stmt in block.iter_mut() {
        super::expr_util::remap_statement_handles(stmt, &mut |h| *redirects.get(h).unwrap_or(&h));
        for nested in nested_blocks_mut(stmt) {
            remap_block_handles(nested, redirects);
        }
    }
}

fn remove_forwarded_stores(
    block: &mut naga::Block,
    exprs: &naga::Arena<naga::Expression>,
    remove_store: &[bool],
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    for (mut stmt, span) in original.span_into_iter() {
        for nested in nested_blocks_mut(&mut stmt) {
            remove_forwarded_stores(nested, exprs, remove_store);
        }
        if let naga::Statement::Store { pointer, .. } = &stmt
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
            && remove_store[l.index()]
        {
            continue;
        }
        rebuilt.push(stmt, span);
    }
    *block = rebuilt;
}

use super::load_dedup::{collect_modified_locals, is_zero_literal};
use super::scoped_map::ScopedMap;
use crate::handle_set::{HandleMap, HandleSet};

/// Dead-branch elimination: the module's four phases, per function, every
/// sweep.
#[derive(Debug, Default)]
pub struct DeadBranchPass;

impl Pass for DeadBranchPass {
    fn name(&self) -> &'static str {
        "dead_branch_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // Built once: the mutable function walk below cannot re-borrow
        // `module.constants`.
        let const_lits = build_const_literal_cache(module);
        let types = &module.types;

        for (_, function) in module.functions.iter_mut() {
            // Order is load-bearing: the re-sugar matches the frontend shape
            // the later phases destroy.
            changed += resugar_short_circuits(function);
            changed += usize::from(forward_single_store_locals(function, types));
            changed += eliminate_redundant_else_stores_in_function(function, &const_lits);
            changed += eliminate_dead_branches(
                &mut function.body,
                &function.expressions,
                &const_lits,
                /*in_loop=*/ false,
                /*break_binds_to_loop=*/ false,
            );
        }
        for entry in module.entry_points.iter_mut() {
            changed += resugar_short_circuits(&mut entry.function);
            changed += usize::from(forward_single_store_locals(&mut entry.function, types));
            changed +=
                eliminate_redundant_else_stores_in_function(&mut entry.function, &const_lits);
            changed += eliminate_dead_branches(
                &mut entry.function.body,
                &entry.function.expressions,
                &const_lits,
                /*in_loop=*/ false,
                /*break_binds_to_loop=*/ false,
            );
        }

        Ok(changed > 0)
    }
}

// MARK: Short-circuit re-sugaring

// naga's WGSL frontend lowers a short-circuit operator to a local plus an
// if/else that stores the intermediate result:
//
//   a && b  =>  var d: bool; if (a)  { d = b; } else { d = false; }
//   a || b  =>  var d: bool; if (!a) { d = b; } else { d = true; }
//
// This phase folds both shapes back into `Binary(LogicalAnd / LogicalOr)`.

/// Re-sugar gate: only locals that are read anywhere fold.  A never-read
/// join is dead code the else-store and dead-branch phases delete outright;
/// folding it first would only append a `Binary` for compaction to sweep.
fn compute_resugar_foldable(function: &naga::Function) -> Vec<bool> {
    let mut loaded = vec![false; function.local_variables.len()];
    for (_, expr) in function.expressions.iter() {
        if let naga::Expression::Load { pointer } = expr
            && let naga::Expression::LocalVariable(l) = function.expressions[*pointer]
        {
            loaded[l.index()] = true;
        }
    }
    loaded
}

/// Fold every lowered short-circuit join, then renumber the arena in
/// emission order.  Each fold appends its `Binary` at the arena END, behind
/// the consumers of the join's load that naga created earlier (`select(..,
/// d)`, `!d`, the next join's condition), and register promotion refuses to
/// forward a value past an earlier-indexed consumer.  The rebuild puts the
/// `Binary` where its `Emit` now sits - ahead of those consumers - so the
/// next phase can collapse `d = a && b; use(d)` into `use(a && b)`, and
/// drops the dead `LogicalNot` an `||` fold unwrapped, which would otherwise
/// pin the same guard.
fn resugar_short_circuits(function: &mut naga::Function) -> usize {
    let foldable = compute_resugar_foldable(function);
    let changed = desugar_short_circuit(&mut function.body, &mut function.expressions, &foldable);
    if changed > 0 {
        super::expr_util::rebuild_function_expressions(function);
    }
    changed
}

fn store_target_foldable(
    pointer: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    foldable: &[bool],
) -> bool {
    matches!(
        expressions[pointer],
        naga::Expression::LocalVariable(l) if foldable[l.index()]
    )
}

fn desugar_short_circuit(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    foldable: &[bool],
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;

    for (mut statement, span) in original.span_into_iter() {
        for nested in nested_blocks_mut(&mut statement) {
            changed += desugar_short_circuit(nested, expressions, foldable);
        }

        match statement {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // `&&`: `if cond { [emits..]; d = val; } else { d = false; }`
                // -> hoisted emits; `d = cond && val;`.  Both operands are in
                // scope in the parent block: `cond`'s Emit dominates the If,
                // and `val`'s producers are the hoisted accept-arm Emits,
                // placed ahead of the Binary.
                if !is_const_bool(condition, expressions)
                    && let Some((ptr_r, val_r)) = single_store_info(&reject)
                    && is_bool_false(expressions, val_r)
                    && let Some((ptr_a, val_a)) = store_with_leading_emits(&accept)
                    && same_local_pointer(ptr_a, ptr_r, expressions)
                    && store_target_foldable(ptr_a, expressions, foldable)
                {
                    let binary = expressions.append(
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::LogicalAnd,
                            left: condition,
                            right: val_a,
                        },
                        naga::Span::default(),
                    );
                    drop(reject);
                    hoist_leading_emits(&mut rebuilt, accept);
                    rebuilt.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(binary, binary)),
                        span,
                    );
                    rebuilt.push(
                        naga::Statement::Store {
                            pointer: ptr_a,
                            value: binary,
                        },
                        span,
                    );
                    changed += 1;
                    continue;
                }

                // `||`: `if !cond { [emits..]; d = val; } else { d = true; }`
                // -> hoisted emits; `d = cond || val;`.  Same scope argument;
                // `cond` (the `LogicalNot` operand) is in scope transitively.
                // The negation is recovered two ways: a literal `LogicalNot`
                // unwraps, and an `==` / `!=` un-flips into a FRESH
                // expression (const_fold's De Morgan turns the lowering's
                // `!(x == y)` into `x != y` before this phase, so
                // equality-left `||` only reaches here flipped).  Un-flipping
                // the equality pair is NaN-safe in both directions, unlike
                // the ordered comparisons.  The fresh handle references only
                // `condition`'s own operands (their Emits dominate the If) and
                // is covered by starting the Emit range at it.  Structural
                // guards run FIRST so a failed match appends nothing.
                if let Some((ptr_r, val_r)) = single_store_info(&reject)
                    && is_bool_true(expressions, val_r)
                    && let Some((ptr_a, val_a)) = store_with_leading_emits(&accept)
                    && same_local_pointer(ptr_a, ptr_r, expressions)
                    && store_target_foldable(ptr_a, expressions, foldable)
                    && let Some((inner_cond, synthesized)) =
                        match unwrap_logical_not(condition, expressions) {
                            Some(inner) if !is_const_bool(inner, expressions) => {
                                Some((inner, false))
                            }
                            Some(_) => None,
                            None => {
                                let unflipped = match &expressions[condition] {
                                    naga::Expression::Binary {
                                        op: naga::BinaryOperator::Equal,
                                        left,
                                        right,
                                    } => Some((naga::BinaryOperator::NotEqual, *left, *right)),
                                    naga::Expression::Binary {
                                        op: naga::BinaryOperator::NotEqual,
                                        left,
                                        right,
                                    } => Some((naga::BinaryOperator::Equal, *left, *right)),
                                    _ => None,
                                };
                                unflipped.map(|(op, left, right)| {
                                    (
                                        expressions.append(
                                            naga::Expression::Binary { op, left, right },
                                            naga::Span::default(),
                                        ),
                                        true,
                                    )
                                })
                            }
                        }
                {
                    let binary = expressions.append(
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::LogicalOr,
                            left: inner_cond,
                            right: val_a,
                        },
                        naga::Span::default(),
                    );
                    drop(reject);
                    hoist_leading_emits(&mut rebuilt, accept);
                    // A synthesized comparison sits directly before `binary`.
                    let emit_from = if synthesized { inner_cond } else { binary };
                    rebuilt.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(emit_from, binary)),
                        span,
                    );
                    rebuilt.push(
                        naga::Statement::Store {
                            pointer: ptr_a,
                            value: binary,
                        },
                        span,
                    );
                    changed += 1;
                    continue;
                }

                rebuilt.push(
                    naga::Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            other => {
                rebuilt.push(other, span);
            }
        }
    }

    *block = rebuilt;
    changed
}

/// Match a block that is exactly one `Store` and return its `(pointer,
/// value)`: the re-sugar's REJECT arm (`d = false` / `d = true`).  A
/// constant store is a `Literal` needing no `Emit`, so any `Emit` means the
/// arm is not the declarative short-circuit value and is rejected.
fn single_store_info(
    block: &naga::Block,
) -> Option<(
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
)> {
    let mut result = None;
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(_) => return None,
            naga::Statement::Store { pointer, value } => {
                if result.is_some() {
                    return None;
                }
                result = Some((*pointer, *value));
            }
            _ => return None,
        }
    }
    result
}

fn same_local_pointer(
    a: naga::Handle<naga::Expression>,
    b: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    if let (naga::Expression::LocalVariable(la), naga::Expression::LocalVariable(lb)) =
        (&expressions[a], &expressions[b])
    {
        la == lb
    } else {
        false
    }
}

fn unwrap_logical_not(
    condition: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[condition]
    {
        Some(*inner)
    } else {
        None
    }
}

// MARK: Constant-condition elimination

/// `block` (recursively) holds a statement that produces a result
/// expression (`Call` / `Atomic` / `WorkGroupUniformLoad` / `RayQuery` /
/// `Subgroup*`).  Constant-condition collapses keep such branches intact:
/// dropping the producer orphans its result expression, which fails
/// validation and rolls the whole pass back every sweep.  Variant-only, so a
/// result-less `Call` also trips it at the cost of one kept-but-dead branch.
fn block_has_result_producer(block: &naga::Block) -> bool {
    block.iter().any(statement_has_result_producer)
}

/// Empty, or a lone bare `Break` (which targets the switch itself).
fn case_body_is_switch_local_noop(body: &naga::Block) -> bool {
    let mut statements = body.iter();
    matches!(
        (statements.next(), statements.next()),
        (None, _) | (Some(naga::Statement::Break), None)
    )
}

fn statement_has_result_producer(stmt: &naga::Statement) -> bool {
    use naga::Statement as S;
    matches!(
        stmt,
        S::Call { .. }
            | S::Atomic { .. }
            | S::WorkGroupUniformLoad { .. }
            | S::RayQuery { .. }
            | S::SubgroupBallot { .. }
            | S::SubgroupGather { .. }
            | S::SubgroupCollectiveOperation { .. }
    ) || nested_blocks(stmt).any(block_has_result_producer)
}

/// Fold constant `If` / `Switch` / `break if` conditions and apply the
/// module doc's structural cleanups; returns the transformation count.
///
/// `in_loop` is true anywhere inside a loop (a dropped `Return` exits it
/// from any depth); `break_binds_to_loop` only where a bare `Break` would
/// target that loop (false inside switch cases, which capture `Break`).
/// Both feed the guards that keep a block carrying what tint counts as the
/// enclosing loop's only exit.
fn eliminate_dead_branches(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
    in_loop: bool,
    break_binds_to_loop: bool,
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;

    let mut statements = original.span_into_iter();
    while let Some((mut statement, span)) = statements.next() {
        // Leaves are enumerated (no catch-all) so a new block-bearing naga
        // variant fails to compile rather than skipping recursion.
        match &mut statement {
            naga::Statement::Block(inner) => {
                changed += eliminate_dead_branches(
                    inner,
                    expressions,
                    const_lits,
                    in_loop,
                    break_binds_to_loop,
                );
            }
            naga::Statement::If { accept, reject, .. } => {
                changed += eliminate_dead_branches(
                    accept,
                    expressions,
                    const_lits,
                    in_loop,
                    break_binds_to_loop,
                );
                changed += eliminate_dead_branches(
                    reject,
                    expressions,
                    const_lits,
                    in_loop,
                    break_binds_to_loop,
                );
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    // A case captures bare `Break`; `Return` still exits the
                    // loop.
                    changed += eliminate_dead_branches(
                        &mut case.body,
                        expressions,
                        const_lits,
                        in_loop,
                        false,
                    );
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed += eliminate_dead_branches(body, expressions, const_lits, true, true);
                // Valid IR bars a bare `Break` in `continuing`.
                changed +=
                    eliminate_dead_branches(continuing, expressions, const_lits, true, false);
            }
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
            | naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Return { .. }
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. }
            | naga::Statement::Call { .. }
            | naga::Statement::Atomic { .. }
            | naga::Statement::RayQuery { .. }
            | naga::Statement::RayPipelineFunction(_)
            | naga::Statement::WorkGroupUniformLoad { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. }
            | naga::Statement::CooperativeStore { .. } => {}
        }

        match statement {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => match resolve_to_literal(expressions, condition, const_lits) {
                Some(naga::Literal::Bool(taken)) => {
                    // One arm for both conditions so the hazard list cannot
                    // drift between them.  Keep the `if` when dropping the
                    // untaken arm would orphan a statement result (invalid IR:
                    // whole-pass rollback every sweep) or lose a loop exit
                    // under tint's analysis, which sequences reachability
                    // without const-evaluating conditions: the dropped arm may
                    // carry the loop's only exit, and splicing a kept arm that
                    // never falls through strands the rest of the block,
                    // un-crediting a trailing `break` / `return`.
                    let (kept, dropped) = if taken {
                        (&accept, &reject)
                    } else {
                        (&reject, &accept)
                    };
                    if block_has_result_producer(dropped)
                        || (in_loop && contains_return(dropped))
                        || (break_binds_to_loop && contains_bare_break(dropped))
                        || (in_loop && splice_loses_tint_loop_exit(kept, break_binds_to_loop))
                    {
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                    } else {
                        splice_block(&mut rebuilt, if taken { accept } else { reject });
                        changed += 1;
                    }
                }
                _ => {
                    if accept.is_empty() && reject.is_empty() {
                        changed += 1;
                    } else if !reject.is_empty() && block_definitely_terminates(&accept) {
                        // Else-elision: `if c { return; } else { x; }` ->
                        // `if c { return; } x;`.  Fires even when `reject`
                        // also terminates: the dropped `else {}` outweighs
                        // the dead-but-conditional `if c { return v1; }`,
                        // and only symmetric value-less returns would make
                        // the trailing return redundant (not collapsed).
                        let hoisted = reject;
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject: naga::Block::new(),
                            },
                            span,
                        );
                        splice_block(&mut rebuilt, hoisted);
                        changed += 1;
                    } else {
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                    }
                }
            },

            naga::Statement::Switch { selector, cases } => {
                // Every arm a no-op: the switch is dead, selector included
                // (naga expressions carry no effects; calls / atomics are
                // statements and make an arm non-trivial).  Checked before
                // the constant-selector path, whose bare-break guard would
                // otherwise keep a `case: { break; }` forever.
                if cases
                    .iter()
                    .all(|c| case_body_is_switch_local_noop(&c.body))
                {
                    changed += 1;
                    continue;
                }
                if let Some(value) = resolve_switch_value(selector, expressions, const_lits) {
                    // Splicing the matched chain drops the other cases, so
                    // keep the switch when any case carries a result producer
                    // (orphaned result: invalid IR, whole-pass rollback) or,
                    // inside a loop, a `Return` that may be the exit tint
                    // credits the loop with (bare `Break`s here target the
                    // switch, so only Returns matter).  Checking ALL cases
                    // over-keeps when the matched chain owns the Return -
                    // rare, and the cost is a kept-but-dead wrapper.  The
                    // degenerate splice below drops only empty prefix cases
                    // and needs no such guard.
                    if cases.iter().any(|c| block_has_result_producer(&c.body))
                        || (in_loop && cases.iter().any(|c| contains_return(&c.body)))
                    {
                        rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                    } else {
                        match find_matching_case_index(&cases, value) {
                            Some(start_idx)
                                if !(case_body_has_bare_break(&cases, start_idx)
                                    || (in_loop
                                        && switch_chain_splice_loses_tint_loop_exit(
                                            &cases,
                                            start_idx,
                                            break_binds_to_loop,
                                        ))) =>
                            {
                                let body = collect_case_body(cases, start_idx);
                                splice_block(&mut rebuilt, body);
                                changed += 1;
                            }
                            None => {
                                // No match, no default: nothing runs.
                                changed += 1;
                            }
                            // The chain holds a switch-targeting bare `Break`
                            // (splicing would mis-target it) or its splice
                            // would shade the loop's trailing exit.
                            Some(_) => {
                                rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                            }
                        }
                    }
                } else {
                    // naga lowers `case X, Y, default: { body }` (and a lone
                    // `default`) to empty fall-through prefix cases plus a
                    // `default` carrying the body, which therefore always
                    // runs exactly once: splice it, unless it holds a bare
                    // `Break` that would mis-target without the wrapper.
                    let degenerate = cases.split_last().is_some_and(|(last, prefix)| {
                        last.value == naga::SwitchValue::Default
                            && !last.fall_through
                            && prefix.iter().all(|c| c.fall_through && c.body.is_empty())
                    });
                    if degenerate && !contains_bare_break(&cases.last().unwrap().body) {
                        let body = cases.into_iter().next_back().unwrap().body;
                        splice_block(&mut rebuilt, body);
                        changed += 1;
                    } else {
                        rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                    }
                }
            }

            naga::Statement::Loop {
                body,
                continuing,
                break_if: Some(bi),
            } => match resolve_to_literal(expressions, bi, const_lits) {
                Some(naga::Literal::Bool(true)) => {
                    // `break if true`: body + continuing run once.  Unwrap
                    // only when no bare Break / Continue targets this loop
                    // (it would mis-target).
                    if !contains_bare_loop_control(&body)
                        && !contains_bare_loop_control(&continuing)
                    {
                        splice_block(&mut rebuilt, body);
                        splice_block(&mut rebuilt, continuing);
                        changed += 1;
                    } else {
                        rebuilt.push(
                            naga::Statement::Loop {
                                body,
                                continuing,
                                break_if: Some(bi),
                            },
                            span,
                        );
                    }
                }
                // `break if false` never fires, but tint's loop-exit
                // analysis is syntactic: when it is the loop's only lexical
                // exit, dropping it turns tint-valid input into "loop does
                // not exit" (naga validates an exit-less `loop {}`).  Drop
                // it only when the body proves another exit; `continuing`
                // cannot carry a bare Break or Return in valid IR.
                Some(naga::Literal::Bool(false))
                    if contains_bare_break(&body) || contains_return(&body) =>
                {
                    rebuilt.push(
                        naga::Statement::Loop {
                            body,
                            continuing,
                            break_if: None,
                        },
                        span,
                    );
                    changed += 1;
                }
                _ => {
                    rebuilt.push(
                        naga::Statement::Loop {
                            body,
                            continuing,
                            break_if: Some(bi),
                        },
                        span,
                    );
                }
            },

            // An emptied nested block would otherwise ship as a vacuous `{}`.
            naga::Statement::Block(inner) if inner.is_empty() => {
                changed += 1;
            }

            naga::Statement::Block(inner) => {
                rebuilt.push(naga::Statement::Block(inner), span);
            }

            other => {
                rebuilt.push(other, span);
            }
        }

        // Everything after a terminator is dead.  Gated like the collapses
        // above: an unreachable result producer must stay (orphaned result:
        // whole-pass rollback), and then the WHOLE tail stays, since dropping
        // its neighbours could strip an `Emit` covering its operands.
        if block_definitely_terminates(&rebuilt) {
            let tail: Vec<_> = statements.by_ref().collect();
            if tail
                .iter()
                .any(|(stmt, _)| statement_has_result_producer(stmt))
            {
                for (stmt, sp) in tail {
                    rebuilt.push(stmt, sp);
                }
            } else {
                changed += tail.len();
            }
            break;
        }
    }

    *block = rebuilt;
    changed
}

fn splice_block(target: &mut naga::Block, source: naga::Block) {
    for (stmt, sp) in source.span_into_iter() {
        target.push(stmt, sp);
    }
}

/// Control never falls off the end of `stmt`.  `bare_break_terminates` is the
/// ONE thing the two callers answer differently: in a plain block a `Break`
/// leaves the construct, in a switch case it only resumes after the switch.
/// Everything else - `Return`, `Continue`, the `Loop` exit analysis, the
/// switch's exhaustiveness test - is the same either way, and one copy of it
/// is what stops a fix landing on one side only.
///
/// `Kill` is NOT a terminator: under demote-to-helper execution continues
/// past `discard`, and tint requires the statements after it (a trailing
/// `return`, a loop's exit) to stay; treating it as terminating strips
/// reachable code and yields "missing return" / "loop does not exit"
/// rejections.
fn tail_terminates(stmt: &naga::Statement, bare_break_terminates: bool) -> bool {
    let block = |b: &naga::Block| block_tail_terminates(b, bare_break_terminates);
    match stmt {
        naga::Statement::Return { .. } | naga::Statement::Continue => true,
        naga::Statement::Break => bare_break_terminates,
        naga::Statement::Block(inner) => block(inner),
        naga::Statement::If { accept, reject, .. } => block(accept) && block(reject),
        // naga admits exactly two exits: a bare `Break` in `body`, or
        // `break_if` - `continuing` may carry neither by IR contract, and
        // `Return` / `Kill` leave the function, not the loop.  `break_if` is
        // armed only from the end of `body` or a bare `Continue`, so a body
        // that reaches neither disarms it; the body's tail is otherwise
        // irrelevant, falling off its end going around again.  Demanding it
        // unconditionally kept naga's appended `return` on every
        // `fn f() -> T { loop { ... return v; ... } }`, failing validation
        // and shipping uncompiled.  A `Break` here binds to THIS loop
        // whatever construct the caller asked about, so the flag is moot.
        naga::Statement::Loop { body, break_if, .. } => {
            !contains_bare_break(body)
                && (break_if.is_none()
                    || (block_definitely_terminates(body) && !contains_bare_continue(body)))
        }
        // Terminates iff every non-fall-through case exits BEYOND the switch
        // (a bare Break only resumes after it), a Default exists, and the
        // last case does not fall through - a shape naga's frontend never
        // emits but the inlining / CSE rebuilders can, and falling off the
        // end is Break-equivalent.  Cases are always asked the beyond-switch
        // question, so the flag is moot here as well.
        naga::Statement::Switch { cases, .. } => {
            let last_falls_through = cases.last().is_some_and(|c| c.fall_through);
            cases
                .iter()
                .all(|c| c.fall_through || case_body_terminates_beyond_switch(&c.body))
                && cases.iter().any(|c| c.value == naga::SwitchValue::Default)
                && !last_falls_through
                // A bare `break` anywhere in a case (not only last, e.g.
                // `case 1: { if (c) { break; } return; }`) resumes after the
                // switch; the per-case tail check cannot see it.
                && !cases.iter().any(|c| contains_bare_break(&c.body))
        }
        _ => false,
    }
}

fn block_tail_terminates(block: &naga::Block, bare_break_terminates: bool) -> bool {
    block
        .last()
        .is_some_and(|stmt| tail_terminates(stmt, bare_break_terminates))
}

/// Control never falls off the end of `block`.  `pub(crate)`: the generator
/// synthesises a trailing zero-value return only when the body provably
/// never falls through, and sharing the predicate keeps that guard in
/// lockstep with the return-stripping here.
pub(crate) fn block_definitely_terminates(block: &naga::Block) -> bool {
    block_tail_terminates(block, /*bare_break_terminates=*/ true)
}

/// The case body's tail exits BEYOND the switch (function or enclosing
/// loop): `Break` only exits the switch and resumes after it, so it does not
/// count; `Continue` jumps past the switch to the loop's continuing block,
/// so it does.
fn case_body_terminates_beyond_switch(block: &naga::Block) -> bool {
    block_tail_terminates(block, /*bare_break_terminates=*/ false)
}

fn resolve_switch_value(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> Option<naga::SwitchValue> {
    match resolve_to_literal(expressions, handle, const_lits)? {
        naga::Literal::I32(v) => Some(naga::SwitchValue::I32(v)),
        naga::Literal::U32(v) => Some(naga::SwitchValue::U32(v)),
        _ => None,
    }
}

/// Index of the case matching `value`, else of the `Default` case.
fn find_matching_case_index(cases: &[naga::SwitchCase], value: naga::SwitchValue) -> Option<usize> {
    cases.iter().position(|c| c.value == value).or_else(|| {
        cases
            .iter()
            .position(|c| c.value == naga::SwitchValue::Default)
    })
}

/// The statements entering case `start_idx` runs: its body plus each
/// fall-through successor (naga encodes `case 1, 2:` as an empty
/// fall-through case) up to the first non-fall-through case.
fn collect_case_body(cases: Vec<naga::SwitchCase>, start_idx: usize) -> naga::Block {
    let mut combined = naga::Block::new();
    for case in cases.into_iter().skip(start_idx) {
        let done = !case.fall_through;
        splice_block(&mut combined, case.body);
        if done {
            break;
        }
    }
    combined
}

/// A bare `Break` anywhere in the fall-through chain starting at `start_idx`.
fn case_body_has_bare_break(cases: &[naga::SwitchCase], start_idx: usize) -> bool {
    for case in &cases[start_idx..] {
        if contains_bare_break(&case.body) {
            return true;
        }
        if !case.fall_through {
            break;
        }
    }
    false
}

/// Search `block` for a bare `Break` / `Continue` bound to the enclosing
/// loop, descending only through constructs that do not rebind them: a
/// `Loop` rebinds both, a `Switch` rebinds `Break` but not `Continue`.
/// Everything else delegates its descent to `nested_blocks`, so a new
/// block-bearing naga variant is entered by construction.
fn contains_loop_control(block: &naga::Block, want_break: bool, want_continue: bool) -> bool {
    block.iter().any(|stmt| match stmt {
        naga::Statement::Break => want_break,
        naga::Statement::Continue => want_continue,
        naga::Statement::Loop { .. } => false,
        naga::Statement::Switch { cases, .. } => {
            want_continue
                && cases
                    .iter()
                    .any(|case| contains_loop_control(&case.body, false, true))
        }
        _ => nested_blocks(stmt)
            .any(|nested| contains_loop_control(nested, want_break, want_continue)),
    })
}

/// A bare `Continue` targeting the enclosing loop, which reaches its
/// `continuing` block and so arms `break_if`.
fn contains_bare_continue(block: &naga::Block) -> bool {
    contains_loop_control(
        block, /*want_break=*/ false, /*want_continue=*/ true,
    )
}

/// A bare `Break` or `Continue` targeting the immediately enclosing loop.
fn contains_bare_loop_control(block: &naga::Block) -> bool {
    contains_loop_control(
        block, /*want_break=*/ true, /*want_continue=*/ true,
    )
}

/// A bare `Break` targeting the immediately enclosing `Switch` or `Loop`.
/// Guards case splicing (without the wrapper the `Break` would mis-target)
/// and load_dedup's switch meet (a case-level `Break` reaches post-switch
/// code with the PRE-break cache state the fall-off-the-end meet never
/// sees).
pub(crate) fn contains_bare_break(block: &naga::Block) -> bool {
    contains_loop_control(
        block, /*want_break=*/ true, /*want_continue=*/ false,
    )
}

/// A `Return` at ANY depth, nested loops and switches included: it exits
/// the function, hence every enclosing loop.  Drives the
/// loop-exit-preservation guards: tint requires every `loop` to exit (a
/// `break` targeting it, `break if`, or an inner `Return`) and does not
/// const-evaluate conditions, so an exit inside `if false { .. }` still
/// counts for tint while naga validates an exit-less `loop {}`; folding
/// such a block turns tint-valid input into "loop does not exit".
/// Reachability-blind, over-approximating what tint credits - safe for the
/// KEEP direction it drives; `tint_block_behavior` is the reachability-aware
/// complement for the splice direction.  `Kill` does not count: tint
/// demotes `discard` to a helper-invocation exit, not a loop exit.
fn contains_return(block: &naga::Block) -> bool {
    block.iter().any(|stmt| {
        matches!(stmt, naga::Statement::Return { .. }) || nested_blocks(stmt).any(contains_return)
    })
}

/// A construct's behavior set under tint's (WGSL-spec) analysis - which of
/// {Next, Return, Break, Continue} executing it can produce - WITHOUT
/// const-evaluating conditions.  `brk` / `cont` are the bare behaviors at
/// the construct's own level; `Switch` and `Loop` absorb them structurally,
/// so no binding context is threaded.  `Kill` is `next` (demote-to-helper).
#[derive(Clone, Copy)]
struct TintBehavior {
    next: bool,
    ret: bool,
    brk: bool,
    cont: bool,
}

impl TintBehavior {
    const NEXT_ONLY: Self = Self {
        next: true,
        ret: false,
        brk: false,
        cont: false,
    };

    /// Behavior of `self` followed by `then`: `then` is unreachable (and
    /// contributes nothing) unless `self` can fall through.
    fn then(self, then: Self) -> Self {
        if !self.next {
            return self;
        }
        Self {
            next: then.next,
            ret: self.ret || then.ret,
            brk: self.brk || then.brk,
            cont: self.cont || then.cont,
        }
    }

    /// Behavior of exclusive alternatives (e.g. the two arms of an `If`).
    fn union(self, other: Self) -> Self {
        Self {
            next: self.next || other.next,
            ret: self.ret || other.ret,
            brk: self.brk || other.brk,
            cont: self.cont || other.cont,
        }
    }
}

/// Sequenced behavior of a block: statements after one that cannot fall
/// through are unreachable and contribute nothing - the property the
/// reachability-blind `contains_*` scans cannot express.
fn tint_block_behavior(block: &naga::Block) -> TintBehavior {
    let mut acc = TintBehavior::NEXT_ONLY;
    for stmt in block.iter() {
        if !acc.next {
            break;
        }
        acc = acc.then(tint_stmt_behavior(stmt));
    }
    acc
}

fn tint_stmt_behavior(stmt: &naga::Statement) -> TintBehavior {
    match stmt {
        naga::Statement::Return { .. } => TintBehavior {
            next: false,
            ret: true,
            brk: false,
            cont: false,
        },
        naga::Statement::Break => TintBehavior {
            next: false,
            ret: false,
            brk: true,
            cont: false,
        },
        naga::Statement::Continue => TintBehavior {
            next: false,
            ret: false,
            brk: false,
            cont: true,
        },
        naga::Statement::Block(inner) => tint_block_behavior(inner),
        naga::Statement::If { accept, reject, .. } => {
            tint_block_behavior(accept).union(tint_block_behavior(reject))
        }
        naga::Statement::Switch { cases, .. } => {
            // Case `i` runs its body then, on `fall_through`, case `i + 1`'s,
            // so effective behaviors fold right-to-left.  Falling past the
            // last case, a case-level Break, and a selector matching no case
            // (no Default) all yield Next.
            let mut union = TintBehavior {
                next: false,
                ret: false,
                brk: false,
                cont: false,
            };
            let mut next_case = TintBehavior::NEXT_ONLY;
            for case in cases.iter().rev() {
                let body = tint_block_behavior(&case.body);
                let effective = if case.fall_through {
                    body.then(next_case)
                } else {
                    body
                };
                union = union.union(effective);
                next_case = effective;
            }
            let has_default = cases.iter().any(|c| c.value == naga::SwitchValue::Default);
            TintBehavior {
                next: union.next || union.brk || !has_default,
                ret: union.ret,
                brk: false,
                cont: union.cont,
            }
        }
        naga::Statement::Loop {
            body,
            continuing,
            break_if,
        } => {
            // A loop falls through only via a bare Break or `break if`
            // (valid IR bars Break in `continuing`; including it costs
            // nothing) and propagates only Return.
            let body = tint_block_behavior(body);
            let continuing = tint_block_behavior(continuing);
            TintBehavior {
                next: body.brk || continuing.brk || break_if.is_some(),
                ret: body.ret || continuing.ret,
                brk: false,
                cont: false,
            }
        }
        // Straight-line, `Kill` included: demote-to-helper continues past it.
        naga::Statement::Emit(_)
        | naga::Statement::Store { .. }
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::ImageStore { .. }
        | naga::Statement::ImageAtomic { .. }
        | naga::Statement::Call { .. }
        | naga::Statement::Atomic { .. }
        | naga::Statement::RayQuery { .. }
        | naga::Statement::RayPipelineFunction(_)
        | naga::Statement::WorkGroupUniformLoad { .. }
        | naga::Statement::SubgroupBallot { .. }
        | naga::Statement::SubgroupGather { .. }
        | naga::Statement::SubgroupCollectiveOperation { .. }
        | naga::Statement::CooperativeStore { .. } => TintBehavior::NEXT_ONLY,
    }
}

/// Splicing `spliced` in place of a const-folded wrapper would deny the
/// enclosing loop its tint-credited exit.  The wrapper's dead arm kept
/// `Next` alive under tint's no-const-eval sequencing; the bare content is
/// safe only if it still falls through (every following statement, e.g. a
/// trailing `break`, stays reachable) or itself carries a credited exit: a
/// `Return`, or a bare `Break` where one binds to the loop.  A `return`
/// sequenced behind a `continue` is never credited and cannot rescue the
/// splice.
fn splice_loses_tint_loop_exit(spliced: &naga::Block, break_binds_to_loop: bool) -> bool {
    let b = tint_block_behavior(spliced);
    !(b.next || b.ret || (break_binds_to_loop && b.brk))
}

/// [`splice_loses_tint_loop_exit`] over the fall-through chain a constant
/// selector would splice.
fn switch_chain_splice_loses_tint_loop_exit(
    cases: &[naga::SwitchCase],
    start_idx: usize,
    break_binds_to_loop: bool,
) -> bool {
    let mut b = TintBehavior::NEXT_ONLY;
    for case in &cases[start_idx..] {
        if !b.next {
            break;
        }
        b = b.then(tint_block_behavior(&case.body));
        if !case.fall_through {
            break;
        }
    }
    !(b.next || b.ret || (break_binds_to_loop && b.brk))
}

// Redundant else-store elimination.  naga lowers `&&` / `||` chains to
//
//     var d: bool;                               // zero-init: false
//     if (a)  { d = b; } else { d = false; }     // &&
//     if (!a) { d = b; } else { d = true; }      // ||
//
// where an arm often stores what `d` already holds: with condition
// `Load(d)`, `d` is false in the reject arm and true in the accept arm;
// with `!Load(d)`, the reverse; and an unmodified zero-init `d` makes
// `d = false` a no-op.  Such arms are emptied.

/// A value that a local variable is known to hold at a given program point.
#[derive(Clone, Debug, PartialEq)]
enum KnownValue {
    /// The type's zero/default value (matches any zero literal or `ZeroValue`).
    Zero,
    Literal(naga::Literal),
}

// MARK: Redundant else-store elimination

/// Module constants whose init is already a `Literal`; anything else stays
/// unresolvable.
fn build_const_literal_cache(module: &naga::Module) -> HandleMap<naga::Constant, naga::Literal> {
    module
        .constants
        .iter()
        .filter_map(|(ch, c)| {
            if let naga::Expression::Literal(lit) = module.global_expressions[c.init] {
                Some((ch, lit))
            } else {
                None
            }
        })
        .collect()
}

/// WGSL zero-initialises locals without an initialiser; literal inits are
/// known outright.
fn init_known_values(
    locals: &naga::Arena<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> FxHashMap<naga::Handle<naga::LocalVariable>, KnownValue> {
    locals
        .iter()
        .filter_map(|(lh, lv)| match lv.init {
            None => Some((lh, KnownValue::Zero)),
            Some(init_h) => {
                let lit = resolve_to_literal(expressions, init_h, const_lits)?;
                if is_zero_literal(&lit) {
                    Some((lh, KnownValue::Zero))
                } else {
                    Some((lh, KnownValue::Literal(lit)))
                }
            }
        })
        .collect()
}

fn eliminate_redundant_else_stores_in_function(
    function: &mut naga::Function,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> usize {
    let mut known_values = ScopedMap::new();
    for (lh, kv) in init_known_values(&function.local_variables, &function.expressions, const_lits)
    {
        known_values.insert(lh, kv);
    }
    // Per local, its latest materialised `Load` with no store since.
    // Condition narrowing is sound only on such a fresh load: a stale
    // forwarded one (`let t = d; d = false; if t {..}`) reflects the
    // pre-store value, and narrowing on it would clobber the post-store
    // known value and drop a live branch.
    let mut fresh_loads = Default::default();
    eliminate_redundant_else_stores(
        &mut function.body,
        &function.expressions,
        const_lits,
        &mut known_values,
        &mut fresh_loads,
    )
}

/// `condition` is `Load(d)` or `!Load(d)` and that load is the freshest
/// recorded for `d` (no store since), so it equals `d`'s current value.
fn condition_load_is_fresh(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    fresh_loads: &HandleMap<naga::LocalVariable, naga::Handle<naga::Expression>>,
) -> bool {
    match &expressions[*condition] {
        naga::Expression::Load { pointer } => {
            if let naga::Expression::LocalVariable(d) = expressions[*pointer] {
                fresh_loads.get(d) == Some(condition)
            } else {
                false
            }
        }
        naga::Expression::Unary {
            op: naga::UnaryOperator::LogicalNot,
            expr: inner,
        } => {
            if let naga::Expression::Load { pointer } = &expressions[*inner]
                && let naga::Expression::LocalVariable(d) = expressions[*pointer]
            {
                fresh_loads.get(d) == Some(inner)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Recursively walk a block, tracking known values of locals, and clear
/// branches that only store a value the variable already holds.
fn eliminate_redundant_else_stores(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    fresh_loads: &mut HandleMap<naga::LocalVariable, naga::Handle<naga::Expression>>,
) -> usize {
    let mut changed = 0usize;

    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // Each arm: narrow, recurse, roll back to the narrowed entry
                // for the redundancy check, then roll back to the pre-if
                // state.  `cond_fresh` is decided once, up front: an
                // accept-arm store must not perturb the reject decision.
                // Written out per arm: one shared body needs the narrowing as
                // a function pointer, which stops it inlining (+1,836 B).
                let cond_fresh = condition_load_is_fresh(condition, expressions, fresh_loads);
                let cp_pre_if = known_values.checkpoint();

                if cond_fresh {
                    narrow_for_accept(condition, expressions, known_values);
                }
                let cp_accept_entry = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    accept,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_accept_entry);
                let accept_redundant = !accept.is_empty()
                    && block_only_has_redundant_known_stores(
                        accept,
                        expressions,
                        known_values.as_map(),
                        const_lits,
                    );
                known_values.rollback_to(cp_pre_if);

                // Same shape for `reject`, and the same order for the same
                // reason.
                if cond_fresh {
                    narrow_for_reject(condition, expressions, known_values);
                }
                let cp_reject_entry = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    reject,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_reject_entry);
                let reject_redundant = !reject.is_empty()
                    && block_only_has_redundant_known_stores(
                        reject,
                        expressions,
                        known_values.as_map(),
                        const_lits,
                    );
                known_values.rollback_to(cp_pre_if);

                if reject_redundant {
                    *reject = naga::Block::new();
                    changed += 1;
                }
                if accept_redundant {
                    *accept = naga::Block::new();
                    changed += 1;
                }

                // A conditional store leaves the local unknown and its prior
                // load stale; the removal is logged for outer rollback.
                let mut modified = Default::default();
                collect_modified_locals(accept, expressions, &mut modified);
                collect_modified_locals(reject, expressions, &mut modified);
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(lh);
                }
            }

            naga::Statement::Emit(range) => {
                // Record the freshest load; every store arm below
                // invalidates it.
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = expressions[h]
                        && let naga::Expression::LocalVariable(d) = expressions[pointer]
                    {
                        fresh_loads.insert(d, h);
                    }
                }
            }

            naga::Statement::Store { pointer, value } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    fresh_loads.remove(lh);
                    if let Some(lit) = resolve_to_literal(expressions, *value, const_lits) {
                        if is_zero_literal(&lit) {
                            known_values.insert(lh, KnownValue::Zero);
                        } else {
                            known_values.insert(lh, KnownValue::Literal(lit));
                        }
                    } else if is_zero_value(expressions, *value, const_lits) {
                        known_values.insert(lh, KnownValue::Zero);
                    } else {
                        known_values.remove(&lh);
                    }
                } else if let Some(lh) = root_local_var(*pointer, expressions) {
                    // Partial store: value unknown.
                    known_values.remove(&lh);
                    fresh_loads.remove(lh);
                }
            }

            naga::Statement::Switch { cases, .. } => {
                let cp = known_values.checkpoint();
                for case in cases.iter_mut() {
                    changed += eliminate_redundant_else_stores(
                        &mut case.body,
                        expressions,
                        const_lits,
                        known_values,
                        fresh_loads,
                    );
                    known_values.rollback_to(cp);
                }
                let mut modified = Default::default();
                for case in cases.iter() {
                    collect_modified_locals(&case.body, expressions, &mut modified);
                }
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(lh);
                }
            }

            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Loop-carried locals are unknown on the back edge: wipe them
                // (permanently) before entering.
                let mut modified = Default::default();
                collect_modified_locals(body, expressions, &mut modified);
                collect_modified_locals(continuing, expressions, &mut modified);

                for lh in &modified {
                    known_values.remove(lh);
                    fresh_loads.remove(lh);
                }
                let cp_loop = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    body,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                // `continuing` is entered from every `continue` edge and the
                // body's fall-through, not sequentially after its tail, so a
                // fact the body set (e.g. `known[d] = true` before a `break`)
                // may not hold there and inheriting it would delete a live
                // continuing store.  Rolling back to the post-wipe state is a
                // sound meet.  `fresh_loads` needs no reset: a surviving entry
                // means `d` is unwritten since that load, which naga emits on
                // every path into continuing.
                known_values.rollback_to(cp_loop);
                changed += eliminate_redundant_else_stores(
                    continuing,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_loop);
            }

            naga::Statement::Block(inner) => {
                changed += eliminate_redundant_else_stores(
                    inner,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
            }

            // Pointer writes by callees / atomics / ray / cooperative ops.
            other => super::expr_util::visit_statement_write_pointers(other, &mut |p| {
                if let Some(lh) = root_local_var(p, expressions) {
                    known_values.remove(&lh);
                    fresh_loads.remove(lh);
                }
            }),
        }
    }

    changed
}

/// Accept-arm narrowing: `Load(d)` implies `d == true`, `!Load(d)` implies
/// `d == false`.  Inserts go through `ScopedMap` so the caller's checkpoint
/// rolls them back.
fn narrow_for_accept(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
) {
    if let naga::Expression::Load { pointer } = &expressions[*condition]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Literal(naga::Literal::Bool(true)));
    }
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[*condition]
        && let naga::Expression::Load { pointer } = &expressions[*inner]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Zero);
    }
}

/// Reject-arm narrowing: `Load(d)` implies `d == false`, `!Load(d)` implies
/// `d == true`.
fn narrow_for_reject(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
) {
    if let naga::Expression::Load { pointer } = &expressions[*condition]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Zero);
    }
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[*condition]
        && let naga::Expression::Load { pointer } = &expressions[*inner]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Literal(naga::Literal::Bool(true)));
    }
}

/// Every statement is an `Emit` or a `Store` of the value its local is
/// known to hold, with at least one such `Store`.
fn block_only_has_redundant_known_stores(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &FxHashMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> bool {
    let mut has_store = false;
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(_) => continue,
            naga::Statement::Store { pointer, value } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer]
                    && let Some(known) = known_values.get(&lh)
                    && expr_matches_known(expressions, *value, known, const_lits)
                {
                    has_store = true;
                    continue;
                }
                return false;
            }
            // A brace-wrapped sub-block is a flat passthrough; its own
            // at-least-one-store rule keeps an empty inner block `false`.
            naga::Statement::Block(inner) => {
                if block_only_has_redundant_known_stores(
                    inner,
                    expressions,
                    known_values,
                    const_lits,
                ) {
                    has_store = true;
                    continue;
                }
                return false;
            }
            _ => return false,
        }
    }
    has_store
}

fn expr_matches_known(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    known: &KnownValue,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> bool {
    match known {
        KnownValue::Zero => is_zero_value(expressions, handle, const_lits),
        KnownValue::Literal(lit) => resolve_to_literal(expressions, handle, const_lits)
            .is_some_and(|resolved| literal_bit_eq(&resolved, lit)),
    }
}

/// Literal value of `handle`, resolving `Constant` through `const_lits` (so
/// folds fire before `const_fold` inlines named constants; abstract
/// literals are filtered upstream).  `Override` is deliberately not
/// resolved: its init is only a default the pipeline can replace at draw
/// time, so folding through it would erase code on a value that changes
/// post-compile.
fn resolve_to_literal(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> Option<naga::Literal> {
    match &expressions[handle] {
        naga::Expression::Literal(lit) => Some(*lit),
        naga::Expression::Constant(c) => const_lits.get(c).copied(),
        _ => None,
    }
}

fn is_zero_value(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HandleMap<naga::Constant, naga::Literal>,
) -> bool {
    match &expressions[handle] {
        naga::Expression::Literal(lit) => is_zero_literal(lit),
        naga::Expression::ZeroValue(_) => true,
        naga::Expression::Constant(c) => const_lits.get(c).is_some_and(is_zero_literal),
        _ => false,
    }
}

// MARK: Tests

#[cfg(test)]
#[path = "dead_branch_tests.rs"]
mod tests;
