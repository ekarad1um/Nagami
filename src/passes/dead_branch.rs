//! Dead-branch elimination.  Three primary phases plus several
//! structural simplifications folded into the constant-condition pass:
//!
//! 1. Short-circuit re-sugaring folds the `if/else store false`
//!    patterns naga's WGSL frontend emits for `&&` and `||` back into
//!    compact [`naga::Expression::Binary`] `LogicalAnd` / `LogicalOr`
//!    expressions before the next phase destroys their shape.
//! 2. Redundant `else`-store elimination removes writes that assign
//!    the same known literal already present in the variable on the
//!    opposite branch, shrinking unbalanced two-arm ifs into one arm.
//! 3. Constant-condition / structural cleanup (`eliminate_dead_branches`)
//!    strips `if (true)` / `if (false)` arms AND performs several
//!    cooperative simplifications in the same walk:
//!    * empty-If / empty-Switch / empty-Block elision,
//!    * else-block elision when the accept branch unconditionally
//!      terminates (return / break / continue; `discard` continues under
//!      demote-to-helper and never counts),
//!    * dead code after a terminating statement,
//!    * `loop { ... break if true; }` unwrap when no bare break/continue
//!      would mis-target after unwrapping,
//!    * `break if false` dropped when the loop body proves another exit,
//!    * single-default-case Switch splicing.
//!
//!    Every fold that could delete - or, by splicing a never-falls-through
//!    arm, make unreachable - what tint credits as a loop's only exit is
//!    guarded; see `contains_return` and `splice_loses_tint_loop_exit`.
//!
//! Phase order is load-bearing: short-circuit patterns rely on the
//! untransformed frontend output, so re-sugaring must run before the
//! later phases mutate the statement shape.
//!
//! The "branch flipping" optimisation listed in the project README
//! (`if c {} else { x; }` -> `if !c { x; }`) is NOT implemented here;
//! it lives at emit time in `generator::stmt_emit` (search for
//! "flip the condition").

use std::collections::{HashMap, HashSet};

use super::expr_util::{nested_blocks, nested_blocks_mut};
use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Like [`single_store_info`], but ALSO accepts a leading run of `Emit`
/// statements before the single trailing `Store`.  Returns the store's
/// `(pointer, value)` on match.
///
/// Used by the short-circuit re-sugar to recognise a branch that computes
/// an intermediate value (`{ let _e = a < b; ...; d = _e; }`) before
/// storing it.  Those leading `Emit`s are hoisted into the parent block by
/// [`hoist_leading_emits`], so they evaluate UNCONDITIONALLY after the
/// fold.  That is sound: `Emit` expressions are side-effect-free (every
/// side-effecting / control-flow construct is a distinct `Statement` kind
/// that this recogniser rejects via the `_ => None` arm), WGSL bounds-
/// checks out-of-range indexing rather than trapping, the hoisted value is
/// discarded by the `&&` / `||` whenever the guard fails, and lifting a
/// computation OUT of a conditional can only reduce non-uniformity (it can
/// never push a derivative / implicit-LOD sample into non-uniform control
/// flow).  Yields `None` for any non-`Emit`/`Store` statement, a second
/// store, zero stores, or any statement after the store.
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
                    return None; // an Emit after the store
                }
            }
            naga::Statement::Store { pointer, value } => {
                if store.is_some() {
                    return None; // a second store
                }
                store = Some((*pointer, *value));
            }
            _ => return None, // a side-effecting or control-flow statement
        }
    }
    store
}

/// `true` when `cond` is a constant `bool` literal.  The short-circuit
/// re-sugar skips such conditions: `const_fold` runs before this pass, so
/// a constant guard means the whole `if` is dead-branch-eliminable (the
/// strictly better rewrite).  Re-sugaring it instead to `var d = false &&
/// x;` would block that elimination and can leave a larger residue (e.g. a
/// dead loop gated on a zero-init bool the loop-elimination cannot prove).
fn is_const_bool(
    cond: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    matches!(
        expressions[cond],
        naga::Expression::Literal(naga::Literal::Bool(_))
    )
}

/// Move the leading `Emit` statements of a re-sugared accept arm into
/// `rebuilt` (the parent block), dropping its trailing `Store` - whose
/// value has become the right operand of the `&&` / `||`.  The arm shape
/// was already validated by [`store_with_leading_emits`].
fn hoist_leading_emits(rebuilt: &mut naga::Block, accept: naga::Block) {
    for (stmt, span) in accept.span_into_iter() {
        if matches!(stmt, naga::Statement::Emit(_)) {
            rebuilt.push(stmt, span);
        }
    }
}

// MARK: Single-store local forwarding (register promotion)

/// Forward a local that is assigned exactly once and otherwise only read
/// (`var t; ...; t = E; ...; use(t)`): redirect every `Load(t)` to the
/// stored value `E` and drop the store, leaving `t` dead.
///
/// This collapses the per-join `&&`/`||` re-sugaring - `t1 = a && b;
/// t2 = t1 && c; if t2 {..}` - all the way to `if (a && b && c) {..}`, and
/// (crucially) makes that re-sugaring IDEMPOTENT: re-parsing `if (a && b
/// && c)` re-lowers to fresh single-use temps that fold straight back to
/// the same shape, a stable fixed point.  It is also a useful general
/// optimisation (single-def/single-use locals vanish).
///
/// Soundness gates:
///  * `t` has exactly one whole-variable store (value `E`) and every other
///    reference is a `Load(t)` - no element access, pointer-arg, or second
///    store.  Verified by an EXHAUSTIVE census (`total_refs == 1 + loads`)
///    built from the canonical handle visitors, so any missed reference
///    fails the equality and bails.
///  * `t` has no initialiser.
///  * the store and the materialising `Emit` of every load are top-level
///    statements of the SAME block, store first - so each load observes the
///    stored value, never a stale / pre-store (zero-init) one.
///
/// `E`'s evaluation position is unchanged (its `Emit` stays put; the store
/// captured `E` immediately after that `Emit`), so the forwarded value
/// equals what `t` held.  The generator's `must_bind_loads` still guards
/// any load-versus-write hazard when it later inlines `E`.
///
/// This pass only redirects loads and removes the store; it leaves the
/// now-dead local declaration and the orphaned `Load(t)` expression in the
/// arena.  Pruning them is delegated to the downstream dead-local / dead-
/// expression cleanup (e.g. [`super::load_dedup`]'s dead-init removal),
/// which runs later in the fixpoint - so reordering or removing that pass
/// would leave a stray `var t;` (a size regression only, never a
/// miscompile).
fn forward_single_store_locals(function: &mut naga::Function) -> bool {
    let nlocals = function.local_variables.len();
    if nlocals == 0 {
        return false;
    }

    // === Exhaustive per-local reference census ===
    let mut total_refs = vec![0u32; nlocals];
    let mut whole_stores = vec![0u32; nlocals];
    let mut store_value: Vec<Option<naga::Handle<naga::Expression>>> = vec![None; nlocals];
    let mut load_count = vec![0usize; nlocals];
    // `min_consumer[h]` = the lowest-indexed EXPRESSION that references `h`
    // as an operand (`u32::MAX` if none).  The re-sugared `&&`/`||` Binary
    // is appended at the END of the arena, so forwarding a `Load` to it can
    // make an earlier expression reference a later handle - a forward
    // reference naga rejects.  Forwarding is allowed only when the stored
    // value precedes every EXPRESSION consumer; statement consumers impose
    // no arena-order constraint, and their value-correctness rests on the
    // store-before-load-`Emit` gate in `collect_forwards`, not on this guard.
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
        // Single store AND single load: forwarding inlines `E` at exactly
        // one site, so it always shrinks (the `var` + store vanish) and
        // never duplicates `E` - a multi-load forward could, growing output.
        if whole_stores[t] == 1
            && load_count[t] == 1
            && total_refs[t] as usize == 2
            && lvar.init.is_none()
        {
            candidate[t] = true;
        }
    }
    if !candidate.iter().any(|&c| c) {
        return false;
    }

    // === Locate the forwards that satisfy block-local dominance ===
    let mut redirects: HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>> =
        HashMap::new();
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
    );
    if redirects.is_empty() {
        return false;
    }

    // === Apply: redirect loads to the stored value, then drop the stores ===
    for (_, expr) in function.expressions.iter_mut() {
        let _ = super::expr_util::try_map_expression_handles_in_place(expr, &mut |h| {
            Some(*redirects.get(&h).unwrap_or(&h))
        });
    }
    remap_block_handles(&mut function.body, &redirects);
    remove_forwarded_stores(&mut function.body, &function.expressions, &remove_store);
    true
}

/// Count whole-variable `Store`s per local (recording the lone value) so
/// [`forward_single_store_locals`] can spot single-def locals.
fn count_whole_stores(
    block: &naga::Block,
    exprs: &naga::Arena<naga::Expression>,
    whole_stores: &mut [u32],
    store_value: &mut [Option<naga::Handle<naga::Expression>>],
) {
    for stmt in block.iter() {
        if let naga::Statement::Store { pointer, value } = stmt
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
        {
            whole_stores[l.index()] += 1;
            store_value[l.index()] = Some(*value);
        }
        for nested in nested_blocks(stmt) {
            count_whole_stores(nested, exprs, whole_stores, store_value);
        }
    }
}

/// Read-only census slices threaded through [`collect_forwards`].
struct ForwardCensus<'a> {
    candidate: &'a [bool],
    load_count: &'a [usize],
    store_value: &'a [Option<naga::Handle<naga::Expression>>],
    min_consumer: &'a [u32],
}

/// `true` when `value` is a bare `Load` of a forwarding-candidate local -
/// i.e. a copy `var t = c;`.  Forwarding such a `t` in the same pass as
/// `c` would chain two redirects (`Load(t) -> Load(c) -> E_c`) that the
/// non-transitive apply cannot follow, so the caller defers it.
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

/// For each block, redirect a candidate local's loads to its stored value
/// when the store and the materialising `Emit` of every load are top-level
/// statements of THIS block with the store first, so each load observes the
/// stored value and all of the local's loads are accounted for here.
fn collect_forwards(
    block: &naga::Block,
    exprs: &naga::Arena<naga::Expression>,
    census: &ForwardCensus<'_>,
    redirects: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    remove_store: &mut [bool],
) {
    // Top-level candidate stores in this block, keyed by local, with index.
    let mut store_idx: HashMap<usize, usize> = HashMap::new();
    for (i, stmt) in block.iter().enumerate() {
        if let naga::Statement::Store { pointer, .. } = stmt
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
            && census.candidate[l.index()]
        {
            store_idx.insert(l.index(), i);
        }
    }
    // A candidate's `Load` is forwardable only when its MATERIALISATION - the
    // `Emit` that defines it - is a top-level statement of THIS block at an
    // index after the store.  Keying on the Emit position (not on a later
    // statement that merely consumes the load) is load-bearing: a
    // read-before-write `let snap = t; t = E; use(snap)` emits `Load(t)`
    // BEFORE the store yet consumes it after, so forwarding to `E` would swap
    // the pre-store / zero-init value `snap` actually holds - a silent
    // miscompile.  Scanning only this block's own `Emit`s also keeps
    // forwarding SAME-block: a load materialised in a nested loop / if belongs
    // to that block and is matched (against an inner store) only when the
    // recursion below reaches it, never forwarded out to an enclosing store
    // (which could fold a loop-invariant into a guard and expose an infinite
    // loop Tint rejects).
    let mut found: HashMap<usize, HashSet<naga::Handle<naga::Expression>>> = HashMap::new();
    for (i, stmt) in block.iter().enumerate() {
        let naga::Statement::Emit(range) = stmt else {
            continue;
        };
        for h in range.clone() {
            if let naga::Expression::Load { pointer } = exprs[h]
                && let naga::Expression::LocalVariable(l) = exprs[pointer]
                && let Some(&si) = store_idx.get(&l.index())
                && i > si
            {
                found.entry(l.index()).or_default().insert(h);
            }
        }
    }
    for (&t, _) in store_idx.iter() {
        let Some(loads_here) = found.get(&t) else {
            continue;
        };
        if loads_here.len() == census.load_count[t]
            && census.load_count[t] > 0
            && let Some(e) = census.store_value[t]
            // Forward-reference guard: `e` must precede every expression that
            // consumes each load, or the redirect makes an earlier expression
            // reference a later handle (invalid IR).
            && loads_here
                .iter()
                .all(|&lh| (e.index() as u32) < census.min_consumer[lh.index()])
            // Copy-chain guard: if `e` is a bare `Load` of ANOTHER forwarded
            // candidate (`var t = c;`), redirecting `Load(t) -> Load(c)` while
            // `c` is also forwarded would resolve non-transitively - removing
            // c's store orphans `Load(c)`, which then reads c's zero-init.
            // Defer `t` to a later sweep, after `c` has been folded into `e`.
            && !is_candidate_copy(e, exprs, census.candidate)
        {
            for &lh in loads_here {
                redirects.insert(lh, e);
            }
            remove_store[t] = true;
        }
    }
    // Recurse into nested blocks (each handled independently).
    for stmt in block.iter() {
        for nested in nested_blocks(stmt) {
            collect_forwards(nested, exprs, census, redirects, remove_store);
        }
    }
}

/// Apply the load-redirect map to every statement (recursively).
fn remap_block_handles(
    block: &mut naga::Block,
    redirects: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    for stmt in block.iter_mut() {
        super::expr_util::remap_statement_handles(stmt, &mut |h| *redirects.get(&h).unwrap_or(&h));
        for nested in nested_blocks_mut(stmt) {
            remap_block_handles(nested, redirects);
        }
    }
}

/// Drop the (now redundant) whole stores of forwarded locals.
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
            continue; // drop the forwarded store
        }
        rebuilt.push(stmt, span);
    }
    *block = rebuilt;
}

use super::load_dedup::{collect_modified_locals, get_stored_local, is_zero_literal};
use super::scoped_map::ScopedMap;

/// Dead-branch pass.  See the module-level doc for the three phases
/// this pass runs per function on every sweep.
#[derive(Debug, Default)]
pub struct DeadBranchPass;

impl Pass for DeadBranchPass {
    fn name(&self) -> &'static str {
        "dead_branch_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // Compute the constant-handle -> `Literal` cache once so the
        // mutable function iteration below does not have to re-borrow
        // `module.constants` on every lookup.
        let const_lits = build_const_literal_cache(module);

        for (_, function) in module.functions.iter_mut() {
            // Phase 0: short-circuit re-sugaring runs before the else
            // store phase destroys the lowered patterns.
            let foldable = compute_resugar_foldable(function);
            changed +=
                desugar_short_circuit(&mut function.body, &mut function.expressions, &foldable);
            changed += usize::from(forward_single_store_locals(function));
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
            changed += {
                let foldable = compute_resugar_foldable(&entry.function);
                desugar_short_circuit(
                    &mut entry.function.body,
                    &mut entry.function.expressions,
                    &foldable,
                )
            };
            changed += usize::from(forward_single_store_locals(&mut entry.function));
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

// naga's WGSL front-end lowers short-circuit operators into explicit
// if/else statements that write the intermediate result to a local:
//
//   // `a && b`
//   var d: bool;
//   if (a) { d = b; } else { d = false; }
//
//   // `a || b`
//   var d: bool;
//   if (!a) { d = b; } else { d = true; }
//
// This phase detects both shapes and folds them back into
// `Binary(LogicalAnd)` / `Binary(LogicalOr)` expressions so downstream
// passes (and the generator) see the compact form.

/// Per-local gate for the re-sugar: `true` when a local has exactly one
/// load and that load has NO expression consumer (only statement consumers
/// like an `if` condition or a `Store` value).  Folding a join into a
/// `d = cond && val` store is kept ONLY for such locals, because they are
/// the ones [`forward_single_store_locals`] can then forward into a
/// condition without a forward reference.  Value-position results whose
/// load feeds an expression (`B = false | d`, `select(.., d)`) are left
/// lowered - exactly as before the re-sugar existed - so the output stays
/// idempotent (re-minifying never flip-flops between the two shapes).
fn compute_resugar_foldable(function: &naga::Function) -> Vec<bool> {
    let nlocals = function.local_variables.len();
    let exprs = &function.expressions;
    // Pass 1: `has_expr_consumer[h]` = some EXPRESSION uses `h` as an operand.
    // (Statement consumers - `if` conditions, `Store`/`Return` values - are
    // NOT expression operands and so leave this `false`.)
    let mut has_expr_consumer = vec![false; exprs.len()];
    for (_, expr) in exprs.iter() {
        super::expr_util::visit_expression_children(expr, |child| {
            has_expr_consumer[child.index()] = true;
        });
    }
    // Pass 2: a local is foldable when it has at least one load and NONE of
    // its loads feed an expression - i.e. every load is a condition / value
    // that [`forward_single_store_locals`] can later forward into place
    // without a forward reference.
    let mut load_count = vec![0usize; nlocals];
    let mut load_expr_consumed = vec![false; nlocals];
    for (hc, expr) in exprs.iter() {
        if let naga::Expression::Load { pointer } = expr
            && let naga::Expression::LocalVariable(l) = exprs[*pointer]
        {
            load_count[l.index()] += 1;
            if has_expr_consumer[hc.index()] {
                load_expr_consumed[l.index()] = true;
            }
        }
    }
    (0..nlocals)
        .map(|t| load_count[t] >= 1 && !load_expr_consumed[t])
        .collect()
}

/// `true` when the whole-variable store through `pointer` targets a local
/// the re-sugar is allowed to fold (see [`compute_resugar_foldable`]).
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

/// Recursively fold short-circuit if/else store patterns into
/// `Binary` logical-and / logical-or expressions.  Returns the number
/// of replacements performed so the caller can aggregate a change
/// count across phases.  `foldable` gates which store targets may fold
/// (see [`compute_resugar_foldable`]).
fn desugar_short_circuit(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    foldable: &[bool],
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;

    for (mut statement, span) in original.span_into_iter() {
        // Step 1: recurse into nested blocks.
        for nested in nested_blocks_mut(&mut statement) {
            changed += desugar_short_circuit(nested, expressions, foldable);
        }

        // Step 2: check for short-circuit patterns.
        match statement {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // Pattern 1 (&&): `if cond { [emits...]; d = val; } else { d = false; }`
                // -> hoist the leading emits, then `d = cond && val;`.
                //
                // Scope: the rewritten Binary lives in the parent block, so
                // both operands must be in scope there.  `condition`'s Emit
                // dominates the If (WGSL evaluates it before either branch).
                // `val_a` and its intermediates are produced by the accept
                // arm's leading `Emit`s, which [`hoist_leading_emits`] moves
                // into the parent ahead of the Binary - see
                // [`store_with_leading_emits`] for why evaluating them
                // unconditionally is sound (side-effect-free, bounds-checked,
                // discarded on guard failure, uniformity-monotonic).  The
                // reject arm is a single declarative `d = false` store with
                // nothing to preserve.
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

                // Pattern 2 (||): `if !cond { [emits...]; d = val; } else { d = true; }`
                // -> hoist the leading emits, then `d = !cond || val;`
                // (rewritten as `inner_cond || val`).
                //
                // Same hoisting rationale and scope invariants as pattern 1.
                // `inner_cond` (the operand of the LogicalNot condition) is
                // in scope transitively: the LogicalNot's Emit dominates the
                // If, and it can only reference operands whose Emit dominates
                // IT.
                if let Some(inner_cond) = unwrap_logical_not(condition, expressions)
                    && !is_const_bool(inner_cond, expressions)
                    && let Some((ptr_r, val_r)) = single_store_info(&reject)
                    && is_bool_true(expressions, val_r)
                    && let Some((ptr_a, val_a)) = store_with_leading_emits(&accept)
                    && same_local_pointer(ptr_a, ptr_r, expressions)
                    && store_target_foldable(ptr_a, expressions, foldable)
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

                // No pattern matched - keep the If.
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

/// Return the pointer and value of a block whose sole statement is a
/// single `Store`.  Yields `None` when the block contains any `Emit`,
/// has zero stores, more than one store, or any other statement.
///
/// The short-circuit re-sugar uses this to recognise the REJECT arm of a
/// lowered pattern - the `d = false` (`&&`) / `d = true` (`||`) constant
/// store that becomes the operator's short-circuit value.  Rejecting any
/// `Emit` keeps that arm purely declarative: the constant `false`/`true`
/// is a `Literal` needing no `Emit`, and there is nothing in the arm to
/// hoist.  (The ACCEPT arm, by contrast, goes through
/// [`store_with_leading_emits`], which deliberately permits and hoists
/// leading `Emit`s - see [`hoist_leading_emits`] for why that is sound.)
fn single_store_info(
    block: &naga::Block,
) -> Option<(
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
)> {
    // Any `Emit` (a load, binary op, array index, ...) disqualifies the
    // arm: a constant-storing reject arm never needs one, so its presence
    // means this is not the declarative short-circuit value we expect.
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

/// `true` when `a` and `b` both resolve to the same `LocalVariable`
/// handle.  Only direct local references are compared; swizzle,
/// access, and pointer arithmetic all bail out as "not same".
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

/// Check if an expression is the boolean literal `false`.
fn is_bool_false(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        &expressions[handle],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Check if an expression is the boolean literal `true`.
fn is_bool_true(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        &expressions[handle],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

/// Unwrap `!expr` and return the inner handle, or `None` when the
/// condition is not a `LogicalNot`.
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

/// `true` when `block` (recursively) contains a statement that produces a
/// result EXPRESSION (`Call`/`Atomic`/`WorkGroupUniformLoad`/`RayQuery`/
/// `Subgroup*`).  A constant-condition collapse KEEPS such an `if`/`switch`
/// branch intact rather than dropping it: dropping it would orphan the result
/// expression (its producer statement is gone), which fails validation and
/// rolls the whole pass back every sweep.  Variant-only and conservative - a
/// result-less `Call` also trips it, at the cost of one kept-but-dead branch.
fn block_has_result_producer(block: &naga::Block) -> bool {
    use naga::Statement as S;
    block.iter().any(|stmt| {
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
    })
}

/// Recursively walk `block`, folding branches whose condition is a
/// compile-time literal `true` / `false` and pruning unreachable switch
/// cases.  Returns the number of transformations applied so the caller can
/// aggregate change counts across phases.
///
/// `in_loop` / `break_binds_to_loop` thread the enclosing-loop context for
/// the loop-exit-preservation guards (see [`contains_return`]): a fold must
/// not drop a block carrying what tint counts as the enclosing loop's only
/// exit.  `in_loop` is true anywhere inside a loop (a dropped `Return` exits
/// it from any depth); `break_binds_to_loop` is true only where a bare
/// `Break` would target that loop (false inside switch cases, which capture
/// `Break`).
fn eliminate_dead_branches(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    in_loop: bool,
    break_binds_to_loop: bool,
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;
    let total = original.len();
    let mut processed = 0usize;

    for (mut statement, span) in original.span_into_iter() {
        processed += 1;

        // Step 1: recurse into nested blocks first.  Leaf statements
        // are enumerated explicitly so a future naga release adding
        // a new block-bearing variant breaks the build instead of
        // silently bypassing recursion.
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
                    // A bare `Break` inside a case targets the SWITCH, so it
                    // stops binding to any enclosing loop; `Return` still
                    // exits the loop from inside a case.
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
                // `continuing` cannot hold a bare `Break` in valid IR;
                // `false` is the precise binding context either way.
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

        // Step 2: structural simplification - fold constant
        // If/Switch conditions, drain empty branches, unwrap
        // degenerate Switch, drop `break if true/false`, elide
        // empty nested Blocks.  See the per-arm comments below.
        match statement {
            // If with constant boolean condition
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => match resolve_to_literal(expressions, condition, const_lits) {
                // `resolve_to_literal` also folds an
                // `Expression::Constant(c)` whose init is a bool
                // literal, so a branch on a named `const X: bool = ...;`
                // is eliminated even before const_fold inlines it.
                Some(naga::Literal::Bool(true)) => {
                    // `accept` is taken; `reject` is dropped.  If `reject`
                    // produced a statement result (`Call`/`Atomic`/...), dropping
                    // it would orphan that result EXPRESSION (its producer is
                    // gone) - invalid IR the validator rejects, rolling the whole
                    // pass back every sweep.  Keep the `if` intact in that case:
                    // the branch is dead-but-valid, and leaving it avoids both
                    // the invalid IR and the wasted per-sweep revalidation.
                    //
                    // Also keep it for either loop-exit hazard - tint counts
                    // exits syntactically, without const-evaluating this
                    // condition, but WITH reachability sequencing:
                    // (a) the dropped block may carry the enclosing loop's
                    //     only exit (see `contains_return`);
                    // (b) splicing a kept block that never falls through
                    //     (e.g. `continue`) makes every statement after the
                    //     `if` unreachable to tint, un-crediting a trailing
                    //     `break`/`return` even though its text survives
                    //     (see `splice_loses_tint_loop_exit`).
                    if block_has_result_producer(&reject)
                        || (in_loop && contains_return(&reject))
                        || (break_binds_to_loop && contains_bare_break(&reject))
                        || (in_loop && splice_loses_tint_loop_exit(&accept, break_binds_to_loop))
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
                        splice_block(&mut rebuilt, accept);
                        changed += 1;
                    }
                }
                Some(naga::Literal::Bool(false)) => {
                    // `accept` is dropped: same orphaned-result and
                    // loop-exit hazards as the `true` arm above.
                    if block_has_result_producer(&accept)
                        || (in_loop && contains_return(&accept))
                        || (break_binds_to_loop && contains_bare_break(&accept))
                        || (in_loop && splice_loses_tint_loop_exit(&reject, break_binds_to_loop))
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
                        splice_block(&mut rebuilt, reject);
                        changed += 1;
                    }
                }
                _ => {
                    // After recursion, if both branches are empty the
                    // entire If is a no-op and can be discarded.
                    if accept.is_empty() && reject.is_empty() {
                        changed += 1;
                    } else if !reject.is_empty() && block_definitely_terminates(&accept) {
                        // Else-elision: `if c { return; } else { x; }` ->
                        // `if c { return; } x;` drops the `else{}`
                        // scaffolding.  Still fires when reject ALSO
                        // terminates with a different value-bearing
                        // return - the saved scaffolding outweighs the
                        // dead-but-conditional `if c { return v1; }`.
                        // Only symmetric value-less returns make the
                        // trailing return redundant; we don't collapse
                        // that case.
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

            // Switch with constant integer selector
            naga::Statement::Switch { selector, cases } => {
                if let Some(value) = resolve_switch_value(selector, expressions, const_lits) {
                    // A constant-selector collapse splices the matched case and
                    // DROPS the others (or all, on no-match).  Dropping a case
                    // that produced a statement result orphans that result
                    // expression - invalid IR that rolls the WHOLE pass back
                    // every sweep (also discarding unrelated dead_branch work).
                    // Keep the switch intact when any case body carries a result
                    // producer, mirroring the `if`-arm guard.  (The degenerate
                    // splice below emits the default body and drops only empty
                    // prefix cases, orphaning no result producer, so it needs no
                    // such guard.)
                    //
                    // Inside a loop, also keep it when any case holds a
                    // `Return` - a dropped case's Return can be the exit tint
                    // credits the loop with (bare `Break`s here target the
                    // switch itself, so only Returns matter).  Checking ALL
                    // cases (not just dropped ones) over-keeps when the
                    // matched chain has the Return; that shape is rare and
                    // the cost is a kept-but-dead wrapper.
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
                                // No match and no default -> the switch is a no-op.
                                changed += 1;
                            }
                            // Matched chain has a bare Break that targets the
                            // switch (splicing would mis-target it) or would
                            // shade the loop's trailing exit (see
                            // `splice_loses_tint_loop_exit`); keep the switch
                            // as-is.
                            Some(_) => {
                                rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                            }
                        }
                    }
                } else {
                    // Degenerate switch: a trailing `default` reached by every
                    // selector because each preceding case is an empty
                    // fall-through - naga lowers `case X[, Y..], default: {body}`
                    // (and a sole `default`) to empty `fall_through` prefix cases
                    // plus the `default` carrying the body, so the body always
                    // runs exactly once and the switch wrapper is dead.  Splice
                    // it, unless the body holds a bare Break targeting the switch
                    // (which would mis-target once the wrapper is gone).  A
                    // non-empty or non-fall-through prefix case means some
                    // selector runs a different (or no) body - then the switch is
                    // meaningful and is kept.
                    let degenerate = cases.split_last().is_some_and(|(last, prefix)| {
                        last.value == naga::SwitchValue::Default
                            && !last.fall_through
                            && prefix.iter().all(|c| c.fall_through && c.body.is_empty())
                    });
                    if degenerate && !contains_bare_break(&cases.last().unwrap().body) {
                        let body = cases.into_iter().next_back().unwrap().body;
                        splice_block(&mut rebuilt, body);
                        changed += 1;
                    }
                    // After recursion, if every case body is empty the
                    // entire Switch is a no-op and can be discarded.
                    else if cases.iter().all(|c| c.body.is_empty()) {
                        changed += 1;
                    } else {
                        rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                    }
                }
            }

            // Loop with constant break_if.  Routed through
            // `resolve_to_literal` so `break if NAMED_CONST` folds
            // without waiting for `const_fold` to inline the
            // constant - matches the If / Switch arms above.
            naga::Statement::Loop {
                body,
                continuing,
                break_if: Some(bi),
            } => match resolve_to_literal(expressions, bi, const_lits) {
                Some(naga::Literal::Bool(true)) => {
                    // `break if true` -> loop executes body + continuing once.
                    // Only safe when body/continuing have no bare Break/Continue
                    // that would target this loop (those would mis-target after
                    // unwrapping).
                    if !contains_bare_loop_control(&body)
                        && !contains_bare_loop_control(&continuing)
                    {
                        splice_block(&mut rebuilt, body);
                        splice_block(&mut rebuilt, continuing);
                        changed += 1;
                    } else {
                        // Unsafe to unwrap - keep the loop.
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
                // `break if false` never breaks at RUNTIME, but tint's
                // loop-exit analysis is syntactic (it does not
                // const-evaluate the condition): when this is the loop's
                // only lexical exit, dropping it turns tint-valid input
                // into tint-rejected output ("loop does not exit"), while
                // naga 30 validates the exit-less `loop{}`.  Drop it only
                // when the body proves another exit (the guard); otherwise
                // fall through to the keep-as-is arm below.  `continuing`
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

            // Drop nested `Block(inner)` if recursion emptied it;
            // mirrors the empty-If / empty-Switch eliminations
            // above.  Without this an upstream fold that drains
            // the body leaves a vacuous `{}` in the output.
            naga::Statement::Block(inner) if inner.is_empty() => {
                changed += 1;
            }

            // Non-empty Block: kept verbatim.  Explicit arm (vs
            // catch-all) so a reader can see the empty-Block
            // elision above doesn't accidentally swallow live blocks.
            naga::Statement::Block(inner) => {
                rebuilt.push(naga::Statement::Block(inner), span);
            }

            // Catch-all on a transform rewriter (not a walker): the
            // safe default for an unknown statement variant is
            // "preserve unchanged".  A future naga variant that
            // warrants a dedicated transform must be added above; if
            // none does, this keeps the pass sound.
            other => {
                rebuilt.push(other, span);
            }
        }

        // Step 3: if the last statement in `rebuilt` definitely terminates,
        // all remaining statements in the original block are dead.
        if rebuilt.last().is_some_and(definitely_terminates) {
            let dead = total - processed;
            changed += dead;
            break;
        }
    }

    *block = rebuilt;
    changed
}

/// Move all statements from `source` into `target`, preserving spans.
fn splice_block(target: &mut naga::Block, source: naga::Block) {
    for (stmt, sp) in source.span_into_iter() {
        target.push(stmt, sp);
    }
}

/// Returns `true` when `stmt` unconditionally terminates control flow in
/// its enclosing block (i.e. no path through `stmt` falls through to the
/// next statement).
fn definitely_terminates(stmt: &naga::Statement) -> bool {
    match stmt {
        // NOTE: `Kill` (discard) is deliberately NOT a terminator.  Under WGSL's
        // demote-to-helper semantics execution CONTINUES past `discard`, and
        // tint's control-flow analysis requires the statements after it to stay
        // present - a trailing `return` in a non-void function, or a loop's
        // `return`/`break` exit.  Treating `discard` as terminating strips that
        // reachable code and yields output tint rejects ("missing return at end
        // of function" / "loop does not exit").
        naga::Statement::Return { .. } | naga::Statement::Break | naga::Statement::Continue => true,
        naga::Statement::Block(inner) => block_definitely_terminates(inner),
        naga::Statement::If { accept, reject, .. } => {
            block_definitely_terminates(accept) && block_definitely_terminates(reject)
        }
        // A loop whose body always terminates *without* Break/Continue
        // (which would exit/restart the loop rather than the enclosing
        // scope) never falls through to the next statement.
        naga::Statement::Loop { body, .. } => {
            block_definitely_terminates(body) && !contains_bare_loop_control(body)
        }
        // A switch terminates the outer block iff every non-
        // fall-through case exits *beyond* the switch
        // (Return/Continue), a Default case exists, AND the
        // last case does NOT have `fall_through: true`.  A bare
        // Break exits the switch only (resumes after it), so it
        // does not qualify; see [`case_body_terminates_beyond_switch`].
        // The last-case fall-through guard catches IR rebuilt by
        // inlining / CSE - naga's frontend never emits this shape,
        // but the rebuilders could; without the guard those switches
        // mis-classify as terminating.
        naga::Statement::Switch { cases, .. } => {
            let last_falls_through = cases.last().is_some_and(|c| c.fall_through);
            cases
                .iter()
                .all(|c| c.fall_through || case_body_terminates_beyond_switch(&c.body))
                && cases.iter().any(|c| c.value == naga::SwitchValue::Default)
                && !last_falls_through
                // A reachable bare `break` ANYWHERE in a case body (not just as
                // the case's last statement) exits the switch and falls through
                // to the statement after it, so the switch does NOT terminate
                // the outer block.  `case_body_terminates_beyond_switch` only
                // inspects `block.last()` and so misses a break nested inside an
                // earlier `if` (e.g. `case 1: { if (c) { break; } return; }`);
                // guard against it here, mirroring the `!contains_bare_loop_control`
                // guard on the `Loop` arm above.
                && !cases.iter().any(|c| contains_bare_break(&c.body))
        }
        _ => false,
    }
}

/// Returns `true` when the last statement of `block` definitely terminates -
/// i.e. control never falls through to the statement that would follow it.
///
/// `pub(crate)` so the generator can reuse this exact divergence judgement: it
/// synthesises a trailing zero-value return only when the body provably never
/// falls through (the appended return is then provably dead), and sharing the
/// predicate keeps that soundness guard in lockstep with the return-stripping
/// done here.
pub(crate) fn block_definitely_terminates(block: &naga::Block) -> bool {
    block.last().is_some_and(definitely_terminates)
}
/// Returns `true` when a statement inside a switch case terminates control
/// flow *beyond* the switch itself (i.e., exits the function or enclosing
/// loop, not just the switch).
///
/// `Break` inside a switch case exits the switch and lets execution resume
/// after the switch statement - it does NOT prevent subsequent statements in
/// the outer block from running.  Therefore `Break` must NOT count here.
/// `Continue` inside a switch-case-body-inside-a-loop does jump past the
/// switch to the loop's continuing block, so it IS a beyond-switch terminator.
fn case_body_terminates_beyond_switch(block: &naga::Block) -> bool {
    block.last().is_some_and(|stmt| match stmt {
        // `Kill` (discard) is NOT a terminator - execution continues past it;
        // see the note in `definitely_terminates`.
        naga::Statement::Return { .. } | naga::Statement::Continue => true,
        // Break in a switch case exits the switch only.
        naga::Statement::Break => false,
        naga::Statement::Block(inner) => case_body_terminates_beyond_switch(inner),
        naga::Statement::If { accept, reject, .. } => {
            case_body_terminates_beyond_switch(accept) && case_body_terminates_beyond_switch(reject)
        }
        // A `loop` inside a switch case only falls through if it contains a
        // bare `break` (which exits the loop, not the switch).  When the loop
        // body definitely terminates *and* has no bare loop-control
        // statements, the only way out is Return, which exits beyond the
        // switch.  Matches the reasoning in `definitely_terminates`.
        naga::Statement::Loop { body, .. } => {
            block_definitely_terminates(body) && !contains_bare_loop_control(body)
        }
        // A nested switch only terminates beyond if all its cases do
        // so AND the last case does not fall through.  See the
        // identical reasoning on `definitely_terminates`'s Switch
        // arm: a `fall_through: true` last case means execution
        // falls past the (nonexistent) next case and out of the
        // switch, which is Break-equivalent and therefore does NOT
        // terminate beyond.  Without this gate, a nested switch
        // wrapped in another switch case (or a function body using
        // this predicate transitively) would mis-classify and let
        // upstream dead-code elimination drop statements that
        // execution can actually reach.
        naga::Statement::Switch { cases, .. } => {
            let last_falls_through = cases.last().is_some_and(|c| c.fall_through);
            cases
                .iter()
                .all(|c| c.fall_through || case_body_terminates_beyond_switch(&c.body))
                && cases.iter().any(|c| c.value == naga::SwitchValue::Default)
                && !last_falls_through
                // A bare `break` reachable in any nested case body exits THIS
                // nested switch and falls through to whatever follows it within
                // the outer case body, so the nested switch does not terminate
                // beyond.  See the identical guard in `definitely_terminates`.
                && !cases.iter().any(|c| contains_bare_break(&c.body))
        }
        _ => false,
    })
}

/// Try to resolve a switch selector expression to a concrete `SwitchValue`.
///
/// Consults `const_lits` so a selector that is `Expression::Constant(c)`
/// (a named constant whose init is an integer literal) is also resolved.
fn resolve_switch_value(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> Option<naga::SwitchValue> {
    match resolve_to_literal(expressions, handle, const_lits)? {
        naga::Literal::I32(v) => Some(naga::SwitchValue::I32(v)),
        naga::Literal::U32(v) => Some(naga::SwitchValue::U32(v)),
        _ => None,
    }
}

/// Find the body to execute for a given constant switch value.
///
/// Handles fall-through: in naga IR a case with `fall_through: true` and an
/// empty body represents a multi-value match (e.g. `case 1, 2:`).  We walk
/// forward from the matched case collecting bodies until we reach a case
/// with `fall_through: false`.
///
/// If no case matches and a `Default` case exists, its index is returned.
fn find_matching_case_index(cases: &[naga::SwitchCase], value: naga::SwitchValue) -> Option<usize> {
    cases.iter().position(|c| c.value == value).or_else(|| {
        cases
            .iter()
            .position(|c| c.value == naga::SwitchValue::Default)
    })
}

/// Collect the combined case body starting at `start_idx`, following
/// fall-through chains.  Consumes the `cases` vector.
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

/// Returns `true` if the combined case body starting at `start_idx`
/// (following fall-through) contains a bare `Break`.
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

/// Returns `true` if `block` contains a bare `Break` or `Continue` that
/// targets the immediately enclosing loop.
///
/// In naga IR `Break` exits the innermost `Loop` **or** `Switch`, while
/// `Continue` targets only the innermost `Loop`.  Therefore:
/// - do NOT recurse into `Loop` (captures both Break and Continue).
/// - do NOT recurse into `Switch` for `Break` (captured by Switch).
/// - DO recurse into `Switch` for `Continue` (Switch does not capture it).
fn contains_bare_loop_control(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Break | naga::Statement::Continue => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_loop_control(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_loop_control(accept) || contains_bare_loop_control(reject) {
                    return true;
                }
            }
            naga::Statement::Switch { cases, .. } => {
                // Switch captures Break but NOT Continue.  A `continue`
                // inside a switch case still targets the enclosing loop.
                for case in cases {
                    if contains_bare_continue(&case.body) {
                        return true;
                    }
                }
            }
            // Do NOT recurse into Loop - it captures both Break and Continue.
            naga::Statement::Loop { .. } => {}
            // Leaf statements that cannot carry a bare Break / Continue
            // and have no nested blocks.  Enumerated explicitly so a
            // future naga release adding a new block-bearing variant
            // breaks the build here instead of silently bypassing
            // the loop-control detection.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
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
    }
    false
}

/// Returns `true` if `block` contains a bare `Continue` targeting the
/// enclosing loop (not captured by any nested `Loop`).
///
/// Unlike [`contains_bare_loop_control`] this only looks for `Continue` and
/// therefore recurses into `Switch` (which does not capture `Continue`).
fn contains_bare_continue(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Continue => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_continue(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_continue(accept) || contains_bare_continue(reject) {
                    return true;
                }
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    if contains_bare_continue(&case.body) {
                        return true;
                    }
                }
            }
            // Loop captures Continue - do not recurse.
            naga::Statement::Loop { .. } => {}
            // Leaf statements that cannot carry a bare Continue and
            // have no nested blocks.  Enumerated explicitly so a
            // future naga release adding a new block-bearing variant
            // breaks the build here instead of silently missing a
            // Continue that targets the enclosing loop.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
            | naga::Statement::Break
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
    }
    false
}

/// Returns `true` if `block` contains a bare `Break` targeting the
/// immediately enclosing `Switch` or `Loop`.
///
/// Used to guard switch-case splicing: after removing the switch wrapper a
/// bare `Break` would mis-target the next enclosing construct.
fn contains_bare_break(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Break => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_break(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_break(accept) || contains_bare_break(reject) {
                    return true;
                }
            }
            // Both Loop and Switch capture Break - do not recurse.
            naga::Statement::Loop { .. } | naga::Statement::Switch { .. } => {}
            // Leaf statements that cannot carry a bare Break and have
            // no nested blocks.  Enumerated explicitly so a future
            // naga release adding a new block-bearing variant breaks
            // the build here instead of silently missing a Break
            // that targets the enclosing Switch / Loop.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
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
    }
    false
}

/// Returns `true` if `block` contains a `Return` at ANY nesting depth,
/// including inside nested loops and switches - a `Return` exits the function
/// (and therefore every enclosing loop) from anywhere.
///
/// Used by the loop-exit-preservation guards: tint's behavior analysis
/// requires every `loop` to exit (`break` targeting it, `break if`, or a
/// `Return` inside), and does NOT const-evaluate conditions - so a
/// `Return`/`Break` inside `if false { .. }` still counts for tint while
/// naga 30 validates a fully exit-less `loop{}` as legal.  Folding away
/// such a block can turn tint-valid input into tint-rejected output ("loop
/// does not exit"); the guards keep those dead-but-load-bearing blocks.
/// This scan is reachability-blind, over-approximating what tint credits -
/// safe for the KEEP direction it drives; [`tint_block_behavior`] is the
/// reachability-aware complement guarding the splice direction.  `Kill`
/// deliberately does NOT count: tint demotes `discard` to a
/// helper-invocation exit, not a loop exit (mirrors the
/// Kill-is-not-a-terminator rule in `definitely_terminates`).
fn contains_return(block: &naga::Block) -> bool {
    block.iter().any(|stmt| {
        // Unlike the bare Break/Continue scans above, recurse into every
        // nested block including loops - a Return exits the function from
        // any depth.
        matches!(stmt, naga::Statement::Return { .. }) || nested_blocks(stmt).any(contains_return)
    })
}

/// A construct's behavior set under tint's (WGSL-spec) analysis: which of
/// {Next, Return, Break, Continue} executing it can produce, WITHOUT
/// const-evaluating conditions.  `brk`/`cont` are the raw bare-Break /
/// bare-Continue behaviors at the construct's own level; `Switch` and `Loop`
/// absorb them structurally, so no binding context needs threading.  `Kill`
/// is `next` (demote-to-helper), mirroring `definitely_terminates`.
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
            // Entering case `i` executes its body, then - on
            // `fall_through` - the next case's, so effective behaviors
            // fold right-to-left; falling past the last case exits the
            // switch (Next).  The switch absorbs its cases' bare Breaks
            // into Next, and a selector matching no case (no Default)
            // also falls through.
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
            // The loop absorbs its body's Break/Continue; it falls
            // through only via a bare Break or a `break if` (valid IR
            // bars Break in `continuing`, but including it costs
            // nothing), and propagates only Return outward.
            let body = tint_block_behavior(body);
            let continuing = tint_block_behavior(continuing);
            TintBehavior {
                next: body.brk || continuing.brk || break_if.is_some(),
                ret: body.ret || continuing.ret,
                brk: false,
                cont: false,
            }
        }
        // Straight-line statements, including `Kill`: tint's
        // demote-to-helper semantics continue past `discard`.
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

/// `true` when splicing `spliced` in place of a const-folded wrapper would
/// deny the enclosing loop its tint-credited exit.  The wrapper's dead arm
/// kept `Next` alive under tint's no-const-eval sequencing (the input was
/// valid); the bare spliced content is safe only if it still falls through
/// (every following statement - e.g. a trailing `break` - stays reachable)
/// or itself carries an exit tint credits the loop with: a `Return`, or a
/// bare `Break` where one binds to the loop.  Reachability matters where
/// the `contains_*` scans are blind: a `return` sequenced behind a
/// `continue` is never credited, so it cannot rescue the splice.
fn splice_loses_tint_loop_exit(spliced: &naga::Block, break_binds_to_loop: bool) -> bool {
    let b = tint_block_behavior(spliced);
    !(b.next || b.ret || (break_binds_to_loop && b.brk))
}

/// [`splice_loses_tint_loop_exit`] over the exact statement sequence
/// `collect_case_body` would splice: the matched case plus its
/// fall-through successors, ending at the first non-fall-through case.
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

// Redundant else-store elimination.  naga's WGSL frontend lowers short-circuit
// `&&` into if-else chains:
//
//     var d: bool;                          // zero-initialized to false
//     if (a) { d = b; } else { d = false; }
//     if (d) { d = c; } else { d = false; }
//
// The else branches store the same value the variable already holds:
//   - Pattern A: condition is `load(d)` -> d must be false in the else branch,
//     so `d = false` is always a no-op.
//   - Pattern B: variable was zero-initialized (init: None) and has not been
//     modified, so `d = false` is a no-op.
//
// Similarly, `||` is lowered as (note the negated condition):
//
//     var d: bool;                          // zero-initialized to false
//     if (!a) { d = b; } else { d = true; }
//     if (!d) { d = c; } else { d = true; }
//
// The else-branches store `true` to a variable already known to be `true`:
//   - Pattern A': condition is `Load(d)` -> d must be true in the accept
//     branch, so `if (d) { d = true; }` is always a no-op.
//   - Pattern A'': condition is `!Load(d)` -> d must be true in the reject
//     branch, so `else { d = true; }` is always a no-op.  (This is the
//     form naga's WGSL frontend actually emits for `||`.)
//
// This pass detects and removes such redundant branches.

/// A value that a local variable is known to hold at a given program point.
#[derive(Clone, Debug, PartialEq)]
enum KnownValue {
    /// The type's zero/default value (matches any zero literal or `ZeroValue`).
    Zero,
    /// A specific literal value.
    Literal(naga::Literal),
}

// MARK: Redundant else-store elimination

/// Build a map from module-level `Constant` handles to their
/// `Literal` values.  Only constants whose init expression is
/// already a `Literal` are included; compound-expression inits fall
/// back to a runtime lookup.  The cache is populated once per run
/// so the else-store phase can read constant values without
/// re-borrowing `module.constants` in the inner loop.
fn build_const_literal_cache(
    module: &naga::Module,
) -> HashMap<naga::Handle<naga::Constant>, naga::Literal> {
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

/// Initialise known values for local variables.  Variables declared without
/// an explicit initializer are zero-initialised by WGSL, so they start as
/// `KnownValue::Zero`.  Variables with a literal init get
/// `KnownValue::Literal`.
fn init_known_values(
    locals: &naga::Arena<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> HashMap<naga::Handle<naga::LocalVariable>, KnownValue> {
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

/// Entry point: run redundant-store elimination on a single function.
/// Seed the known-values scoped map with every uninitialised local
/// that has an obvious default, then recurse into the function body.
/// Returns the change count so the caller can aggregate.
fn eliminate_redundant_else_stores_in_function(
    function: &mut naga::Function,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> usize {
    let mut known_values = ScopedMap::new();
    for (lh, kv) in init_known_values(&function.local_variables, &function.expressions, const_lits)
    {
        known_values.insert(lh, kv);
    }
    // Tracks, per local, the handle of its most recently materialised
    // `Load(LocalVariable)` that is still *fresh* - i.e. no store to that
    // local has happened since.  Condition narrowing is only sound when the
    // condition's `Load` is fresh; a stale forwarded load (e.g. `let t = d;
    // d = false; if t {...}`) reflects the value BEFORE the store, so
    // narrowing on it would clobber the correct post-store known value and
    // drop a live branch.  See `condition_load_is_fresh`.
    let mut fresh_loads = HashMap::new();
    eliminate_redundant_else_stores(
        &mut function.body,
        &function.expressions,
        const_lits,
        &mut known_values,
        &mut fresh_loads,
    )
}

/// `true` when `condition`'s underlying `Load(LocalVariable)` is still the
/// freshest load of that local recorded in `fresh_loads` - meaning no store
/// to the local has intervened since the load was materialised, so the
/// loaded value equals the local's current value.  Covers the bare
/// `Load(d)` (Pattern A) and `!Load(d)` (Pattern A'') condition shapes that
/// [`narrow_for_accept`] / [`narrow_for_reject`] act on; returns `false` for
/// any other shape (which those helpers ignore anyway).
fn condition_load_is_fresh(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    fresh_loads: &HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::Expression>>,
) -> bool {
    match &expressions[*condition] {
        naga::Expression::Load { pointer } => {
            if let naga::Expression::LocalVariable(d) = expressions[*pointer] {
                fresh_loads.get(&d) == Some(condition)
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
                fresh_loads.get(&d) == Some(inner)
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
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    fresh_loads: &mut HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::Expression>>,
) -> usize {
    let mut changed = 0usize;

    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // Per-branch scoped undo-log walk:
                //   1. Snapshot pre-if state.
                //   2. Apply accept-narrowing, recurse, roll back the
                //      recursion's mutations to the post-narrowing
                //      state, then run the redundancy check.
                //   3. Roll back to pre-if state, repeat for reject.
                //   4. Roll back fully, then permanently drop any
                //      locals modified in either branch (logged for an
                //      outer scope's rollback).
                // Condition narrowing is sound only when the condition's
                // `Load` reflects the local's CURRENT value (no store since
                // the load was materialised).  Decide this once, before
                // recursing, so an accept-branch store cannot perturb the
                // reject-phase decision.
                let cond_fresh = condition_load_is_fresh(condition, expressions, fresh_loads);
                let cp_pre_if = known_values.checkpoint();

                // Accept phase
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

                // Reject phase
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

                // Permanent update: conservatively remove any locals
                // modified in either branch.  Logged so outer scopes can
                // roll back if needed.  A branch may conditionally store a
                // local, so any prior load of it is no longer fresh.
                let mut modified = HashSet::new();
                collect_modified_locals(accept, expressions, &mut modified);
                collect_modified_locals(reject, expressions, &mut modified);
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::Emit(range) => {
                // Materialising a `Load(LocalVariable)` records it as the
                // freshest load of that local; the entry is invalidated by
                // any subsequent store to the local (below and in the branch
                // / call / atomic arms).  This is what lets condition
                // narrowing tell a fresh `if d {...}` from a stale forwarded
                // `let t = d; d = ...; if t {...}`.
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
                    // The store changes the local, so any earlier load of it
                    // is no longer fresh for narrowing.
                    fresh_loads.remove(&lh);
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
                } else if let Some(lh) = get_stored_local(expressions, *pointer) {
                    // Partial store - conservatively remove.
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
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
                let mut modified = HashSet::new();
                for case in cases.iter() {
                    collect_modified_locals(&case.body, expressions, &mut modified);
                }
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Variables modified inside the loop may not hold the same
                // value on subsequent iterations - strip them from the
                // known-value set before recursing.  The removals are
                // permanent (persisted past the loop); the body's interior
                // mutations are rolled back after the loop body is done.
                let mut modified = HashSet::new();
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
                // continuing is entered from every `continue` edge and body
                // fall-through, not sequentially after the body's tail, so a fact
                // the body set there (e.g. `known[d]=true` before a `break`) may
                // not hold on those edges - inheriting it would delete a live
                // continuing store as redundant.  Roll back to the post-wipe state
                // (a sound meet over entry edges).  `fresh_loads` needs no reset:
                // any Store clears its entry, so a surviving `fresh_loads[d]=h`
                // means `d` is unwritten after `h`, which naga emits on every path
                // into continuing, so `d==h` on all edges and its narrowing stays
                // sound.
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

            naga::Statement::Call { arguments, .. } => {
                // Pointer arguments may be written through by the callee.
                for &arg in arguments.iter() {
                    if let Some(lh) = get_stored_local(expressions, arg) {
                        known_values.remove(&lh);
                        fresh_loads.remove(&lh);
                    }
                }
            }

            naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::RayQuery { query, .. } => {
                if let Some(lh) = get_stored_local(expressions, *query) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(lh) = get_stored_local(expressions, *payload) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::CooperativeStore { data, .. } => {
                // `CooperativeStore { target, data }`: `data.pointer` is
                // the validator-required STORE-space write destination;
                // `target` is the matrix value being read.  Invalidate
                // the local rooted at the write side.
                if let Some(lh) = get_stored_local(expressions, data.pointer) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            // Statements that neither modify known-value tracking nor
            // contain nested blocks - enumerated explicitly so a
            // future naga release adding a new pointer-bearing variant
            // breaks the build here instead of silently leaving a
            // known_values entry stale across the new statement type.
            // (`Emit` has its own arm above: it materialises loads that
            // feed condition narrowing.)
            naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Return { .. }
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. }
            | naga::Statement::WorkGroupUniformLoad { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. } => {}
        }
    }

    changed
}

/// Apply "accept branch" condition-derived narrowing to `known_values`.
///
/// Pattern A: condition is `Load(cond_local)` -> in the accept branch,
/// `cond_local` must be `true`.
/// Pattern A'': condition is `!Load(cond_local)` -> in the accept branch,
/// `cond_local` must be zero/false.
///
/// All mutations go through `ScopedMap::insert` so the narrowing can be
/// rolled back via a caller-held checkpoint.
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

/// Apply "reject branch" condition-derived narrowing.  Mirror image of
/// [`narrow_for_accept`].
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

/// Return `true` when every statement in `block` is either an `Emit` (no
/// side-effect) or a `Store` whose value matches the known value of the
/// target local.  At least one such `Store` must be present.
fn block_only_has_redundant_known_stores(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &HashMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
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
            // A brace-wrapped sub-block (`if (c) { { d = false; } }`) is a
            // flat passthrough at the IR level: recurse so its redundant
            // stores are recognised too.  The recursive call's own
            // "at least one store" requirement makes an empty inner block
            // yield `false`, keeping the conservative behaviour for a
            // branch that does nothing observable.
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

/// Check whether an expression's value matches a `KnownValue`.
fn expr_matches_known(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    known: &KnownValue,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> bool {
    match known {
        KnownValue::Zero => is_zero_value(expressions, handle, const_lits),
        KnownValue::Literal(lit) => resolve_to_literal(expressions, handle, const_lits)
            .is_some_and(|resolved| literal_bit_eq(&resolved, lit)),
    }
}

/// Bit-exact literal equality.  Float variants (`F16`/`F32`/`F64`/
/// `AbstractFloat`) compare bit patterns, so `+0.0` and `-0.0` - IEEE-equal
/// but distinct bits - are NOT conflated, and two identical NaN payloads
/// ARE.  That is exactly the right test for "does storing `a` actually
/// change a value already known to be `b`": a store is redundant only when
/// it writes the identical bit pattern.  naga's derived `PartialEq` uses
/// IEEE `==` (where `+0.0 == -0.0`), which would let the redundant-store
/// elimination drop a sign-flipping store - a silent miscompile for
/// sign-of-zero-sensitive ops (`1.0/x`, `sign`, `copysign`, bit reinterpret).
/// Int/bool variants are already bit-exact under `==`, and mismatched
/// variants are correctly unequal.
fn literal_bit_eq(a: &naga::Literal, b: &naga::Literal) -> bool {
    use naga::Literal as L;
    match (a, b) {
        (L::F16(x), L::F16(y)) => x.to_bits() == y.to_bits(),
        (L::F32(x), L::F32(y)) => x.to_bits() == y.to_bits(),
        (L::F64(x), L::F64(y)) => x.to_bits() == y.to_bits(),
        (L::AbstractFloat(x), L::AbstractFloat(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Resolve an expression to a concrete `naga::Literal`, or `None`
/// when the value isn't known at compile time.
///
/// `Override` is omitted on purpose: its init is only a *default*
/// that the pipeline can replace at draw time, so folding through it
/// would let dead_branch erase code based on a value that changes
/// post-compile.  `Constant` resolves through `const_lits`, which
/// already filters out abstract literals upstream.
fn resolve_to_literal(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> Option<naga::Literal> {
    match &expressions[handle] {
        naga::Expression::Literal(lit) => Some(*lit),
        naga::Expression::Constant(c) => const_lits.get(c).copied(),
        _ => None,
    }
}

/// Check whether an expression evaluates to zero / false.
fn is_zero_value(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
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
