//! Statement-level WGSL emission: walks each function's [`naga::Block`] in
//! source order, reconstructs `for` loops from naga's `Loop` shape, and hosts
//! the expression use-detection behind the for-header preload safety checks.

use crate::error::Error;

use super::core::{FunctionCtx, Generator};
use super::syntax::{expression_kind, statement_kind};
use crate::handle_set::{HandleMap, HandleSet};

/// Rewrite a function-tail statement so its arms no longer end in a void
/// `return;` (`None` = unchanged).  Falling off an if-arm, a non-fall-through
/// switch case, or a nested block in tail position already ends the void
/// function, and naga's `ensure_block_returns` re-synthesises the returns on
/// every re-parse, so keeping them grows the text by one `else{return;}` per
/// round trip.  Only `Return { value: None }` is touched (a valued return
/// cannot sit in a void function's IR), loops are left alone (a return inside
/// a loop exits it, falling off the body does not), and fall-through cases
/// flow into the next case, so their tail is not the function tail.
fn elide_tail_void_returns(stmt: &naga::Statement) -> Option<naga::Statement> {
    fn rewrite_block(block: &naga::Block) -> Option<naga::Block> {
        let len = block.len();
        let last = block.iter().last()?;
        let replacement: Option<Option<naga::Statement>> = match last {
            naga::Statement::Return { value: None } => Some(None),
            other => elide_tail_void_returns(other).map(Some),
        };
        let replacement = replacement?;
        let mut rebuilt = naga::Block::with_capacity(len);
        for stmt in block.iter().take(len - 1) {
            rebuilt.push(stmt.clone(), naga::Span::UNDEFINED);
        }
        if let Some(stmt) = replacement {
            rebuilt.push(stmt, naga::Span::UNDEFINED);
        }
        Some(rebuilt)
    }

    match stmt {
        naga::Statement::If {
            condition,
            accept,
            reject,
        } => {
            let new_accept = rewrite_block(accept);
            let new_reject = rewrite_block(reject);
            if new_accept.is_none() && new_reject.is_none() {
                return None;
            }
            Some(naga::Statement::If {
                condition: *condition,
                accept: new_accept.unwrap_or_else(|| accept.clone()),
                reject: new_reject.unwrap_or_else(|| reject.clone()),
            })
        }
        naga::Statement::Switch { selector, cases } => {
            let rewrites: Vec<Option<naga::Block>> = cases
                .iter()
                .map(|c| {
                    if c.fall_through {
                        None
                    } else {
                        rewrite_block(&c.body)
                    }
                })
                .collect();
            if rewrites.iter().all(Option::is_none) {
                return None;
            }
            Some(naga::Statement::Switch {
                selector: *selector,
                cases: cases
                    .iter()
                    .zip(rewrites)
                    .map(|(c, rw)| naga::SwitchCase {
                        value: c.value,
                        body: rw.unwrap_or_else(|| c.body.clone()),
                        fall_through: c.fall_through,
                    })
                    .collect(),
            })
        }
        naga::Statement::Block(inner) => rewrite_block(inner).map(naga::Statement::Block),
        _ => None,
    }
}

/// Cap on the rendered nesting cost of one emitted expression, in tint parser
/// recursion frames.  tint's hard limit is 512 frames: a parenthesized / call
/// / index level costs ~4, a parenthesized unary level ~8, a flat binary chain
/// element ~1, and statement braces draw from the same budget; naga's own
/// frontend stops near 197 nesting levels.  [`Generator::render_depth`] weighs
/// every node at 4 (8 for `Unary`), so this cap keeps one expression at <= 64
/// call/paren or 32 unary levels, inside both limits with ~64 brace levels to
/// spare.  Single-use `let` inlining is the only unbounded depth source; chain
/// interiors bind at most one node past the cap.
const MAX_RENDER_DEPTH: u16 = 256;

/// Number of times emitting the tree rooted at `root` materialises `target`, a
/// for-loop preload result bound to inline `workgroupUniformLoad(&p)` text:
/// loop-body-dependent intermediates cannot be hoisted to a `let`, so every
/// distinct DAG path to `target` re-emits it.  Paths-to-`target` is a per-node
/// property, so memoising makes a shared sub-DAG count in linear time.  A path
/// through a short-circuit right operand runs zero or one times per
/// evaluation, never exactly once, so it counts as two: the caller wants
/// exactly one barrier per iteration and must decline it either way.
fn count_inline_emissions(
    root: naga::Handle<naga::Expression>,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    cache: &mut HandleMap<naga::Expression, usize>,
) -> usize {
    if root == target {
        return 1;
    }
    if let Some(&c) = cache.get(root) {
        return c;
    }
    let conditional = crate::passes::expr_util::short_circuit_rhs(&expressions[root]);
    // Saturating: a chain of shared diamonds has exponentially many paths,
    // and a wrapped count could read as the one the caller accepts.
    let mut total = 0usize;
    crate::passes::expr_util::visit_expression_children(&expressions[root], |child| {
        let paths = count_inline_emissions(child, target, expressions, cache);
        total = total.saturating_add(if Some(child) == conditional && paths > 0 {
            2
        } else {
            paths
        });
    });
    cache.insert(root, total);
    total
}

/// [`count_inline_emissions`] summed over the operands of a for-update
/// statement (`Store` / `Call` / `ImageStore`), one cache across operands.
fn count_update_stmt_emissions(
    stmt: &naga::Statement,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    cache: &mut HandleMap<naga::Expression, usize>,
) -> usize {
    let mut total = 0usize;
    let mut add = |h: naga::Handle<naga::Expression>,
                   cache: &mut HandleMap<naga::Expression, usize>| {
        total = total.saturating_add(count_inline_emissions(h, target, expressions, cache));
    };
    match stmt {
        naga::Statement::Store { pointer, value } => {
            add(*pointer, cache);
            add(*value, cache);
        }
        naga::Statement::Call { arguments, .. } => {
            for a in arguments {
                add(*a, cache);
            }
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
            ..
        } => {
            add(*image, cache);
            add(*coordinate, cache);
            if let Some(ai) = array_index {
                add(*ai, cache);
            }
            add(*value, cache);
        }
        _ => {}
    }
    total
}

/// Parsed for-loop shape.  The emitter, the preload-safety predicate and the
/// counter-`var` suppression decision all read one parse, so they cannot drift.
pub(super) struct ForLoopShape<'a> {
    /// `(pointer, result)` of each leading body `WorkGroupUniformLoad`.
    pub(super) guard_preloads: Vec<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    )>,
    /// Body indices of those preloads (excluded from the for-body).
    pub(super) guard_preload_stmt_indices: Vec<usize>,
    /// Body index of the if-break guard.
    pub(super) guard_idx: usize,
    pub(super) condition: naga::Handle<naga::Expression>,
    /// The guard is `if cond { break; }` (exit form; negate for `for`).
    pub(super) needs_negation: bool,
    /// `(pointer, result)` of each leading `continuing` `WorkGroupUniformLoad`.
    pub(super) update_preloads: Vec<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    )>,
    /// The single core `continuing` statement; its kind is not validated here.
    pub(super) update_stmt: Option<&'a naga::Statement>,
}

/// Parse a `Loop` into a [`ForLoopShape`], or `None` when it is not
/// for-convertible: a `break_if`, no leading if-break guard (after optional
/// `Emit` / `WorkGroupUniformLoad` preloads), or more than one core statement
/// in `continuing`.  Purely structural; every consumer layers its own policy
/// (preload safety, update-statement kind) on this one parse.
pub(super) fn parse_for_loop_shape<'a>(
    body: &'a naga::Block,
    continuing: &'a naga::Block,
    break_if: &Option<naga::Handle<naga::Expression>>,
) -> Option<ForLoopShape<'a>> {
    if break_if.is_some() {
        return None;
    }

    let body_stmts: Vec<_> = body.iter().collect();
    let mut guard_preloads = Vec::new();
    let mut guard_preload_stmt_indices = Vec::new();
    let mut guard_idx = 0;
    while guard_idx < body_stmts.len() {
        match body_stmts[guard_idx] {
            naga::Statement::Emit(_) => guard_idx += 1,
            naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                guard_preloads.push((*pointer, *result));
                guard_preload_stmt_indices.push(guard_idx);
                guard_idx += 1;
            }
            _ => break,
        }
    }
    if guard_idx >= body_stmts.len() {
        return None;
    }
    let (condition, needs_negation) = match body_stmts[guard_idx] {
        // `if cond {} else { break; }` - cond is the continue condition.
        naga::Statement::If {
            condition,
            accept,
            reject,
        } if accept.is_empty()
            && reject.len() == 1
            && matches!(reject.iter().next(), Some(naga::Statement::Break)) =>
        {
            (*condition, false)
        }
        // `if cond { break; }` - cond is the exit condition (negate for `for`).
        naga::Statement::If {
            condition,
            accept,
            reject,
        } if reject.is_empty()
            && accept.len() == 1
            && matches!(accept.iter().next(), Some(naga::Statement::Break)) =>
        {
            (*condition, true)
        }
        _ => return None,
    };

    let mut update_preloads = Vec::new();
    let mut update_stmt: Option<&naga::Statement> = None;
    for s in continuing.iter() {
        match s {
            naga::Statement::Emit(_) => continue,
            naga::Statement::WorkGroupUniformLoad { pointer, result } if update_stmt.is_none() => {
                update_preloads.push((*pointer, *result));
            }
            _ => {
                if update_stmt.is_some() {
                    return None;
                }
                update_stmt = Some(s);
            }
        }
    }

    Some(ForLoopShape {
        guard_preloads,
        guard_preload_stmt_indices,
        guard_idx,
        condition,
        needs_negation,
        update_preloads,
        update_stmt,
    })
}

/// `true` when rendering `shape`'s guard condition, update statement, or a
/// preload pointer inline in the `for(...)` header would exceed
/// [`MAX_RENDER_DEPTH`].  Header clauses bypass the `S::Emit` depth gate (the
/// for-conversion consumes their `Emit` ranges), so an over-deep chain must
/// stay on the plain `loop` path where the gate binds it; init `Emit`s precede
/// the loop and are gated normally.  Ignores binding state on purpose: the
/// counter-`var` suppression decision and the emitter both call this on the
/// same [`ForLoopShape`], so they cannot drift toward "suppressed + undeclared".
/// `true` when a header expression would render as the `T1(T2(x))` operand
/// of a unary that Dawn's Metal backend misparses
/// ([`super::const_hazard::msl_cast_ambiguity_operand`]).  The `for(...)`
/// header has nowhere to put the `let` that fixes it - hoisting one above
/// the loop would freeze a value the loop updates - so the shape declines
/// and the plain `loop` form binds it in the body.
pub(super) fn for_header_has_msl_cast_ambiguity(
    shape: &ForLoopShape,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut pending = vec![shape.condition];
    if let Some(stmt) = shape.update_stmt {
        crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
            pending.push(h)
        });
    }
    // Cone-sized, not arena-sized: a header is small and this runs per loop.
    let mut seen = std::collections::BTreeSet::new();
    while let Some(h) = pending.pop() {
        if !seen.insert(h) {
            continue;
        }
        if super::const_hazard::msl_cast_ambiguity_operand_in(expressions, &|_| false, h).is_some()
        {
            return true;
        }
        crate::passes::expr_util::visit_expression_children(&expressions[h], |c| pending.push(c));
    }
    false
}

pub(super) fn for_header_exceeds_depth_cap(
    shape: &ForLoopShape,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    // Cone-sized memo: a dense one would zero the whole arena per loop.
    fn depth(
        h: naga::Handle<naga::Expression>,
        expressions: &naga::Arena<naga::Expression>,
        memo: &mut std::collections::BTreeMap<naga::Handle<naga::Expression>, u16>,
    ) -> u16 {
        if let Some(&d) = memo.get(&h) {
            return d;
        }
        let mut children = Vec::new();
        crate::passes::expr_util::visit_expression_children(&expressions[h], |c| children.push(c));
        let mut max_child = 0u16;
        for child in children {
            max_child = max_child.max(depth(child, expressions, memo));
        }
        let weight = match expressions[h] {
            naga::Expression::Unary { .. } => 8,
            _ => 4,
        };
        let d = max_child.saturating_add(weight);
        memo.insert(h, d);
        d
    }
    let mut memo = std::collections::BTreeMap::new();
    let mut exceeded = depth(shape.condition, expressions, &mut memo) > MAX_RENDER_DEPTH;
    if let Some(stmt) = shape.update_stmt {
        crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
            exceeded |= depth(h, expressions, &mut memo) > MAX_RENDER_DEPTH;
        });
    }
    // A preload result is a childless leaf, so the condition / update walks
    // miss its POINTER, which the header renders inline as
    // `workgroupUniformLoad(&p)`; the `+4` is that wrapper.
    for &(pointer, _) in shape.guard_preloads.iter().chain(&shape.update_preloads) {
        exceeded |= depth(pointer, expressions, &mut memo).saturating_add(4) > MAX_RENDER_DEPTH;
    }
    exceeded
}

/// Single source of truth for whether a for-shaped loop's
/// `WorkGroupUniformLoad` preloads may be inlined into the `for(...)` header;
/// the emitter and the counter-`var` suppression decision both call it on the
/// same [`ForLoopShape`], so they never disagree (a disagreement would leave
/// the counter undeclared).
///
/// A preload carries a barrier and must execute exactly once per iteration,
/// yet in a `for` it is materialised only where its `result` is emitted (the
/// condition for a guard preload, the update statement for an update
/// preload).  Three hazards refuse the conversion, so plain-loop emission
/// keeps the preload as its own statement: a guard result used after the
/// guard (body tail or `continuing`) would lose its `let`; a result reused
/// within the condition / update would run the barrier twice; a result never
/// referenced (including a `continuing` with preloads but no core update
/// statement) would drop the barrier.  Each preload must count exactly one
/// emission.  The update clause relocates ahead of the body as well, so it
/// may not read a must-bind `Load` or a result some loop statement binds.
pub(super) fn for_loop_preload_inlining_is_safe(
    shape: &ForLoopShape,
    body: &naga::Block,
    continuing: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    must_bind_loads: &HandleSet<naga::Expression>,
) -> bool {
    // The update clause is relocated into the header, emitted BEFORE the body,
    // while a must-bind `Load` (its place is overwritten between its `Emit` and
    // its use) gets its `let` at its body `Emit`, AFTER the header: a relocated
    // clause referencing such a load would inline the bare post-write place,
    // the very miscompile the must-bind analysis prevents.  Two clauses
    // relocate: the update statement, and each `continuing` preload whose
    // POINTER (e.g. `&A[snap]`) is hoisted into the update slot as bare place
    // text.  A condition with no relocated barrier is safe: evaluated at
    // iteration top it coincides with the body-top load position, outer loads
    // it reads are `let`-bound before the loop, and back-edge writes are
    // covered by the must-bind loop pre-marking.  A guard preload relocated
    // into the condition IS a barrier, the guard-preload hazard.
    if !must_bind_loads.is_empty()
        && let Some(stmt) = shape.update_stmt
    {
        let update_hazard = stmt_references_any(stmt, must_bind_loads, expressions)
            || shape.update_preloads.iter().any(|&(pointer, _)| {
                let mut visited = Default::default();
                cone_intersects_set(pointer, must_bind_loads, expressions, &mut visited)
            });
        if update_hazard {
            return false;
        }
    }

    // The same relocation puts the update clause BEFORE every statement of
    // the loop, so a result one of them binds (`CallResult`, `AtomicResult`,
    // a body-side `workgroupUniformLoad`) has no name where the clause
    // renders and would print as naga's `_e<n>` placeholder.  Only the
    // `continuing` preloads relocate together with the clause.
    if let Some(stmt) = shape.update_stmt {
        let mut loop_bound = HandleSet::default();
        for block in [body, continuing] {
            crate::passes::expr_util::for_each_statement(block, &mut |s| {
                if let Some(result) = crate::passes::expr_util::statement_result(s) {
                    loop_bound.insert(result);
                }
            });
        }
        for &(_, result) in &shape.update_preloads {
            loop_bound.remove(result);
        }
        if !loop_bound.is_empty() && stmt_references_any(stmt, &loop_bound, expressions) {
            return false;
        }
    }

    if !shape.guard_preloads.is_empty() {
        let body_stmts: Vec<_> = body.iter().collect();
        // A guard preload inlined into the condition runs its barrier BEFORE
        // the body.  A must-bind load defined in the pre-guard region (only
        // `Emit` / `WorkGroupUniformLoad` statements) snapshots the PRE-barrier
        // value; for-reconstruction would re-emit its `let` in the body or
        // inline it into the condition after the barrier operand, reading the
        // POST-barrier value.  Plain-loop emission keeps the snapshot in place.
        if !must_bind_loads.is_empty()
            && body_stmts[..shape.guard_idx].iter().any(|s| {
                matches!(s, naga::Statement::Emit(range)
                    if range.clone().any(|h| must_bind_loads.contains(h)))
            })
        {
            return false;
        }
        let tail = if shape.guard_idx + 1 < body_stmts.len() {
            &body_stmts[shape.guard_idx + 1..]
        } else {
            &[]
        };
        let continuing_stmts: Vec<_> = continuing.iter().collect();
        let mut cache = HandleMap::default();
        for &(_, result) in &shape.guard_preloads {
            if stmts_use_expr(tail, result, expressions)
                || stmts_use_expr(&continuing_stmts, result, expressions)
            {
                return false;
            }
            cache.clear();
            // 0 drops the barrier, >1 duplicates it.
            if count_inline_emissions(shape.condition, result, expressions, &mut cache) != 1 {
                return false;
            }
        }
    }

    match shape.update_stmt {
        // No update clause to carry the preloads: they would drop with their barrier.
        None => {
            if !shape.update_preloads.is_empty() {
                return false;
            }
        }
        Some(stmt) => {
            let mut cache = HandleMap::default();
            for &(_, result) in &shape.update_preloads {
                cache.clear();
                if count_update_stmt_emissions(stmt, result, expressions, &mut cache) != 1 {
                    return false;
                }
            }
        }
    }

    true
}

// MARK: Expression use detection

/// `true` when any handle in `set` lies in the operand cone (transitive
/// children of every expression `stmt` reads) of `stmt`.  A result the
/// statement defines is not a read: a `Call` update clause whose value goes
/// unused still carries one.
fn stmt_references_any(
    stmt: &naga::Statement,
    set: &HandleSet<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut visited = Default::default();
    let mut found = false;
    crate::passes::expr_util::visit_statement_operands(stmt, false, &mut |root| {
        if !found {
            found = cone_intersects_set(root, set, expressions, &mut visited);
        }
    });
    found
}

/// `true` when `root` or a transitive child is in `set`; `visited` memoises
/// proven-absent nodes so a shared sub-DAG is walked once.
fn cone_intersects_set(
    root: naga::Handle<naga::Expression>,
    set: &HandleSet<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    visited: &mut HandleSet<naga::Expression>,
) -> bool {
    if set.contains(root) {
        return true;
    }
    if !visited.insert(root) {
        return false;
    }
    let mut found = false;
    crate::passes::expr_util::visit_expression_children(&expressions[root], |child| {
        if !found {
            found = cone_intersects_set(child, set, expressions, visited);
        }
    });
    found
}

/// `true` when `target` appears in the subtree rooted at `root` (inclusive).
/// A use almost always reaches a statement through an enclosing expression
/// (`w + 1`, `f(w)`), so flat handle equality on statement operands is unsound.
fn expr_subtree_contains(
    root: naga::Handle<naga::Expression>,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    // Proven-absent nodes (per fixed `target`); without the memo the path count
    // through a diamond-shaped DAG is super-linear.
    visited: &mut HandleSet<naga::Expression>,
) -> bool {
    if root == target {
        return true;
    }
    if visited.contains(root) {
        return false;
    }
    let mut found = false;
    crate::passes::expr_util::visit_expression_children(&expressions[root], |child| {
        if !found {
            found = expr_subtree_contains(child, target, expressions, visited);
        }
    });
    // Memoise only absence: a hit short-circuits the whole walk.
    if !found {
        visited.insert(root);
    }
    found
}

/// `true` when `stmt` references `target` in any operand position, nested
/// inside emitted expressions and control-flow blocks alike.  Built on the
/// exhaustive statement visitor, so a new naga variant forces an update there
/// instead of a silent false negative here.
fn stmt_uses_expr(
    stmt: &naga::Statement,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut visited = Default::default();
    let mut found = false;
    crate::passes::expr_util::visit_statement_expression_handles(
        stmt,
        /*include_emit_handles=*/ true,
        &mut |h| {
            found = found || expr_subtree_contains(h, target, expressions, &mut visited);
        },
    );
    found
}

fn stmts_use_expr(
    stmts: &[&naga::Statement],
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    stmts.iter().any(|s| stmt_uses_expr(s, target, expressions))
}

// MARK: Block emission

impl<'a> Generator<'a> {
    pub(super) fn generate_block(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, false, None)
    }

    /// Function-body variant: a trailing void `return;` is optional in WGSL.
    pub(super) fn generate_block_elide_trailing_return(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, true, None)
    }

    /// `continuing` variant: `break if` is the block's last statement and
    /// reads what the block bound (a const-hazard `let` for its condition),
    /// so it renders inside the block's scope, before that scope is released.
    fn generate_continuing(
        &mut self,
        block: &naga::Block,
        break_if: Option<naga::Handle<naga::Expression>>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, false, break_if)
    }

    fn generate_block_inner(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
        elide_trailing_void_return: bool,
        break_if: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let mut stmts: Vec<_> = block.iter().collect();
        let mut rewritten_tail: Option<naga::Statement> = None;
        if elide_trailing_void_return {
            if let Some(naga::Statement::Return { value: None }) = stmts.last() {
                stmts.pop();
            }
            if let Some(last) = stmts.last()
                && let Some(rewritten) = elide_tail_void_returns(last)
            {
                stmts.pop();
                rewritten_tail = Some(rewritten);
            }
        }
        // The rewritten tail must stay in the slice `emit_stmts` scans, or the
        // for-init absorption's later-use check misses a counter read only
        // inside the tail and scopes it into a for-header `var`.  A rewrite can
        // drain the statement (`if c { return; }` -> `if c {}`); conditions are
        // pure, so the shell is dead bytes and skipped as dead_branch would on
        // re-minify - unless its condition cone holds a stashed single-use
        // call, whose only emission site is this consumer; the kept `if f(){}`
        // converges to `f();` on re-minify.
        let vacuous = match &rewritten_tail {
            Some(naga::Statement::If { accept, reject, .. }) => {
                accept.is_empty() && reject.is_empty()
            }
            Some(naga::Statement::Switch { cases, .. }) => cases.iter().all(|c| c.body.is_empty()),
            Some(naga::Statement::Block(inner)) => inner.is_empty(),
            Some(_) | None => false,
        } && !tail_cone_has_stashed_call(&rewritten_tail, ctx);
        if let Some(rewritten) = &rewritten_tail
            && !vacuous
        {
            stmts.push(rewritten);
        }
        let hazard_mark = ctx.const_hazard_bindings.len();
        let result = self.emit_stmts(&stmts, ctx).and_then(|()| match break_if {
            Some(condition) => {
                self.push_indent();
                self.out.push_str("break if ");
                let text = self.emit_expr(condition, ctx)?;
                self.out.push_str(&text);
                self.out.push(';');
                self.push_newline();
                Ok(())
            }
            None => Ok(()),
        });
        release_hazard_scope(ctx, hazard_mark);
        result
    }

    /// `let`-bind a `const_hazard` operand here and route later uses through
    /// the name; the caller positions the line.
    fn emit_const_hazard_binding(
        &mut self,
        operand: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let (annotation, value) = self.const_hazard_binding_value(operand, ctx)?;
        let name = ctx.next_expr_name();
        self.out.push_str("let ");
        self.out.push_str(&name);
        if let Some(ty) = annotation {
            self.out.push(':');
            self.out.push_str(&ty);
        }
        self.push_assign();
        self.out.push_str(&value);
        self.out.push(';');
        ctx.expr_names.insert(operand, name);
        ctx.const_hazard_bindings.push(operand);
        Ok(())
    }
}

/// Hazard bindings are block-scoped `let`s over pre-emitted operands that later
/// blocks may use again, so every block-emission path pairs a mark with this.
fn release_hazard_scope(ctx: &mut FunctionCtx<'_, '_>, mark: usize) {
    for operand in ctx.const_hazard_bindings.drain(mark..) {
        ctx.expr_names.remove(operand);
    }
}

/// Whether a drained tail's condition / selector cone holds a call whose text
/// is stashed for inline emission: the shell is then that call's only route
/// into the output.
fn tail_cone_has_stashed_call(tail: &Option<naga::Statement>, ctx: &FunctionCtx<'_, '_>) -> bool {
    let root = match tail {
        Some(naga::Statement::If { condition, .. }) => *condition,
        Some(naga::Statement::Switch { selector, .. }) => *selector,
        _ => return false,
    };
    let mut stack = vec![root];
    let mut seen = HandleSet::default();
    while let Some(h) = stack.pop() {
        if !seen.insert(h) {
            continue;
        }
        if ctx.inlineable_calls.contains(h) {
            return true;
        }
        crate::passes::expr_util::visit_expression_children(&ctx.exprs[h], |child| {
            stack.push(child)
        });
    }
    false
}

impl<'a> Generator<'a> {
    /// Emit statements, reconstructing `for` loops from a `Loop` optionally
    /// preceded by a deferred-var `Store` that becomes the init clause.
    fn emit_stmts(
        &mut self,
        stmts: &[&naga::Statement],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let len = stmts.len();
        let mut i = 0;
        while i < len {
            let stmt = stmts[i];

            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = stmt
                && self.try_emit_for_loop(body, continuing, break_if, None, ctx)?
            {
                i += 1;
                continue;
            }
            if i + 1 < len
                && let naga::Statement::Store { pointer, value } = stmt
                && let naga::Expression::LocalVariable(lh) = ctx.exprs[*pointer]
                && ctx.deferred_vars[lh.index()]
            {
                // The for-init scopes `lh` to the loop; a later use would be out of scope.
                let safe = i + 2 >= len
                    || !super::module_emit::local_var_in_stmts(&stmts[i + 2..], lh, ctx.exprs);
                if safe
                    && let naga::Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    } = stmts[i + 1]
                    && self.try_emit_for_loop(
                        body,
                        continuing,
                        break_if,
                        Some((*pointer, *value)),
                        ctx,
                    )?
                {
                    ctx.deferred_vars[lh.index()] = false;
                    i += 2;
                    continue;
                }
            }

            let before = self.out.len();
            self.push_indent();
            let after_indent = self.out.len();
            self.generate_statement(stmt, ctx)?;
            if self.out.len() > after_indent {
                self.push_newline();
            } else {
                // Nothing emitted (e.g. a fully inlined `Emit`): drop the speculative indent.
                self.out.truncate(before);
            }
            i += 1;
        }
        Ok(())
    }

    fn generate_statement(
        &mut self,
        stmt: &naga::Statement,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        use naga::Statement as S;
        match stmt {
            S::Emit(range) => self.generate_emit_stmt(range, ctx)?,
            S::Block(block) => {
                self.out.push('{');
                self.push_newline();
                self.indent_depth += 1;
                self.generate_block(block, ctx)?;
                self.close_brace();
            }
            S::If {
                condition,
                accept,
                reject,
            } => {
                if accept.is_empty() && !reject.is_empty() {
                    self.out.push_str("if ");
                    self.out
                        .push_str(&self.emit_negated_condition(*condition, ctx)?);
                    self.open_brace();
                    self.generate_block(reject, ctx)?;
                    self.close_brace();
                } else {
                    self.out.push_str("if ");
                    self.out.push_str(&self.emit_expr(*condition, ctx)?);
                    self.open_brace();
                    self.generate_block(accept, ctx)?;
                    self.close_brace();
                    if !reject.is_empty() {
                        self.push_else();
                        self.open_brace();
                        self.generate_block(reject, ctx)?;
                        self.close_brace();
                    }
                }
            }
            S::Switch { selector, cases } => self.generate_switch_stmt(selector, cases, ctx)?,
            S::Loop {
                body,
                continuing,
                break_if,
            } => {
                self.out.push_str("loop");
                self.open_brace();
                self.generate_block(body, ctx)?;
                if !continuing.is_empty() || break_if.is_some() {
                    self.push_indent();
                    self.out.push_str("continuing");
                    self.open_brace();
                    self.generate_continuing(continuing, *break_if, ctx)?;
                    self.close_brace();
                    self.push_newline();
                }
                self.close_brace();
            }
            S::Break => self.out.push_str("break;"),
            S::Continue => self.out.push_str("continue;"),
            S::Return { value } => {
                self.out.push_str("return");
                if let Some(v) = value {
                    self.out.push(' ');
                    self.out.push_str(&self.emit_expr(*v, ctx)?);
                }
                self.out.push(';');
            }
            S::Kill => self.out.push_str("discard;"),
            S::ControlBarrier(flags) => {
                self.emit_barrier_calls(*flags);
            }
            S::MemoryBarrier(flags) => {
                self.emit_barrier_calls(*flags);
            }
            S::Store { pointer, value } => self.generate_store_stmt(pointer, value, ctx)?,
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                self.emit_image_store(*image, *coordinate, *array_index, *value, ctx)?;
                self.out.push(';');
            }
            S::Call {
                function,
                arguments,
                result,
            } => {
                let call = self.emit_call(*function, arguments, ctx)?;
                // A `CallResult` has no expression children (the arguments hang
                // off the statement), so the stashed text's true depth is
                // recorded here for the stash gate and `render_depth`; chains
                // accumulate through it.
                if let Some(h) = *result
                    && ctx.inlineable_calls.contains(h)
                {
                    let mut depth = 0u16;
                    for &arg in arguments {
                        depth = depth.max(self.render_depth(arg, ctx));
                    }
                    ctx.stashed_call_depth.insert(h, depth.saturating_add(4));
                }
                self.emit_call_result(&call, *result, ctx);
            }
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => self.generate_atomic_stmt(pointer, fun, value, result, ctx)?,
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                self.generate_image_atomic_stmt(image, coordinate, array_index, fun, value, ctx)?
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("workgroupUniformLoad(");
                self.out
                    .push_str(&self.emit_pointer_operand(*pointer, ctx)?);
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupBallot { result, predicate } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("subgroupBallot(");
                if let Some(pred) = predicate {
                    self.out.push_str(&self.emit_expr(*pred, ctx)?);
                }
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                let fn_name = subgroup_collective_name(*op, *collective_op)?;
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str(fn_name);
                self.out.push('(');
                let arg_hint = self.expr_scalar_hint(*result, ctx);
                self.out
                    .push_str(&self.emit_expr_with_scalar_hint(*argument, arg_hint, ctx)?);
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                let sep = self.comma_sep();
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                let (fn_name, index) = subgroup_gather_name_and_index(mode);
                self.out.push_str(fn_name);
                self.out.push('(');
                let arg_hint = self.expr_scalar_hint(*result, ctx);
                self.out
                    .push_str(&self.emit_expr_with_scalar_hint(*argument, arg_hint, ctx)?);
                if let Some(idx) = index {
                    self.out.push_str(sep);
                    let u32_hint = Some(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    });
                    self.out
                        .push_str(&self.emit_expr_with_scalar_hint(idx, u32_hint, ctx)?);
                }
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    let sep = self.comma_sep();
                    self.out.push_str("traceRay(");
                    self.out
                        .push_str(&self.emit_expr(*acceleration_structure, ctx)?);
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*descriptor, ctx)?);
                    self.out.push_str(sep);
                    self.out
                        .push_str(&self.emit_pointer_operand(*payload, ctx)?);
                    self.out.push_str(");");
                }
            },
            S::RayQuery { query, fun } => self.generate_ray_query_stmt(query, fun, ctx)?,
            // The generator cannot render `cooperative_matrix<...>`, and naga's
            // WGSL fallback emits the store without `enable
            // wgpu_cooperative_matrix;` (the type lives inline on the
            // load/store, never in `module.types`, so its enable-detection
            // misses it) and fails re-validation, so this error surfaces
            // cleanly.  `rename` reserves the `A`/`B`/`C` role names for when
            // this path round-trips.
            S::CooperativeStore { .. } => {
                return Err(Error::Emit(format!(
                    "cooperative-matrix store is not supported by nagami's \
                     generator in '{}', and naga's WGSL fallback cannot \
                     round-trip it either",
                    ctx.display_name,
                )));
            }
        }
        Ok(())
    }

    fn generate_emit_stmt(
        &mut self,
        range: &naga::Range<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let mut emitted_any = false;
        for h in range.clone() {
            // Bind a tint-rejected const-expression's operand first so the
            // consumer evaluates at runtime as the input did (`const_hazard`).
            if ctx.ref_counts[h.index()] > 0
                && let Some(operand) = super::const_hazard::creation_error_operand(
                    self.module,
                    ctx,
                    h,
                    &self.options.float_precision,
                )
                .or_else(|| super::const_hazard::msl_cast_ambiguity_operand(ctx, h))
                && !ctx.expr_names.contains_key(operand)
            {
                if emitted_any {
                    self.push_newline();
                    self.push_indent();
                }
                emitted_any = true;
                self.emit_const_hazard_binding(operand, ctx)?;
            }
            // A `Load` whose place is written between this `Emit` and a use
            // must be bound or it reads the post-write value; uniformity-pinned
            // expressions share the force-bind.  Both override the short-name
            // skip in `should_bind_expression` and the ref-count threshold.
            let force_bind = ctx.must_bind_loads.contains(h) || self.is_uniformity_pinned(h, ctx);
            // Both parsers bound nesting ([`MAX_RENDER_DEPTH`]): once inlining
            // `h` would render past the cap, bind it regardless of byte cost.
            // Restricted to bindable shapes (never pointers / statement-result
            // names) and live values; depth passes through unbindable nodes to
            // their first bindable ancestor.
            let depth_capped = !force_bind
                && ctx.ref_counts[h.index()] > 0
                && self.should_bind_expression(h, ctx)
                && self.render_depth(h, ctx) > MAX_RENDER_DEPTH;
            if !force_bind && !depth_capped && !self.should_bind_expression(h, ctx) {
                continue;
            }
            let refs = ctx.ref_counts[h.index()];
            if !force_bind && !depth_capped && refs < self.min_binding_refs(h, ctx) {
                continue;
            }
            if emitted_any {
                self.push_newline();
                self.push_indent();
            }
            emitted_any = true;
            let name = ctx.next_expr_name();
            let value = self.emit_expr_uncached(h, ctx)?;
            self.out.push_str("let ");
            self.out.push_str(&name);
            self.push_assign();
            self.out.push_str(&value);
            self.out.push(';');
            ctx.expr_names.insert(h, name);
        }
        Ok(())
    }

    fn generate_switch_stmt(
        &mut self,
        selector: &naga::Handle<naga::Expression>,
        cases: &[naga::SwitchCase],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.out.push_str("switch ");
        // Case labels carry a type suffix (`0u` for `SwitchValue::U32`), so an
        // uncached literal selector must keep its suffix too: the bare-literal
        // gate would emit `switch 0{case 0u:...}`, which naga rejects.
        if !ctx.expr_names.contains_key(selector)
            && let naga::Expression::Literal(lit) = ctx.exprs[*selector]
        {
            self.out.push_str(&super::syntax::literal_to_wgsl(
                lit,
                &self.options.float_precision,
            ));
        } else {
            self.out.push_str(&self.emit_expr(*selector, ctx)?);
        }
        self.open_brace();
        let mut new_case = true;
        for case in cases {
            if case.fall_through && !case.body.is_empty() {
                return Err(Error::Emit(format!(
                    "fall-through switch case with non-empty body \
                             in function '{}' is not representable in WGSL",
                    ctx.display_name,
                )));
            }
            if new_case {
                self.push_indent();
            }
            match case.value {
                naga::SwitchValue::I32(v) => {
                    if new_case {
                        self.out.push_str("case ");
                    }
                    self.out.push_str(&v.to_string());
                }
                naga::SwitchValue::U32(v) => {
                    if new_case {
                        self.out.push_str("case ");
                    }
                    self.out.push_str(&v.to_string());
                    self.out.push('u');
                }
                naga::SwitchValue::Default => {
                    if new_case && case.fall_through {
                        self.out.push_str("case ");
                    }
                    self.out.push_str("default");
                }
            }
            new_case = !case.fall_through;
            if case.fall_through {
                self.out.push_str(self.comma_sep());
            } else {
                self.open_brace();
                self.generate_block(&case.body, ctx)?;
                self.close_brace();
                self.push_newline();
            }
        }
        self.close_brace();
        Ok(())
    }

    fn generate_store_stmt(
        &mut self,
        pointer: &naga::Handle<naga::Expression>,
        value: &naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        if let Some(atomic_scalar) = self.atomic_scalar_for_expr(*pointer, ctx) {
            self.emit_atomic_store(*pointer, *value, atomic_scalar, ctx)?;
            self.out.push(';');
            return Ok(());
        }

        // Drop the no-op `p = p`, an UNCACHED `Load` of the same place: naga
        // re-lowers a folded `var d = a && b` into a temp-plus-copy that
        // coalescing collapses to `d = d`, which would grow the text by one
        // line per re-minify.  `uncached` is essential: `let t = p; ..p..;
        // p = t` restores a stale value and is not a no-op.
        if !ctx.expr_names.contains_key(value)
            && let naga::Expression::Load { pointer: loaded } = ctx.exprs[*value]
            && ptrs_structurally_equal(loaded, *pointer, ctx.exprs)
        {
            return Ok(());
        }

        let deferred_local = if let naga::Expression::LocalVariable(lh) = ctx.exprs[*pointer] {
            if ctx.deferred_vars[lh.index()] {
                Some(lh)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(lh) = deferred_local {
            ctx.deferred_vars[lh.index()] = false;
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&lh]);
            // The first store IS the declaration, so a zero value is WGSL's
            // zero-init: emit the shorter `:type` / `=0i` tail instead.
            if !ctx.expr_names.contains_key(value)
                && crate::passes::load_dedup::is_zero_init(ctx.exprs, *value)
            {
                self.emit_zero_init_tail(ctx.func.local_variables[lh].ty)?;
            } else {
                if ctx.needs_declared_type(lh, *value) {
                    self.push_colon();
                    self.out
                        .push_str(&self.type_ref(ctx.func.local_variables[lh].ty)?);
                }
                self.push_assign();
                // An uncached literal keeps its type suffix so the `var` gets the concrete type.
                if !ctx.expr_names.contains_key(value) {
                    if let naga::Expression::Literal(lit) = ctx.exprs[*value] {
                        self.out.push_str(&super::syntax::literal_to_wgsl(
                            lit,
                            &self.options.float_precision,
                        ));
                    } else {
                        self.out.push_str(&self.emit_expr(*value, ctx)?);
                    }
                } else {
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
                }
            }
            self.out.push(';');
        } else {
            self.emit_assignment(*pointer, *value, ctx)?;
            self.out.push(';');
        }
        Ok(())
    }

    /// `lvalue <op>= rhs` / `lvalue++` when `value` is `lvalue <op> rhs`, else
    /// `lvalue = value`; no terminator, so the for-update slot shares it.
    fn emit_assignment(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        // Decide the form before rendering the lvalue: the decision reads the
        // binding map, which rendering may extend.
        let compound = self.try_compound_assign(pointer, value, ctx);
        self.out.push_str(&self.emit_lvalue(pointer, ctx)?);
        if let Some((cop, other)) = compound {
            if let Some(inc) = self.try_increment(cop, other, value, ctx) {
                self.out.push_str(inc);
            } else {
                let sp = self.bin_op_sep();
                self.out.push_str(sp);
                self.out.push_str(cop);
                self.out.push_str(sp);
                self.out
                    .push_str(&self.emit_compound_assign_rhs(cop, other, ctx)?);
            }
        } else {
            self.push_assign();
            self.out.push_str(&self.emit_expr(value, ctx)?);
        }
        Ok(())
    }

    /// `textureStore(image, coordinate[, layer], value)` without the terminator.
    fn emit_image_store(
        &mut self,
        image: naga::Handle<naga::Expression>,
        coordinate: naga::Handle<naga::Expression>,
        array_index: Option<naga::Handle<naga::Expression>>,
        value: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let sep = self.comma_sep();
        self.out.push_str("textureStore(");
        self.out.push_str(&self.emit_expr(image, ctx)?);
        self.out.push_str(sep);
        self.out.push_str(&self.emit_expr(coordinate, ctx)?);
        if let Some(index) = array_index {
            self.out.push_str(sep);
            self.out.push_str(&self.emit_expr(index, ctx)?);
        }
        self.out.push_str(sep);
        self.out.push_str(&self.emit_expr(value, ctx)?);
        self.out.push(')');
        Ok(())
    }

    fn generate_atomic_stmt(
        &mut self,
        pointer: &naga::Handle<naga::Expression>,
        fun: &naga::AtomicFunction,
        value: &naga::Handle<naga::Expression>,
        result: &Option<naga::Handle<naga::Expression>>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let atomic_scalar = self.atomic_scalar_for_expr(*pointer, ctx);
        let sep = self.comma_sep();
        let fn_name = match *fun {
            naga::AtomicFunction::Add => "atomicAdd",
            naga::AtomicFunction::Subtract => "atomicSub",
            naga::AtomicFunction::And => "atomicAnd",
            naga::AtomicFunction::ExclusiveOr => "atomicXor",
            naga::AtomicFunction::InclusiveOr => "atomicOr",
            naga::AtomicFunction::Min => "atomicMin",
            naga::AtomicFunction::Max => "atomicMax",
            naga::AtomicFunction::Exchange { compare: None } => "atomicExchange",
            naga::AtomicFunction::Exchange {
                compare: Some(compare),
            } => {
                let mut call = String::from("atomicCompareExchangeWeak(&");
                call.push_str(&self.emit_expr(*pointer, ctx)?);
                call.push_str(sep);
                if let Some(scalar) = atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(compare, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(compare, ctx)?);
                }
                call.push_str(sep);
                if let Some(scalar) = atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(*value, ctx)?);
                }
                call.push(')');
                self.emit_call_result(&call, *result, ctx);
                return Ok(());
            }
        };
        let mut call = String::from(fn_name);
        call.push('(');
        call.push_str(&self.emit_pointer_operand(*pointer, ctx)?);
        call.push_str(sep);
        if let Some(scalar) = atomic_scalar {
            call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
        } else {
            call.push_str(&self.emit_expr(*value, ctx)?);
        }
        call.push(')');
        self.emit_call_result(&call, *result, ctx);
        Ok(())
    }

    fn generate_image_atomic_stmt(
        &mut self,
        image: &naga::Handle<naga::Expression>,
        coordinate: &naga::Handle<naga::Expression>,
        array_index: &Option<naga::Handle<naga::Expression>>,
        fun: &naga::AtomicFunction,
        value: &naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let image_atomic_scalar = self.image_atomic_scalar_for_expr(*image, ctx);
        let fn_name = match *fun {
            naga::AtomicFunction::Add => "textureAtomicAdd",
            naga::AtomicFunction::Subtract => "textureAtomicSub",
            naga::AtomicFunction::And => "textureAtomicAnd",
            naga::AtomicFunction::ExclusiveOr => "textureAtomicXor",
            naga::AtomicFunction::InclusiveOr => "textureAtomicOr",
            naga::AtomicFunction::Min => "textureAtomicMin",
            naga::AtomicFunction::Max => "textureAtomicMax",
            naga::AtomicFunction::Exchange { compare: None } => "textureAtomicExchange",
            naga::AtomicFunction::Exchange {
                compare: Some(compare),
            } => {
                let sep = self.comma_sep();
                let mut call = String::from("textureAtomicCompareExchangeWeak(");
                call.push_str(&self.emit_expr(*image, ctx)?);
                call.push_str(sep);
                call.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    call.push_str(sep);
                    call.push_str(&self.emit_expr(*index, ctx)?);
                }
                call.push_str(sep);
                if let Some(scalar) = image_atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(compare, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(compare, ctx)?);
                }
                call.push_str(sep);
                if let Some(scalar) = image_atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(*value, ctx)?);
                }
                call.push(')');
                self.out.push_str(&call);
                self.out.push(';');
                return Ok(());
            }
        };
        let sep = self.comma_sep();
        self.out.push_str(fn_name);
        self.out.push('(');
        self.out.push_str(&self.emit_expr(*image, ctx)?);
        self.out.push_str(sep);
        self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
        if let Some(index) = array_index {
            self.out.push_str(sep);
            self.out.push_str(&self.emit_expr(*index, ctx)?);
        }
        self.out.push_str(sep);
        if let Some(scalar) = image_atomic_scalar {
            self.out
                .push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
        } else {
            self.out.push_str(&self.emit_expr(*value, ctx)?);
        }
        self.out.push_str(");");
        Ok(())
    }

    fn generate_ray_query_stmt(
        &mut self,
        query: &naga::Handle<naga::Expression>,
        fun: &naga::RayQueryFunction,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        // naga's validator restricts `query` to a `LocalVariable`
        // (`InvalidRayQueryExpression`), so its text is a bare identifier and
        // `&` always forms the pointer operand; should naga relax this, pointer
        // arguments must emit verbatim without `&`.  The exhaustive match makes
        // a new naga ray-query variant a compile error rather than a silent
        // emission gap.
        debug_assert!(
            matches!(ctx.exprs[*query], naga::Expression::LocalVariable(_)),
            "Statement::RayQuery.query must be a LocalVariable per \
                     naga's validator; got {}",
            expression_kind(&ctx.exprs[*query])
        );
        let query_text = self.emit_expr(*query, ctx)?;
        match fun {
            naga::RayQueryFunction::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                let sep = self.comma_sep();
                self.out.push_str("rayQueryInitialize(&");
                self.out.push_str(&query_text);
                self.out.push_str(sep);
                self.out
                    .push_str(&self.emit_expr(*acceleration_structure, ctx)?);
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*descriptor, ctx)?);
                self.out.push_str(");");
            }
            naga::RayQueryFunction::Proceed { result } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("rayQueryProceed(&");
                self.out.push_str(&query_text);
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                let sep = self.comma_sep();
                self.out.push_str("rayQueryGenerateIntersection(&");
                self.out.push_str(&query_text);
                self.out.push_str(sep);
                // naga's lowerer propagates no expected type into this argument,
                // so a bare literal would concretize to i32 ("Hit distance must
                // be an f32").
                self.out.push_str(&self.emit_expr_with_scalar_hint(
                    *hit_t,
                    Some(naga::Scalar::F32),
                    ctx,
                )?);
                self.out.push_str(");");
            }
            naga::RayQueryFunction::ConfirmIntersection => {
                self.out.push_str("rayQueryConfirmIntersection(&");
                self.out.push_str(&query_text);
                self.out.push_str(");");
            }
            naga::RayQueryFunction::Terminate => {
                self.out.push_str("rayQueryTerminate(&");
                self.out.push_str(&query_text);
                self.out.push_str(");");
            }
        }
        Ok(())
    }

    fn emit_barrier_calls(&mut self, flags: naga::Barrier) {
        let mut emitted = false;
        let barriers: &[(naga::Barrier, &str)] = &[
            (naga::Barrier::WORK_GROUP, "workgroupBarrier();"),
            (naga::Barrier::STORAGE, "storageBarrier();"),
            (naga::Barrier::TEXTURE, "textureBarrier();"),
            (naga::Barrier::SUB_GROUP, "subgroupBarrier();"),
        ];
        for &(flag, call) in barriers {
            if flags.contains(flag) {
                if emitted {
                    self.push_newline();
                    self.push_indent();
                }
                self.out.push_str(call);
                emitted = true;
            }
        }
    }

    /// Force-bound at its naga-placed `Emit` so single-use inlining cannot
    /// sink a [`crate::passes::expr_util::is_uniformity_constrained_expr`]
    /// into a possibly non-uniform branch: naga's validator does not enforce
    /// it, tint/Dawn reject it.
    fn is_uniformity_pinned(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        crate::passes::expr_util::is_uniformity_constrained_expr(&ctx.exprs[h])
    }

    /// Rendered nesting cost of `h` if inlined at its use site now, in the frame
    /// units of [`MAX_RENDER_DEPTH`]: a bound expression (per `expr_names`)
    /// renders as an identifier and costs one leaf; anything else adds its
    /// weight - 8 for `Unary` (a parenthesized `-(...)` level costs tint double),
    /// 4 otherwise - over its deepest child.  Memoised per function so a
    /// whole-arena sweep is linear; unmemoised the walk is 2^depth.
    fn render_depth(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> u16 {
        // Stashed call text sits in `expr_names` like a binding but renders at
        // its recorded full depth, not as a leaf; checked first so a chain of
        // stashed calls cannot hide from the cap.
        if let Some(&stashed) = ctx.stashed_call_depth.get(h) {
            return stashed;
        }
        if ctx.expr_names.contains_key(h) {
            return 4;
        }
        let memoized = ctx.render_depth_memo[h.index()];
        if memoized != 0 {
            return memoized;
        }
        // The visitor borrows the arena while the recursion needs `ctx` mutably.
        let mut children = Vec::new();
        crate::passes::expr_util::visit_expression_children(&ctx.exprs[h], |c| children.push(c));
        let mut max_child = 0u16;
        for child in children {
            max_child = max_child.max(self.render_depth(child, ctx));
        }
        let weight = match ctx.exprs[h] {
            naga::Expression::Unary { .. } => 8,
            _ => 4,
        };
        let depth = max_child.saturating_add(weight);
        ctx.render_depth_memo[h.index()] = depth;
        depth
    }

    fn should_bind_expression(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::Expression as E;
        match &ctx.exprs[h] {
            E::CallResult(_)
            | E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult => return false,
            // A bare local/global load already renders as a short name; a `let`
            // over it saves nothing.
            E::Load { pointer } => {
                if matches!(
                    ctx.exprs[*pointer],
                    E::LocalVariable(_) | E::GlobalVariable(_)
                ) {
                    return false;
                }
            }
            _ => {}
        }

        let inner = ctx.ty(h).inner_with(&self.module.types);
        !matches!(
            inner,
            naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. }
        )
    }

    /// Minimum reference count before a `let` pays for itself: `let X=EXPR;`
    /// costs about `len + 7` bytes and saves `len - 1` per use, so binding wins
    /// when `refs > (len + 7) / (len - 1)`.
    fn min_binding_refs(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> usize {
        use naga::Expression as E;
        match &ctx.exprs[h] {
            // `-x` ~= 2 chars -> need 10+ refs to break even
            E::Unary { expr: child, .. } => {
                if self.expr_resolves_to_name(*child, ctx) {
                    return 10;
                }
                2
            }
            // `v.x` on a vector ~= 3 chars -> need 6+ refs
            E::AccessIndex { base, .. } => {
                if self.expr_resolves_to_name(*base, ctx) {
                    let inner = ctx.ty(*base).inner_with(&self.module.types);
                    if matches!(inner, naga::TypeInner::Vector { .. }) {
                        return 6;
                    }
                }
                2
            }
            // `a*b` with both operands named ~= 3 chars -> need 6+ refs
            E::Binary { left, right, .. } => {
                if self.expr_resolves_to_name(*left, ctx) && self.expr_resolves_to_name(*right, ctx)
                {
                    return 6;
                }
                2
            }
            _ => 2,
        }
    }

    /// Whether `h` renders as a short name (1-2 chars) rather than a
    /// sub-expression.  A stashed single-use call sits in `expr_names` as its
    /// whole call text and is NOT one: a wrapper priced as cheap over it would
    /// re-render per use, and each rendering runs the call again - an impure
    /// callee's write repeated, a pure one's texture sample multiplied.
    fn expr_resolves_to_name(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        if ctx.expr_names.contains_key(h) {
            return !ctx.inlineable_calls.contains(h);
        }
        use naga::Expression as E;
        match &ctx.exprs[h] {
            E::FunctionArgument(_) | E::Constant(_) | E::Override(_) => true,
            E::Load { pointer } => matches!(
                ctx.exprs[*pointer],
                E::LocalVariable(_) | E::GlobalVariable(_)
            ),
            _ => false,
        }
    }

    /// Emit a `Loop` as `for(init; cond; update)`.  `Ok(false)` means the loop
    /// is not for-convertible or header inlining is unsafe; nothing has been
    /// written, so the caller falls back to plain `loop` emission.
    fn try_emit_for_loop(
        &mut self,
        body: &naga::Block,
        continuing: &naga::Block,
        break_if: &Option<naga::Handle<naga::Expression>>,
        init: Option<(
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        )>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<bool, Error> {
        if break_if.is_some() {
            return Ok(false);
        }

        let Some(shape) = parse_for_loop_shape(body, continuing, break_if) else {
            return Ok(false);
        };

        // Only Store / Call / ImageStore fit the update slot.  Every bail-out
        // precedes the first write so the fallback starts clean.
        if let Some(stmt) = shape.update_stmt
            && !matches!(
                stmt,
                naga::Statement::Store { .. }
                    | naga::Statement::Call { .. }
                    | naga::Statement::ImageStore { .. }
            )
        {
            return Ok(false);
        }

        if !for_loop_preload_inlining_is_safe(
            &shape,
            body,
            continuing,
            ctx.exprs,
            &ctx.must_bind_loads,
        ) {
            return Ok(false);
        }

        if for_header_exceeds_depth_cap(&shape, ctx.exprs)
            || for_header_has_msl_cast_ambiguity(&shape, ctx.exprs)
        {
            return Ok(false);
        }

        let body_stmts: Vec<_> = body.iter().collect();
        let ForLoopShape {
            guard_preloads,
            guard_preload_stmt_indices,
            guard_idx,
            condition,
            needs_negation,
            update_preloads,
            update_stmt,
        } = shape;

        // Without an external init, absorb a for_loop_var counter; one with no
        // init value gets a bare `var name:type` in the init clause.
        let mut decl_only_local: Option<naga::Handle<naga::LocalVariable>> = None;

        let init = if init.is_some() {
            // An external init occupies the init slot, which would leave a
            // suppressed-`var` counter undeclared: bail so the Store emits as a
            // statement and the plain `Loop` path absorbs the counter.
            if let Some(naga::Statement::Store { pointer, .. }) = update_stmt
                && let naga::Expression::LocalVariable(lh) = ctx.exprs[*pointer]
                && ctx.for_loop_vars[lh.index()]
            {
                return Ok(false);
            }
            init
        } else if let Some(naga::Statement::Store { pointer, .. }) = update_stmt {
            if let naga::Expression::LocalVariable(lh) = ctx.exprs[*pointer] {
                if ctx.for_loop_vars[lh.index()] {
                    // Consumed: never absorbed twice.
                    ctx.for_loop_vars[lh.index()] = false;
                    if let Some(init_handle) = ctx.func.local_variables[lh].init {
                        Some((*pointer, init_handle))
                    } else {
                        decl_only_local = Some(lh);
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Header expressions render inline, bypassing `generate_emit_stmt`:
        // bind their hazards before the loop, in the enclosing scope.
        let mut pending = vec![condition];
        if let Some(stmt) = update_stmt {
            crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
                pending.push(h)
            });
        }
        let mut cone = std::collections::BTreeSet::new();
        while let Some(h) = pending.pop() {
            if cone.insert(h) {
                crate::passes::expr_util::visit_expression_children(&ctx.exprs[h], |c| {
                    pending.push(c)
                });
            }
        }
        for h in cone {
            if let Some(operand) = super::const_hazard::creation_error_operand(
                self.module,
                ctx,
                h,
                &self.options.float_precision,
            ) && !ctx.expr_names.contains_key(operand)
            {
                self.push_indent();
                self.emit_const_hazard_binding(operand, ctx)?;
                self.push_newline();
            }
        }

        self.push_indent();
        self.push_for_open();

        // Init clause.
        let mut deferred_for_init_local: Option<naga::Handle<naga::LocalVariable>> = None;
        if let Some((pointer, value)) = init {
            if let naga::Expression::LocalVariable(lh) = ctx.exprs[pointer] {
                self.out.push_str("var ");
                self.out.push_str(&ctx.local_names[&lh]);
                if ctx.needs_declared_type(lh, value) {
                    self.push_colon();
                    self.out
                        .push_str(&self.type_ref(ctx.func.local_variables[lh].ty)?);
                }
                self.push_assign();
                if !ctx.expr_names.contains_key(value) {
                    if let naga::Expression::Literal(lit) = ctx.exprs[value] {
                        self.out.push_str(&super::syntax::literal_to_wgsl(
                            lit,
                            &self.options.float_precision,
                        ));
                    } else {
                        self.out.push_str(&self.emit_expr(value, ctx)?);
                    }
                } else {
                    self.out.push_str(&self.emit_expr(value, ctx)?);
                }
                // The for-init declares `lh`, so a body Store to it (a counter
                // update that stayed in the body) must assign rather than
                // re-declare a second `var` that shadows the for-init copy and
                // freezes the counter; clearing here is what body emission sees.
                deferred_for_init_local = Some(lh);
            } else {
                self.out.push_str(&self.emit_lvalue(pointer, ctx)?);
                self.push_assign();
                self.out.push_str(&self.emit_expr(value, ctx)?);
            }
        } else if let Some(lh) = decl_only_local {
            // No explicit init (WGSL zero-initialises): the shorter of
            // `var name:type` / `var name=0i`.
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&lh]);
            let ty = ctx.func.local_variables[lh].ty;
            self.emit_zero_init_tail(ty)?;
            deferred_for_init_local = Some(lh);
        }
        if let Some(lh) = deferred_for_init_local {
            ctx.deferred_vars[lh.index()] = false;
        }
        self.push_for_sep();

        // Condition clause: rendered inline, its `Emit`s are never processed.
        let mut preload_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
            Vec::new();
        for (pointer, result) in &guard_preloads {
            let mut preload = String::from("workgroupUniformLoad(");
            preload.push_str(&self.emit_pointer_operand(*pointer, ctx)?);
            preload.push(')');
            let old = ctx.expr_names.insert(*result, preload);
            preload_old_bindings.push((*result, old));
        }

        if needs_negation {
            self.out
                .push_str(&self.emit_negated_condition(condition, ctx)?);
        } else {
            self.out.push_str(&self.emit_expr(condition, ctx)?);
        }

        for (result, old) in preload_old_bindings {
            if let Some(name) = old {
                ctx.expr_names.insert(result, name);
            } else {
                ctx.expr_names.remove(result);
            }
        }

        self.push_for_sep();

        // Update clause.
        if let Some(stmt) = update_stmt {
            let mut update_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
                Vec::new();
            for (pointer, result) in &update_preloads {
                let mut preload = String::from("workgroupUniformLoad(");
                preload.push_str(&self.emit_pointer_operand(*pointer, ctx)?);
                preload.push(')');
                let old = ctx.expr_names.insert(*result, preload);
                update_old_bindings.push((*result, old));
            }
            self.emit_statement_inline(stmt, ctx)?;
            for (result, old) in update_old_bindings {
                if let Some(name) = old {
                    ctx.expr_names.insert(result, name);
                } else {
                    ctx.expr_names.remove(result);
                }
            }
        }

        self.out.push(')');
        self.open_brace();

        // The header rendered the guard / update cones inline, bypassing the
        // `Emit` -> `let` path; decrement their children's ref counts so values
        // shared with the body (e.g. load_dedup'd loads) are inlined rather
        // than bound to dead `let`s.
        for s in &body_stmts[..guard_idx] {
            if let naga::Statement::Emit(range) = s {
                for h in range.clone() {
                    crate::passes::expr_util::visit_expression_children(&ctx.exprs[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }
        for stmt in continuing.iter() {
            if let naga::Statement::Emit(range) = stmt {
                for h in range.clone() {
                    crate::passes::expr_util::visit_expression_children(&ctx.exprs[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }

        let remaining: Vec<_> = body_stmts
            .iter()
            .enumerate()
            .filter_map(|(j, s)| {
                if j == guard_idx || guard_preload_stmt_indices.contains(&j) {
                    None
                } else {
                    Some(*s)
                }
            })
            .collect();

        // naga wraps the user's for-body in a `Statement::Block`; when it is
        // the only non-`Emit` statement, emit its contents directly to avoid
        // `for(...){{body}}`.
        let mut non_emit: Vec<usize> = Vec::new();
        for (k, s) in remaining.iter().enumerate() {
            if !matches!(s, naga::Statement::Emit(_)) {
                non_emit.push(k);
            }
        }
        // The body is its own scope; leading Emits bind here too.
        let hazard_mark = ctx.const_hazard_bindings.len();
        if non_emit.len() == 1
            && let naga::Statement::Block(inner) = remaining[non_emit[0]]
        {
            for (k, s) in remaining.iter().enumerate() {
                if k == non_emit[0] {
                    continue;
                }
                let before = self.out.len();
                self.push_indent();
                let after_indent = self.out.len();
                self.generate_statement(s, ctx)?;
                if self.out.len() > after_indent {
                    self.push_newline();
                } else {
                    self.out.truncate(before);
                }
            }
            self.generate_block(inner, ctx)?;
            release_hazard_scope(ctx, hazard_mark);
            self.close_brace();
            self.push_newline();
            return Ok(true);
        }

        self.emit_stmts(&remaining, ctx)?;
        release_hazard_scope(ctx, hazard_mark);

        self.close_brace();
        self.push_newline();
        Ok(true)
    }

    /// `atomicStore(&p, v)` without terminator, shared by the statement and
    /// for-update `Store` paths: assigning a scalar straight to `atomic<T>` is a
    /// type error strict consumers reject.
    fn emit_atomic_store(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        atomic_scalar: naga::Scalar,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let sep = self.comma_sep();
        self.out.push_str("atomicStore(");
        self.out.push_str(&self.emit_pointer_operand(pointer, ctx)?);
        self.out.push_str(sep);
        self.out
            .push_str(&self.emit_expr_for_atomic(value, atomic_scalar, ctx)?);
        self.out.push(')');
        Ok(())
    }

    /// Pointer operand of the atomic builtins, `workgroupUniformLoad`,
    /// `arrayLength` and `traceRay`: a `var`-rooted place needs `&`, a pointer
    /// parameter is one already and `&p` on it is a type error.
    pub(super) fn emit_pointer_operand(
        &self,
        pointer: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if self.pointer_is_ptr_value(pointer, ctx) {
            self.emit_expr(pointer, ctx)
        } else {
            Ok(format!("&{}", self.emit_lvalue(pointer, ctx)?))
        }
    }

    /// `true` when `pointer` is already a `ptr<>` value (a `ptr<...>` parameter).
    pub(super) fn pointer_is_ptr_value(
        &self,
        pointer: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        matches!(
            ctx.exprs[pointer],
            naga::Expression::FunctionArgument(idx)
            if matches!(
                self.module.types[ctx.func.arguments[idx as usize].ty].inner,
                naga::TypeInner::Pointer { .. }
            )
        )
    }

    /// One statement without indent / newline / terminator, for the for-update clause.
    fn emit_statement_inline(
        &mut self,
        stmt: &naga::Statement,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        match stmt {
            naga::Statement::Store { pointer, value } => {
                if let Some(atomic_scalar) = self.atomic_scalar_for_expr(*pointer, ctx) {
                    self.emit_atomic_store(*pointer, *value, atomic_scalar, ctx)?;
                } else {
                    self.emit_assignment(*pointer, *value, ctx)?;
                }
            }
            naga::Statement::Call {
                function,
                arguments,
                result: _,
            } => {
                self.out
                    .push_str(&self.emit_call(*function, arguments, ctx)?);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => self.emit_image_store(*image, *coordinate, *array_index, *value, ctx)?,
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported statement in for-loop update clause \
                     in function '{}': {}",
                    ctx.display_name,
                    statement_kind(stmt),
                )));
            }
        }
        Ok(())
    }

    /// Recognise `Store(ptr, Binary(op, Load(ptr), rhs))` or its commutative
    /// mirror: the compound-assign token plus the "other" operand.
    fn try_compound_assign(
        &self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<(&'static str, naga::Handle<naga::Expression>)> {
        // A bound value renders as a name, not as the binary.
        if ctx.expr_names.contains_key(value) {
            return None;
        }
        let naga::Expression::Binary { op, left, right } = &ctx.exprs[value] else {
            return None;
        };
        let (cop, commutative) = compound_assign_info(*op)?;

        let left_is_self = !ctx.expr_names.contains_key(left) && {
            if let naga::Expression::Load { pointer: p } = &ctx.exprs[*left] {
                ptrs_structurally_equal(*p, pointer, ctx.exprs)
            } else {
                false
            }
        };
        if left_is_self {
            return Some((cop, *right));
        }

        // The mirror swaps operands, which transposes a matrix product; the
        // left-self fold keeps operand order and needs no guard.
        if commutative
            && !ctx.expr_names.contains_key(right)
            && let naga::Expression::Load { pointer: p } = &ctx.exprs[*right]
            && ptrs_structurally_equal(*p, pointer, ctx.exprs)
            && (*op != naga::BinaryOperator::Multiply
                || self.multiply_is_commutative(*left, *right, ctx))
        {
            return Some((cop, *left));
        }

        None
    }

    /// WGSL `*` is commutative for scalar and component-wise products but not
    /// for linear-algebra ones (`mat*mat`, `mat*vec`, `vec*mat`), where swapping
    /// operands computes the transposed product.
    fn multiply_is_commutative(
        &self,
        left: naga::Handle<naga::Expression>,
        right: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::TypeInner as TI;
        let l = ctx.ty(left).inner_with(&self.module.types);
        let r = ctx.ty(right).inner_with(&self.module.types);
        let l_mat = matches!(l, TI::Matrix { .. });
        let r_mat = matches!(r, TI::Matrix { .. });
        let l_vec = matches!(l, TI::Vector { .. });
        let r_vec = matches!(r, TI::Vector { .. });
        !((l_mat && (r_mat || r_vec)) || (r_mat && l_vec))
    }

    /// `++` / `--` for `lhs += 1` / `lhs -= 1` on a concrete 32-bit integer
    /// scalar, one byte shorter than the compound form; WGSL allows them only on
    /// integer scalar references, so floats, vectors, bools and 64-bit integers
    /// keep `+= 1`.  `value` is the `Store`'s binary, typed as the lvalue.
    fn try_increment(
        &self,
        cop: &str,
        other: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<&'static str> {
        let token = match cop {
            "+=" => "++",
            "-=" => "--",
            _ => return None,
        };
        // A bound `1` renders as a name, not the literal.
        if ctx.expr_names.contains_key(other) {
            return None;
        }
        let is_one = matches!(
            ctx.exprs[other],
            naga::Expression::Literal(
                naga::Literal::I32(1) | naga::Literal::U32(1) | naga::Literal::AbstractInt(1)
            )
        );
        if !is_one {
            return None;
        }
        matches!(
            ctx.ty(value).inner_with(&self.module.types),
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Sint | naga::ScalarKind::Uint,
                width: 4,
            })
        )
        .then_some(token)
    }

    /// Compound-assignment right-hand side, with splat elision for a Splat /
    /// splat-Compose operand under an arithmetic operator.
    fn emit_compound_assign_rhs(
        &self,
        cop: &str,
        other: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if matches!(cop, "+=" | "-=" | "*=" | "/=" | "%=") {
            let cached = ctx.expr_names.contains_key(other);
            if let Some(scalar) = self.try_splat_scalar(other, ctx.exprs, cached) {
                return self.emit_constructor_arg(scalar, ctx);
            }
        }
        self.emit_expr(other, ctx)
    }

    /// Emit a call statement, binding its result only when something reads it:
    /// a never-read result or no result handle emits the bare `<call>;`; a
    /// single-use result in an inline-safe zone (`inlineable_calls`) stashes the
    /// call text for emission at the use site; otherwise `let <name> = <call>;`.
    fn emit_call_result(
        &mut self,
        call: &str,
        result: Option<naga::Handle<naga::Expression>>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) {
        if let Some(handle) = result {
            // ref_count 0 means unread (only genuine uses bump a Call/Atomic
            // result, never its producing statement): keep the side effect, an
            // impure call cannot be DCE'd, but drop the dead `let`.
            if ctx.ref_counts[handle.index()] == 0 {
                self.out.push_str(call);
                self.out.push(';');
                return;
            }
            if ctx.inlineable_calls.contains(handle) {
                // Stash only within the depth budget: past it the text would
                // keep nesting at its consumer, and tint's parser limit rejects
                // what naga's re-parse accepts, so the self-check cannot catch
                // it; a binding renders as a leaf and resets the chain.
                if ctx
                    .stashed_call_depth
                    .get(handle)
                    .is_none_or(|d| *d <= MAX_RENDER_DEPTH)
                {
                    ctx.expr_names.insert(handle, call.to_string());
                    return;
                }
                // Bound after all: `inlineable_calls` then means "stashed"
                // for everything that prices or scans the result by it.
                ctx.stashed_call_depth.remove(handle);
                ctx.inlineable_calls.remove(handle);
            }
            let name = ctx.next_expr_name();
            self.out.push_str("let ");
            self.out.push_str(&name);
            self.push_assign();
            self.out.push_str(call);
            self.out.push(';');
            ctx.expr_names.insert(handle, name);
        } else {
            self.out.push_str(call);
            self.out.push(';');
        }
    }
}

impl<'a> Generator<'a> {
    pub(super) fn atomic_scalar_for_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        match ctx.ty(expr).inner_with(&self.module.types) {
            naga::TypeInner::Atomic(s) => Some(*s),
            naga::TypeInner::Pointer { base, .. } => match &self.module.types[*base].inner {
                naga::TypeInner::Atomic(s) => Some(*s),
                _ => None,
            },
            _ => None,
        }
    }

    fn emit_expr_for_atomic(
        &self,
        expr: naga::Handle<naga::Expression>,
        target: naga::Scalar,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        use naga::Literal as L;

        let literal = match &ctx.exprs[expr] {
            E::Literal(lit) => Some(*lit),
            E::Constant(h) => {
                let c = &self.module.constants[*h];
                if c.name.is_none() {
                    if let E::Literal(lit) = self.module.global_expressions[c.init] {
                        Some(lit)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(lit) = literal {
            let forced = match (lit, target.kind, target.width) {
                (L::U32(v), naga::ScalarKind::Uint, 4) => Some(format!("{v}u")),
                (L::U64(v), naga::ScalarKind::Uint, 8) => Some(format!("{v}lu")),
                (L::I32(v), naga::ScalarKind::Sint, 4) => Some(format!("{v}i")),
                (L::I64(v), naga::ScalarKind::Sint, 8) => Some(format!("{v}li")),
                (L::AbstractInt(v), naga::ScalarKind::Sint, 4) => Some(format!("{v}i")),
                (L::AbstractInt(v), naga::ScalarKind::Uint, 4) => Some(format!("{v}u")),
                (L::AbstractInt(v), naga::ScalarKind::Sint, 8) => Some(format!("{v}li")),
                (L::AbstractInt(v), naga::ScalarKind::Uint, 8) => Some(format!("{v}lu")),
                _ => None,
            };
            if let Some(v) = forced {
                return Ok(v);
            }
        }

        self.emit_expr(expr, ctx)
    }

    fn image_atomic_scalar_for_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        fn resolve_global_image(
            expr: naga::Handle<naga::Expression>,
            exprs: &naga::Arena<naga::Expression>,
        ) -> Option<naga::Handle<naga::GlobalVariable>> {
            match exprs[expr] {
                naga::Expression::GlobalVariable(h) => Some(h),
                naga::Expression::Access { base, .. }
                | naga::Expression::AccessIndex { base, .. } => resolve_global_image(base, exprs),
                _ => None,
            }
        }

        let image_inner = if let Some(gh) = resolve_global_image(expr, ctx.exprs) {
            &self.module.types[self.module.global_variables[gh].ty].inner
        } else {
            let inner = ctx.ty(expr).inner_with(&self.module.types);
            match inner {
                naga::TypeInner::Image { .. } => inner,
                naga::TypeInner::Pointer { base, .. } => &self.module.types[*base].inner,
                _ => return None,
            }
        };

        let naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, .. },
            ..
        } = image_inner
        else {
            return None;
        };

        use naga::StorageFormat as F;
        match format {
            F::R64Uint => Some(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 8,
            }),
            F::R32Uint | F::Rg32Uint | F::Rgba32Uint => Some(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 4,
            }),
            F::R32Sint | F::Rg32Sint | F::Rgba32Sint => Some(naga::Scalar {
                kind: naga::ScalarKind::Sint,
                width: 4,
            }),
            _ => None,
        }
    }
}

// MARK: Statement-level helpers

/// WGSL compound-assignment token for a binary operator plus whether it is
/// commutative, i.e. whether `lhs = rhs <op> lhs` may fold by matching the
/// right operand (`Multiply` is flagged commutative; the caller excludes
/// matrix products).  `None` when the operator has no compound form.
fn compound_assign_info(op: naga::BinaryOperator) -> Option<(&'static str, bool)> {
    use naga::BinaryOperator as B;
    match op {
        B::Add => Some(("+=", true)),
        B::Subtract => Some(("-=", false)),
        B::Multiply => Some(("*=", true)),
        B::Divide => Some(("/=", false)),
        B::Modulo => Some(("%=", false)),
        B::And => Some(("&=", true)),
        B::ExclusiveOr => Some(("^=", true)),
        B::InclusiveOr => Some(("|=", true)),
        B::ShiftLeft => Some(("<<=", false)),
        B::ShiftRight => Some((">>=", false)),
        _ => None,
    }
}

/// `true` when two pointer expressions walk identically through `Access` /
/// `AccessIndex` chains from the same root (global, local or argument).
fn ptrs_structurally_equal(
    a: naga::Handle<naga::Expression>,
    b: naga::Handle<naga::Expression>,
    exprs: &naga::Arena<naga::Expression>,
) -> bool {
    if a == b {
        return true;
    }
    use naga::Expression as E;
    match (&exprs[a], &exprs[b]) {
        (E::GlobalVariable(ga), E::GlobalVariable(gb)) => ga == gb,
        (E::LocalVariable(la), E::LocalVariable(lb)) => la == lb,
        (E::FunctionArgument(ia), E::FunctionArgument(ib)) => ia == ib,
        (
            E::AccessIndex {
                base: ba,
                index: ia,
            },
            E::AccessIndex {
                base: bb,
                index: ib,
            },
        ) => ia == ib && ptrs_structurally_equal(*ba, *bb, exprs),
        (
            E::Access {
                base: ba,
                index: ia,
            },
            E::Access {
                base: bb,
                index: ib,
            },
        ) => {
            // Same index handle, or equal literals; distinct dynamic indices never match.
            if ia == ib {
                return ptrs_structurally_equal(*ba, *bb, exprs);
            }
            if let (E::Literal(la), E::Literal(lb)) = (&exprs[*ia], &exprs[*ib]) {
                la == lb && ptrs_structurally_equal(*ba, *bb, exprs)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// WGSL built-in name for a `(SubgroupOperation, CollectiveOperation)` pair.
fn subgroup_collective_name(
    op: naga::SubgroupOperation,
    collective_op: naga::CollectiveOperation,
) -> Result<&'static str, Error> {
    use naga::CollectiveOperation as C;
    use naga::SubgroupOperation as S;
    match (collective_op, op) {
        (C::Reduce, S::All) => Ok("subgroupAll"),
        (C::Reduce, S::Any) => Ok("subgroupAny"),
        (C::Reduce, S::Add) => Ok("subgroupAdd"),
        (C::Reduce, S::Mul) => Ok("subgroupMul"),
        (C::Reduce, S::Min) => Ok("subgroupMin"),
        (C::Reduce, S::Max) => Ok("subgroupMax"),
        (C::Reduce, S::And) => Ok("subgroupAnd"),
        (C::Reduce, S::Or) => Ok("subgroupOr"),
        (C::Reduce, S::Xor) => Ok("subgroupXor"),
        (C::InclusiveScan, S::Add) => Ok("subgroupInclusiveAdd"),
        (C::InclusiveScan, S::Mul) => Ok("subgroupInclusiveMul"),
        (C::ExclusiveScan, S::Add) => Ok("subgroupExclusiveAdd"),
        (C::ExclusiveScan, S::Mul) => Ok("subgroupExclusiveMul"),
        _ => Err(Error::Emit(format!(
            "unsupported subgroup collective operation: {:?}/{:?}",
            collective_op, op,
        ))),
    }
}

/// WGSL built-in name and optional index operand for a subgroup `GatherMode`.
fn subgroup_gather_name_and_index(
    mode: &naga::GatherMode,
) -> (&'static str, Option<naga::Handle<naga::Expression>>) {
    match *mode {
        naga::GatherMode::BroadcastFirst => ("subgroupBroadcastFirst", None),
        naga::GatherMode::Broadcast(h) => ("subgroupBroadcast", Some(h)),
        naga::GatherMode::Shuffle(h) => ("subgroupShuffle", Some(h)),
        naga::GatherMode::ShuffleDown(h) => ("subgroupShuffleDown", Some(h)),
        naga::GatherMode::ShuffleUp(h) => ("subgroupShuffleUp", Some(h)),
        naga::GatherMode::ShuffleXor(h) => ("subgroupShuffleXor", Some(h)),
        naga::GatherMode::QuadBroadcast(h) => ("quadBroadcast", Some(h)),
        naga::GatherMode::QuadSwap(naga::Direction::X) => ("quadSwapX", None),
        naga::GatherMode::QuadSwap(naga::Direction::Y) => ("quadSwapY", None),
        naga::GatherMode::QuadSwap(naga::Direction::Diagonal) => ("quadSwapDiagonal", None),
    }
}
