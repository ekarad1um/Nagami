//! Statement-level WGSL emission.
//!
//! Drives the body of every function by walking its
//! [`naga::Block`] in source order and dispatching to the matching
//! emitter.  The file also hosts the expression use-detection helpers
//! (`stmt_uses_expr` / `stmts_use_expr`) that back the for-loop preload
//! tail-use scan in [`for_loop_preload_inlining_is_safe`]; the call-inlining
//! decision itself lives in `module_emit::find_inlineable_calls`.

use crate::error::Error;

use super::core::{FunctionCtx, Generator};

/// Count how many times emitting the expression tree rooted at `root` will
/// materialise `target`.  `target` is a for-loop preload result that the
/// caller has bound to inline `workgroupUniformLoad(&p)` text; every distinct
/// path from `root` to `target` through the expression DAG re-emits that
/// text, because loop-body-dependent intermediates cannot be hoisted to a
/// `let` and so are inlined on each visit.  Memoised (paths-to-`target` is a
/// per-node property) so a shared sub-DAG is counted in linear time.
fn count_inline_emissions(
    root: naga::Handle<naga::Expression>,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    cache: &mut std::collections::HashMap<naga::Handle<naga::Expression>, usize>,
) -> usize {
    if root == target {
        return 1;
    }
    if let Some(&c) = cache.get(&root) {
        return c;
    }
    let mut total = 0usize;
    crate::passes::expr_util::visit_expression_children(&expressions[root], |child| {
        total += count_inline_emissions(child, target, expressions, cache);
    });
    cache.insert(root, total);
    total
}

/// Sum [`count_inline_emissions`] over the operand expressions of a for-loop
/// update statement (`Store` / `Call` / `ImageStore`), so a preload reused
/// across the update (e.g. `out[w] = w`) is detected as multi-emit.  The
/// cache is shared across operands (same `target`).
fn count_update_stmt_emissions(
    stmt: &naga::Statement,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    cache: &mut std::collections::HashMap<naga::Handle<naga::Expression>, usize>,
) -> usize {
    let mut total = 0usize;
    let mut add =
        |h: naga::Handle<naga::Expression>,
         cache: &mut std::collections::HashMap<naga::Handle<naga::Expression>, usize>| {
            total += count_inline_emissions(h, target, expressions, cache);
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

/// Parsed for-loop shape shared by the emitter, the preload-safety predicate,
/// and the candidate check, so they cannot drift.  Holds the leading body
/// `WorkGroupUniformLoad` guard preloads, the if-break guard, and the
/// `continuing` update preloads plus the single core update statement.
pub(super) struct ForLoopShape<'a> {
    /// `(pointer, result)` for each leading body `WorkGroupUniformLoad`.
    pub(super) guard_preloads: Vec<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    )>,
    /// Body statement indices of those preloads (excluded from the for-body).
    pub(super) guard_preload_stmt_indices: Vec<usize>,
    /// Body index of the if-break guard statement.
    pub(super) guard_idx: usize,
    /// The guard's condition expression.
    pub(super) condition: naga::Handle<naga::Expression>,
    /// `true` when the guard is `if cond { break; }` (exit form; negate for `for`).
    pub(super) needs_negation: bool,
    /// `(pointer, result)` for each leading `continuing` `WorkGroupUniformLoad`.
    pub(super) update_preloads: Vec<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
    )>,
    /// The single core update statement in `continuing`, if any.  Its KIND
    /// (Store / Call / ImageStore) is NOT validated here - callers that emit it
    /// check that separately.
    pub(super) update_stmt: Option<&'a naga::Statement>,
}

/// Parse a `Loop` into a [`ForLoopShape`], or `None` when it is not
/// for-convertible: it carries a `break_if`, lacks a leading if-break guard
/// (after optional `Emit` / `WorkGroupUniformLoad` preloads), or its
/// `continuing` block holds more than one core (non-`Emit`, non-leading-
/// preload) statement.  Purely structural - applies no preload-inlining
/// safety check and no update-statement-kind check, so every consumer shares
/// exactly one parse and layers its own policy on top.
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
                    return None; // more than one core update statement
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

/// Single source of truth for "can a for-shaped loop's `WorkGroupUniformLoad`
/// preloads be safely inlined into the `for(...)` header?".  Both
/// [`Generator::try_emit_for_loop`] (which emits) and
/// `module_emit::is_for_loop_candidate` (which decides whether to suppress
/// the counter's top-level `var`) call this on the SAME [`ForLoopShape`], so
/// the two never disagree (a disagreement would leave the counter undeclared).
/// Returns `false` only when inlining a preload would be wrong.
///
/// A `WorkGroupUniformLoad` preload carries a barrier side effect, so it must
/// execute EXACTLY ONCE per iteration.  When the loop becomes a `for(...)`,
/// a preload is materialised only where its `result` is emitted (inlined into
/// the condition for a guard preload, into the update statement for an update
/// preload); it has no statement of its own.  Three hazards are checked:
/// * tail use - a guard preload result used AFTER the guard (in the body tail
///   or the `continuing` block, both in scope for it) would lose its `let`
///   binding when the preload is inlined into the condition;
/// * multi-emit - a preload reused within the condition or update expression
///   would execute the barrier more than once per iteration;
/// * dropped - a preload whose `result` is NOT referenced by the condition /
///   update (count 0) would never be emitted at all, silently deleting the
///   barrier.  This also covers a `continuing` block that holds preloads but
///   no core update statement (no update clause to carry them).
///
/// Each preload must therefore be emitted exactly once (`count == 1`); any
/// other count refuses the for-loop conversion so plain-loop emission - which
/// keeps the preload as its own statement - preserves the barrier.
pub(super) fn for_loop_preload_inlining_is_safe(
    shape: &ForLoopShape,
    body: &naga::Block,
    continuing: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    // The update clause is RELOCATED into the `for(...; ...; update)` header,
    // which is emitted BEFORE the body.  A `Load` that must be bound (its place
    // is overwritten between its `Emit` and its use) has its `let` emitted at
    // its body `Emit` - which comes AFTER the header.  So if a relocated clause
    // references such a load it is inlined as the bare (post-write) place,
    // re-introducing the very miscompile the must-bind analysis prevents.  Fall
    // back to plain `loop` emission, where the body binding precedes the
    // `continuing` use.  Two relocated clauses are at risk:
    // * the update statement itself, and
    // * each `continuing` `workgroupUniformLoad` preload, whose POINTER (e.g.
    //   `&A[snap]`) is hoisted into the for-update slot - its index expression
    //   is emitted as bare place text there.
    // A condition with NO relocated barrier is safe: evaluated at iteration top, it
    // coincides with the body-top load position, so an inlined must-bind load reads
    // the value its body `let` would; outer loads it reads are already `let`-bound
    // before the loop, and back-edge writes are covered by the must-bind loop
    // pre-marking.  But a `workgroupUniformLoad` GUARD PRELOAD relocated into the
    // condition IS a barrier - that third hazard is handled in the guard-preload
    // block below.
    if !must_bind_loads.is_empty()
        && let Some(stmt) = shape.update_stmt
    {
        let update_hazard = stmt_references_must_bind_load(stmt, must_bind_loads, expressions)
            || shape.update_preloads.iter().any(|&(pointer, _)| {
                let mut visited = std::collections::HashSet::new();
                cone_intersects_set(pointer, must_bind_loads, expressions, &mut visited)
            });
        if update_hazard {
            return false;
        }
    }

    if !shape.guard_preloads.is_empty() {
        let body_stmts: Vec<_> = body.iter().collect();
        // Guard-preload barrier hazard: a `workgroupUniformLoad` guard preload is
        // inlined into the condition, which `for` evaluates at iteration top - BEFORE
        // the body - and it carries a barrier (a peer invocation's store becomes
        // visible only across it).  A must-bind load DEFINED in the pre-guard region
        // (`body_stmts[..guard_idx]`, which holds only `Emit` / `WorkGroupUniformLoad`)
        // is snapshotted there to capture the PRE-barrier value, but for-reconstruction
        // re-emits its `let` in the body (after the relocated barrier) and/or inlines
        // it into the condition after the barrier operand - either way reading the
        // POST-barrier value, plain-loop emission keeps the snapshot before the barrier,
        // so bail.
        if !must_bind_loads.is_empty()
            && body_stmts[..shape.guard_idx].iter().any(|s| {
                matches!(s, naga::Statement::Emit(range)
                    if range.clone().any(|h| must_bind_loads.contains(&h)))
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
        let mut cache = std::collections::HashMap::new();
        for &(_, result) in &shape.guard_preloads {
            if stmts_use_expr(tail, result, expressions)
                || stmts_use_expr(&continuing_stmts, result, expressions)
            {
                return false;
            }
            cache.clear();
            // Must be inlined into the condition exactly once: 0 drops the
            // barrier, >1 duplicates it.
            if count_inline_emissions(shape.condition, result, expressions, &mut cache) != 1 {
                return false;
            }
        }
    }

    match shape.update_stmt {
        // No core update statement: any leading `WorkGroupUniformLoad` preload
        // in `continuing` has nowhere to be emitted (the for-update clause is
        // empty), so it would be dropped along with its barrier.
        None => {
            if !shape.update_preloads.is_empty() {
                return false;
            }
        }
        Some(stmt) => {
            let mut cache = std::collections::HashMap::new();
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

/// `true` when any handle in `set` appears anywhere in the operand cone of
/// `stmt` (the transitive children of every expression the statement
/// references).  Used to detect a `for`-update clause that would inline a
/// load which the must-bind analysis requires be `let`-bound.
fn stmt_references_must_bind_load(
    stmt: &naga::Statement,
    set: &std::collections::HashSet<naga::Handle<naga::Expression>>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut found = false;
    crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |root| {
        if !found {
            found = cone_intersects_set(root, set, expressions, &mut visited);
        }
    });
    found
}

/// `true` when `root` or any of its transitive children is in `set`.  Mirrors
/// [`expr_subtree_contains`] but tests membership in a set; `visited` memoises
/// proven-absent nodes so a shared sub-DAG is walked once.
fn cone_intersects_set(
    root: naga::Handle<naga::Expression>,
    set: &std::collections::HashSet<naga::Handle<naga::Expression>>,
    expressions: &naga::Arena<naga::Expression>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    if set.contains(&root) {
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

/// `true` when `target` appears anywhere in the expression subtree rooted at
/// `root` (including `root` itself).  A use of `target` almost always reaches
/// a statement through an enclosing expression (`w + 1`, `f(w)`), so a flat
/// handle-equality check on statement operands misses it - the recursion here
/// is what makes [`stmt_uses_expr`] sound.
fn expr_subtree_contains(
    root: naga::Handle<naga::Expression>,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    // Nodes already proven NOT to contain `target` (per fixed `target`), so a
    // shared sub-DAG is explored once - without this the path count through a
    // diamond-shaped expression DAG is super-linear.  Mirrors the memo in the
    // sibling `count_inline_emissions`.
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    if root == target {
        return true;
    }
    if visited.contains(&root) {
        return false;
    }
    let mut found = false;
    crate::passes::expr_util::visit_expression_children(&expressions[root], |child| {
        if !found {
            found = expr_subtree_contains(child, target, expressions, visited);
        }
    });
    // Only memoise the absent result; if `found`, the caller short-circuits the
    // whole walk so `root` is never revisited.
    if !found {
        visited.insert(root);
    }
    found
}

/// `true` when `stmt` references `target` in ANY operand position - including
/// nested inside an emitted expression (`out[i] = w + 1` uses `w`) and inside
/// nested control-flow blocks.  Built on the exhaustive
/// [`crate::passes::expr_util::visit_statement_expression_handles`], so every
/// handle-bearing statement variant is covered and a new naga variant forces
/// an update there rather than silently returning a false negative here.
fn stmt_uses_expr(
    stmt: &naga::Statement,
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    // One memo per (target) query, shared across this statement's operands.
    let mut visited = std::collections::HashSet::new();
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

/// Slice variant of [`stmt_uses_expr`] used when scanning a subset of
/// statements (for example the tail of a loop body).
fn stmts_use_expr(
    stmts: &[&naga::Statement],
    target: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    stmts.iter().any(|s| stmt_uses_expr(s, target, expressions))
}

// MARK: Block emission

impl<'a> Generator<'a> {
    /// Emit every statement in `block` in source order.
    pub(super) fn generate_block(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, false)
    }

    /// Emit a function body, dropping a trailing `return;` on
    /// void-returning functions because WGSL makes it optional.
    pub(super) fn generate_block_elide_trailing_return(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, true)
    }

    /// Shared driver for [`generate_block`] and
    /// [`generate_block_elide_trailing_return`].
    fn generate_block_inner(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
        elide_trailing_void_return: bool,
    ) -> Result<(), Error> {
        let mut stmts: Vec<_> = block.iter().collect();
        // Optionally drop trailing `return;` (void return) that WGSL does not require.
        if elide_trailing_void_return
            && let Some(naga::Statement::Return { value: None }) = stmts.last()
        {
            stmts.pop();
        }
        self.emit_stmts(&stmts, ctx)
    }

    /// Emit a slice of statement references while attempting to
    /// reconstruct a `for` loop from the `Loop` / `Store` init pattern
    /// whenever possible.  Invoked from
    /// [`generate_block_inner`](Self::generate_block_inner) and from
    /// the for-loop body emission in `try_emit_for_loop`.
    fn emit_stmts(
        &mut self,
        stmts: &[&naga::Statement],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let len = stmts.len();
        let mut i = 0;
        while i < len {
            let stmt = stmts[i];

            // Try to reconstruct a for-loop from a Loop statement
            // (optionally preceded by a deferred-var Store as the init).
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
            // Look ahead: if stmt[i] is a deferred-var Store and stmt[i+1]
            // is a for-loop-shaped Loop, absorb the Store as the for-init.
            if i + 1 < len
                && let naga::Statement::Store { pointer, value } = stmt
                && let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer]
                && ctx.deferred_vars[lh.index()]
            {
                // Safety check: the for-init scopes `lh` inside the
                // loop body.  If `lh` is referenced after the Loop
                // (stmts[i+2..]), absorbing it would leave those
                // later uses out of scope -> skip the absorption.
                let safe = i + 2 >= len
                    || !super::module_emit::local_var_in_stmts(
                        &stmts[i + 2..],
                        lh,
                        &ctx.func.expressions,
                    );
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
                    // Mark deferred var as emitted.
                    ctx.deferred_vars[lh.index()] = false;
                    i += 2; // skip both Store and Loop
                    continue;
                }
            }

            let before = self.out.len();
            self.push_indent();
            let after_indent = self.out.len();
            self.generate_statement(stmt, ctx)?;
            if self.out.len() > after_indent {
                // Statement produced output; terminate the line.
                self.push_newline();
            } else {
                // Statement produced nothing (e.g. all-single-use Emit);
                // remove the indent we speculatively pushed.
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
            S::Emit(range) => {
                let mut emitted_any = false;
                for h in range.clone() {
                    // A `Load` whose place is written between this `Emit` and a
                    // use MUST be bound; inlining it would read the post-write
                    // value (silent miscompile).  `is_uniformity_pinned` shares
                    // this force-bind for its own reason (see its doc).  Both
                    // override the bare-local/global short-name skip in
                    // `should_bind_expression` and the `min_binding_refs`
                    // threshold below.
                    let force_bind =
                        ctx.must_bind_loads.contains(&h) || self.is_uniformity_pinned(h, ctx);
                    if !force_bind && !self.should_bind_expression(h, ctx) {
                        continue;
                    }
                    // Skip binding when the expression has too few references
                    // to justify the `let X=...;` overhead.  For trivially short
                    // expressions (e.g. `-x`, `v.y`, `a*b`) the threshold is
                    // higher because inlining them at each use site is cheaper.
                    let refs = ctx.ref_counts[h.index()];
                    if !force_bind && refs < self.min_binding_refs(h, ctx) {
                        continue;
                    }
                    // For subsequent bindings, start a new indented line.
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
            }
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
                    // Empty true-branch: flip the condition and emit only
                    // the false-branch, e.g. `if c{}else{break;}` -> `if c>=32{break;}`
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
            S::Switch { selector, cases } => {
                self.out.push_str("switch ");
                // Switch case labels carry an explicit type suffix
                // (`0u` for `SwitchValue::U32`, bare `0` for `I32`);
                // the selector must match.  If the selector is itself
                // a `Literal::U32(0)` (rare but legal naga IR), the
                // generic `emit_expr` path goes through the bare-
                // literal gate and emits `0`, producing `switch 0 {
                // case 0u: ... }` which naga rejects on selector /
                // case-value type mismatch.  Force the typed form
                // when the selector is an uncached literal to keep
                // emit consistent with the case labels.
                if !ctx.expr_names.contains_key(selector)
                    && let naga::Expression::Literal(lit) = ctx.func.expressions[*selector]
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
            }
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
                    if !continuing.is_empty() {
                        self.generate_block(continuing, ctx)?;
                    }
                    // `break if` must be the last statement inside `continuing`.
                    if let Some(expr) = break_if {
                        self.push_indent();
                        self.out.push_str("break if ");
                        self.out.push_str(&self.emit_expr(*expr, ctx)?);
                        self.out.push(';');
                        self.push_newline();
                    }
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
                // Only emit barriers for the scopes explicitly requested.
                // An empty flag set produces no output (consistent with
                // MemoryBarrier handling).
                self.emit_barrier_calls(*flags);
            }
            S::MemoryBarrier(flags) => {
                // Only emit barriers for the scopes explicitly requested.
                self.emit_barrier_calls(*flags);
            }
            S::Store { pointer, value } => {
                if let Some(atomic_scalar) = self.atomic_scalar_for_expr(*pointer, ctx) {
                    self.emit_atomic_store(*pointer, *value, atomic_scalar, ctx)?;
                    self.out.push(';');
                    return Ok(());
                }

                // Drop a no-op self-store `p = p`: an UNCACHED `Load` of the
                // same place reads `p`'s current value and writes it straight
                // back, changing nothing.  naga re-lowers a folded
                // `var d = a && b` into a temp-plus-copy that coalescing
                // collapses to `d = d`; emitting it would accumulate one such
                // line on every re-minify (a non-idempotence growth).  The
                // `uncached` guard is essential - a `let t = p; ..p..; p = t`
                // restores a stale value and is NOT a no-op.
                if !ctx.expr_names.contains_key(value)
                    && let naga::Expression::Load { pointer: loaded } = ctx.func.expressions[*value]
                    && ptrs_structurally_equal(loaded, *pointer, &ctx.func.expressions)
                {
                    return Ok(());
                }

                // Check if this is the first store to a deferred local var.
                let deferred_local =
                    if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
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
                    // A deferred var's first store IS its declaration, so a
                    // first store of the zero value is equivalent to relying
                    // on WGSL's zero-init.  Drop the value and emit the
                    // shorter `:type` / `=0i` tail (e.g. `var E=vec3f(0)` ->
                    // `var E:vec3f`).
                    if !ctx.expr_names.contains_key(value)
                        && crate::passes::load_dedup::is_zero_init(&ctx.func.expressions, *value)
                    {
                        self.emit_zero_init_tail(ctx.func.local_variables[lh].ty)?;
                    } else {
                        self.push_assign();
                        // When the value is an uncached Literal, use the typed
                        // suffix so WGSL infers the correct concrete type.
                        if !ctx.expr_names.contains_key(value) {
                            if let naga::Expression::Literal(lit) = ctx.func.expressions[*value] {
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
                } else if let Some((cop, other)) = self.try_compound_assign(*pointer, *value, ctx) {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    if let Some(inc) = self.try_increment(cop, other, *value, ctx) {
                        self.out.push_str(inc);
                    } else {
                        let sp = self.bin_op_sep();
                        self.out.push_str(sp);
                        self.out.push_str(cop);
                        self.out.push_str(sp);
                        self.out
                            .push_str(&self.emit_compound_assign_rhs(cop, other, ctx)?);
                    }
                    self.out.push(';');
                } else {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    self.push_assign();
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
                    self.out.push(';');
                }
            }
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                self.out.push_str("textureStore(");
                self.out.push_str(&self.emit_expr(*image, ctx)?);
                let sep = self.comma_sep();
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*index, ctx)?);
                }
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*value, ctx)?);
                self.out.push_str(");");
            }
            S::Call {
                function,
                arguments,
                result,
            } => {
                let call = self.emit_call(*function, arguments, ctx)?;
                self.emit_call_result(&call, *result, ctx);
            }
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
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
                call.push_str("(&");
                call.push_str(&self.emit_expr(*pointer, ctx)?);
                call.push_str(sep);
                if let Some(scalar) = atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(*value, ctx)?);
                }
                call.push(')');
                self.emit_call_result(&call, *result, ctx);
            }
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
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
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("workgroupUniformLoad(&");
                self.out.push_str(&self.emit_expr(*pointer, ctx)?);
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
                    self.out.push('&');
                    self.out.push_str(&self.emit_expr(*payload, ctx)?);
                    self.out.push_str(");");
                }
            },
            // Ray-query intrinsics.  Each function takes the ray-query
            // local through a pointer (`&q`); WGSL expects this exact
            // shape per the `wgpu_ray_query` enable extension.  All
            // arms are listed explicitly so a future naga release that
            // grows a new ray-query function variant produces a compile
            // error here instead of silently bypassing emission.
            S::RayQuery { query, fun } => {
                // naga's validator restricts `query` to
                // `Expression::LocalVariable` (else `InvalidRayQueryExpression`),
                // so `emit_expr(query)` is a bare identifier and prepending `&`
                // for the `&q` operand is always correct.  The debug_assert guards
                // this: if naga ever relaxes it, branch on the variant here -
                // pointer function-arguments emit verbatim (no `&`), local/Access
                // chains keep the `&`.
                debug_assert!(
                    matches!(
                        ctx.func.expressions[*query],
                        naga::Expression::LocalVariable(_)
                    ),
                    "Statement::RayQuery.query must be a LocalVariable per \
                     naga's validator; got {:?}",
                    ctx.func.expressions[*query]
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
                        self.out.push_str(&self.emit_expr(*hit_t, ctx)?);
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
            }
            // Cooperative-matrix store.  nagami's generator can't render the
            // `cooperative_matrix<...>` type, so it errors and the pipeline falls
            // back to naga's WGSL backend - but naga emits the store while
            // OMITTING `enable wgpu_cooperative_matrix;` (the coop type is held
            // inline on the load/store, never interned in `module.types`, so
            // naga's own enable-detection misses it), so the fallback fails
            // re-validation and the store surfaces this error cleanly
            // (empirically verified).  (`rename` reserves the `A`/`B`/`C` role
            // names for when this path round-trips.)
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

    /// Whether `h` must stay in uniform control flow per the WGSL
    /// uniformity rules: an implicit derivative (`dpdx`/`dpdy`/`fwidth`)
    /// or an implicit-LOD texture sample (`textureSample` /
    /// `textureSampleBias` / `textureSampleCompare` - a non-gather sample
    /// whose LOD is `Auto` or `Bias`).  Such an expression is force-bound
    /// at its naga-placed Emit site so single-use inlining cannot sink it
    /// into a deeper, possibly non-uniform branch.  naga's validator does
    /// not enforce derivative uniformity, but strict downstream compilers
    /// (Tint/Dawn/browsers) reject the violation, so the un-pinned form
    /// would ship a shader that fails to compile there.
    ///
    /// Explicit forms (`textureSampleLevel` / `Grad` / `CompareLevel` /
    /// `BaseClampToEdge`) and all gathers use no implicit derivatives and
    /// stay freely inlinable.
    fn is_uniformity_pinned(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            E::Derivative { .. } => true,
            E::ImageSample { gather, level, .. } => {
                gather.is_none()
                    && matches!(level, naga::SampleLevel::Auto | naga::SampleLevel::Bias(_))
            }
            _ => false,
        }
    }

    fn should_bind_expression(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            E::CallResult(_)
            | E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult
            | E::RayQueryGetIntersection { .. } => return false,
            // Load from a bare local/global variable emits as just the
            // variable name (1-2 chars after rename).  A `let` binding
            // would introduce overhead (`let X=Y;`) with zero savings
            // per use since both names come from the same short-name pool.
            E::Load { pointer } => {
                if matches!(
                    ctx.func.expressions[*pointer],
                    E::LocalVariable(_) | E::GlobalVariable(_)
                ) {
                    return false;
                }
            }
            _ => {}
        }

        let inner = ctx.info[h].ty.inner_with(&self.module.types);
        !matches!(
            inner,
            naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. }
        )
    }

    /// Minimum reference count before a `let` binding pays for itself.
    ///
    /// A `let X = EXPR;` costs roughly `textLen + 7` characters (in compact
    /// mode), saving `textLen - 1` characters per use compared to inlining.
    /// For very short expressions the overhead exceeds the savings at low
    /// reference counts, so we require more references before introducing a
    /// binding.
    ///
    /// Thresholds are derived from:
    ///   binding wins when  `refs > (textLen + 7) / (textLen - 1)`
    fn min_binding_refs(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> usize {
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
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
                    let inner = ctx.info[*base].ty.inner_with(&self.module.types);
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

    /// Whether an expression handle will be emitted as a short name (1-2
    /// characters) rather than a sub-expression.
    fn expr_resolves_to_name(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        if ctx.expr_names.contains_key(&h) {
            return true;
        }
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            E::FunctionArgument(_) | E::Constant(_) | E::Override(_) => true,
            // Load from a bare local/global emits as the variable name.
            E::Load { pointer } => matches!(
                ctx.func.expressions[*pointer],
                E::LocalVariable(_) | E::GlobalVariable(_)
            ),
            _ => false,
        }
    }

    /// Try to emit a `Loop` as a WGSL `for(init; cond; update)` statement.
    ///
    /// Returns `Ok(true)` if a for-loop was emitted, `Ok(false)` if the
    /// loop doesn't match the pattern and should use the regular `loop`
    /// emission path.
    ///
    /// Pattern recognised:
    /// - `break_if` is `None`
    /// - `body` starts with optional `WorkGroupUniformLoad` preloads then
    ///   `If { condition, accept: [], reject: [Break] }`
    /// - `continuing` has at most one statement (the update)
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
        // Only convert loops without `break if` in continuing.
        if break_if.is_some() {
            return Ok(false);
        }

        // Parse the loop into its for-loop shape via the shared parser, so the
        // emitter, the preload-safety predicate, and `is_for_loop_candidate`
        // all read the SAME structure and never drift.  `None` => not
        // for-convertible.
        let Some(shape) = parse_for_loop_shape(body, continuing, break_if) else {
            return Ok(false);
        };

        // Pre-validate: only Store / Call / ImageStore fit the for-loop update
        // slot.  Bail out *before* writing any output so the caller can fall
        // back to normal `loop` emission cleanly.
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

        // Reject when inlining a guard/update preload into the for-header would
        // be unsafe (tail use, or reuse that duplicates the barrier).  The SAME
        // predicate gates `is_for_loop_candidate`'s counter-var suppression, so
        // the two never disagree (else a suppressed counter is left undeclared).
        if !for_loop_preload_inlining_is_safe(
            &shape,
            body,
            continuing,
            &ctx.func.expressions,
            &ctx.must_bind_loads,
        ) {
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

        // If no init was provided (from deferred-var absorption), try to
        // absorb an initialized local whose refs are confined to this loop.

        // Track a for-loop counter consumed without an init value, so we
        // can emit a bare `var name:type` declaration in the init clause.
        let mut decl_only_local: Option<naga::Handle<naga::LocalVariable>> = None;

        let init = if init.is_some() {
            // An external init occupies the for-loop init slot.  If the
            // loop counter is a for_loop_var it would be left undeclared
            // (its `var` was suppressed).  Bail out so the Store is emitted
            // as a normal statement and the Loop falls through to path 1,
            // which can absorb the for_loop_var.
            if let Some(naga::Statement::Store { pointer, .. }) = update_stmt
                && let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer]
                && ctx.for_loop_vars[lh.index()]
            {
                return Ok(false);
            }
            init
        } else if let Some(naga::Statement::Store { pointer, .. }) = update_stmt {
            if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
                if ctx.for_loop_vars[lh.index()] {
                    // Mark as consumed so try_emit_for_loop won't absorb again.
                    ctx.for_loop_vars[lh.index()] = false;
                    if let Some(init_handle) = ctx.func.local_variables[lh].init {
                        Some((*pointer, init_handle))
                    } else {
                        // Variable has no explicit init (zero-initialised by
                        // WGSL semantics, e.g. after dead-init removal).
                        // Record it for a bare declaration in the init clause.
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

        // Emit the for-loop
        self.push_indent();
        self.push_for_open();

        // Init clause.  A local declared here is no longer deferred (see below).
        let mut deferred_for_init_local: Option<naga::Handle<naga::LocalVariable>> = None;
        if let Some((pointer, value)) = init {
            if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[pointer] {
                self.out.push_str("var ");
                self.out.push_str(&ctx.local_names[&lh]);
                self.push_assign();
                if !ctx.expr_names.contains_key(&value) {
                    if let naga::Expression::Literal(lit) = ctx.func.expressions[value] {
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
                // The `var <lh> = ...` for-init declares `lh`, so it is no longer
                // deferred: a later Store to `lh` in the body (e.g. a counter
                // update `b = b + 1` that stayed in the body rather than the
                // continuing block) must emit a plain assignment, not a second
                // `var` that shadows the for-init copy and freezes the counter.
                // Clearing here (not at the Path-2 absorption site, which only
                // runs after this whole call returns) is what the body emission
                // sees.
                deferred_for_init_local = Some(lh);
            } else {
                self.out.push_str(&self.emit_lvalue(pointer, ctx)?);
                self.push_assign();
                self.out.push_str(&self.emit_expr(value, ctx)?);
            }
        } else if let Some(lh) = decl_only_local {
            // For-loop counter with no explicit init (WGSL zero-initialises).
            // Emit the shorter of `var name:type` / `var name=0i` so an
            // un-aliased scalar counter keeps the one-byte-cheaper `=0i`
            // form instead of regressing to `:i32`.
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&lh]);
            let ty = ctx.func.local_variables[lh].ty;
            self.emit_zero_init_tail(ty)?;
            // The for-init clause emitted this counter's `var` declaration;
            // clearing its deferred flag stops a later body Store from
            // re-declaring it as a second `var`.
            deferred_for_init_local = Some(lh);
        }
        if let Some(lh) = deferred_for_init_local {
            ctx.deferred_vars[lh.index()] = false;
        }
        self.push_for_sep();

        // Condition clause (inlined - Emits haven't been processed yet).
        let mut preload_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
            Vec::new();
        for (pointer, result) in &guard_preloads {
            let mut preload = String::from("workgroupUniformLoad(&");
            preload.push_str(&self.emit_expr(*pointer, ctx)?);
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
                ctx.expr_names.remove(&result);
            }
        }

        self.push_for_sep();

        // Update clause.
        if let Some(stmt) = update_stmt {
            let mut update_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
                Vec::new();
            for (pointer, result) in &update_preloads {
                let mut preload = String::from("workgroupUniformLoad(&");
                preload.push_str(&self.emit_expr(*pointer, ctx)?);
                preload.push(')');
                let old = ctx.expr_names.insert(*result, preload);
                update_old_bindings.push((*result, old));
            }
            self.emit_statement_inline(stmt, ctx)?;
            for (result, old) in update_old_bindings {
                if let Some(name) = old {
                    ctx.expr_names.insert(result, name);
                } else {
                    ctx.expr_names.remove(&result);
                }
            }
        }

        self.out.push(')');
        self.open_brace();

        // The condition (and update) were emitted inline into the for-header,
        // bypassing the normal Emit -> let-binding path.  Decrement ref counts
        // for children of expressions in the guard's Emit ranges so that
        // expressions shared between the condition and body (e.g. loads
        // deduplicated by load_dedup) get correct effective ref counts and
        // are inlined rather than bound to unnecessary `let` variables.
        for s in &body_stmts[..guard_idx] {
            if let naga::Statement::Emit(range) = s {
                for h in range.clone() {
                    super::module_emit::visit_expr_children(&ctx.func.expressions[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }
        for stmt in continuing.iter() {
            if let naga::Statement::Emit(range) = stmt {
                for h in range.clone() {
                    super::module_emit::visit_expr_children(&ctx.func.expressions[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }

        // Emit all body statements except the If-break guard,
        // using the shared helper so nested for-loops are also reconstructed.
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

        // Unwrap a single trailing Block statement to avoid double-braces.
        // naga wraps the user's for-body in `Statement::Block`, producing
        // `for(...){ { body } }` - this removes the inner braces.
        //
        // `non_emit` collects the indices of non-Emit statements in the
        // remaining body.  When there is exactly one and it is a
        // `Statement::Block`, we emit its contents directly (unwrapping
        // the inner braces).  Non-`Block` single-statement bodies
        // (e.g. `for(...){ i+=1; }`) don't have a double-brace problem,
        // so they fall through to the normal `emit_stmts` path below.
        let mut non_emit: Vec<usize> = Vec::new();
        for (k, s) in remaining.iter().enumerate() {
            if !matches!(s, naga::Statement::Emit(_)) {
                non_emit.push(k);
            }
        }
        if non_emit.len() == 1
            && let naga::Statement::Block(inner) = remaining[non_emit[0]]
        {
            // Emit leading Emits, then the inner block contents directly.
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
            self.close_brace();
            self.push_newline();
            return Ok(true);
        }

        self.emit_stmts(&remaining, ctx)?;

        self.close_brace();
        self.push_newline();
        Ok(true)
    }

    /// Emit `atomicStore(&p, v)` (or `atomicStore(p, v)` when the pointer is
    /// already a `ptr<>` value) WITHOUT a trailing terminator.  Shared by the
    /// top-level `Store` handler and the for-loop-update inline `Store`
    /// handler so an atomic-pointer `Store` lowers to the `atomicStore` builtin
    /// in either position; assigning a scalar directly to `atomic<T>` (`p = v`)
    /// is a WGSL type error that strict consumers reject.
    fn emit_atomic_store(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        atomic_scalar: naga::Scalar,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let sep = self.comma_sep();
        if self.pointer_is_ptr_value(pointer, ctx) {
            self.out.push_str("atomicStore(");
        } else {
            self.out.push_str("atomicStore(&");
        }
        self.out.push_str(&self.emit_expr(pointer, ctx)?);
        self.out.push_str(sep);
        self.out
            .push_str(&self.emit_expr_for_atomic(value, atomic_scalar, ctx)?);
        self.out.push(')');
        Ok(())
    }

    /// `true` when `pointer` is ALREADY a `ptr<>` value - a function parameter
    /// declared `ptr<...>` - so the atomic builtins (`atomicLoad` /
    /// `atomicStore`) take it directly; a `var<>`-rooted place needs `&` to form
    /// the pointer, but emitting `&p` against a `ptr` value is a type error.
    pub(super) fn pointer_is_ptr_value(
        &self,
        pointer: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        matches!(
            ctx.func.expressions[pointer],
            naga::Expression::FunctionArgument(idx)
            if matches!(
                self.module.types[ctx.func.arguments[idx as usize].ty].inner,
                naga::TypeInner::Pointer { .. }
            )
        )
    }

    /// Emit a single statement's text without indent/newline/trailing semicolon.
    /// Used for for-loop update clauses.
    fn emit_statement_inline(
        &mut self,
        stmt: &naga::Statement,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        match stmt {
            naga::Statement::Store { pointer, value } => {
                if let Some(atomic_scalar) = self.atomic_scalar_for_expr(*pointer, ctx) {
                    // For-loop update slot can hold an `atomicStore`; lower it
                    // the same way the top-level Store handler does (see the
                    // for-loop pre-validation that accepts atomic Stores).
                    self.emit_atomic_store(*pointer, *value, atomic_scalar, ctx)?;
                } else if let Some((cop, other)) = self.try_compound_assign(*pointer, *value, ctx) {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    if let Some(inc) = self.try_increment(cop, other, *value, ctx) {
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
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    self.push_assign();
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
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
            } => {
                self.out.push_str("textureStore(");
                self.out.push_str(&self.emit_expr(*image, ctx)?);
                let sep = self.comma_sep();
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*index, ctx)?);
                }
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*value, ctx)?);
                self.out.push(')');
            }
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported statement in for-loop update clause \
                     in function '{}': {:?}",
                    ctx.display_name, stmt,
                )));
            }
        }
        Ok(())
    }

    /// Try to recognise `Store(ptr, Binary(op, Load(ptr), rhs))` or the
    /// commutative mirror and return the compound-assign operator token
    /// plus the "other" operand handle.
    fn try_compound_assign(
        &self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<(&'static str, naga::Handle<naga::Expression>)> {
        // The value expression must not already be cached (bound to a let).
        if ctx.expr_names.contains_key(&value) {
            return None;
        }
        let naga::Expression::Binary { op, left, right } = &ctx.func.expressions[value] else {
            return None;
        };
        let (cop, commutative) = compound_assign_info(*op)?;

        // Check if left is Load of a structurally-equivalent pointer.
        let left_is_self = !ctx.expr_names.contains_key(left) && {
            if let naga::Expression::Load { pointer: p } = &ctx.func.expressions[*left] {
                ptrs_structurally_equal(*p, pointer, &ctx.func.expressions)
            } else {
                false
            }
        };
        if left_is_self {
            return Some((cop, *right));
        }

        // For commutative ops, also check the right side - but a `Multiply`
        // must additionally be type-commutative, or swapping a matrix product
        // would silently transpose it.  The left-self fold above preserves
        // operand order unconditionally and needs no such guard.
        if commutative
            && !ctx.expr_names.contains_key(right)
            && let naga::Expression::Load { pointer: p } = &ctx.func.expressions[*right]
            && ptrs_structurally_equal(*p, pointer, &ctx.func.expressions)
            && (*op != naga::BinaryOperator::Multiply
                || self.multiply_is_commutative(*left, *right, ctx))
        {
            return Some((cop, *left));
        }

        None
    }

    /// WGSL `*` is commutative for scalar and component-wise products
    /// (`s*t`, `v*v`, `s*v`, `s*m`, ...) but NOT for linear-algebra products
    /// (`mat*mat`, `mat*vec`, `vec*mat`).  Swapping operands when folding
    /// `lhs = rhs * lhs` into `lhs *= rhs` only preserves meaning for the
    /// commutative cases; for a matrix product it would compute the opposite
    /// (transposed) product.  Resolve both operand value types to decide.
    fn multiply_is_commutative(
        &self,
        left: naga::Handle<naga::Expression>,
        right: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::TypeInner as TI;
        let l = ctx.info[left].ty.inner_with(&self.module.types);
        let r = ctx.info[right].ty.inner_with(&self.module.types);
        let l_mat = matches!(l, TI::Matrix { .. });
        let r_mat = matches!(r, TI::Matrix { .. });
        let l_vec = matches!(l, TI::Vector { .. });
        let r_vec = matches!(r, TI::Vector { .. });
        // Non-commutative exactly when a matrix multiplies another matrix or a
        // vector (on either side); matrix*scalar and scalar*matrix stay safe.
        !((l_mat && (r_mat || r_vec)) || (r_mat && l_vec))
    }

    /// Recognise `lhs += 1` / `lhs -= 1` on a concrete 32-bit integer
    /// scalar and return the WGSL increment / decrement statement token
    /// (`++` / `--`), which is one byte shorter than the compound form.
    /// WGSL permits `++`/`--` only on references to integer scalars, so
    /// floats, vectors, bools, and 64-bit integers keep `+= 1` / `-= 1`.
    /// `value` is the `Store`'s value expression (the `lhs <op> 1` binary)
    /// whose resolved type is exactly the lvalue's store type.
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
        // The operand that `++`/`--` drops must be exactly the literal 1.
        if ctx.expr_names.contains_key(&other) {
            return None;
        }
        let is_one = matches!(
            ctx.func.expressions[other],
            naga::Expression::Literal(
                naga::Literal::I32(1) | naga::Literal::U32(1) | naga::Literal::AbstractInt(1)
            )
        );
        if !is_one {
            return None;
        }
        // The store target must be a concrete integer scalar (i32/u32);
        // anything else (float, vector, bool, 64-bit) is not a valid
        // increment target in WGSL.
        matches!(
            ctx.info[value].ty.inner_with(&self.module.types),
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Sint | naga::ScalarKind::Uint,
                width: 4,
            })
        )
        .then_some(token)
    }

    /// Emit the right-hand side of a compound assignment, applying splat
    /// elision when the operand is a Splat / splat-Compose and the compound
    /// operator is arithmetic (`+=`, `-=`, `*=`, `/=`, `%=`).
    fn emit_compound_assign_rhs(
        &self,
        cop: &str,
        other: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if matches!(cop, "+=" | "-=" | "*=" | "/=" | "%=") {
            let cached = ctx.expr_names.contains_key(&other);
            if let Some(scalar) = self.try_splat_scalar(other, &ctx.func.expressions, cached) {
                return self.emit_constructor_arg(scalar, ctx);
            }
        }
        self.emit_expr(other, ctx)
    }

    /// Emit a call statement, binding its result only when something reads
    /// it.  A never-read result (`ref_count == 0`) or no result handle emits
    /// the bare `<call>;`, preserving the side effect without a dead `let`.
    /// A single-use result in an inline-safe zone (pre-computed in
    /// `inlineable_calls`) stashes the call text for inline emission at the
    /// use site.  Otherwise it binds `let <name> = <call>;`.
    fn emit_call_result(
        &mut self,
        call: &str,
        result: Option<naga::Handle<naga::Expression>>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) {
        if let Some(handle) = result {
            // Result is never read (e.g. `atomicAdd(&ctr, 5u);` written as a
            // bare statement): emit the bare call so its side effect still
            // runs, but drop the dead `let` binding.  An impure call cannot be
            // DCE'd, so without this it would always carry an unread `let`.
            // A Call/Atomic result is bumped only by genuine uses (statement
            // operands or children of live expressions), never by its own
            // producing statement, so ref_count 0 means the result is unread.
            if ctx.ref_counts[handle.index()] == 0 {
                self.out.push_str(call);
                self.out.push(';');
                return;
            }
            if ctx.inlineable_calls.contains(&handle) {
                // Single-use in a safe zone: store call text for inline emission
                ctx.expr_names.insert(handle, call.to_string());
                return;
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
        match ctx.info[expr].ty.inner_with(&self.module.types) {
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

        let literal = match &ctx.func.expressions[expr] {
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

        let image_inner = if let Some(gh) = resolve_global_image(expr, &ctx.func.expressions) {
            &self.module.types[self.module.global_variables[gh].ty].inner
        } else {
            let inner = ctx.info[expr].ty.inner_with(&self.module.types);
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

/// Return the WGSL compound-assignment token (`+=`, `*=`, and so on) for a
/// binary operator, plus whether the operator is commutative - i.e. whether a
/// `lhs = rhs <op> lhs` store may also fold by matching the right operand.
/// (`Multiply` is flagged commutative here, but the caller further excludes
/// non-commutative matrix products.)  `None` when the operator has no
/// compound form.
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

/// `true` when two pointer expressions walk identically through
/// `Access` and `AccessIndex` chains and share the same root
/// (`GlobalVariable`, `LocalVariable`, or `FunctionArgument`).
/// Used to detect `lhs = lhs <op> rhs` patterns safe to fold into
/// `lhs <op>= rhs`.
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
            // Only match when both indices are identical literals.
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

/// Render a `(SubgroupOperation, CollectiveOperation)` pair as its
/// WGSL built-in name (`subgroupInclusiveAdd`, and so on).
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

/// Render a subgroup `GatherMode` as the
/// `(builtin_name, Option<index_handle>)` pair the caller needs to
/// build a WGSL call expression for broadcast / shuffle /
/// quad-broadcast variants.
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
