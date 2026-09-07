//! Deferred-variable and for-loop-variable analysis: which locals can
//! defer their declaration to their first whole store (any initializer
//! is provably dead there), and which loop counters can be absorbed
//! into a `for` header.

use super::local_resolve::resolve_local_var;
use crate::handle_set::HandleSet;

/// Mark in `seen` every local `block` references anywhere in its subtree.  A
/// value read is a `Load` in an `Emit` range (`expr_reads`); every other
/// reference is a pointer chain in a statement operand, which
/// `resolve_local_var` roots (value operands root nothing).
fn collect_block_local_refs(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    seen: &mut [bool],
) {
    crate::passes::expr_util::for_each_statement(block, &mut |stmt| match stmt {
        naga::Statement::Emit(range) => {
            for h in range.clone() {
                if let Some(lh) = expr_reads[h.index()] {
                    seen[lh.index()] = true;
                }
            }
        }
        other => crate::passes::expr_util::visit_statement_operands(other, false, &mut |h| {
            if let Some(lh) = resolve_local_var(h, expressions) {
                seen[lh.index()] = true;
            }
        }),
    });
}

/// `(deferrable, dead)` bitmaps indexed by local handle: locals whose
/// declaration can defer to their first `Store` (at any depth), and locals never
/// referenced.  A local defers when its first reference in a block (reads,
/// writes and sub-block references alike) is a direct whole-variable `Store`
/// at that level and every reference stays inside that block, so the `var`
/// emitted at the store stays in scope.  The first condition is what makes any
/// initialiser dead: a full overwrite precedes every observation, so the
/// deferred `var` drops the init; a live init means the first reference is a
/// read, which fails it.
pub(in crate::generator) fn find_deferrable_vars(func: &naga::Function) -> (Vec<bool>, Vec<bool>) {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    let candidates = vec![true; local_len];

    let mut deferrable = vec![false; local_len];
    scan_block_deferrable_vars(
        &func.body,
        &func.expressions,
        &expr_reads,
        &candidates,
        &mut deferrable,
    );

    let mut seen = vec![false; local_len];
    collect_block_local_refs(&func.body, &func.expressions, &expr_reads, &mut seen);
    let mut dead = vec![false; local_len];
    for (h, _) in func.local_variables.iter() {
        if !seen[h.index()] {
            dead[h.index()] = true;
        }
    }

    (deferrable, dead)
}

/// Mark a candidate deferrable when its first program-order touch at this
/// block level is a direct whole-variable `Store`, recursing into sub-blocks
/// that own all of a candidate's references.  `candidates[i]`: local `i` has
/// every reference within `block` and is not yet deferrable.
fn scan_block_deferrable_vars(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut Vec<bool>,
) {
    let local_len = result.len();
    if !candidates.iter().any(|&b| b) {
        return;
    }

    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);

    // A direct Store to a candidate not yet seen at this level defers it.
    let mut seen = vec![false; local_len];
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()]
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    if candidates[lh.index()] && !seen[lh.index()] && !result[lh.index()] {
                        result[lh.index()] = true;
                    }
                    seen[lh.index()] = true;
                } else if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    // An indirect store never defers, but marks the variable seen.
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            other => {
                crate::passes::expr_util::visit_statement_operands(other, false, &mut |h| {
                    if let Some(lh) = resolve_local_var(h, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                });
                for nested in crate::passes::expr_util::nested_blocks(other) {
                    collect_block_local_refs(nested, expressions, expr_reads, &mut seen);
                }
            }
        }
    }

    // A candidate owned by a single compound statement may defer inside it.
    for (idx, stmt) in block.iter().enumerate() {
        let any_owned =
            (0..local_len).any(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx));
        if !any_owned {
            continue;
        }

        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                let mut seen_a = vec![false; local_len];
                let mut seen_r = vec![false; local_len];
                collect_block_local_refs(accept, expressions, expr_reads, &mut seen_a);
                collect_block_local_refs(reject, expressions, expr_reads, &mut seen_r);
                let a_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_a[i]
                            && !seen_r[i]
                    })
                    .collect();
                let r_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_r[i]
                            && !seen_a[i]
                    })
                    .collect();
                scan_block_deferrable_vars(accept, expressions, expr_reads, &a_cands, result);
                scan_block_deferrable_vars(reject, expressions, expr_reads, &r_cands, result);
            }
            naga::Statement::Switch { cases, .. } => {
                let case_seen: Vec<Vec<bool>> = cases
                    .iter()
                    .map(|case| {
                        let mut s = vec![false; local_len];
                        collect_block_local_refs(&case.body, expressions, expr_reads, &mut s);
                        s
                    })
                    .collect();
                for (ci, case) in cases.iter().enumerate() {
                    let cands: Vec<bool> = (0..local_len)
                        .map(|i| {
                            candidates[i]
                                && !result[i]
                                && ref_owner[i] == Some(idx)
                                && case_seen[ci][i]
                                && !case_seen.iter().enumerate().any(|(cj, s)| cj != ci && s[i])
                        })
                        .collect();
                    scan_block_deferrable_vars(&case.body, expressions, expr_reads, &cands, result);
                }
            }
            naga::Statement::Block(inner) => {
                let cands: Vec<bool> = (0..local_len)
                    .map(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx))
                    .collect();
                scan_block_deferrable_vars(inner, expressions, expr_reads, &cands, result);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                let mut seen_b = vec![false; local_len];
                let mut seen_c = vec![false; local_len];
                collect_block_local_refs(body, expressions, expr_reads, &mut seen_b);
                collect_block_local_refs(continuing, expressions, expr_reads, &mut seen_c);
                let b_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_b[i]
                            && !seen_c[i]
                    })
                    .collect();
                scan_block_deferrable_vars(body, expressions, expr_reads, &b_cands, result);
            }
            _ => {}
        }
    }
}

/// Per-local bitmap of counters whose references are confined to exactly one
/// `Loop` (at any depth), absorbable into that loop's `for(var x=init;...)` /
/// `for(var x:type;...)` header.
pub(super) fn find_for_loop_vars(
    func: &naga::Function,
    must_bind_loads: &HandleSet<naga::Expression>,
) -> Vec<bool> {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    // A local without an explicit init is zero-initialised and still a counter
    // candidate.
    let candidates = vec![true; local_len];

    let mut result = vec![false; local_len];
    scan_block_for_loop_vars(
        &func.body,
        &func.expressions,
        &func.local_variables,
        &expr_reads,
        &candidates,
        &mut result,
        must_bind_loads,
        false,
    );
    result
}

/// Owner sentinel: referenced by more than one statement.
const MULTI_OWNER: usize = usize::MAX;

fn mark_owner(ref_owner: &mut [Option<usize>], lh_idx: usize, idx: usize) {
    match ref_owner[lh_idx] {
        None => ref_owner[lh_idx] = Some(idx),
        Some(prev) if prev == idx => {}
        _ => ref_owner[lh_idx] = Some(MULTI_OWNER),
    }
}

/// Per local, the statement index in `block` owning all of its references:
/// `None` when unreferenced, `Some(MULTI_OWNER)` when several statements do.
fn compute_block_ownership(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    local_len: usize,
) -> Vec<Option<usize>> {
    let mut ref_owner: Vec<Option<usize>> = vec![None; local_len];
    let mut tmp_seen = vec![false; local_len];

    for (idx, stmt) in block.iter().enumerate() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()] {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            other => crate::passes::expr_util::visit_statement_operands(other, false, &mut |h| {
                if let Some(lh) = resolve_local_var(h, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }),
        }
        let mut nested = crate::passes::expr_util::nested_blocks(stmt).peekable();
        if nested.peek().is_none() {
            continue;
        }
        tmp_seen.fill(false);
        for sub in nested {
            collect_block_local_refs(sub, expressions, expr_reads, &mut tmp_seen);
        }
        for (i, &s) in tmp_seen.iter().enumerate() {
            if s {
                mark_owner(&mut ref_owner, i, idx);
            }
        }
    }

    ref_owner
}

/// Whether the emitter will render this `Loop` as a `for`: the same parse,
/// update-kind check, preload-safety predicate and header depth cap as the
/// emitter, so counter-`var` suppression can never disagree with the emission
/// decision (a disagreement leaves the counter undeclared).
fn is_for_loop_candidate(
    body: &naga::Block,
    continuing: &naga::Block,
    break_if: &Option<naga::Handle<naga::Expression>>,
    expressions: &naga::Arena<naga::Expression>,
    must_bind_loads: &HandleSet<naga::Expression>,
) -> bool {
    let Some(shape) = crate::generator::stmt_emit::parse_for_loop_shape(body, continuing, break_if)
    else {
        return false;
    };
    if let Some(stmt) = shape.update_stmt
        && !matches!(
            stmt,
            naga::Statement::Store { .. }
                | naga::Statement::Call { .. }
                | naga::Statement::ImageStore { .. }
        )
    {
        return false;
    }
    crate::generator::stmt_emit::for_loop_preload_inlining_is_safe(
        &shape,
        body,
        continuing,
        expressions,
        must_bind_loads,
    ) && !crate::generator::stmt_emit::for_header_exceeds_depth_cap(&shape, expressions)
        && !crate::generator::stmt_emit::for_header_has_msl_cast_ambiguity(&shape, expressions)
}

/// Find for-shaped `Loop`s that fully confine candidate locals, recursing into
/// nested blocks.  `candidates[i]`: local `i` has every reference within
/// `block` and is eligible for absorption.
#[allow(clippy::too_many_arguments)]
fn scan_block_for_loop_vars(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    local_variables: &naga::Arena<naga::LocalVariable>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut Vec<bool>,
    must_bind_loads: &HandleSet<naga::Expression>,
    // `true` once the walk has descended through a `Loop`.  Absorbing a counter
    // via its declaration / zero-init (not an explicit pre-loop `Store`) is
    // sound only at top level: nested in another loop that init would
    // re-execute every outer iteration, whereas the source declared the
    // counter once.  Nested `for`s re-init via a `Store` (the deferred-var
    // path, which ignores this flag).
    inside_loop: bool,
) {
    let local_len = result.len();
    if !candidates.iter().any(|&b| b) {
        return;
    }

    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);
    let stmts: Vec<_> = block.iter().collect();

    for (h, _local) in local_variables.iter() {
        let i = h.index();
        if !candidates[i] || result[i] {
            continue;
        }
        if let Some(owner) = ref_owner[i] {
            if owner == MULTI_OWNER {
                continue;
            }
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = stmts[owner]
                && is_for_loop_candidate(body, continuing, break_if, expressions, must_bind_loads)
            {
                let has_update = continuing.iter().any(|s| {
                    if let naga::Statement::Store { pointer, .. } = s {
                        matches!(
                            expressions[*pointer],
                            naga::Expression::LocalVariable(lh) if lh == h
                        )
                    } else {
                        false
                    }
                });
                if has_update && !inside_loop {
                    result[i] = true;
                }
            }
        }
    }

    for (idx, stmt) in block.iter().enumerate() {
        let any_owned =
            (0..local_len).any(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx));
        if !any_owned {
            continue;
        }

        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                let mut seen_a = vec![false; local_len];
                let mut seen_r = vec![false; local_len];
                collect_block_local_refs(accept, expressions, expr_reads, &mut seen_a);
                collect_block_local_refs(reject, expressions, expr_reads, &mut seen_r);
                let a_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_a[i]
                            && !seen_r[i]
                    })
                    .collect();
                let r_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_r[i]
                            && !seen_a[i]
                    })
                    .collect();
                scan_block_for_loop_vars(
                    accept,
                    expressions,
                    local_variables,
                    expr_reads,
                    &a_cands,
                    result,
                    must_bind_loads,
                    inside_loop,
                );
                scan_block_for_loop_vars(
                    reject,
                    expressions,
                    local_variables,
                    expr_reads,
                    &r_cands,
                    result,
                    must_bind_loads,
                    inside_loop,
                );
            }
            naga::Statement::Switch { cases, .. } => {
                let case_seen: Vec<Vec<bool>> = cases
                    .iter()
                    .map(|case| {
                        let mut s = vec![false; local_len];
                        collect_block_local_refs(&case.body, expressions, expr_reads, &mut s);
                        s
                    })
                    .collect();
                for (ci, case) in cases.iter().enumerate() {
                    let cands: Vec<bool> = (0..local_len)
                        .map(|i| {
                            candidates[i]
                                && !result[i]
                                && ref_owner[i] == Some(idx)
                                && case_seen[ci][i]
                                && !case_seen.iter().enumerate().any(|(cj, s)| cj != ci && s[i])
                        })
                        .collect();
                    scan_block_for_loop_vars(
                        &case.body,
                        expressions,
                        local_variables,
                        expr_reads,
                        &cands,
                        result,
                        must_bind_loads,
                        inside_loop,
                    );
                }
            }
            naga::Statement::Block(inner) => {
                let cands: Vec<bool> = (0..local_len)
                    .map(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx))
                    .collect();
                scan_block_for_loop_vars(
                    inner,
                    expressions,
                    local_variables,
                    expr_reads,
                    &cands,
                    result,
                    must_bind_loads,
                    inside_loop,
                );
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Whether or not this Loop absorbed a counter, locals confined
                // to its body may still be absorbed by an inner loop.
                let mut seen_b = vec![false; local_len];
                let mut seen_c = vec![false; local_len];
                collect_block_local_refs(body, expressions, expr_reads, &mut seen_b);
                collect_block_local_refs(continuing, expressions, expr_reads, &mut seen_c);
                let b_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_b[i]
                            && !seen_c[i]
                    })
                    .collect();
                scan_block_for_loop_vars(
                    body,
                    expressions,
                    local_variables,
                    expr_reads,
                    &b_cands,
                    result,
                    must_bind_loads,
                    // Inside this Loop a counter is nested.
                    true,
                );
            }
            _ => {}
        }
    }
}
