//! Deferred-variable and for-loop-variable analysis: which locals can
//! defer their declaration to their first whole store (any initializer
//! is provably dead there), and which loop counters can be absorbed
//! into a `for` header.

use crate::handle_set::HandleSet;
use crate::passes::expr_util::root_local_var;

/// Mark in `seen` every local `block` references anywhere in its subtree.  A
/// value read is a `Load` in an `Emit` range (`expr_reads`); every other
/// reference is a pointer chain in a statement operand, which
/// `root_local_var` roots (value operands root nothing).
fn collect_block_local_refs(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    seen: &mut [bool],
) {
    crate::ir::visit::for_each_statement(block, &mut |stmt| match stmt {
        naga::Statement::Emit(range) => {
            for h in range.clone() {
                if let Some(lh) = expr_reads[h.index()] {
                    seen[lh.index()] = true;
                }
            }
        }
        other => crate::ir::visit::visit_statement_operands(other, false, &mut |h| {
            if let Some(lh) = root_local_var(h, expressions) {
                seen[lh.index()] = true;
            }
        }),
    });
}

/// Per-expression: the local a `Load` reads through.  Both scans below index
/// by expression and building it is a whole pass over the arena, so it is
/// built once here.
fn load_source_locals(func: &naga::Function) -> Vec<Option<naga::Handle<naga::LocalVariable>>> {
    let mut expr_reads = vec![None; func.expressions.len()];
    for (eh, expr) in func.expressions.iter() {
        if let naga::Expression::Load { pointer } = *expr
            && let Some(lh) = root_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }
    expr_reads
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
    let local_len = func.local_variables.len();
    let expr_reads = load_source_locals(func);

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

/// What [`for_each_owned_arm`] hands each arm it descends into: the arm, the
/// candidates that may still resolve inside it, whether the descent crossed a
/// `Loop`, and the per-local result table to record into.
type OwnedArmVisit<'v, R> = dyn FnMut(&naga::Block, &[bool], bool, &mut R) + 'v;

/// Descend into each compound statement that SOLELY owns a candidate, handing
/// `visit` the arm and the candidates that may still resolve inside it.  A
/// candidate survives into an arm only where that arm alone references it: an
/// `If` arm its sibling also touches, or a `Switch` case another case touches,
/// would move the declaration out from under one of the uses.  A `Block` owns
/// its candidates outright, and a `Loop`'s `continuing` is a sibling of its
/// body.
#[allow(clippy::too_many_arguments)]
fn for_each_owned_arm<R>(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut R,
    resolved: fn(&R, usize) -> bool,
    ref_owner: &[Option<usize>],
    visit: &mut OwnedArmVisit<'_, R>,
) {
    let local_len = candidates.len();
    for (idx, stmt) in block.iter().enumerate() {
        let owned: Vec<bool> = (0..local_len)
            .map(|i| candidates[i] && !resolved(result, i) && ref_owner[i] == Some(idx))
            .collect();
        if !owned.contains(&true) {
            continue;
        }
        let seen_in = |b: &naga::Block| {
            let mut s = vec![false; local_len];
            collect_block_local_refs(b, expressions, expr_reads, &mut s);
            s
        };
        let only = |mine: &[bool], theirs: &[bool]| -> Vec<bool> {
            (0..local_len)
                .map(|i| owned[i] && mine[i] && !theirs[i])
                .collect()
        };
        // Choosing every arm's candidates from one `owned` snapshot before any
        // `visit` runs is exact: the arms of a statement are DISJOINT in their
        // candidates (each keeps only what it alone references) and a recursion
        // resolves nothing outside the candidates it was handed, so no arm can
        // change what a later arm of the same statement would have taken.
        let arms: Vec<(&naga::Block, Vec<bool>, bool)> = match stmt {
            naga::Statement::If { accept, reject, .. } => {
                let (seen_a, seen_r) = (seen_in(accept), seen_in(reject));
                vec![
                    (accept, only(&seen_a, &seen_r), false),
                    (reject, only(&seen_r, &seen_a), false),
                ]
            }
            naga::Statement::Switch { cases, .. } => {
                let case_seen: Vec<Vec<bool>> =
                    cases.iter().map(|case| seen_in(&case.body)).collect();
                cases
                    .iter()
                    .enumerate()
                    .map(|(ci, case)| {
                        let cands = (0..local_len)
                            .map(|i| {
                                owned[i]
                                    && case_seen[ci][i]
                                    && !case_seen.iter().enumerate().any(|(cj, s)| cj != ci && s[i])
                            })
                            .collect();
                        (&case.body, cands, false)
                    })
                    .collect()
            }
            naga::Statement::Block(inner) => vec![(inner, owned.clone(), false)],
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                let (seen_b, seen_c) = (seen_in(body), seen_in(continuing));
                vec![(body, only(&seen_b, &seen_c), true)]
            }
            _ => Vec::new(),
        };
        for (arm, cands, in_loop) in arms {
            visit(arm, &cands, in_loop, result);
        }
    }
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
                } else if let Some(lh) = root_local_var(*pointer, expressions) {
                    // An indirect store never defers, but marks the variable seen.
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            other => {
                crate::ir::visit::visit_statement_operands(other, false, &mut |h| {
                    if let Some(lh) = root_local_var(h, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                });
                for nested in crate::ir::visit::nested_blocks(other) {
                    collect_block_local_refs(nested, expressions, expr_reads, &mut seen);
                }
            }
        }
    }

    for_each_owned_arm(
        block,
        expressions,
        expr_reads,
        candidates,
        result,
        |done, i| done[i],
        &ref_owner,
        &mut |arm, cands, _in_loop, result| {
            scan_block_deferrable_vars(arm, expressions, expr_reads, cands, result)
        },
    );
}

/// Per local, the guard of the for-shaped `Loop` (at any depth) whose header
/// declares it - `for(var x=init;...)` / `for(var x:type;...)` - because the
/// loop alone references it.  One local per loop, the most loop-variable-like:
/// the one `continuing` stores, else - unless `deferred` declares it at its
/// first store (`var G=i;` there beats `for(var G=0i;..){G=i;`) or a pre-loop
/// `[Store, Loop]` init holds the header - the one the guard reads, else one
/// the body whole-stores, ties to the one the body touches first.  The
/// emitter keys the header on the same guard; loops may share one, so it
/// also checks the local is its own.
pub(super) fn find_for_loop_vars(
    func: &naga::Function,
    must_bind: &HandleSet<naga::Expression>,
    deferred: &[bool],
) -> Vec<Option<naga::Handle<naga::Expression>>> {
    let local_len = func.local_variables.len();
    let expr_reads = load_source_locals(func);

    // A local without an explicit init is zero-initialised and still a
    // candidate.
    let candidates = vec![true; local_len];

    let mut result = vec![None; local_len];
    scan_block_for_loop_vars(
        &func.body,
        &func.expressions,
        &func.local_variables,
        &expr_reads,
        &candidates,
        &mut result,
        must_bind,
        deferred,
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
            other => crate::ir::visit::visit_statement_operands(other, false, &mut |h| {
                if let Some(lh) = root_local_var(h, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }),
        }
        let mut nested = crate::ir::visit::nested_blocks(stmt).peekable();
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

/// `true` when `condition`'s cone loads through `local`.
fn guard_reads(
    condition: naga::Handle<naga::Expression>,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
) -> bool {
    let mut pending = vec![condition];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(h) = pending.pop() {
        if !seen.insert(h) {
            continue;
        }
        if expr_reads[h.index()] == Some(local) {
            return true;
        }
        crate::ir::visit::visit_expression_children(&expressions[h], |c| pending.push(c));
    }
    false
}

/// Preorder index of the first statement in `block` whose own operands or
/// `Emit` loads reference `local`; `usize::MAX` when none does.
fn first_reference(
    block: &naga::Block,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
) -> usize {
    let (mut pos, mut first) = (0, usize::MAX);
    crate::ir::visit::for_each_statement(block, &mut |stmt| {
        if first == usize::MAX {
            let hit = match stmt {
                naga::Statement::Emit(range) => {
                    range.clone().any(|h| expr_reads[h.index()] == Some(local))
                }
                other => {
                    let mut hit = false;
                    crate::ir::visit::visit_statement_operands(other, false, &mut |h| {
                        hit |= root_local_var(h, expressions) == Some(local);
                    });
                    hit
                }
            };
            if hit {
                first = pos;
            }
        }
        pos += 1;
    });
    first
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
    result: &mut Vec<Option<naga::Handle<naga::Expression>>>,
    must_bind: &HandleSet<naga::Expression>,
    deferred: &[bool],
    // `true` once the walk has descended through a `Loop`.  Absorbing a counter
    // via its declaration / zero-init (not an explicit pre-loop `Store`) is
    // sound only at top level: nested in another loop that init would
    // re-execute every outer iteration, whereas the source declared the
    // counter once.  Nested `for`s re-init via a `Store` (the deferred-var
    // path, which ignores this flag).
    inside_loop: bool,
) {
    let local_len = result.len();
    if inside_loop || !candidates.iter().any(|&b| b) {
        return;
    }

    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);
    let stmts: Vec<_> = block.iter().collect();

    // A whole `Store` to a deferred local right before the loop is the
    // emitter's `[Store, Loop]` init.  It keeps the header unless the loop's
    // counter claims it: handing the slot to another local is byte-neutral
    // and detaches the counter from the header, which costs tint its
    // finiteness proof.
    let external_init = |owner: usize| {
        if owner > 0
            && let naga::Statement::Store { pointer, .. } = stmts[owner - 1]
            && let naga::Expression::LocalVariable(x) = expressions[*pointer]
        {
            deferred[x.index()]
                && !super::local_var_in_stmts(&stmts[..owner - 1], x, expressions)
                && !super::local_var_in_stmts(&stmts[owner + 1..], x, expressions)
        } else {
            false
        }
    };

    // Per loop, the most loop-variable-like of the locals it alone references
    // takes the header: the one `continuing` stores, else the one the guard
    // reads, else one the body whole-stores; ties go to the one the body
    // touches first, then the first declared.  A `for(var i=0;i<n;){..i=i+1..}`
    // re-parsed from this emitter's own text (naga lowers a top-level `for`
    // init to a declaration) would otherwise render `var i=0;for(;i<n;)`,
    // and the rank keeps the choice stable across passes (declaration order
    // is not).  The header holds one declaration.
    type Key = (u8, std::cmp::Reverse<usize>);
    let mut best: Vec<Option<(Key, usize, naga::Handle<naga::Expression>)>> =
        vec![None; stmts.len()];
    for (h, _local) in local_variables.iter() {
        let i = h.index();
        if !candidates[i] || result[i].is_some() {
            continue;
        }
        let Some(owner) = ref_owner[i] else { continue };
        if owner == MULTI_OWNER {
            continue;
        }
        let naga::Statement::Loop {
            body,
            continuing,
            break_if,
        } = stmts[owner]
        else {
            continue;
        };
        let Some(shape) = crate::generator::stmt_emit::emittable_for_loop_shape(
            body,
            continuing,
            break_if,
            expressions,
            must_bind,
        ) else {
            continue;
        };
        let stores_whole = |block: &naga::Block| {
            let mut hit = false;
            crate::ir::visit::for_each_statement(block, &mut |s| {
                hit |= matches!(s, naga::Statement::Store { pointer, .. }
                    if matches!(expressions[*pointer], naga::Expression::LocalVariable(lh) if lh == h));
            });
            hit
        };
        let rank = if stores_whole(continuing) {
            3
        } else if deferred[i] || external_init(owner) {
            continue;
        } else if guard_reads(shape.condition, h, expressions, expr_reads) {
            2
        } else if stores_whole(body) {
            1
        } else {
            0
        };
        let key = (
            rank,
            std::cmp::Reverse(first_reference(body, h, expressions, expr_reads)),
        );
        if best[owner].is_none_or(|(k, ..)| key > k) {
            best[owner] = Some((key, i, shape.condition));
        }
    }
    for (_, i, condition) in best.into_iter().flatten() {
        result[i] = Some(condition);
    }

    for_each_owned_arm(
        block,
        expressions,
        expr_reads,
        candidates,
        result,
        |done, i| done[i].is_some(),
        &ref_owner,
        &mut |arm, cands, arm_in_loop, result| {
            scan_block_for_loop_vars(
                arm,
                expressions,
                local_variables,
                expr_reads,
                cands,
                result,
                must_bind,
                deferred,
                inside_loop || arm_in_loop,
            )
        },
    );
}
