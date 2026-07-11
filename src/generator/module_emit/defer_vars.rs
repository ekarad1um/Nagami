//! Deferred-variable and for-loop-variable analysis: which locals can
//! defer their declaration to their first whole store (any initializer
//! is provably dead there), and which loop counters can be absorbed
//! into a `for` header.

use super::local_resolve::resolve_local_var;

/// Mark in `seen` (a bitmap indexed by local handle) every local that
/// `block` references anywhere in its subtree.  The deferral and
/// for-loop analyses use per-sub-block runs of this census to decide
/// which single sub-block, if any, owns all of a candidate's uses.
fn collect_block_local_refs(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    seen: &mut Vec<bool>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Store { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    seen[lh.index()] = true;
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx) = index
                    && let Some(lh) = resolve_local_var(idx, expressions)
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions) {
                            seen[lh.index()] = true;
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions) {
                    seen[lh.index()] = true;
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions) {
                                seen[lh.index()] = true;
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions) {
                            seen[lh.index()] = true;
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            _ => {}
        }
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            collect_block_local_refs(nested, expressions, expr_reads, seen);
        }
    }
}

/// Identify locals whose declaration can be deferred to the site of
/// their first `Store` (at any nesting depth) and locals that turn
/// out to be entirely dead.  The returned vectors are indexed by
/// local handle.
///
/// A variable is deferrable when BOTH of these hold (the analysis does
/// NOT inspect `local.init` directly):
///
/// 1. its first reference in the enclosing block (considering both
///    reads in `Emit` ranges and writes in `Store` statements, plus
///    any sub-block references) is a *direct* whole-variable `Store`
///    at that block level; and
/// 2. all of its references are confined to that block, so the
///    `var` declaration emitted at the store site stays in scope
///    for every use.
///
/// Condition (1) is exactly what makes any initialiser dead-on-arrival:
/// the first thing that happens to the variable is a full overwrite, so
/// the init value is never observed.  The deferred `var` therefore drops
/// the init and re-emits it as the store's value at the deferred site;
/// a variable whose init IS live fails condition (1) (its first
/// reference is a read) and is not deferred.
pub(in crate::generator) fn find_deferrable_vars(func: &naga::Function) -> (Vec<bool>, Vec<bool>) {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Map expression handles that **read** a local variable (via Load whose
    // pointer chain resolves to a LocalVariable).
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    // All locals are candidates at the top level (the function body contains
    // every possible reference).
    let candidates = vec![true; local_len];

    let mut deferrable = vec![false; local_len];
    scan_block_deferrable_vars(
        &func.body,
        &func.expressions,
        &expr_reads,
        &candidates,
        &mut deferrable,
    );

    // Variables never referenced (no stores, no loads) are dead.
    // Use collect_block_local_refs on the whole function body to build
    // the complete `seen` set.
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

/// DFS walker for [`find_deferrable_vars`]: marks a candidate deferrable
/// when its first program-order touch at this block level is a direct
/// whole-variable `Store`, recursing into sub-blocks that own all of a
/// candidate's references.  `candidates[i]` is `true` when local `i` has
/// all of its references within `block` and is not already deferrable.
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

    // We need ownership info for the recursion step (determining which
    // sub-block a candidate is confined to).
    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);

    // Walk statements in program order, tracking which candidates have been
    // "seen" (any reference).  A direct Store to an unseen candidate is
    // deferrable at this block level.
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
                    // Direct store to the whole variable.
                    if candidates[lh.index()] && !seen[lh.index()] && !result[lh.index()] {
                        result[lh.index()] = true;
                    }
                    seen[lh.index()] = true;
                } else if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    // Indirect store (e.g. field/index access) - not deferrable,
                    // but the variable is now "seen".
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx) = index
                    && let Some(lh) = resolve_local_var(idx, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions)
                            && candidates[lh.index()]
                        {
                            seen[lh.index()] = true;
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions)
                                && candidates[lh.index()]
                            {
                                seen[lh.index()] = true;
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions)
                            && candidates[lh.index()]
                        {
                            seen[lh.index()] = true;
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            _ => {
                // For control-flow and other compound statements, conservatively
                // mark every candidate local referenced in sub-blocks as seen.
                for nested in crate::passes::expr_util::nested_blocks(stmt) {
                    collect_block_local_refs(nested, expressions, expr_reads, &mut seen);
                }
            }
        }
    }

    // Recurse into compound statements for candidates not resolved at this
    // level.  A candidate that is owned by a single compound statement can
    // potentially be deferred inside that statement's sub-block.
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
                // Recurse into body for candidates confined to body only.
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

/// Identify locals whose references are confined to exactly one `Loop`
/// statement (at any nesting depth), so their declaration can be absorbed
/// into `for(var x=init;...)` or `for(var x:type;...)` (no init).
/// Return the per-local bitmap of init-once locals whose uses stay
/// confined to a single `Loop` and can therefore be absorbed into
/// that loop's `for (var ...; ...; ...)` header.
pub(super) fn find_for_loop_vars(
    func: &naga::Function,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> Vec<bool> {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Build Load->LocalVariable map, same pattern as find_deferrable_vars.
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    // All locals are candidates at the top level.
    // Variables without an explicit init are zero-initialised in WGSL and can
    // still serve as for-loop counters (e.g. after a dead-init removal pass).
    let mut candidates = vec![false; local_len];
    for (h, _local) in func.local_variables.iter() {
        candidates[h.index()] = true;
    }

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

/// Sentinel value indicating a local is referenced by multiple statements.
const MULTI_OWNER: usize = usize::MAX;

/// Mark a local as referenced by statement `idx` in a block.  If it was
/// already referenced by a different statement, set it to `MULTI_OWNER`.
/// Tag each local's `ref_owner` slot with the loop index that owns
/// it.  `None` means no owner yet, `Some(idx)` pins the local to one
/// loop, and `Some(sentinel)` marks it as conflicted (used outside
/// any candidate loop).
fn mark_owner(ref_owner: &mut [Option<usize>], lh_idx: usize, idx: usize) {
    match ref_owner[lh_idx] {
        None => ref_owner[lh_idx] = Some(idx),
        Some(prev) if prev == idx => {} // same stmt, no change
        _ => ref_owner[lh_idx] = Some(MULTI_OWNER),
    }
}

/// For each local variable, compute which statement index in `block` "owns"
/// all of its references.  Returns `None` if the local is not referenced in
/// this block, `Some(idx)` if all references are within statement `idx`, or
/// `Some(MULTI_OWNER)` if referenced by multiple statements.
/// Traverse `block`, assigning loop ownership to every local
/// reference and flagging cross-loop escapes.  Cooperating helper
/// for [`find_for_loop_vars`].
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
            naga::Statement::Store { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx_h) = index
                    && let Some(lh) = resolve_local_var(idx_h, expressions)
                {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions) {
                            mark_owner(&mut ref_owner, lh.index(), idx);
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions) {
                                mark_owner(&mut ref_owner, lh.index(), idx);
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions) {
                            mark_owner(&mut ref_owner, lh.index(), idx);
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            _ => {}
        }
        // For compound statements, scan sub-blocks and attribute to `idx`.
        let mut nested = crate::passes::expr_util::nested_blocks(stmt).peekable();
        if nested.peek().is_none() {
            continue; // Leaf statement - no sub-block refs to drain.
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

/// Check whether a `Loop` statement matches the for-loop pattern
/// recognised by `try_emit_for_loop`:
///
/// - `break_if` is `None`;
/// - `continuing` has at most one non-`Emit` statement (the update);
/// - the update, when present, is a `Store`, `Call`, or `ImageStore`;
/// - `body` starts with an if-break guard.
fn is_for_loop_candidate(
    body: &naga::Block,
    continuing: &naga::Block,
    break_if: &Option<naga::Handle<naga::Expression>>,
    expressions: &naga::Arena<naga::Expression>,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    // Parse via the SHARED parser so this var-suppression decision and
    // `try_emit_for_loop`'s emission decision can never drift.  `None` => not
    // for-convertible (break_if present, no if-break guard, or >1 continuing
    // core update statement).
    let Some(shape) = crate::generator::stmt_emit::parse_for_loop_shape(body, continuing, break_if)
    else {
        return false;
    };
    // Update must be Store / Call / ImageStore, matching try_emit_for_loop's
    // pre-validation.
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
    // And the preloads must be safe to inline into the for-header, else
    // try_emit_for_loop bails to plain `loop` emission and the suppressed
    // counter `var` would be left undeclared.  The header depth cap is
    // mirrored for the same reason.
    crate::generator::stmt_emit::for_loop_preload_inlining_is_safe(
        &shape,
        body,
        continuing,
        expressions,
        must_bind_loads,
    ) && !crate::generator::stmt_emit::for_header_exceeds_depth_cap(&shape, expressions)
}

/// Recursively scan a block (and its nested sub-blocks) looking for
/// for-loop-shaped `Loop` statements that fully confine candidate locals.
///
/// `candidates[i]` is `true` when local `i` has all of its references
/// within `block` and is eligible for for-loop absorption.
/// DFS walker that identifies candidate loops and records which
/// locals are safe to absorb into each.  Complements
/// [`compute_block_ownership`], which handles the liveness half of
/// the decision.
#[allow(clippy::too_many_arguments)]
fn scan_block_for_loop_vars(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    local_variables: &naga::Arena<naga::LocalVariable>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut Vec<bool>,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
    // `true` once the walk has descended through a `Loop`.  A counter absorbed
    // into a for-init via its declaration/zero-init (not an explicit pre-loop
    // re-init `Store`) is sound only at top level: nested in another loop that
    // init would re-execute every outer iteration, whereas the source declared
    // the counter once.  Legitimate nested `for`s re-init via a `Store` (the
    // deferred-var path, which ignores this flag), so gating here is safe.
    inside_loop: bool,
) {
    let local_len = result.len();
    if !candidates.iter().any(|&b| b) {
        return;
    }

    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);
    let stmts: Vec<_> = block.iter().collect();

    // Check for-loop candidates: locals owned by a single Loop at this level.
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

    // Recurse into compound statements for remaining candidates.
    for (idx, stmt) in block.iter().enumerate() {
        // Collect candidates that are owned by this statement and not yet resolved.
        let any_owned =
            (0..local_len).any(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx));
        if !any_owned {
            continue;
        }

        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                // Determine which sub-block each candidate is confined to.
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
                // This Loop was either already handled as a for-loop candidate
                // above, or it doesn't match the pattern.  Either way, recurse
                // into the body for locals that are confined to body only
                // (not referenced in continuing).
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
                    // Descending through this Loop: any counter marked below is
                    // nested and must not be absorbed via declaration/zero init.
                    true,
                );
            }
            _ => {}
        }
    }
}
