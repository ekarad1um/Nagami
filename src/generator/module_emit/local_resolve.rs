//! Local-variable reference resolution: root a pointer chain at its
//! function-local and detect whether statements reference a given local.

/// Walk a chain of `AccessIndex` / `Access` / `LocalVariable`
/// expressions and return the root local when the chain ultimately
/// resolves to one, or `None` otherwise.
pub(super) fn resolve_local_var(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match &expressions[expr] {
        naga::Expression::LocalVariable(lh) => Some(*lh),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            resolve_local_var(*base, expressions)
        }
        _ => None,
    }
}

/// Check if `local` is referenced (read *or* written) in any of `stmts`.
/// Scans recursively into nested blocks.  Used in the deferred-var look-ahead
/// safety check: a deferred-var Store + Loop may only be absorbed into a
/// `for(var x=init;...)` when `x` is not referenced after the Loop,
/// because the for-init scopes `x` inside the loop body.
/// `true` when any statement in `stmts` references the local `lh`,
/// either directly, through an access chain, or inside a nested block.
pub(in crate::generator) fn local_var_in_stmts(
    stmts: &[&naga::Statement],
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    stmts
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

/// Block-scoped variant of [`local_var_in_stmts`].
fn local_var_in_block(
    block: &naga::Block,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    block
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

/// Single-statement variant of [`local_var_in_stmts`].  Recurses
/// into nested control-flow blocks and into expression operands via
/// [`resolve_local_var`].
fn local_var_in_stmt(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Expression as E;
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => range.clone().any(|h| {
            matches!(&expressions[h], E::Load { pointer }
                    if resolve_local_var(*pointer, expressions) == Some(local))
        }),
        // These statement-level arms resolve only the bare POINTER/place operand
        // (a write target or taken address).  A local used BY VALUE - an atomic
        // `value`/`compare`, a call/image/ray operand - is a `Load(LocalVariable)`
        // expression, which `resolve_local_var` does not see through (it returns
        // `None` for a `Load`); such reads are already caught by the `Emit` arm
        // above.  So intentionally omitting value/compare operands here cannot
        // miss a reference - the same place/value split
        // `collect_block_local_refs` relies on.
        S::Store { pointer, .. }
        | S::WorkGroupUniformLoad { pointer, .. }
        | S::Atomic { pointer, .. } => resolve_local_var(*pointer, expressions) == Some(local),
        S::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => [Some(*image), Some(*coordinate), *array_index, Some(*value)]
            .into_iter()
            .flatten()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::Call { arguments, .. } => arguments
            .iter()
            .any(|&a| resolve_local_var(a, expressions) == Some(local)),
        S::Return { value: Some(v) } => resolve_local_var(*v, expressions) == Some(local),
        S::If { accept, reject, .. } => {
            local_var_in_block(accept, local, expressions)
                || local_var_in_block(reject, local, expressions)
        }
        S::Switch { cases, .. } => cases
            .iter()
            .any(|c| local_var_in_block(&c.body, local, expressions)),
        S::Loop {
            body, continuing, ..
        } => {
            local_var_in_block(body, local, expressions)
                || local_var_in_block(continuing, local, expressions)
        }
        S::Block(inner) => local_var_in_block(inner, local, expressions),
        // Pointer/operand-bearing statements that
        // `collect_block_local_refs` also handles.  Without these arms a
        // post-loop reference to the absorbed induction local through one of
        // them would be missed, leaving the for-init-scoped local referenced
        // out of scope (invalid WGSL).  The genuinely-reachable case for a
        // function-local is `CooperativeStore`'s `data.pointer`.
        S::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            ..
        } => [Some(*image), Some(*coordinate), *array_index, Some(*value)]
            .into_iter()
            .flatten()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::SubgroupBallot {
            predicate: Some(p), ..
        } => resolve_local_var(*p, expressions) == Some(local),
        S::SubgroupGather { mode, argument, .. } => {
            let hits = |e| resolve_local_var(e, expressions) == Some(local);
            hits(*argument)
                || match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => hits(*h),
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => false,
                }
        }
        S::SubgroupCollectiveOperation { argument, .. } => {
            resolve_local_var(*argument, expressions) == Some(local)
        }
        S::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            acceleration_structure,
            descriptor,
            payload,
        }) => [*acceleration_structure, *descriptor, *payload]
            .into_iter()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::CooperativeStore { target, data } => [*target, data.pointer, data.stride]
            .into_iter()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::RayQuery { query, fun } => {
            let hits = |e| resolve_local_var(e, expressions) == Some(local);
            hits(*query)
                || match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => hits(*acceleration_structure) || hits(*descriptor),
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => hits(*hit_t),
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => false,
                }
        }
        _ => false,
    }
}
