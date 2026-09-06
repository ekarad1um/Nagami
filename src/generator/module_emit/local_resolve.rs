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

/// Single-statement variant of [`local_var_in_stmts`].  A value read is a
/// `Load` in an `Emit` range; every other reference is a pointer operand,
/// which `resolve_local_var` roots (value operands root nothing).
fn local_var_in_stmt(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    if let naga::Statement::Emit(range) = stmt {
        return range.clone().any(|h| {
            matches!(&expressions[h], naga::Expression::Load { pointer }
                    if resolve_local_var(*pointer, expressions) == Some(local))
        });
    }
    let mut hit = false;
    crate::passes::expr_util::visit_statement_operands(stmt, false, &mut |h| {
        hit |= resolve_local_var(h, expressions) == Some(local);
    });
    hit || crate::passes::expr_util::nested_blocks(stmt)
        .any(|block| local_var_in_block(block, local, expressions))
}
