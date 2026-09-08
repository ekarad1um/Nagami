//! Whether statements reference a given function-local; the pointer-chain
//! root itself comes from [`crate::passes::expr_util::root_local_var`].

use crate::passes::expr_util::root_local_var;

/// `true` when any statement in `stmts` reads or writes `local`, access chains
/// and nested blocks included.  Gates the deferred-var `Store` + `Loop`
/// absorption: the for-init scopes the local to the loop body, so nothing after
/// the loop may reference it.
pub(in crate::generator) fn local_var_in_stmts(
    stmts: &[&naga::Statement],
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    stmts
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

fn local_var_in_block(
    block: &naga::Block,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    block
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

/// A value read is a `Load` in an `Emit` range; every other reference is a
/// pointer operand, which `root_local_var` roots (value operands root
/// nothing).
fn local_var_in_stmt(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    if let naga::Statement::Emit(range) = stmt {
        return range.clone().any(|h| {
            matches!(&expressions[h], naga::Expression::Load { pointer }
                    if root_local_var(*pointer, expressions) == Some(local))
        });
    }
    let mut hit = false;
    crate::passes::expr_util::visit_statement_operands(stmt, false, &mut |h| {
        hit |= root_local_var(h, expressions) == Some(local);
    });
    hit || crate::passes::expr_util::nested_blocks(stmt)
        .any(|block| local_var_in_block(block, local, expressions))
}
