//! Expression ref-count analysis: how many live consumers each
//! expression handle has, and which handles appear in `Emit` ranges.

use crate::generator::core::FunctionExprInfo;

/// [`crate::passes::expr_util::live_expression_ref_counts`] in the shape the
/// generator caches it.
pub(super) fn compute_expression_ref_counts(func: &naga::Function) -> FunctionExprInfo {
    let (ref_counts, live) = crate::passes::expr_util::live_expression_ref_counts(func);
    FunctionExprInfo { ref_counts, live }
}

/// Discount references made from initializer trees that no body `let` can
/// serve: a hoisted `var x = <init>;` renders its tree at the top of the body,
/// before any `Emit` range is priced, and a deferred or dead local never
/// renders it; only a `for`-header local renders its init after the `Emit`,
/// where a name does serve it.  Without the discount a sub-expression the tree
/// shares (`array(l(1,0), .., l(1,0))` after CSE) is bound on the tree's
/// account alone, dead text.  Each child occurrence counts once, so a subtree
/// two initializers share is discounted once.
pub(super) fn discount_initializer_refs(
    func: &naga::Function,
    for_loop_vars: &[bool],
    counts: &mut [usize],
) {
    let mut visited = vec![false; func.expressions.len()];
    let mut stack: Vec<_> = func
        .local_variables
        .iter()
        .filter(|(lh, _)| !for_loop_vars[lh.index()])
        .filter_map(|(_, local)| local.init)
        .collect();
    while let Some(h) = stack.pop() {
        if std::mem::replace(&mut visited[h.index()], true) {
            continue;
        }
        crate::passes::expr_util::visit_expression_children(&func.expressions[h], |child| {
            counts[child.index()] = counts[child.index()].saturating_sub(1);
            stack.push(child);
        });
    }
}
