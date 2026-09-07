//! Expression ref-count analysis: how many live consumers each
//! expression handle has, and which handles appear in `Emit` ranges.

use crate::generator::core::FunctionExprInfo;

/// Per-handle reference counts from live expressions (those in `Emit` ranges)
/// and from statements, plus the live bitmap, in one walk; dead expressions
/// never inflate a count.
pub(super) fn compute_expression_ref_counts(func: &naga::Function) -> FunctionExprInfo {
    let len = func.expressions.len();
    let mut counts: Vec<usize> = vec![0; len];

    let mut live = vec![false; len];
    collect_emitted_handles(&func.body, &mut live);

    for (h, expr) in func.expressions.iter() {
        if live[h.index()] {
            count_expr_children(expr, &mut counts);
        }
    }

    count_block_refs(&func.body, &mut counts);

    FunctionExprInfo {
        ref_counts: counts,
        live,
    }
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

/// `Emit`-range membership across all control flow, the authoritative liveness
/// signal for ref counting and literal extraction.
fn collect_emitted_handles(block: &naga::Block, live: &mut [bool]) {
    crate::passes::expr_util::for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                live[h.index()] = true;
            }
        }
    });
}

fn bump(counts: &mut [usize], h: naga::Handle<naga::Expression>) {
    counts[h.index()] += 1;
}

fn count_expr_children(expr: &naga::Expression, counts: &mut [usize]) {
    crate::passes::expr_util::visit_expression_children(expr, |h| bump(counts, h));
}

/// Bump `counts[h]` per statement operand, nested blocks included.  Defined
/// results are not uses: a call / atomic result stays at 0 until consumed,
/// which single-use call inlining and dead-`let` elision key on.
fn count_block_refs(block: &naga::Block, counts: &mut [usize]) {
    crate::passes::expr_util::for_each_statement(block, &mut |stmt| {
        crate::passes::expr_util::visit_statement_operands(stmt, false, &mut |h| bump(counts, h));
    });
}
