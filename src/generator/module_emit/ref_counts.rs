//! Expression ref-count analysis: how many live consumers each
//! expression handle has, and which handles appear in `Emit` ranges.

use crate::generator::core::FunctionExprInfo;

/// Count how many times each expression handle is referenced by
/// other *live* expressions (those in `Emit` ranges) and by
/// statements in the function.  Dead expressions are excluded so
/// they never inflate reference counts.
///
/// Returns both the per-handle reference count vector and the
/// `live` bitmap in a single [`FunctionExprInfo`].  Both are
/// consumed downstream (`generate_function` reads `ref_counts`,
/// `literal_extract` reads both); producing them in one walk
/// avoids a second body traversal.
pub(super) fn compute_expression_ref_counts(func: &naga::Function) -> FunctionExprInfo {
    let len = func.expressions.len();
    let mut counts: Vec<usize> = vec![0; len];

    // Collect handles that appear in Emit ranges (live emitted expressions).
    let mut live = vec![false; len];
    collect_emitted_handles(&func.body, &mut live);

    // Count references only from live expressions.
    for (h, expr) in func.expressions.iter() {
        if live[h.index()] {
            count_expr_children(expr, &mut counts);
        }
    }

    // Count references from statements.
    count_block_refs(&func.body, &mut counts);

    FunctionExprInfo {
        ref_counts: counts,
        live,
    }
}

/// Mark every handle that appears inside an `Emit` range of `block`
/// (recursively across control flow).  Emission-range membership is
/// the authoritative liveness signal for literal extraction and
/// expression ref counting.
fn collect_emitted_handles(block: &naga::Block, live: &mut [bool]) {
    crate::passes::expr_util::for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                live[h.index()] = true;
            }
        }
    });
}

/// Increment `counts[h]` by one.  The maximum ref count is bounded by
/// the total expression-reference count in the function, which fits
/// comfortably in `usize` for any realistic shader, so a checked add
/// is unwarranted - on overflow the program is already pathological
/// and `usize::MAX` writes would have been the least of our worries.
fn bump(counts: &mut [usize], h: naga::Handle<naga::Expression>) {
    counts[h.index()] += 1;
}

/// Shortcut helper that bumps `counts` for every child handle of
/// `expr`.  Used by [`compute_expression_ref_counts`] in the
/// arena-traversal loop.
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
