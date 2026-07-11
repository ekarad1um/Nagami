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
fn collect_emitted_handles(block: &naga::Block, live: &mut Vec<bool>) {
    for stmt in block {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                live[h.index()] = true;
            }
        }
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            collect_emitted_handles(nested, live);
        }
    }
}

/// Increment `counts[h]` by one.  The maximum ref count is bounded by
/// the total expression-reference count in the function, which fits
/// comfortably in `usize` for any realistic shader, so a checked add
/// is unwarranted - on overflow the program is already pathological
/// and `usize::MAX` writes would have been the least of our worries.
fn bump(counts: &mut [usize], h: naga::Handle<naga::Expression>) {
    counts[h.index()] += 1;
}

/// Generator-local alias for the shared exhaustive child walker in
/// [`crate::passes::expr_util::visit_expression_children`].  Kept as a
/// thin indirection so the generator's call sites don't take a direct
/// dependency on the passes layer.
pub(in crate::generator) fn visit_expr_children(
    expr: &naga::Expression,
    f: impl FnMut(naga::Handle<naga::Expression>),
) {
    crate::passes::expr_util::visit_expression_children(expr, f);
}

/// Shortcut helper that bumps `counts` for every child handle of
/// `expr`.  Used by [`compute_expression_ref_counts`] in the
/// arena-traversal loop.
fn count_expr_children(expr: &naga::Expression, counts: &mut [usize]) {
    visit_expr_children(expr, |h| bump(counts, h));
}

/// Walk `block` and increment `counts[h]` for every expression
/// handle referenced from a statement operand, recursing into nested
/// blocks.  Ensures statement-level uses contribute to liveness in
/// the same way expression-level uses do.
fn count_block_refs(block: &naga::Block, counts: &mut Vec<usize>) {
    for stmt in block {
        match stmt {
            naga::Statement::If { condition, .. } => {
                bump(counts, *condition);
            }
            naga::Statement::Switch { selector, .. } => {
                bump(counts, *selector);
            }
            naga::Statement::Loop {
                break_if: Some(h), ..
            } => {
                bump(counts, *h);
            }
            naga::Statement::Return { value: Some(h) } => {
                bump(counts, *h);
            }
            naga::Statement::Store { pointer, value } => {
                bump(counts, *pointer);
                bump(counts, *value);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                bump(counts, *image);
                bump(counts, *coordinate);
                if let Some(i) = array_index {
                    bump(counts, *i);
                }
                bump(counts, *value);
            }
            naga::Statement::Atomic {
                pointer,
                fun,
                value,
                ..
            } => {
                bump(counts, *pointer);
                bump(counts, *value);
                crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                    bump(counts, h)
                });
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                bump(counts, *image);
                bump(counts, *coordinate);
                if let Some(i) = array_index {
                    bump(counts, *i);
                }
                bump(counts, *value);
                crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                    bump(counts, h)
                });
            }
            naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                bump(counts, *pointer);
            }
            naga::Statement::Call { arguments, .. } => {
                for a in arguments {
                    bump(counts, *a);
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                bump(counts, *query);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        bump(counts, *acceleration_structure);
                        bump(counts, *descriptor);
                    }
                    naga::RayQueryFunction::Proceed { .. } => {}
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        bump(counts, *hit_t);
                    }
                    naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(h), ..
            } => {
                bump(counts, *h);
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                bump(counts, *argument);
                match mode {
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => {
                        bump(counts, *h);
                    }
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                bump(counts, *argument);
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    bump(counts, *acceleration_structure);
                    bump(counts, *descriptor);
                    bump(counts, *payload);
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                bump(counts, *target);
                bump(counts, data.pointer);
                bump(counts, data.stride);
            }
            _ => {}
        }
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            count_block_refs(nested, counts);
        }
    }
}
