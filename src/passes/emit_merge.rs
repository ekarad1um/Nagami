//! Coalesce consecutive [`naga::Statement::Emit`] statements whose
//! expression ranges are contiguous.  Back-to-back emits block naga's
//! WGSL writer from inlining single-use expressions at their use sites
//! (each `Emit` becomes a separate `let` binding).  Merging them into
//! one `Emit` over the union range restores the inliner's ability to
//! fold the expressions back into their consumers.

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Merge adjacent `Emit` statements with contiguous ranges so the WGSL
/// writer can inline the underlying expressions at their uses.
/// Recurses into every nested block so branches, loops, and switch
/// cases are handled uniformly.
#[derive(Debug, Default)]
pub struct EmitMergePass;

impl Pass for EmitMergePass {
    fn name(&self) -> &'static str {
        "emit_merge"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for (_, function) in module.functions.iter_mut() {
            changed |= merge_emits_in_block(&mut function.body);
        }
        for entry in module.entry_points.iter_mut() {
            changed |= merge_emits_in_block(&mut entry.function.body);
        }
        Ok(changed)
    }
}

/// Merge contiguous `Emit` ranges in `block` and recurse into nested
/// blocks.  Returns `true` if any merge or nested change occurred.
///
/// Two-pass: the scan detects whether this level needs a rebuild
/// (contiguous-Emit pair or empty Emit to drop) and recurses into
/// nested blocks unconditionally; if nothing here changed, the
/// `mem::take` + `Block::with_capacity` rebuild is skipped.  On
/// already-converged IR this saves the allocator pressure across
/// the pipeline's ~16-sweep fixed-point.
fn merge_emits_in_block(block: &mut naga::Block) -> bool {
    let mut nested_changed = false;
    let mut has_work = false;
    let mut prev_emit_last: Option<naga::Handle<naga::Expression>> = None;
    for stmt in block.iter_mut() {
        if let naga::Statement::Emit(range) = stmt {
            let mut iter = range.clone();
            let Some(first) = iter.next() else {
                // Empty Emit ranges are dropped by the slow path; mark
                // work so the rebuild runs.  Don't touch
                // `prev_emit_last` so two non-empty Emits separated
                // only by empty ones (from prior passes that emptied
                // a range) still register as mergeable.
                has_work = true;
                continue;
            };
            let last = iter.last().unwrap_or(first);
            if let Some(prev_last) = prev_emit_last
                && first.index() == prev_last.index() + 1
            {
                has_work = true;
            }
            prev_emit_last = Some(last);
        } else {
            prev_emit_last = None;
            match stmt {
                naga::Statement::Block(inner) => {
                    nested_changed |= merge_emits_in_block(inner);
                }
                naga::Statement::If { accept, reject, .. } => {
                    nested_changed |= merge_emits_in_block(accept);
                    nested_changed |= merge_emits_in_block(reject);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases.iter_mut() {
                        nested_changed |= merge_emits_in_block(&mut case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    nested_changed |= merge_emits_in_block(body);
                    nested_changed |= merge_emits_in_block(continuing);
                }
                // Leaf statements (Emit is already handled by the
                // outer `if let`).  Listed exhaustively so a new
                // block-bearing variant trips the build instead of
                // silently bypassing recursion.
                naga::Statement::Emit(_)
                | naga::Statement::Store { .. }
                | naga::Statement::Break
                | naga::Statement::Continue
                | naga::Statement::Return { .. }
                | naga::Statement::Kill
                | naga::Statement::ControlBarrier(_)
                | naga::Statement::MemoryBarrier(_)
                | naga::Statement::ImageStore { .. }
                | naga::Statement::ImageAtomic { .. }
                | naga::Statement::Call { .. }
                | naga::Statement::Atomic { .. }
                | naga::Statement::RayQuery { .. }
                | naga::Statement::RayPipelineFunction(_)
                | naga::Statement::WorkGroupUniformLoad { .. }
                | naga::Statement::SubgroupBallot { .. }
                | naga::Statement::SubgroupGather { .. }
                | naga::Statement::SubgroupCollectiveOperation { .. }
                | naga::Statement::CooperativeStore { .. } => {}
            }
        }
    }

    if !has_work {
        return nested_changed;
    }

    // Rebuild this level - nested recursion already happened above,
    // so the rebuild only merges adjacent Emits and drops empties.
    let mut changed = nested_changed;
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    // Currently-accumulating merged range stored as
    // `(first_handle, last_handle, span, emit_count)`.  `emit_count`
    // tracks how many source emits feed this run so the flusher only
    // reports `changed` when two or more emits actually merged.
    let mut pending: Option<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
        naga::Span,
        usize,
    )> = None;

    for (statement, span) in original.span_into_iter() {
        if let naga::Statement::Emit(ref range) = statement {
            // Range::Iterator: `next()` consumes first, `last()`
            // walks the tail and returns the last - one pass, no
            // intermediate allocation.
            let mut iter = range.clone();
            let Some(first) = iter.next() else {
                // Dropping an empty Emit shrinks the block.  Without
                // this flip, a block whose only "work" is empty-Emit
                // removal would `mem::take` itself into a shorter
                // block but report `changed = false`, defeating the
                // pipeline's convergence signal.
                changed = true;
                continue;
            };
            let last = iter.last().unwrap_or(first);

            if let Some((pf, pl, ps, pc)) = pending {
                if first.index() == pl.index() + 1 {
                    pending = Some((pf, last, ps, pc + 1));
                } else {
                    // Non-contiguous: flush pending, start new run.
                    if pc > 1 {
                        changed = true;
                    }
                    rebuilt.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(pf, pl)),
                        ps,
                    );
                    pending = Some((first, last, span, 1));
                }
            } else {
                pending = Some((first, last, span, 1));
            }
            continue;
        }

        // Non-Emit statement: flush the pending run and recurse into
        // any nested blocks the statement carries.
        if let Some((pf, pl, ps, pc)) = pending.take() {
            if pc > 1 {
                changed = true;
            }
            rebuilt.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(pf, pl)),
                ps,
            );
        }

        // Nested blocks already recursed above - pass through.
        rebuilt.push(statement, span);
    }

    // Flush any run that reached end-of-block without a follow-up
    // non-Emit statement.
    if let Some((pf, pl, ps, pc)) = pending {
        if pc > 1 {
            changed = true;
        }
        rebuilt.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(pf, pl)),
            ps,
        );
    }

    *block = rebuilt;
    changed
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PassContext;

    fn make_test_module() -> naga::Module {
        let src = r#"
            fn test_fn(a: vec3<f32>) -> f32 {
                var e = a;
                let x = e.x;
                let y = e.y;
                let z = e.z;
                return fract((x + y) * z);
            }
        "#;
        naga::front::wgsl::parse_str(src).expect("parse failed")
    }

    #[test]
    fn merges_consecutive_emits() {
        let mut module = make_test_module();
        let config = crate::config::Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let mut pass = EmitMergePass;
        let changed = pass.run(&mut module, &ctx).unwrap();
        assert!(changed, "should merge consecutive emits");

        // Validate the result.
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module);
        assert!(info.is_ok(), "validation failed: {:?}", info.err());
    }

    #[test]
    fn no_change_when_emits_already_merged() {
        // A single-expression function has only one Emit; nothing to merge.
        let src = "fn f(a: f32) -> f32 { return a; }";
        let mut module = naga::front::wgsl::parse_str(src).expect("parse failed");
        let config = crate::config::Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let mut pass = EmitMergePass;
        let changed = pass.run(&mut module, &ctx).unwrap();
        assert!(!changed, "should report no change");
    }

    #[test]
    fn merges_inside_control_flow() {
        let src = r#"
            fn f(a: f32) -> f32 {
                if (a > 0.0) {
                    let x = a * 2.0;
                    let y = x + 1.0;
                    return y;
                }
                return a;
            }
        "#;
        let mut module = naga::front::wgsl::parse_str(src).expect("parse failed");
        let config = crate::config::Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let mut pass = EmitMergePass;
        let _ = pass.run(&mut module, &ctx).unwrap();

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module);
        assert!(
            info.is_ok(),
            "validation failed after merging inside if: {:?}",
            info.err()
        );
    }

    /// Fast-path regression: on already-merged IR the pass must
    /// return `changed = false`, signalling pipeline convergence.
    /// The allocator-skip behind that flag is not directly observable
    /// in safe Rust, so we test the load-bearing surface.
    #[test]
    fn fast_path_reports_no_change_on_already_merged_nested_blocks() {
        // Control flow forces nested-block recursion; every inner
        // block holds a single Emit or none, so no merge is possible
        // and the second run must report `false`.
        let src = r#"
            fn f(a: f32, c: bool) -> f32 {
                if c {
                    return a;
                }
                return a * 2.0;
            }
            @compute @workgroup_size(1) fn main() { _ = f(1.0, true); }
        "#;
        let mut module = naga::front::wgsl::parse_str(src).expect("parses");
        let config = crate::config::Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let mut pass = EmitMergePass;

        // First run: real work may happen if naga emits non-merged
        // emits.  Second run must converge to no-change because IR
        // is now at fixed point.
        let _ = pass.run(&mut module, &ctx).unwrap();
        let changed2 = pass.run(&mut module, &ctx).unwrap();
        assert!(
            !changed2,
            "second run must report no-change (fast-path bypasses rebuild on converged IR)"
        );
    }

    /// Regression: a block whose rebuild only drops empty Emit ranges
    /// (and never merges contiguous pairs) shrinks but used to return
    /// `changed = false`, hiding the mutation from the pipeline's
    /// convergence detector.
    #[test]
    fn dropping_only_empty_emits_reports_changed() {
        use naga::{Span, Statement};

        // Hand-build a function body of two empty Emit ranges.  naga's
        // WGSL frontend won't produce these directly, but upstream
        // passes (e.g. const_fold removing every expression in a run)
        // can leave them behind.
        let mut module = naga::Module::default();
        let mut function = naga::Function::default();
        let empty = naga::Range::from_index_range(0..0, &function.expressions);
        function
            .body
            .push(Statement::Emit(empty.clone()), Span::UNDEFINED);
        function
            .body
            .push(Statement::Emit(empty.clone()), Span::UNDEFINED);
        let _ = module.functions.append(function, Span::UNDEFINED);

        let config = crate::config::Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let mut pass = EmitMergePass;
        let changed = pass.run(&mut module, &ctx).unwrap();
        assert!(
            changed,
            "dropping empty Emit ranges must report `changed = true`"
        );
        let body = &module.functions.iter().next().unwrap().1.body;
        assert!(
            body.is_empty(),
            "expected empty body after drop, got {} statements",
            body.len()
        );
    }
}
