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

/// Walk `block` once, merging runs of contiguous `Emit` statements
/// and recursing into nested blocks.  Returns `true` when at least
/// one merge or nested merge occurred; the block is overwritten in
/// place with the rebuilt statement list regardless.
fn merge_emits_in_block(block: &mut naga::Block) -> bool {
    let mut changed = false;
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
            let handles: Vec<_> = range.clone().collect();
            if handles.is_empty() {
                continue;
            }

            let first = handles[0];
            let last = handles[handles.len() - 1];

            if let Some((pf, pl, ps, pc)) = pending {
                if first.index() == pl.index() + 1 {
                    // Contiguous with the pending run; extend it.
                    pending = Some((pf, last, ps, pc + 1));
                } else {
                    // Gap in the handle sequence; flush the pending
                    // run and start a new one rooted at this emit.
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

        let mut stmt = statement;
        match &mut stmt {
            naga::Statement::Block(inner) => {
                changed |= merge_emits_in_block(inner);
            }
            naga::Statement::If { accept, reject, .. } => {
                changed |= merge_emits_in_block(accept);
                changed |= merge_emits_in_block(reject);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    changed |= merge_emits_in_block(&mut case.body);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed |= merge_emits_in_block(body);
                changed |= merge_emits_in_block(continuing);
            }
            _ => {}
        }
        rebuilt.push(stmt, span);
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
}
