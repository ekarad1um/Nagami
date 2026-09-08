//! Coalesce consecutive [`naga::Statement::Emit`] statements with
//! contiguous expression ranges: each separate `Emit` becomes its own `let`
//! binding in the WGSL writer, blocking single-use inlining at the use
//! site; one `Emit` over the union range restores it.

use super::expr_util::{for_each_function_mut, nested_blocks_mut};
use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Merges adjacent contiguous `Emit` statements, recursing into every
/// nested block.
#[derive(Debug, Default)]
pub struct EmitMergePass;

impl Pass for EmitMergePass {
    fn name(&self) -> &'static str {
        "emit_merge"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
            changed |= merge_emits_in_block(&mut f.body);
        });
        Ok(changed)
    }
}

/// The scan recurses into nested blocks and decides whether this level
/// needs a rebuild (a contiguous pair or an empty Emit to drop), so
/// converged IR skips the `mem::take` + `with_capacity` rebuild on every
/// sweep.
fn merge_emits_in_block(block: &mut naga::Block) -> bool {
    let mut nested_changed = false;
    let mut has_work = false;
    let mut prev_emit_last: Option<naga::Handle<naga::Expression>> = None;
    for stmt in block.iter_mut() {
        if let naga::Statement::Emit(range) = stmt {
            let mut iter = range.clone();
            let Some(first) = iter.next() else {
                // An empty Emit needs the rebuild; `prev_emit_last` is kept so
                // two non-empty Emits separated only by empty ones still
                // register as mergeable.
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
            for nested in nested_blocks_mut(stmt) {
                nested_changed |= merge_emits_in_block(nested);
            }
        }
    }

    if !has_work {
        return nested_changed;
    }

    let mut changed = nested_changed;
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    // `(first, last, span, emit_count)`; `changed` is reported only when two
    // or more emits merged.
    let mut pending: Option<(
        naga::Handle<naga::Expression>,
        naga::Handle<naga::Expression>,
        naga::Span,
        usize,
    )> = None;

    for (statement, span) in original.span_into_iter() {
        if let naga::Statement::Emit(ref range) = statement {
            let mut iter = range.clone();
            let Some(first) = iter.next() else {
                // Dropping an empty Emit is a change: otherwise the block
                // shrinks while reporting `false`, defeating the convergence
                // signal.
                changed = true;
                continue;
            };
            let last = iter.last().unwrap_or(first);

            if let Some((pf, pl, ps, pc)) = pending {
                if first.index() == pl.index() + 1 {
                    pending = Some((pf, last, ps, pc + 1));
                } else {
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

        if let Some((pf, pl, ps, pc)) = pending.take() {
            if pc > 1 {
                changed = true;
            }
            rebuilt.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(pf, pl)),
                ps,
            );
        }

        rebuilt.push(statement, span);
    }

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
            name_log: None,
        };
        let mut pass = EmitMergePass;
        let changed = pass.run(&mut module, &ctx).unwrap();
        assert!(changed, "should merge consecutive emits");

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
            name_log: None,
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
            name_log: None,
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

    /// The allocator skip is not observable; its `changed = false` signal is.
    #[test]
    fn fast_path_reports_no_change_on_already_merged_nested_blocks() {
        // Control flow forces nested recursion; every inner block holds at
        // most one Emit.
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
            name_log: None,
        };
        let mut pass = EmitMergePass;

        let _ = pass.run(&mut module, &ctx).unwrap();
        let changed2 = pass.run(&mut module, &ctx).unwrap();
        assert!(
            !changed2,
            "second run must report no-change (fast-path bypasses rebuild on converged IR)"
        );
    }

    /// A rebuild that only drops empty Emits still mutates the block and
    /// must say so.
    #[test]
    fn dropping_only_empty_emits_reports_changed() {
        use naga::{Span, Statement};

        // naga's front-end never produces empty Emits; upstream passes
        // (const_fold emptying a run) can.
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
            name_log: None,
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
