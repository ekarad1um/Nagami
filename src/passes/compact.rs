//! Dead-code elimination via `naga::compact`.  Library modules (no entry
//! points) keep every declaration so fragments the caller splices later
//! are not wiped out.

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// DCE pass delegating to naga's arena compactor.  `changed` is inferred
/// from arena lengths: the six module arenas plus every function's and
/// entry point's expression arena.  The compactor culls function-local
/// expressions too (e.g. an initialiser orphaned by dead-local
/// elimination), remapping later handles, and the pipeline trusts
/// `changed == false` enough to skip post-pass validation, so
/// under-reporting such a cull leaves a stale `ModuleInfo` whose shifted
/// per-expression info makes the next `--trace` emission error out.
///
/// Length-only comparison is sound because `naga::compact` only removes
/// entries, never reorders without shrinking, so unchanged lengths mean
/// every arena is bit-identical.  State it touches that is not
/// length-checked (`special_types`, `diagnostic_filters`, `entry_points`,
/// per-function `named_expressions` and statement handle rewrites) only
/// changes as a cascade of a tracked-arena cull.
#[derive(Debug, Default)]
pub struct CompactPass;

impl Pass for CompactPass {
    fn name(&self) -> &'static str {
        "compact_dce"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        fn arena_shape(
            module: &naga::Module,
        ) -> (usize, usize, usize, usize, usize, usize, Vec<usize>) {
            (
                module.types.len(),
                module.constants.len(),
                module.overrides.len(),
                module.global_variables.len(),
                module.global_expressions.len(),
                module.functions.len(),
                module
                    .functions
                    .iter()
                    .map(|(_, f)| f.expressions.len())
                    .chain(
                        module
                            .entry_points
                            .iter()
                            .map(|ep| ep.function.expressions.len()),
                    )
                    .collect(),
            )
        }
        let before = arena_shape(module);
        let keep = if module.entry_points.is_empty() {
            naga::compact::KeepUnused::Yes
        } else {
            naga::compact::KeepUnused::No
        };
        naga::compact::compact(module, keep);
        Ok(before != arena_shape(module))
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn removes_unused_non_entry_function() {
        let source = r#"
fn helper_unused() -> f32 {
    return 1.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
"#;

        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        assert_eq!(
            module.functions.len(),
            1,
            "helper function should exist before compact"
        );

        let mut pass = CompactPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
            .expect("compact pass should run");

        assert!(changed, "compact pass should report it ran");
        assert_eq!(module.functions.len(), 0, "unused helper should be removed");
        assert_eq!(
            module.entry_points.len(),
            1,
            "entry point should be preserved"
        );

        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module should remain valid after compact");
    }

    #[test]
    fn preserves_library_module_without_entry_points() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x * 2.0;
}

fn another(y: f32) -> f32 {
    return helper(y) + 1.0;
}
"#;
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        assert_eq!(module.functions.len(), 2);

        let mut pass = CompactPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
            .expect("compact pass should run");

        assert!(
            !changed,
            "no-entry-point module should not lose declarations"
        );
        assert_eq!(
            module.functions.len(),
            2,
            "both functions should be preserved in library module"
        );
    }
}
