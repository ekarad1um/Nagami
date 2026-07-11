//! Dead-code elimination via `naga::compact`.  Removes arena entries
//! unreachable from the module's entry points.  Library modules
//! (no entry points) keep every declaration so partial fragments the
//! caller intends to splice later are not wiped out.

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// DCE pass that delegates to naga's arena compactor.  `changed` is
/// inferred from arena length deltas: the six module arenas (types,
/// constants, overrides, globals, global expressions, functions) PLUS
/// every function's and entry point's expression arena.  The compactor
/// culls function-local expressions too (e.g. an initialiser orphaned by
/// `dead_local_elimination`), remapping every later handle - and the
/// pipeline trusts `changed == false` enough to skip post-pass
/// validation, so under-reporting such a cull leaves a stale
/// `ModuleInfo` behind and the next `--trace` emission indexes shifted
/// per-expression info (naga's writer errors out mid-run).
///
/// Length-only comparison is sound because `naga::compact` only
/// removes entries - it never reorders without shrinking - so an
/// unchanged length tuple means every arena is bit-identical to its
/// pre-compact state.  Remaining module-level state naga's compactor
/// touches (`special_types`, `diagnostic_filters`, `entry_points`,
/// per-function `named_expressions` / statement handle rewrites)
/// is NOT length-checked here, on the principle that it only changes
/// as a cascade of a tracked-arena cull (a removed predeclared type
/// drops the type arena, a handle rewrite requires a culled
/// expression, etc.).
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
        // A module with no entry points is treated as a library
        // fragment; stripping unreachable items would delete every
        // declaration since there is nothing pulling them in.
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
            trace_run_dir: None,
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
            trace_run_dir: None,
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
