//! Dead-local elimination.  naga's compactor roots every `LocalVariable`
//! (user declarations), so a local whose reads and writes were all
//! optimised away survives DCE forever, pinning its type (transitively
//! struct declarations and `enable` directives) and vetoing inlining,
//! which refuses bodies with locals.  Only removing the arena entry, not
//! skipping the declaration at emit time, unblocks those decisions.
//!
//! Runs right after `CompactPass`, whose culling of statement-unreachable
//! expressions makes "no `LocalVariable(h)` expression in the arena" mean
//! "dead local" without statement walking.  This under-approximates
//! soundly: compact also roots `named_expressions`, which can keep a
//! statement-unreachable reference alive.  A local whose last reference
//! dies later in the sweep is caught by the next sweep's compact ->
//! dead-local prefix, and an orphaned initialiser is valid floating IR
//! that the next compact culls; if the sweep cap lands first it sits
//! outside every `Emit` range and is never rendered.

use super::expr_util::for_each_function_mut;
use crate::error::Error;
use crate::handle_set::HandleSet;
use crate::pipeline::{Pass, PassContext};

/// Remove locals no expression references and remap survivors' handles.
#[derive(Debug, Default)]
pub struct DeadLocalPass;

impl Pass for DeadLocalPass {
    fn name(&self) -> &'static str {
        "dead_local_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
            changed |= remove_dead_locals(f);
        });
        Ok(changed)
    }
}

fn remove_dead_locals(func: &mut naga::Function) -> bool {
    if func.local_variables.is_empty() {
        return false;
    }
    let referenced: HandleSet<naga::LocalVariable> = func
        .expressions
        .iter()
        .filter_map(|(_, e)| match e {
            naga::Expression::LocalVariable(h) => Some(*h),
            _ => None,
        })
        .collect();
    // `referenced` holds only valid local handles, so equal cardinality
    // means every local is referenced.
    if referenced.len() == func.local_variables.len() {
        return false;
    }

    let mut remap: Vec<Option<naga::Handle<naga::LocalVariable>>> =
        vec![None; func.local_variables.len()];
    let mut rebuilt: naga::Arena<naga::LocalVariable> = naga::Arena::new();
    for (h, local) in func.local_variables.iter() {
        if referenced.contains(h) {
            let span = func.local_variables.get_span(h);
            remap[h.index()] = Some(rebuilt.append(local.clone(), span));
        }
    }
    func.local_variables = rebuilt;

    // Total over surviving references: every `LocalVariable` target was
    // just re-appended.
    for (_, expr) in func.expressions.iter_mut() {
        if let naga::Expression::LocalVariable(h) = expr {
            *h = remap[h.index()].expect("referenced local survives the rebuild");
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::io;
    use crate::pipeline::PassContext;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = io::parse_wgsl(source).expect("test source parses");
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            name_log: None,
        };
        let changed = DeadLocalPass
            .run(&mut module, &ctx)
            .expect("pass must not error");
        io::validate_module(&module).expect("pass output must validate");
        (changed, module)
    }

    #[test]
    fn removes_unreferenced_local_and_remaps_survivors() {
        // `dead` precedes `live`, so removal shifts `live`'s handle;
        // validation proves the remap.
        let (changed, module) = run_pass(
            "fn f() -> f32 { var dead: f32; var live: f32 = 2.0; live = live + 1.0; return live; }\
             @fragment fn main() -> @location(0) vec4f { return vec4f(f()); }",
        );
        assert!(changed);
        let (_, f) = module.functions.iter().next().expect("f exists");
        assert_eq!(f.local_variables.len(), 1, "only `live` survives");
        let (_, survivor) = f.local_variables.iter().next().unwrap();
        assert_eq!(survivor.name.as_deref(), Some("live"));
    }

    #[test]
    fn keeps_every_referenced_local() {
        let (changed, module) = run_pass(
            "fn f() -> f32 { var a: f32 = 1.0; var b: f32 = 2.0; return a + b; }\
             @fragment fn main() -> @location(0) vec4f { return vec4f(f()); }",
        );
        assert!(!changed);
        let (_, f) = module.functions.iter().next().expect("f exists");
        assert_eq!(f.local_variables.len(), 2);
    }

    #[test]
    fn cleans_entry_point_locals_too() {
        let (changed, module) = run_pass(
            "@fragment fn main() -> @location(0) vec4f { var dead: vec3f; return vec4f(1.0); }",
        );
        assert!(changed);
        assert!(
            module.entry_points[0].function.local_variables.is_empty(),
            "entry-point dead local must be removed"
        );
    }
}
