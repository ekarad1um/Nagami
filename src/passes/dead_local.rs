//! Dead-local elimination.  naga's compactor deliberately roots every
//! `LocalVariable` (they are user declarations), so a local whose reads and
//! writes have all been optimised away survives DCE forever - pinning its
//! type (and, transitively, struct declarations and `enable` directives
//! through the type arena) and vetoing function inlining, which refuses
//! bodies with locals.  Removing the arena entry, not just skipping its
//! declaration at emit time, is what unblocks those downstream decisions.
//!
//! Scheduled immediately after `CompactPass`: compaction culls
//! statement-unreachable expressions first, so "no `LocalVariable(h)`
//! expression left in the arena" means "dead local" here - no statement
//! walking and no expression-arena surgery.  (A sound under-approximation:
//! naga's compact also roots `named_expressions`, which can keep a
//! statement-unreachable reference - and with it the local - alive.)  A
//! local whose last reference dies later in the same sweep is caught by the
//! next sweep's compact -> dead-local prefix; the convergence loop already
//! runs until no pass reports a change.  A floating expression this removal
//! orphans (a dead local's initialiser) is valid IR and is culled by the
//! next compact; if the sweep cap lands first, it reaches the generator
//! outside any `Emit` range and is simply never rendered.

use std::collections::HashSet;

use crate::error::Error;
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
        for (_, func) in module.functions.iter_mut() {
            changed |= remove_dead_locals(func);
        }
        for entry in module.entry_points.iter_mut() {
            changed |= remove_dead_locals(&mut entry.function);
        }
        Ok(changed)
    }
}

fn remove_dead_locals(func: &mut naga::Function) -> bool {
    if func.local_variables.is_empty() {
        return false;
    }
    let referenced: HashSet<naga::Handle<naga::LocalVariable>> = func
        .expressions
        .iter()
        .filter_map(|(_, e)| match e {
            naga::Expression::LocalVariable(h) => Some(*h),
            _ => None,
        })
        .collect();
    // `referenced` only holds valid local handles, so equal cardinality
    // means every local is referenced.
    if referenced.len() == func.local_variables.len() {
        return false;
    }

    let mut remap: Vec<Option<naga::Handle<naga::LocalVariable>>> =
        vec![None; func.local_variables.len()];
    let mut rebuilt: naga::Arena<naga::LocalVariable> = naga::Arena::new();
    for (h, local) in func.local_variables.iter() {
        if referenced.contains(&h) {
            let span = func.local_variables.get_span(h);
            remap[h.index()] = Some(rebuilt.append(local.clone(), span));
        }
    }
    func.local_variables = rebuilt;

    // The remap is total over surviving references by construction: every
    // `LocalVariable` expression's target was just re-appended.
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
            trace_run_dir: None,
        };
        let changed = DeadLocalPass
            .run(&mut module, &ctx)
            .expect("pass must not error");
        io::validate_module(&module).expect("pass output must validate");
        (changed, module)
    }

    #[test]
    fn removes_unreferenced_local_and_remaps_survivors() {
        // `dead` precedes `live`, so removing it shifts `live`'s handle -
        // validation above proves the remap kept the IR well-typed.
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
