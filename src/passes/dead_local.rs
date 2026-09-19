//! Dead-local elimination, the second half of the driver's normalization
//! (`super::normalize`).  naga's compactor roots every `LocalVariable`
//! (user declarations), so a local whose reads and writes were all
//! optimised away would survive DCE forever, pinning its type (transitively
//! struct declarations and `enable` directives) and vetoing inlining, which
//! refuses bodies with locals.  Only removing the arena entry, not skipping
//! the declaration at emit time, unblocks those decisions.
//!
//! Runs after compaction, whose culling of statement-unreachable
//! expressions makes "no `LocalVariable(h)` expression in the arena" mean
//! "dead local" without statement walking; an initialiser the removal
//! orphans is valid floating IR that the next compaction round culls.

use crate::handle_set::HandleSet;
use crate::ir::visit::for_each_function_mut;

/// Remove the locals no expression references, in every body; `true` when
/// any went.
pub(crate) fn remove_dead_locals(module: &mut naga::Module) -> bool {
    let mut changed = false;
    for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
        changed |= remove_dead_locals_in(f);
    });
    changed
}

/// Remove locals no expression references and remap survivors' handles.
fn remove_dead_locals_in(func: &mut naga::Function) -> bool {
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
    use crate::io;

    fn run(source: &str) -> (bool, naga::Module) {
        let mut module = io::parse_wgsl(source).expect("test source parses");
        let changed = remove_dead_locals(&mut module);
        io::validate_module(&module).expect("output must validate");
        (changed, module)
    }

    #[test]
    fn removes_unreferenced_local_and_remaps_survivors() {
        // `dead` precedes `live`, so removal shifts `live`'s handle;
        // validation proves the remap.
        let (changed, module) = run(
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
        let (changed, module) = run(
            "fn f() -> f32 { var a: f32 = 1.0; var b: f32 = 2.0; return a + b; }\
             @fragment fn main() -> @location(0) vec4f { return vec4f(f()); }",
        );
        assert!(!changed);
        let (_, f) = module.functions.iter().next().expect("f exists");
        assert_eq!(f.local_variables.len(), 2);
    }

    #[test]
    fn cleans_entry_point_locals_too() {
        let (changed, module) = run(
            "@fragment fn main() -> @location(0) vec4f { var dead: vec3f; return vec4f(1.0); }",
        );
        assert!(changed);
        assert!(
            module.entry_points[0].function.local_variables.is_empty(),
            "entry-point dead local must be removed"
        );
    }
}
