//! Dead-parameter elimination.  Strips function arguments that are
//! never read inside the callee and updates every call site to drop
//! the corresponding argument expression.  Entry points are skipped
//! because their signatures are part of the pipeline contract.
//!
//! The pass runs in three phases per sweep:
//!
//! 1. Identify unused parameters via expression liveness analysis.
//! 2. Remove the arguments from the function signatures and rewrite
//!    stray `FunctionArgument` references (live or dead) so the IR
//!    stays well-typed.
//! 3. Drop the corresponding argument expressions at every call site
//!    across both functions and entry points.

use std::collections::{HashMap, HashSet};

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Remove unused parameters from non-entry-point functions and their
/// call sites.  Entry points are left untouched.
#[derive(Debug, Default)]
pub struct DeadParamPass;

impl Pass for DeadParamPass {
    fn name(&self) -> &'static str {
        "dead_param_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        // Phase 1: identify every unused parameter.  A parameter is
        // unused when its `FunctionArgument` expression handle is not
        // transitively reachable from any statement root.
        let mut removals: HashMap<naga::Handle<naga::Function>, Vec<usize>> = HashMap::new();

        for (fh, func) in module.functions.iter() {
            if func.arguments.is_empty() {
                continue;
            }

            let live = compute_live_expr_set(func);

            // One-pass collection so the unused-arg filter is O(1)
            // per parameter instead of O(arena) per parameter.
            let mut live_arg_indices: HashSet<u32> = HashSet::with_capacity(func.arguments.len());
            for (h, e) in func.expressions.iter() {
                if let naga::Expression::FunctionArgument(idx) = e
                    && live.contains(&h)
                {
                    live_arg_indices.insert(*idx);
                }
            }

            let unused: Vec<usize> = (0..func.arguments.len())
                .filter(|&i| {
                    // Non-constructible types would survive the
                    // arg -> ZeroValue rewrite below as invalid IR
                    // (`ZeroValue` is rejected for opaque/resource
                    // types).  Keep them.
                    if is_non_constructible_type(func.arguments[i].ty, &module.types) {
                        return false;
                    }
                    !live_arg_indices.contains(&(i as u32))
                })
                .collect();

            if !unused.is_empty() {
                removals.insert(fh, unused);
            }
        }

        if removals.is_empty() {
            return Ok(false);
        }

        // Phase 2: rewrite the callee signatures and patch any
        // surviving `FunctionArgument` expressions so dead indices
        // collapse to `ZeroValue` and live indices shift to close the
        // gap.
        for (&fh, indices) in &removals {
            let func = &mut module.functions[fh];

            // Snapshot types keyed by original index so the expression
            // rewrite below can resolve a removed slot's type after
            // `func.arguments` has shrunk.
            let removed_types: HashMap<usize, naga::Handle<naga::Type>> =
                indices.iter().map(|&i| (i, func.arguments[i].ty)).collect();

            // Reverse iteration keeps earlier indices valid.
            for &idx in indices.iter().rev() {
                func.arguments.remove(idx);
            }

            // Expression arena fixup: dead args become `ZeroValue`
            // (typed, validator-acceptable); live args shift down by
            // the count of removed slots below them.  `indices` is
            // small (typically <= 5 dead args), so linear `contains`
            // beats hash-set allocation overhead.
            for (_, expr) in func.expressions.iter_mut() {
                if let naga::Expression::FunctionArgument(arg_idx) = expr {
                    let old = *arg_idx as usize;
                    if indices.contains(&old) {
                        *expr = naga::Expression::ZeroValue(removed_types[&old]);
                    } else {
                        let shift = indices.iter().filter(|&&i| i < old).count();
                        *arg_idx = (old - shift) as u32;
                    }
                }
            }
        }

        // Phase 3: drop the corresponding actual arguments at every
        // call site across regular functions and entry points.
        for (_, func) in module.functions.iter_mut() {
            remove_call_args_in_block(&mut func.body, &removals)?;
        }
        for entry in module.entry_points.iter_mut() {
            remove_call_args_in_block(&mut entry.function.body, &removals)?;
        }

        Ok(true)
    }
}

// MARK: Call-site surgery

/// Walk `block` recursively, dropping argument expressions from every
/// `Call` targeting a function whose parameters were stripped in
/// phase 1.  Indices are consumed in reverse order so earlier
/// positions stay valid across removals.
fn remove_call_args_in_block(
    block: &mut naga::Block,
    removals: &HashMap<naga::Handle<naga::Function>, Vec<usize>>,
) -> Result<(), Error> {
    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::Call {
                function,
                arguments,
                ..
            } => {
                if let Some(indices) = removals.get(function) {
                    for &idx in indices.iter().rev() {
                        // Surface caller/callee arity drift as a
                        // structured error attributed to dead_param,
                        // not as a downstream validation crash under
                        // some sibling pass's name.
                        if idx >= arguments.len() {
                            return Err(Error::Validation(format!(
                                "dead_param: removal index {idx} out of bounds for call \
                                 site with {} arguments - caller/callee out of sync",
                                arguments.len()
                            )));
                        }
                        arguments.remove(idx);
                    }
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                remove_call_args_in_block(accept, removals)?;
                remove_call_args_in_block(reject, removals)?;
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    remove_call_args_in_block(&mut case.body, removals)?;
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                remove_call_args_in_block(body, removals)?;
                remove_call_args_in_block(continuing, removals)?;
            }
            naga::Statement::Block(inner) => {
                remove_call_args_in_block(inner, removals)?;
            }
            // Leaf statements - no Call to surgery, no nested blocks.
            // Enumerated explicitly so a future naga release adding
            // a new block-bearing variant breaks the build here
            // instead of silently leaving a Call's argument list
            // out of sync with the renumbered callee signature.
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
    Ok(())
}

// MARK: Liveness analysis

/// Return the set of expression handles transitively reachable from
/// any statement root.  A parameter whose `FunctionArgument` handle
/// is absent from this set has no live read and is eligible for
/// removal.
fn compute_live_expr_set(func: &naga::Function) -> HashSet<naga::Handle<naga::Expression>> {
    let mut live = HashSet::new();
    let mut worklist: Vec<naga::Handle<naga::Expression>> = Vec::new();

    // Seed the worklist with every handle reached directly from a
    // statement.
    collect_stmt_expr_roots(&func.body, &mut worklist);

    // Standard worklist traversal through expression dependencies.
    while let Some(handle) = worklist.pop() {
        if !live.insert(handle) {
            continue;
        }
        super::expr_util::visit_expression_children(&func.expressions[handle], |child| {
            if !live.contains(&child) {
                worklist.push(child);
            }
        });
    }

    live
}

/// Collect the roots of the liveness graph: every expression handle
/// a statement directly references, including Emit'd handles (those
/// are let-bound names reachable from elsewhere in the function).
/// A missed statement variant would under-track liveness and let the
/// pass remove a live parameter - the shared walker's exhaustive
/// match defends against that.
fn collect_stmt_expr_roots(block: &naga::Block, roots: &mut Vec<naga::Handle<naga::Expression>>) {
    super::expr_util::visit_block_expression_handles(
        block,
        /*include_emit_handles=*/ true,
        &mut |h| roots.push(h),
    );
}

// MARK: Constructibility gating

/// `true` when the type at `ty_handle` has no valid `ZeroValue`, and
/// therefore must not be removed by this pass.
///
/// The rewrite turns each removed `FunctionArgument(ty)` into
/// `ZeroValue(ty)` to keep surviving uses well-typed, and the validator
/// rejects `ZeroValue` of any type lacking the `CONSTRUCTIBLE` flag.  We
/// defer to naga's own [`TypeInner::is_constructible`], which exactly mirrors
/// that flag: it rejects the opaque leaves (pointers, samplers, images,
/// atomics, acceleration structures, binding arrays) AND, crucially,
/// recurses into aggregates - so an override- or runtime-sized array
/// (`ArraySize::Pending` / `Dynamic`), or a struct containing one, is
/// correctly rejected.  Such arrays carry the `ARGUMENT` flag (naga accepts
/// them as parameters) but NOT `CONSTRUCTIBLE`, so a hand-rolled flat
/// deny-list that treats every `Array` as constructible would emit an
/// invalid `ZeroValue` and force a whole-pass rollback.
fn is_non_constructible_type(
    ty_handle: naga::Handle<naga::Type>,
    types: &naga::UniqueArena<naga::Type>,
) -> bool {
    !types[ty_handle].inner.is_constructible(types)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = DeadParamPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("pass should run");

        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module should remain valid after pass");

        (changed, module)
    }

    #[test]
    fn removes_single_unused_param() {
        let src = r#"
fn helper(used: f32, unused: f32) -> f32 {
    return used * 2.0;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(1.0, 2.0));
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should report a change");
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(func.arguments.len(), 1, "unused param should be removed");
    }

    #[test]
    fn removes_multiple_unused_params() {
        let src = r#"
fn helper(a: f32, b: f32, c: f32) -> f32 {
    return b;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(1.0, 2.0, 3.0));
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(
            func.arguments.len(),
            1,
            "two unused params should be removed"
        );
    }

    #[test]
    fn preserves_all_used_params() {
        let src = r#"
fn helper(a: f32, b: f32) -> f32 {
    return a + b;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(1.0, 2.0));
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "no change when all params are used");
    }

    #[test]
    fn preserves_entry_point_params() {
        // Entry points cannot have params removed.
        let src = r#"
@fragment fn fs(@location(0) unused: f32) -> @location(0) vec4f {
    return vec4f(1.0);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "entry point params should not be removed");
    }

    #[test]
    fn handles_multiple_call_sites() {
        let src = r#"
fn helper(a: f32, unused: f32) -> f32 {
    return a;
}
@fragment fn fs() -> @location(0) vec4f {
    let x = helper(1.0, 2.0);
    let y = helper(3.0, 4.0);
    return vec4f(x + y);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(func.arguments.len(), 1);
    }

    #[test]
    fn handles_no_params() {
        let src = r#"
fn helper() -> f32 {
    return 1.0;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper());
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "no change when function has no params");
    }

    #[test]
    fn removes_first_param_remaps_second() {
        // Removing the first param must shift `FunctionArgument(1)`
        // down to `FunctionArgument(0)`; a validation failure below
        // signals the remap went wrong.
        let src = r#"
fn helper(unused: f32, used: f32) -> f32 {
    return used;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(1.0, 2.0));
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(func.arguments.len(), 1);
        // Verify the function still returns the correct (remapped) param.
        // Validation passing confirms the remapping is correct.
    }

    // Regression: non-constructible-typed parameters must be left
    // alone.  Substituting `ZeroValue(ty)` for them fails naga
    // validation and would break the module.
    #[test]
    fn preserves_unused_sampler_param() {
        // An unused sampler parameter must not be removed: a ZeroValue of
        // `sampler` is not constructible.
        let src = r#"
@group(0) @binding(0) var s: sampler;
fn helper(unused: sampler, v: f32) -> f32 {
    return v;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(s, 1.0));
}
"#;
        let (_changed, module) = run_pass(src);
        // The helper function must retain both params (no sampler removal).
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(
            func.arguments.len(),
            2,
            "sampler parameter must not be removed"
        );
    }

    #[test]
    fn preserves_unused_texture_param() {
        let src = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
fn helper(unused: texture_2d<f32>, v: f32) -> f32 {
    return v;
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(helper(t, 1.0));
}
"#;
        let (_changed, module) = run_pass(src);
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(
            func.arguments.len(),
            2,
            "texture parameter must not be removed"
        );
    }

    #[test]
    fn preserves_unused_pointer_param() {
        // Pre-existing pointer case (was already covered by the old guard).
        let src = r#"
fn helper(unused: ptr<function, f32>, v: f32) -> f32 {
    return v;
}
@fragment fn fs() -> @location(0) vec4f {
    var x: f32 = 0.0;
    return vec4f(helper(&x, 1.0));
}
"#;
        let (_changed, module) = run_pass(src);
        let func = &module.functions.iter().next().unwrap().1;
        assert_eq!(
            func.arguments.len(),
            2,
            "pointer parameter must not be removed"
        );
    }

    #[test]
    fn preserves_unused_override_sized_array_param() {
        // An override-sized array (`ArraySize::Pending`) carries naga's
        // ARGUMENT flag (accepted as a parameter) but NOT CONSTRUCTIBLE, so
        // `ZeroValue` of it is invalid.  The pass must KEEP such a dead param
        // rather than rewrite its refs to an invalid `ZeroValue` (which
        // `run_pass`'s post-pass validation would reject).
        let src = r#"
override N: u32 = 4u;
var<workgroup> wg: array<f32, N>;
fn helper(unused_arr: array<f32, N>, used: f32) -> f32 {
    return used;
}
@compute @workgroup_size(1) fn cs() {
    let r = helper(wg, 1.0);
}
"#;
        let (_changed, module) = run_pass(src);
        let helper = module
            .functions
            .iter()
            .map(|(_, f)| f)
            .find(|f| f.arguments.len() == 2)
            .expect("helper with both params must survive");
        assert_eq!(
            helper.arguments.len(),
            2,
            "override-sized array param has no valid ZeroValue and must be kept"
        );
    }
}
