//! Dead-parameter elimination: arguments never read inside the callee are
//! stripped from the signature and from every call site.  Entry points are
//! skipped because their signatures are part of the pipeline contract.

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::ir::visit::for_each_function_mut;
use crate::pipeline::{Pass, PassContext};

/// Removes unused parameters from ordinary functions and their call sites;
/// entry points and preserve-listed functions export a fixed signature.
#[derive(Debug, Default)]
pub struct DeadParamPass;

impl Pass for DeadParamPass {
    fn name(&self) -> &'static str {
        "dead_param_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        // Unused: the `FunctionArgument` handle is unreachable from every
        // statement root.
        let mut removals: HandleMap<naga::Function, Vec<usize>> = Default::default();

        for (fh, func) in module.functions.iter() {
            if func.arguments.is_empty() {
                continue;
            }

            // A preserved function's signature is an external contract:
            // `--preserve-symbol` callers and re-prepended `--preamble`
            // definitions pass the original arity.
            if func
                .name
                .as_deref()
                .is_some_and(|n| ctx.config.preserve_symbols.iter().any(|p| p == n))
            {
                continue;
            }

            let live = compute_live_expr_set(func);

            // One arena pass, so the filter is O(1) per parameter.
            let mut live_arg = vec![false; func.arguments.len()];
            for (h, e) in func.expressions.iter() {
                if let naga::Expression::FunctionArgument(idx) = e
                    && live.contains(h)
                {
                    live_arg[*idx as usize] = true;
                }
            }

            let unused: Vec<usize> = (0..func.arguments.len())
                .filter(|&i| {
                    // No valid `ZeroValue` to rewrite the dead uses to.
                    if is_non_constructible_type(func.arguments[i].ty, &module.types) {
                        return false;
                    }
                    !live_arg[i]
                })
                .collect();

            if !unused.is_empty() {
                removals.insert(fh, unused);
            }
        }

        if removals.is_empty() {
            return Ok(false);
        }

        for (&fh, indices) in &removals {
            let func = &mut module.functions[fh];

            // Removed slots' types, parallel to `indices`, needed after
            // `func.arguments` has shrunk.
            let removed_types: Vec<naga::Handle<naga::Type>> =
                indices.iter().map(|&i| func.arguments[i].ty).collect();

            // Reverse iteration keeps earlier indices valid.
            for &idx in indices.iter().rev() {
                func.arguments.remove(idx);
            }

            // Dead args become a typed `ZeroValue`, live args shift down past
            // removed slots; `indices` is tiny, so a linear scan beats a
            // hash set.
            for (_, expr) in func.expressions.iter_mut() {
                if let naga::Expression::FunctionArgument(arg_idx) = expr {
                    let old = *arg_idx as usize;
                    if let Some(removed) = indices.iter().position(|&i| i == old) {
                        *expr = naga::Expression::ZeroValue(removed_types[removed]);
                    } else {
                        let shift = indices.iter().filter(|&&i| i < old).count();
                        *arg_idx = (old - shift) as u32;
                    }
                }
            }
        }

        // A callback cannot `?`: hold the first error, skip the rest of the
        // walk, and raise it after.
        let mut result = Ok(());
        for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
            if result.is_ok() {
                result = remove_call_args_in_block(&mut f.body, &removals);
            }
        });
        result?;

        Ok(true)
    }
}

// MARK: Call-site surgery

/// Indices are consumed in reverse so earlier positions stay valid.
fn remove_call_args_in_block(
    block: &mut naga::Block,
    removals: &HandleMap<naga::Function, Vec<usize>>,
) -> Result<(), Error> {
    let mut result = Ok(());
    crate::ir::visit::for_each_statement_mut(block, &mut |stmt| {
        if result.is_err() {
            return;
        }
        if let naga::Statement::Call {
            function,
            arguments,
            ..
        } = stmt
            && let Some(indices) = removals.get(function)
        {
            for &idx in indices.iter().rev() {
                // Arity drift surfaces as an error attributed to this pass,
                // not a downstream validation failure under another pass's
                // name.
                if idx >= arguments.len() {
                    result = Err(Error::Validation(format!(
                        "dead_param: removal index {idx} out of bounds for call \
                         site with {} arguments - caller/callee out of sync",
                        arguments.len()
                    )));
                    return;
                }
                arguments.remove(idx);
            }
        }
    });
    result
}

// MARK: Liveness analysis

/// Expression handles transitively reachable from any statement root.
fn compute_live_expr_set(func: &naga::Function) -> HandleSet<naga::Expression> {
    let mut live = HandleSet::default();
    let mut worklist: Vec<naga::Handle<naga::Expression>> = Vec::new();

    collect_stmt_expr_roots(&func.body, &mut worklist);

    while let Some(handle) = worklist.pop() {
        if !live.insert(handle) {
            continue;
        }
        crate::ir::visit::visit_expression_children(&func.expressions[handle], |child| {
            if !live.contains(child) {
                worklist.push(child);
            }
        });
    }

    live
}

/// Every handle a statement directly references, Emit'd ones included
/// (let-bound names reachable from elsewhere); a missed statement variant
/// would under-track liveness and remove a live parameter, which the shared
/// walker's exhaustive match defends against.
fn collect_stmt_expr_roots(block: &naga::Block, roots: &mut Vec<naga::Handle<naga::Expression>>) {
    crate::ir::visit::visit_block_expression_handles(
        block,
        /*include_emit_handles=*/ true,
        &mut |h| roots.push(h),
    );
}

// MARK: Constructibility gating

/// Dead uses are rewritten to `ZeroValue(ty)`, which the validator rejects
/// for any type without the `CONSTRUCTIBLE` flag.  naga's own
/// `is_constructible` mirrors that flag exactly: it rejects the opaque
/// leaves (pointers, samplers, images, atomics, acceleration structures,
/// binding arrays) and recurses into aggregates, so an override- or
/// runtime-sized array, or a struct containing one, is rejected too - such
/// arrays carry `ARGUMENT` but not `CONSTRUCTIBLE`, and a flat deny-list
/// treating every `Array` as constructible would force a whole-pass
/// rollback.
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
        let changed =
            PassContext::run_pass(&mut pass, &mut module, &config).expect("pass should run");

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
        // A wrong `FunctionArgument(1)` -> `FunctionArgument(0)` shift fails
        // validation.
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
    }

    #[test]
    fn preserves_unused_sampler_param() {
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
        // `ArraySize::Pending` carries `ARGUMENT` but not `CONSTRUCTIBLE`.
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
