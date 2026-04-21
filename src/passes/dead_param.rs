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

            let unused: Vec<usize> = (0..func.arguments.len())
                .filter(|&i| {
                    // Skip parameters whose type is not constructible:
                    // the rewrite below would turn their
                    // `FunctionArgument(ty)` into `ZeroValue(ty)`, but
                    // naga's validator rejects `ZeroValue` for opaque
                    // or resource-backed types.  See
                    // `is_non_constructible_type` for the exact set.
                    if is_non_constructible_type(func.arguments[i].ty, &module.types) {
                        return false;
                    }
                    // A parameter is unused when no live
                    // `FunctionArgument(i)` expression reaches a
                    // statement root.
                    !func.expressions.iter().any(|(h, e)| {
                        matches!(e, naga::Expression::FunctionArgument(idx) if *idx == i as u32)
                            && live.contains(&h)
                    })
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
            let removed_set: HashSet<usize> = indices.iter().copied().collect();

            // Snapshot the types before the vec is mutated so
            // the rewrite below can still consult them.
            let removed_types: HashMap<usize, naga::Handle<naga::Type>> =
                indices.iter().map(|&i| (i, func.arguments[i].ty)).collect();

            // Reverse-index removal keeps earlier indices valid
            // across the loop.
            for &idx in indices.iter().rev() {
                func.arguments.remove(idx);
            }

            // Expression arena fixup: dead `FunctionArgument` entries
            // become `ZeroValue` so the validator still sees a typed
            // value; live entries shift down by the count of removed
            // slots below them.
            for (_, expr) in func.expressions.iter_mut() {
                if let naga::Expression::FunctionArgument(arg_idx) = expr {
                    let old = *arg_idx as usize;
                    if removed_set.contains(&old) {
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
            remove_call_args_in_block(&mut func.body, &removals);
        }
        for entry in module.entry_points.iter_mut() {
            remove_call_args_in_block(&mut entry.function.body, &removals);
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
) {
    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::Call {
                function,
                arguments,
                ..
            } => {
                if let Some(indices) = removals.get(function) {
                    for &idx in indices.iter().rev() {
                        if idx < arguments.len() {
                            arguments.remove(idx);
                        }
                    }
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                remove_call_args_in_block(accept, removals);
                remove_call_args_in_block(reject, removals);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    remove_call_args_in_block(&mut case.body, removals);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                remove_call_args_in_block(body, removals);
                remove_call_args_in_block(continuing, removals);
            }
            naga::Statement::Block(inner) => {
                remove_call_args_in_block(inner, removals);
            }
            _ => {}
        }
    }
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
        visit_expr_children(&func.expressions[handle], |child| {
            if !live.contains(&child) {
                worklist.push(child);
            }
        });
    }

    live
}

/// Collect every expression handle a statement references directly
/// (the roots of the liveness graph) and push them into `roots`.
/// Recurses into nested blocks; exhaustive statement coverage is
/// required because a missed statement type would understate liveness
/// and let the pass remove live parameters.
fn collect_stmt_expr_roots(block: &naga::Block, roots: &mut Vec<naga::Handle<naga::Expression>>) {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    roots.push(h);
                }
            }
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                roots.push(*condition);
                collect_stmt_expr_roots(accept, roots);
                collect_stmt_expr_roots(reject, roots);
            }
            naga::Statement::Switch { selector, cases } => {
                roots.push(*selector);
                for case in cases {
                    collect_stmt_expr_roots(&case.body, roots);
                }
            }
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                collect_stmt_expr_roots(body, roots);
                collect_stmt_expr_roots(continuing, roots);
                if let Some(bi) = break_if {
                    roots.push(*bi);
                }
            }
            naga::Statement::Block(inner) => {
                collect_stmt_expr_roots(inner, roots);
            }
            naga::Statement::Store { pointer, value } => {
                roots.push(*pointer);
                roots.push(*value);
            }
            naga::Statement::Return { value: Some(v) } => {
                roots.push(*v);
            }
            naga::Statement::Return { value: None } => {}
            naga::Statement::Call {
                arguments, result, ..
            } => {
                roots.extend(arguments.iter().copied());
                if let Some(r) = result {
                    roots.push(*r);
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                roots.push(*image);
                roots.push(*coordinate);
                if let Some(ai) = array_index {
                    roots.push(*ai);
                }
                roots.push(*value);
            }
            naga::Statement::Atomic {
                pointer,
                value,
                result,
                fun,
            } => {
                roots.push(*pointer);
                roots.push(*value);
                if let Some(r) = result {
                    roots.push(*r);
                }
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    roots.push(*c);
                }
            }
            naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                roots.push(*pointer);
                roots.push(*result);
            }
            naga::Statement::RayQuery { query, fun } => {
                roots.push(*query);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        roots.push(*acceleration_structure);
                        roots.push(*descriptor);
                    }
                    naga::RayQueryFunction::Proceed { result } => {
                        roots.push(*result);
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        roots.push(*hit_t);
                    }
                    naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            naga::Statement::SubgroupBallot { result, predicate } => {
                roots.push(*result);
                if let Some(p) = predicate {
                    roots.push(*p);
                }
            }
            naga::Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                roots.push(*argument);
                roots.push(*result);
                match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => {
                        roots.push(*h);
                    }
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                }
            }
            naga::Statement::SubgroupCollectiveOperation {
                argument, result, ..
            } => {
                roots.push(*argument);
                roots.push(*result);
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } = fun;
                roots.push(*acceleration_structure);
                roots.push(*descriptor);
                roots.push(*payload);
            }
            naga::Statement::CooperativeStore { target, data } => {
                roots.push(*target);
                roots.push(data.pointer);
                roots.push(data.stride);
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                roots.push(*image);
                roots.push(*coordinate);
                if let Some(ai) = array_index {
                    roots.push(*ai);
                }
                roots.push(*value);
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    roots.push(*c);
                }
            }
            _ => {}
        }
    }
}

// MARK: Constructibility gating

/// `true` when the type at `ty_handle` has no valid `ZeroValue`, and
/// therefore must not be removed by this pass.
///
/// The rewrite turns each removed `FunctionArgument(ty)` into
/// `ZeroValue(ty)` to keep surviving uses well-typed.  naga's IR
/// validator rejects `ZeroValue` for:
///
/// - [`TypeInner::Pointer`] / [`TypeInner::ValuePointer`]: pointers
///   reference concrete memory and are not constructible.
/// - [`TypeInner::Sampler`]: samplers are opaque resource handles.
/// - [`TypeInner::Image`]: textures are opaque resource handles.
/// - [`TypeInner::AccelerationStructure`] / [`TypeInner::RayQuery`]:
///   opaque ray-tracing resources.
/// - [`TypeInner::Atomic`]: must be backed by a storage or workgroup
///   variable and is not value-constructible.
/// - [`TypeInner::BindingArray`]: binding arrays are resource bundles
///   scoped to globals and never appear as function-argument values.
///
/// Aggregate types ([`Struct`], [`Array`]) are intentionally NOT
/// recursed into.  naga also rejects function parameters whose type
/// is an aggregate carrying any of the above leaves, so such a
/// parameter cannot reach this predicate.  Keeping the check flat
/// avoids a recursive walk through the type arena on every invocation.
///
/// [`Struct`]: naga::TypeInner::Struct
/// [`Array`]: naga::TypeInner::Array
fn is_non_constructible_type(
    ty_handle: naga::Handle<naga::Type>,
    types: &naga::UniqueArena<naga::Type>,
) -> bool {
    matches!(
        types[ty_handle].inner,
        naga::TypeInner::Pointer { .. }
            | naga::TypeInner::ValuePointer { .. }
            | naga::TypeInner::Sampler { .. }
            | naga::TypeInner::Image { .. }
            | naga::TypeInner::AccelerationStructure { .. }
            | naga::TypeInner::RayQuery { .. }
            | naga::TypeInner::Atomic(_)
            | naga::TypeInner::BindingArray { .. }
    )
}

// MARK: Expression child walker

/// Visit every child expression handle of `expr`, invoking `f` for
/// each.  Declarative and result expressions yield nothing.  Walker
/// is exhaustive so a future naga variant forces a deliberate
/// classification decision instead of silently under-reporting
/// liveness.
fn visit_expr_children(expr: &naga::Expression, mut f: impl FnMut(naga::Handle<naga::Expression>)) {
    use naga::Expression as E;
    match expr {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => {}
        E::Compose { components, .. } => {
            for &c in components {
                f(c);
            }
        }
        E::Access { base, index } => {
            f(*base);
            f(*index);
        }
        E::AccessIndex { base, .. } => f(*base),
        E::Splat { value, .. } => f(*value),
        E::Swizzle { vector, .. } => f(*vector),
        E::Load { pointer } => f(*pointer),
        E::Unary { expr: e, .. } => f(*e),
        E::Binary { left, right, .. } => {
            f(*left);
            f(*right);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            f(*condition);
            f(*accept);
            f(*reject);
        }
        E::Derivative { expr: e, .. } => f(*e),
        E::Relational { argument, .. } => f(*argument),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            f(*arg);
            if let Some(a) = arg1 {
                f(*a);
            }
            if let Some(a) = arg2 {
                f(*a);
            }
            if let Some(a) = arg3 {
                f(*a);
            }
        }
        E::As { expr: e, .. } => f(*e),
        E::ArrayLength(e) => f(*e),
        E::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            f(*image);
            f(*sampler);
            f(*coordinate);
            if let Some(ai) = array_index {
                f(*ai);
            }
            if let Some(o) = offset {
                f(*o);
            }
            match level {
                naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                naga::SampleLevel::Exact(h) | naga::SampleLevel::Bias(h) => f(*h),
                naga::SampleLevel::Gradient { x, y } => {
                    f(*x);
                    f(*y);
                }
            }
            if let Some(d) = depth_ref {
                f(*d);
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            f(*image);
            f(*coordinate);
            if let Some(ai) = array_index {
                f(*ai);
            }
            if let Some(s) = sample {
                f(*s);
            }
            if let Some(l) = level {
                f(*l);
            }
        }
        E::ImageQuery { image, query } => {
            f(*image);
            if let naga::ImageQuery::Size { level: Some(l) } = query {
                f(*l);
            }
        }
        E::RayQueryVertexPositions { query, .. } => f(*query),
        E::RayQueryGetIntersection { query, .. } => f(*query),
        E::CooperativeLoad { data, .. } => {
            f(data.pointer);
            f(data.stride);
        }
        E::CooperativeMultiplyAdd { a, b, c } => {
            f(*a);
            f(*b);
            f(*c);
        }
    }
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
}
