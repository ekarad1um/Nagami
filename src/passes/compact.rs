//! Dead-code elimination via `naga::compact`, the first half of the
//! driver's normalization (`super::normalize`); a library module
//! (`is_library_module`) keeps every declaration.

/// Cull what no entry point reaches; `true` when anything went, inferred
/// from arena lengths: the six module arenas plus every function's and
/// entry point's expression arena.  The driver trusts `false` enough to
/// keep its `ModuleInfo`, so under-reporting a cull would leave a stale
/// info whose shifted per-expression view the backend indexes out of
/// bounds.  Length-only comparison is sound because `naga::compact` only
/// removes entries, never reorders without shrinking, so unchanged lengths
/// mean every arena is bit-identical.  State it touches that is not
/// length-checked (`special_types`, `diagnostic_filters`, `entry_points`,
/// per-function `named_expressions` and statement handle rewrites) only
/// changes as a cascade of a tracked-arena cull.
pub(crate) fn compact_module(module: &mut naga::Module, preserved: &dyn Fn(&str) -> bool) -> bool {
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
    if super::expr_util::is_library_module(module) {
        naga::compact::compact(module, naga::compact::KeepUnused::Yes);
    } else {
        // naga roots nothing but the entry points, so the declarations
        // the host names live through a synthetic one: a preserved
        // resource stays in the pipeline's bind-group layout, a
        // preserved function keeps its signature callable, and a named
        // `override` stays a valid pipeline-constant key (Dawn validates
        // the keys against every declared override, used or not).
        compact_behind_anchor(module, preserved);
    }
    before != arena_shape(module)
}

/// `KeepUnused::No` compaction behind the interface anchor: the ONE door for
/// it, so no second call site can drop what the anchor roots (the pre-pipeline
/// pointer specialization did).  `preserved` answers for globals and
/// functions; named overrides are always kept.
pub(crate) fn compact_behind_anchor(module: &mut naga::Module, preserved: &dyn Fn(&str) -> bool) {
    module
        .entry_points
        .push(interface_anchor(module, preserved));
    naga::compact::compact(module, naga::compact::KeepUnused::No);
    module.entry_points.pop();
}

/// An entry point whose statements reference every `preserved` global and
/// function and every named override.  The tracer follows statement operands
/// (an `Emit` range alone is not a use) and never type-checks, so any
/// statement carrying the handle roots it.
fn interface_anchor(module: &naga::Module, preserved: &dyn Fn(&str) -> bool) -> naga::EntryPoint {
    let preserved = |name: Option<&str>| name.is_some_and(preserved);
    let span = naga::Span::default();
    let mut function = naga::Function::default();
    let reference = |function: &mut naga::Function, expr: naga::Expression| {
        let handle = function.expressions.append(expr, span);
        function.body.push(
            naga::Statement::Return {
                value: Some(handle),
            },
            span,
        );
    };
    for (h, global) in module.global_variables.iter() {
        if preserved(global.name.as_deref()) {
            reference(&mut function, naga::Expression::GlobalVariable(h));
        }
    }
    for (h, over) in module.overrides.iter() {
        if over.name.is_some() {
            reference(&mut function, naga::Expression::Override(h));
        }
    }
    for (h, f) in module.functions.iter() {
        if preserved(f.name.as_deref()) {
            function.body.push(
                naga::Statement::Call {
                    function: h,
                    arguments: Vec::new(),
                    result: None,
                },
                span,
            );
        }
    }
    naga::EntryPoint {
        name: String::new(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(module: &naga::Module) {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(module)
        .expect("module should remain valid after compact");
    }

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
        assert!(
            compact_module(&mut module, &|_| false),
            "the cull must report itself"
        );
        assert_eq!(module.functions.len(), 0, "unused helper should be removed");
        assert_eq!(
            module.entry_points.len(),
            1,
            "entry point should be preserved"
        );
        validate(&module);
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
        assert!(
            !compact_module(&mut module, &|_| false),
            "no-entry-point module should not lose declarations"
        );
        assert_eq!(
            module.functions.len(),
            2,
            "both functions should be preserved in library module"
        );
    }

    /// The interface anchor: a preserved global and function and a named
    /// override survive `KeepUnused::No`, an unpreserved dead global does not,
    /// and the anchor itself leaves no trace.
    #[test]
    fn preserved_declarations_and_named_overrides_survive() {
        let source = r#"
@group(0) @binding(0) var<storage, read> b: array<u32>;
@group(0) @binding(1) var<storage, read> dead: array<u32>;
override unused_ov: f32 = 1.0;
fn keep_me() -> f32 { return 1.0; }
@compute @workgroup_size(1) fn main() {}
"#;
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        assert!(compact_module(&mut module, &|name| {
            name == "b" || name == "keep_me"
        }));
        let globals: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.name.clone())
            .collect();
        assert_eq!(globals, ["b"]);
        assert_eq!(module.overrides.len(), 1, "the named override is interface");
        assert_eq!(module.functions.len(), 1, "the preserved function is kept");
        assert_eq!(module.entry_points.len(), 1, "the anchor is popped");
        validate(&module);
    }
}
