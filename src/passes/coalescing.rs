//! Variable coalescing.  Folds multiple same-typed locals whose
//! live ranges are disjoint onto a single backing local, shrinking the
//! declared-locals list and giving the rename pass fewer identifiers
//! to chew through.
//!
//! Live ranges are approximated by `(first, last)` positions assigned
//! in a DFS statement walk.  The approximation is deliberately
//! conservative: any access that overlaps in traversal order is
//! treated as live simultaneously, which rules out only the
//! unambiguously-disjoint cases and never coalesces something that
//! would break.  Locals with initialisers are excluded because their
//! init value would have to be re-materialised at every alias site.

use std::collections::HashMap;

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// Coalesce disjoint same-typed locals onto shared backing slots.
#[derive(Debug, Default)]
pub struct CoalescingPass;

/// Per-local liveness summary gathered from a DFS walk.  `first` /
/// `last` are position indices; `used` distinguishes a live local from
/// one that never appears in the function body.  `init_is_none`
/// disqualifies locals with explicit initialisers from coalescing
/// (see module-level doc).
#[derive(Debug, Clone, Copy)]
struct LocalUse {
    ty: naga::Handle<naga::Type>,
    first: usize,
    last: usize,
    used: bool,
    init_is_none: bool,
}

/// A "lane" is the liveness window currently attached to a
/// representative local; new locals join the lane whose `last`
/// precedes their `first` to form a chain of disjoint windows.
#[derive(Debug, Clone, Copy)]
struct Lane {
    representative: naga::Handle<naga::LocalVariable>,
    last: usize,
}

/// Intermediate record sorted by `(ty, first, last, handle)` so the
/// lane-packing step sees locals in a stable, live-range-friendly order.
#[derive(Debug, Clone, Copy)]
struct LocalSpan {
    handle: naga::Handle<naga::LocalVariable>,
    ty: naga::Handle<naga::Type>,
    first: usize,
    last: usize,
}

impl Pass for CoalescingPass {
    fn name(&self) -> &'static str {
        "variable_coalescing"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        for (_, function) in module.functions.iter_mut() {
            changed += coalesce_function_locals(function);
        }
        for entry in module.entry_points.iter_mut() {
            changed += coalesce_function_locals(&mut entry.function);
        }

        Ok(changed > 0)
    }
}

/// Rewrite every `LocalVariable` expression in `function` to use the
/// coalesced representative.  Returns the number of expression
/// references rewritten (used by the caller as a change flag).
fn coalesce_function_locals(function: &mut naga::Function) -> usize {
    if function.local_variables.is_empty() {
        return 0;
    }

    let usage = collect_local_usage(function);
    let alias = build_alias_map(&usage);
    if alias.is_empty() {
        return 0;
    }

    let mut changed = 0usize;
    for (_, expr) in function.expressions.iter_mut() {
        if let naga::Expression::LocalVariable(local) = expr {
            let mapped = resolve_alias(*local, &alias);
            if mapped != *local {
                *local = mapped;
                changed += 1;
            }
        }
    }

    if changed > 0 {
        function.named_expressions.clear();
    }

    changed
}

/// Build the per-local [`LocalUse`] table.  Maps every `Load`
/// expression to its root `LocalVariable` so the subsequent DFS can
/// attribute reads to the correct handle without re-resolving pointer
/// chains at each use site.
fn collect_local_usage(
    function: &naga::Function,
) -> HashMap<naga::Handle<naga::LocalVariable>, LocalUse> {
    let mut usage = function
        .local_variables
        .iter()
        .map(|(handle, local)| {
            (
                handle,
                LocalUse {
                    ty: local.ty,
                    first: usize::MAX,
                    last: 0,
                    used: false,
                    init_is_none: local.init.is_none(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    // Pre-resolve every `Load` to its root local so the DFS below
    // does not repeat the pointer walk per use.
    let mut load_to_local: HashMap<
        naga::Handle<naga::Expression>,
        naga::Handle<naga::LocalVariable>,
    > = HashMap::new();
    for (eh, expr) in function.expressions.iter() {
        if let naga::Expression::Load { pointer } = *expr {
            if let Some(local) = resolve_ptr_to_local(pointer, &function.expressions) {
                load_to_local.insert(eh, local);
            }
        }
    }

    // DFS the statement tree with monotonic positions; each local
    // records its minimum `first` and maximum `last` across every
    // access to approximate its live range.
    let mut pos = 0usize;
    scan_block_usage(
        &function.body,
        &function.expressions,
        &load_to_local,
        &mut pos,
        &mut usage,
    );

    usage
}

/// Walk a pointer-expression chain back to its root `LocalVariable`
/// handle, unwrapping `AccessIndex` and `Access` wrappers.  Returns
/// `None` if the pointer does not root in a local (function argument
/// pointer, global, etc.), which conservatively excludes the expression
/// from coalescing.
fn resolve_ptr_to_local(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match expressions[expr] {
        naga::Expression::LocalVariable(lh) => Some(lh),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            resolve_ptr_to_local(base, expressions)
        }
        _ => None,
    }
}

/// Record a use of `local` at position `pos`, widening the running
/// `(first, last)` window.  The first touch also flips `used` so the
/// lane packer can skip locals that never appear.
fn mark_used(
    usage: &mut HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
    local: naga::Handle<naga::LocalVariable>,
    pos: usize,
) {
    if let Some(info) = usage.get_mut(&local) {
        if !info.used {
            info.first = pos;
            info.last = pos;
            info.used = true;
        } else {
            info.first = info.first.min(pos);
            info.last = info.last.max(pos);
        }
    }
}

/// DFS traversal that attributes every read, store, call argument,
/// and pointer-flavoured statement operand to the root local it
/// ultimately touches, widening that local's live range.
fn scan_block_usage(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    load_to_local: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::LocalVariable>>,
    pos: &mut usize,
    usage: &mut HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
) {
    for stmt in block {
        let current = *pos;
        *pos += 1;
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(&local) = load_to_local.get(&h) {
                        mark_used(usage, local, current);
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let Some(local) = resolve_ptr_to_local(*pointer, expressions) {
                    mark_used(usage, local, current);
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(local) = resolve_ptr_to_local(arg, expressions) {
                        mark_used(usage, local, current);
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                if let Some(local) = resolve_ptr_to_local(*pointer, expressions) {
                    mark_used(usage, local, current);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                if let Some(local) = resolve_ptr_to_local(*query, expressions) {
                    mark_used(usage, local, current);
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = resolve_ptr_to_local(*payload, expressions) {
                    mark_used(usage, local, current);
                }
            }
            naga::Statement::CooperativeStore { target, .. } => {
                if let Some(local) = resolve_ptr_to_local(*target, expressions) {
                    mark_used(usage, local, current);
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                scan_block_usage(accept, expressions, load_to_local, pos, usage);
                scan_block_usage(reject, expressions, load_to_local, pos, usage);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    scan_block_usage(&case.body, expressions, load_to_local, pos, usage);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                scan_block_usage(body, expressions, load_to_local, pos, usage);
                scan_block_usage(continuing, expressions, load_to_local, pos, usage);
            }
            naga::Statement::Block(inner) => {
                scan_block_usage(inner, expressions, load_to_local, pos, usage);
            }
            _ => {}
        }
    }
}

/// Pack disjoint live ranges into type-keyed lanes and emit an alias
/// map from each coalesced local onto its lane representative.
///
/// Only `used && init_is_none` locals participate; uninitialised
/// locals are the safe set because no init value has to survive
/// aliasing.  Within each `ty` bucket, locals are assigned to the
/// lane whose latest `last` is still strictly earlier than the
/// candidate's `first`, which greedily maximises reuse of already-hot
/// lanes while still respecting non-overlap.
fn build_alias_map(
    usage: &HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
) -> HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::LocalVariable>> {
    let mut locals = usage
        .iter()
        .filter_map(|(&handle, info)| {
            (info.used && info.init_is_none).then_some(LocalSpan {
                handle,
                ty: info.ty,
                first: info.first,
                last: info.last,
            })
        })
        .collect::<Vec<_>>();

    locals.sort_by_key(|s| (s.ty, s.first, s.last, s.handle));

    let mut lanes_by_type: HashMap<naga::Handle<naga::Type>, Vec<Lane>> = HashMap::new();
    let mut alias = HashMap::new();

    for local in locals {
        let lanes = lanes_by_type.entry(local.ty).or_default();

        let selected = lanes
            .iter()
            .enumerate()
            .filter(|(_, lane)| lane.last < local.first)
            .max_by_key(|(_, lane)| lane.last)
            .map(|(idx, _)| idx);

        if let Some(idx) = selected {
            let representative = lanes[idx].representative;
            lanes[idx].last = local.last;
            alias.insert(local.handle, representative);
        } else {
            lanes.push(Lane {
                representative: local.handle,
                last: local.last,
            });
        }
    }

    alias
}

/// Walk `alias` transitively so callers always land on a
/// representative, never an intermediate hop.  Short-circuits on
/// self-loops to avoid infinite chains if the map ever produced one.
fn resolve_alias(
    mut handle: naga::Handle<naga::LocalVariable>,
    alias: &HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::LocalVariable>>,
) -> naga::Handle<naga::LocalVariable> {
    while let Some(next) = alias.get(&handle).copied() {
        if next == handle {
            break;
        }
        handle = next;
    }
    handle
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = CoalescingPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
            .expect("coalescing pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn entry_local_ref_count(module: &naga::Module) -> usize {
        module.entry_points[0]
            .function
            .expressions
            .iter()
            .filter_map(|(_, e)| match e {
                naga::Expression::LocalVariable(h) => Some(*h),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    #[test]
    fn coalesces_non_overlapping_locals_in_straight_line_function() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    a = 1.0;
    let x = a;

    var b: f32;
    b = 2.0;
    let y = b;

    return vec4f(x + y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "coalescing should report change");
        assert_eq!(
            entry_local_ref_count(&module),
            1,
            "non-overlapping locals with same type should coalesce"
        );
    }

    #[test]
    fn no_coalesce_when_both_live_after_branch() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    var b: f32;
    if true {
        a = 1.0;
    } else {
        b = 2.0;
    }
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(!changed, "overlapping locals should not coalesce");
        assert_eq!(
            entry_local_ref_count(&module),
            2,
            "locals both live at return should remain distinct"
        );
    }

    #[test]
    fn coalesces_non_overlapping_locals_across_control_flow() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    a = 1.0;
    let x = a;
    if (x > 0.5) {
        _ = 1;
    }
    var b: f32;
    b = 2.0;
    let y = b;
    return vec4f(x + y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(
            changed,
            "non-overlapping locals across control flow should coalesce"
        );
        assert_eq!(
            entry_local_ref_count(&module),
            1,
            "sequentially-used locals should coalesce despite intervening control flow"
        );
    }

    #[test]
    fn skips_when_local_types_differ() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    a = 1.0;
    let x = a;

    var b: i32;
    b = 2;
    let y = f32(b);

    return vec4f(x + y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(!changed, "locals with different types should not coalesce");
        assert_eq!(
            entry_local_ref_count(&module),
            2,
            "different-typed locals should remain distinct"
        );
    }

    #[test]
    fn coalesces_sequential_locals_around_loop() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    a = 1.0;
    let x = a;
    for (var i: i32 = 0; i < 4; i++) {
        _ = i;
    }
    var b: f32;
    b = 2.0;
    let y = b;
    return vec4f(x + y, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(
            changed,
            "non-overlapping locals around loop should coalesce"
        );
        // 2 distinct locals remain: the coalesced a/b and the loop var i.
        assert_eq!(
            entry_local_ref_count(&module),
            2,
            "sequentially-used locals should coalesce despite intervening loop"
        );
    }

    #[test]
    fn no_coalesce_when_both_live_in_loop() {
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var a: f32;
    var b: f32;
    a = 0.0;
    b = 0.0;
    for (var i: i32 = 0; i < 4; i++) {
        a += 1.0;
        b += a;
    }
    return vec4f(a + b, 0.0, 0.0, 1.0);
}
"#;

        let (changed, _module) = run_pass(source);
        assert!(!changed, "locals both live in loop should not coalesce");
    }

    #[test]
    fn trace_ray_payload_extends_local_live_range() {
        // Construct a function with two locals (a, b) where `a` is used
        // as a TraceRay payload between their usages.  Without tracking
        // TraceRay, the pass would think a's last use ends before b starts,
        // allowing incorrect coalescing.

        let mut module = naga::Module::default();

        let f32_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            },
            naga::Span::UNDEFINED,
        );

        let accel_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::AccelerationStructure {
                    vertex_return: false,
                },
            },
            naga::Span::UNDEFINED,
        );

        let mut function = naga::Function::default();

        let local_a = function.local_variables.append(
            naga::LocalVariable {
                name: Some("a".into()),
                ty: f32_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let local_b = function.local_variables.append(
            naga::LocalVariable {
                name: Some("b".into()),
                ty: f32_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        // Expressions: LocalVariable(a), LocalVariable(b), Load(a), Load(b),
        // a literal, and dummy accel/descriptor expressions.
        let ptr_a = function.expressions.append(
            naga::Expression::LocalVariable(local_a),
            naga::Span::UNDEFINED,
        );
        let ptr_b = function.expressions.append(
            naga::Expression::LocalVariable(local_b),
            naga::Span::UNDEFINED,
        );
        let load_a = function.expressions.append(
            naga::Expression::Load { pointer: ptr_a },
            naga::Span::UNDEFINED,
        );
        let load_b = function.expressions.append(
            naga::Expression::Load { pointer: ptr_b },
            naga::Span::UNDEFINED,
        );
        let lit_one = function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            naga::Span::UNDEFINED,
        );

        // Dummy global var handles for acceleration_structure and descriptor.
        let accel_global = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("accel".into()),
                space: naga::AddressSpace::Handle,
                binding: None,
                ty: accel_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        let accel_expr = function.expressions.append(
            naga::Expression::GlobalVariable(accel_global),
            naga::Span::UNDEFINED,
        );

        let desc_global = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("desc".into()),
                space: naga::AddressSpace::Private,
                binding: None,
                ty: f32_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        let desc_expr = function.expressions.append(
            naga::Expression::GlobalVariable(desc_global),
            naga::Span::UNDEFINED,
        );

        // Build block:
        //   Store(a, 1.0)
        //   Emit(load_a)   -> marks a as used
        //   Store(b, 1.0)  -> marks b as used
        //   TraceRay(accel, desc, payload=ptr_a) -> SHOULD extend a past b.first
        //   Emit(load_b)   -> marks b as used
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_a,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_a, load_a)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: ptr_b,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
                acceleration_structure: accel_expr,
                descriptor: desc_expr,
                payload: ptr_a,
            }),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_b, load_b)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        let usage = collect_local_usage(&function);
        let info_a = usage[&local_a];
        let info_b = usage[&local_b];

        // a's live range must overlap with b's because TraceRay extends a
        // past the point where b starts.
        assert!(
            info_a.last >= info_b.first,
            "TraceRay should extend a's live range to overlap with b (a.last={}, b.first={})",
            info_a.last,
            info_b.first,
        );

        // Verify the alias map does NOT coalesce them.
        let alias = build_alias_map(&usage);
        assert!(
            alias.is_empty(),
            "overlapping locals should not be coalesced when TraceRay extends the range"
        );
    }

    #[test]
    fn cooperative_store_extends_local_live_range() {
        // Construct a function with two locals (a, b) where `a` is the
        // target of a CooperativeStore between their usages.  Without
        // tracking CooperativeStore, the pass would think a's last use
        // ends before b starts, allowing incorrect coalescing.

        let mut module = naga::Module::default();

        let f32_scalar = naga::Scalar::F32;
        let coop_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::CooperativeMatrix {
                    columns: naga::CooperativeSize::Sixteen,
                    rows: naga::CooperativeSize::Sixteen,
                    scalar: f32_scalar,
                    role: naga::CooperativeRole::C,
                },
            },
            naga::Span::UNDEFINED,
        );

        let f32_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(f32_scalar),
            },
            naga::Span::UNDEFINED,
        );

        let mut function = naga::Function::default();

        let local_a = function.local_variables.append(
            naga::LocalVariable {
                name: Some("a".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let local_b = function.local_variables.append(
            naga::LocalVariable {
                name: Some("b".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        let ptr_a = function.expressions.append(
            naga::Expression::LocalVariable(local_a),
            naga::Span::UNDEFINED,
        );
        let ptr_b = function.expressions.append(
            naga::Expression::LocalVariable(local_b),
            naga::Span::UNDEFINED,
        );
        let load_a = function.expressions.append(
            naga::Expression::Load { pointer: ptr_a },
            naga::Span::UNDEFINED,
        );
        let load_b = function.expressions.append(
            naga::Expression::Load { pointer: ptr_b },
            naga::Span::UNDEFINED,
        );
        let lit_one = function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            naga::Span::UNDEFINED,
        );

        // Dummy global for CooperativeData pointer/stride.
        let dummy_global = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("buf".into()),
                space: naga::AddressSpace::Storage {
                    access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                },
                binding: None,
                ty: f32_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        let data_ptr = function.expressions.append(
            naga::Expression::GlobalVariable(dummy_global),
            naga::Span::UNDEFINED,
        );
        let stride = function.expressions.append(
            naga::Expression::Literal(naga::Literal::U32(16)),
            naga::Span::UNDEFINED,
        );

        // Block:
        //   Store(a, lit_one)
        //   Emit(load_a)            -> marks a as used
        //   Store(b, lit_one)       -> marks b as used
        //   CooperativeStore(a, data) -> SHOULD extend a past b.first
        //   Emit(load_b)            -> marks b as used
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_a,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_a, load_a)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: ptr_b,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::CooperativeStore {
                target: ptr_a,
                data: naga::CooperativeData {
                    pointer: data_ptr,
                    stride,
                    row_major: false,
                },
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_b, load_b)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        let usage = collect_local_usage(&function);
        let info_a = usage[&local_a];
        let info_b = usage[&local_b];

        // a's live range must overlap with b's because CooperativeStore
        // extends a past the point where b starts.
        assert!(
            info_a.last >= info_b.first,
            "CooperativeStore should extend a's live range to overlap with b (a.last={}, b.first={})",
            info_a.last,
            info_b.first,
        );

        // Verify the alias map does NOT coalesce them.
        let alias = build_alias_map(&usage);
        assert!(
            alias.is_empty(),
            "overlapping locals should not be coalesced when CooperativeStore extends the range"
        );
    }
}
