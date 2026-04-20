//! Centralised classifiers and walkers for [`naga::Expression`] and
//! [`naga::Statement`] shapes.
//!
//! Multiple passes once carried private deny-lists of expression kinds
//! (see notes in `inlining.rs`).  Each list had to be updated whenever
//! naga added a variant (`CooperativeLoad`, `RayQueryGetIntersection`,
//! and so on), and a missed update silently produced wrong output.
//! The helpers here use exhaustive `match` with no `_` arm, so the
//! compiler refuses to build when naga grows a new variant and forces
//! a deliberate classification decision at that moment.
//!
//! Semantics match the previous per-pass copies exactly; the behaviour
//! is unchanged, only the single source of truth has moved.

// MARK: Expression classifiers

/// `true` when `expression` must appear inside an `Emit` range for the
/// surrounding function to type-check.
///
/// Declarative expressions (`Literal`, `Constant`, `FunctionArgument`,
/// `GlobalVariable`, `LocalVariable`, and so on) and statement-attached
/// result expressions (`CallResult`, `AtomicResult`,
/// `WorkGroupUniformLoadResult`, subgroup and ray-query results) are
/// produced implicitly and need no `Emit`; every other expression does.
pub fn expression_needs_emit(expression: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expression {
        // Declarative references.
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_) => false,
        // Results produced by their originating statement; they come
        // into existence when the statement runs, so no separate `Emit`
        // is needed.
        E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => false,
        // Computational expressions; all must appear in an `Emit` range.
        E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Compose { .. }
        | E::Load { .. }
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Derivative { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. }
        | E::ArrayLength(_)
        | E::RayQueryGetIntersection { .. }
        | E::RayQueryVertexPositions { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => true,
    }
}

/// `true` when `expression` cannot be cloned into a caller during
/// function inlining because it refers to per-invocation state that
/// does not round-trip across function boundaries.
///
/// `LocalVariable` names a function-scoped slot; `CallResult` and the
/// statement-attached result expressions (`AtomicResult`,
/// `WorkGroupUniformLoadResult`, ray-query / subgroup /
/// cooperative-matrix results) exist only at their originating
/// statement's site and cannot be re-rooted.
///
/// NOTE: `GlobalVariable` and `FunctionArgument` are explicitly
/// allowed; globals remap 1-to-1, and arguments are substituted from
/// the caller's actual argument handles during inlining.
pub fn is_disallowed_inline_expression(expression: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expression {
        // Function-local state: cannot be re-rooted in the caller.
        E::LocalVariable(_) => true,
        // Results attached to their originating statement.
        E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::RayQueryVertexPositions { .. }
        | E::RayQueryGetIntersection { .. }
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => true,
        // Everything else is safe to clone into a caller (subject to the
        // usual recursive analysis of its children).
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Compose { .. }
        | E::Load { .. }
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Derivative { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. }
        | E::ArrayLength(_) => false,
    }
}

// MARK: Handle remapping

/// Remap every child-expression handle inside `expression` through
/// `remap`.  Returns `None` if `remap` returns `None` for any handle,
/// leaving `expression` in a partially-remapped state (callers are
/// expected to abandon the expression on that outcome).
///
/// Declarative and result expressions carry no child handles and are
/// skipped.  Like the classifiers above, this walker is exhaustive so
/// new naga variants fail the build instead of silently short-circuiting.
pub fn try_map_expression_handles_in_place(
    expression: &mut naga::Expression,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> Option<naga::Handle<naga::Expression>>,
) -> Option<()> {
    match expression {
        naga::Expression::Literal(_)
        | naga::Expression::Constant(_)
        | naga::Expression::Override(_)
        | naga::Expression::ZeroValue(_)
        | naga::Expression::FunctionArgument(_)
        | naga::Expression::GlobalVariable(_)
        | naga::Expression::CallResult(_)
        | naga::Expression::AtomicResult { .. }
        | naga::Expression::WorkGroupUniformLoadResult { .. }
        | naga::Expression::RayQueryProceedResult
        | naga::Expression::SubgroupBallotResult
        | naga::Expression::SubgroupOperationResult { .. } => {}
        naga::Expression::Compose { components, .. } => {
            for handle in components {
                *handle = remap(*handle)?;
            }
        }
        naga::Expression::Access { base, index } => {
            *base = remap(*base)?;
            *index = remap(*index)?;
        }
        naga::Expression::AccessIndex { base, .. } => {
            *base = remap(*base)?;
        }
        naga::Expression::Splat { value, .. } => {
            *value = remap(*value)?;
        }
        naga::Expression::Swizzle { vector, .. } => {
            *vector = remap(*vector)?;
        }
        naga::Expression::LocalVariable(_) => {}
        naga::Expression::Load { pointer } => {
            *pointer = remap(*pointer)?;
        }
        naga::Expression::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            *image = remap(*image)?;
            *sampler = remap(*sampler)?;
            *coordinate = remap(*coordinate)?;
            if let Some(index) = array_index {
                *index = remap(*index)?;
            }
            if let Some(off) = offset {
                *off = remap(*off)?;
            }
            match level {
                naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                naga::SampleLevel::Exact(handle) | naga::SampleLevel::Bias(handle) => {
                    *handle = remap(*handle)?;
                }
                naga::SampleLevel::Gradient { x, y } => {
                    *x = remap(*x)?;
                    *y = remap(*y)?;
                }
            }
            if let Some(depth) = depth_ref {
                *depth = remap(*depth)?;
            }
        }
        naga::Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            *image = remap(*image)?;
            *coordinate = remap(*coordinate)?;
            if let Some(index) = array_index {
                *index = remap(*index)?;
            }
            if let Some(sample_index) = sample {
                *sample_index = remap(*sample_index)?;
            }
            if let Some(level_expr) = level {
                *level_expr = remap(*level_expr)?;
            }
        }
        naga::Expression::ImageQuery { image, query } => {
            *image = remap(*image)?;
            if let naga::ImageQuery::Size { level: Some(level) } = query {
                *level = remap(*level)?;
            }
        }
        naga::Expression::Unary { expr, .. } => {
            *expr = remap(*expr)?;
        }
        naga::Expression::Binary { left, right, .. } => {
            *left = remap(*left)?;
            *right = remap(*right)?;
        }
        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            *condition = remap(*condition)?;
            *accept = remap(*accept)?;
            *reject = remap(*reject)?;
        }
        naga::Expression::Derivative { expr, .. } => {
            *expr = remap(*expr)?;
        }
        naga::Expression::Relational { argument, .. } => {
            *argument = remap(*argument)?;
        }
        naga::Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            *arg = remap(*arg)?;
            if let Some(value) = arg1 {
                *value = remap(*value)?;
            }
            if let Some(value) = arg2 {
                *value = remap(*value)?;
            }
            if let Some(value) = arg3 {
                *value = remap(*value)?;
            }
        }
        naga::Expression::As { expr, .. } => {
            *expr = remap(*expr)?;
        }
        naga::Expression::ArrayLength(handle) => {
            *handle = remap(*handle)?;
        }
        naga::Expression::RayQueryVertexPositions { query, .. } => {
            *query = remap(*query)?;
        }
        naga::Expression::RayQueryGetIntersection { query, .. } => {
            *query = remap(*query)?;
        }
        naga::Expression::CooperativeLoad { data, .. } => {
            data.pointer = remap(data.pointer)?;
            data.stride = remap(data.stride)?;
        }
        naga::Expression::CooperativeMultiplyAdd { a, b, c } => {
            *a = remap(*a)?;
            *b = remap(*b)?;
            *c = remap(*c)?;
        }
    }

    Some(())
}

/// Remap the optional compare-exchange operand inside an
/// [`naga::AtomicFunction`].  Every other variant is handle-free.
pub fn map_atomic_function_handles(
    fun: &mut naga::AtomicFunction,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    if let naga::AtomicFunction::Exchange {
        compare: Some(compare),
    } = fun
    {
        *compare = remap(*compare);
    }
}

/// Remap the per-lane operand carried by subgroup gather modes.
/// `BroadcastFirst` and `QuadSwap` carry no handle and are skipped.
pub fn map_gather_mode_handles(
    mode: &mut naga::GatherMode,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    match mode {
        naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
        naga::GatherMode::Broadcast(handle)
        | naga::GatherMode::Shuffle(handle)
        | naga::GatherMode::ShuffleDown(handle)
        | naga::GatherMode::ShuffleUp(handle)
        | naga::GatherMode::ShuffleXor(handle)
        | naga::GatherMode::QuadBroadcast(handle) => {
            *handle = remap(*handle);
        }
    }
}

/// Remap every operand handle reachable through an
/// [`naga::RayQueryFunction`].  Control-only variants
/// (`ConfirmIntersection`, `Terminate`) are skipped.
pub fn map_ray_query_function_handles(
    fun: &mut naga::RayQueryFunction,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    match fun {
        naga::RayQueryFunction::Initialize {
            acceleration_structure,
            descriptor,
        } => {
            *acceleration_structure = remap(*acceleration_structure);
            *descriptor = remap(*descriptor);
        }
        naga::RayQueryFunction::Proceed { result } => {
            *result = remap(*result);
        }
        naga::RayQueryFunction::GenerateIntersection { hit_t } => {
            *hit_t = remap(*hit_t);
        }
        naga::RayQueryFunction::ConfirmIntersection | naga::RayQueryFunction::Terminate => {}
    }
}

/// Remap every operand handle reachable through an
/// [`naga::RayPipelineFunction`] (currently `TraceRay`).
pub fn map_ray_pipeline_function_handles(
    fun: &mut naga::RayPipelineFunction,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    match fun {
        naga::RayPipelineFunction::TraceRay {
            acceleration_structure,
            descriptor,
            payload,
        } => {
            *acceleration_structure = remap(*acceleration_structure);
            *descriptor = remap(*descriptor);
            *payload = remap(*payload);
        }
    }
}

/// Remap the `pointer` and `stride` operands of a cooperative-matrix
/// load/store descriptor.
pub fn map_cooperative_data_handles(
    data: &mut naga::CooperativeData,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    data.pointer = remap(data.pointer);
    data.stride = remap(data.stride);
}

// MARK: Statement walkers

/// Remap every expression handle referenced by `statement`.  Control
/// flow is walked exhaustively so a future naga variant must be
/// explicitly classified here before the build succeeds.
pub fn remap_statement_handles(
    statement: &mut naga::Statement,
    remap: &mut impl FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    match statement {
        naga::Statement::Emit(_) | naga::Statement::Block(_) => {}
        naga::Statement::If { condition, .. } => {
            *condition = remap(*condition);
        }
        naga::Statement::Switch { selector, .. } => {
            *selector = remap(*selector);
        }
        naga::Statement::Loop { break_if, .. } => {
            if let Some(handle) = break_if {
                *handle = remap(*handle);
            }
        }
        naga::Statement::Break | naga::Statement::Continue | naga::Statement::Kill => {}
        naga::Statement::Return { value } => {
            if let Some(handle) = value {
                *handle = remap(*handle);
            }
        }
        naga::Statement::ControlBarrier(_) | naga::Statement::MemoryBarrier(_) => {}
        naga::Statement::Store { pointer, value } => {
            *pointer = remap(*pointer);
            *value = remap(*value);
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            *value = remap(*value);
        }
        naga::Statement::Atomic {
            pointer,
            fun,
            value,
            result,
        } => {
            *pointer = remap(*pointer);
            map_atomic_function_handles(fun, remap);
            *value = remap(*value);
            if let Some(handle) = result {
                *handle = remap(*handle);
            }
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            fun,
            value,
        } => {
            *image = remap(*image);
            *coordinate = remap(*coordinate);
            if let Some(index) = array_index {
                *index = remap(*index);
            }
            map_atomic_function_handles(fun, remap);
            *value = remap(*value);
        }
        naga::Statement::WorkGroupUniformLoad { pointer, result } => {
            *pointer = remap(*pointer);
            *result = remap(*result);
        }
        naga::Statement::Call {
            arguments, result, ..
        } => {
            for argument in arguments {
                *argument = remap(*argument);
            }
            if let Some(handle) = result {
                *handle = remap(*handle);
            }
        }
        naga::Statement::RayQuery { query, fun } => {
            *query = remap(*query);
            map_ray_query_function_handles(fun, remap);
        }
        naga::Statement::RayPipelineFunction(fun) => {
            map_ray_pipeline_function_handles(fun, remap);
        }
        naga::Statement::SubgroupBallot { result, predicate } => {
            *result = remap(*result);
            if let Some(handle) = predicate {
                *handle = remap(*handle);
            }
        }
        naga::Statement::SubgroupGather {
            mode,
            argument,
            result,
        } => {
            map_gather_mode_handles(mode, remap);
            *argument = remap(*argument);
            *result = remap(*result);
        }
        naga::Statement::SubgroupCollectiveOperation {
            argument, result, ..
        } => {
            *argument = remap(*argument);
            *result = remap(*result);
        }
        naga::Statement::CooperativeStore { target, data } => {
            *target = remap(*target);
            map_cooperative_data_handles(data, remap);
        }
    }
}

// MARK: Emit-range surgery

/// Drop every handle in `removed` from every `Emit` range inside
/// `block`, rebuilding contiguous sub-ranges around the survivors and
/// discarding `Emit` statements that become empty.  Walks nested
/// control flow recursively.
///
/// Shared by passes that rewrite expression handles in place (CSE,
/// constant folding to `Literal`, and so on).  In all cases some
/// handles must drop out of their old `Emit` range, either because a
/// canonical replacement took over (CSE) or because the new expression
/// shape is no longer allowed inside `Emit` (folded literals).  One
/// implementation here prevents drift between per-pass copies when
/// new control-flow statements are introduced upstream.
pub fn rebuild_emit_ranges_after_removal(
    block: &mut naga::Block,
    removed: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    let original = std::mem::replace(block, naga::Block::new());
    for (mut statement, span) in original.span_into_iter() {
        match &mut statement {
            naga::Statement::Emit(range) => {
                let surviving: Vec<_> = range.clone().filter(|h| !removed.contains(h)).collect();
                if surviving.is_empty() {
                    continue; // Drop an emit that lost every handle.
                }
                // Rebuild contiguous sub-ranges around survivors.
                let mut start = surviving[0];
                let mut end = surviving[0];
                for &h in &surviving[1..] {
                    if h.index() == end.index() + 1 {
                        end = h;
                    } else {
                        block.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                            span,
                        );
                        start = h;
                        end = h;
                    }
                }
                block.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                    span,
                );
                continue;
            }
            naga::Statement::Block(inner) => rebuild_emit_ranges_after_removal(inner, removed),
            naga::Statement::If { accept, reject, .. } => {
                rebuild_emit_ranges_after_removal(accept, removed);
                rebuild_emit_ranges_after_removal(reject, removed);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    rebuild_emit_ranges_after_removal(&mut case.body, removed);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                rebuild_emit_ranges_after_removal(body, removed);
                rebuild_emit_ranges_after_removal(continuing, removed);
            }
            _ => {}
        }
        block.push(statement, span);
    }
}

// MARK: Replacement chain flattening

/// Collapse transitive chains in a replacement map so every key points
/// directly at its terminal target.
///
/// Passes that accumulate `HashMap<H, H>` entries of the form "replace
/// handle A with handle B" can produce chains like `A -> B -> C` when
/// `B` was picked as canonical before `C` supplanted it (CSE) or when
/// a load was store-forwarded to an earlier load that was itself
/// forwarded elsewhere (`load_dedup`).  A single
/// [`try_map_expression_handles_in_place`] walk only resolves one
/// level, so callers MUST flatten before applying the map to the arena,
/// otherwise dangling references survive.
///
/// # Panics
///
/// Loops forever on a cyclic map.  Every current caller produces an
/// acyclic map by construction; maintaining that property is the
/// caller's responsibility.
pub fn flatten_replacement_chains<H>(replacements: &mut std::collections::HashMap<H, H>)
where
    H: Copy + Eq + std::hash::Hash,
{
    let keys: Vec<H> = replacements.keys().copied().collect();
    for key in keys {
        let mut target = replacements[&key];
        while let Some(&next) = replacements.get(&target) {
            target = next;
        }
        replacements.insert(key, target);
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal arena and return a real `Handle<Expression>`
    /// usable as a placeholder inside variants whose body fields the
    /// classifiers never inspect.
    fn dummy_handles() -> (
        naga::Arena<naga::Expression>,
        naga::Handle<naga::Expression>,
    ) {
        let mut arena = naga::Arena::<naga::Expression>::new();
        let h = arena.append(
            naga::Expression::Literal(naga::Literal::I32(0)),
            naga::Span::UNDEFINED,
        );
        (arena, h)
    }

    fn dummy_type_handle() -> naga::Handle<naga::Type> {
        let mut u = naga::UniqueArena::<naga::Type>::new();
        u.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::I32),
            },
            naga::Span::UNDEFINED,
        )
    }

    fn dummy_const_handle() -> naga::Handle<naga::Constant> {
        let mut a = naga::Arena::<naga::Constant>::new();
        // Reuse a placeholder type handle; the classifier never dereferences it.
        let ty = dummy_type_handle();
        a.append(
            naga::Constant {
                name: None,
                ty,
                init: {
                    let (_, h) = dummy_handles();
                    h
                },
            },
            naga::Span::UNDEFINED,
        )
    }

    #[test]
    fn needs_emit_declarative_false() {
        assert!(!expression_needs_emit(&naga::Expression::Literal(
            naga::Literal::I32(0)
        )));
        assert!(!expression_needs_emit(&naga::Expression::Constant(
            dummy_const_handle()
        )));
        assert!(!expression_needs_emit(&naga::Expression::ZeroValue(
            dummy_type_handle()
        )));
        assert!(!expression_needs_emit(&naga::Expression::FunctionArgument(
            0
        )));
    }

    #[test]
    fn needs_emit_statement_results_false() {
        let (_, h) = dummy_handles();
        assert!(!expression_needs_emit(&naga::Expression::CallResult(
            naga::Arena::<naga::Function>::new()
                .append(naga::Function::default(), naga::Span::UNDEFINED)
        )));
        let _ = h;
        assert!(!expression_needs_emit(
            &naga::Expression::RayQueryProceedResult
        ));
        assert!(!expression_needs_emit(
            &naga::Expression::SubgroupBallotResult
        ));
    }

    #[test]
    fn needs_emit_computational_true() {
        let (_, h) = dummy_handles();
        assert!(expression_needs_emit(&naga::Expression::AccessIndex {
            base: h,
            index: 0
        }));
        assert!(expression_needs_emit(&naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: h
        }));
        assert!(expression_needs_emit(&naga::Expression::Load {
            pointer: h
        }));
        assert!(expression_needs_emit(&naga::Expression::ArrayLength(h)));
    }

    #[test]
    fn disallowed_inline_statement_results_true() {
        let mut locals = naga::Arena::<naga::LocalVariable>::new();
        let lv = locals.append(
            naga::LocalVariable {
                name: None,
                ty: dummy_type_handle(),
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        assert!(is_disallowed_inline_expression(
            &naga::Expression::LocalVariable(lv)
        ));
        assert!(is_disallowed_inline_expression(
            &naga::Expression::RayQueryProceedResult
        ));
        assert!(is_disallowed_inline_expression(
            &naga::Expression::SubgroupBallotResult
        ));
    }

    #[test]
    fn disallowed_inline_declarative_false() {
        // `FunctionArgument` and `GlobalVariable` are explicitly allowed
        // because they map 1-to-1 into the caller's context during
        // inlining; only per-invocation state is forbidden.
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::Literal(naga::Literal::I32(0))
        ));
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::FunctionArgument(0)
        ));

        let mut globals = naga::Arena::<naga::GlobalVariable>::new();
        let gv = globals.append(
            naga::GlobalVariable {
                name: None,
                space: naga::AddressSpace::Private,
                binding: None,
                ty: dummy_type_handle(),
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        assert!(!is_disallowed_inline_expression(
            &naga::Expression::GlobalVariable(gv)
        ));
    }

    #[test]
    fn disallowed_inline_computational_false() {
        let (_, h) = dummy_handles();
        assert!(!is_disallowed_inline_expression(&naga::Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: h,
        }));
        assert!(!is_disallowed_inline_expression(&naga::Expression::Load {
            pointer: h,
        }));
    }

    #[test]
    fn flatten_replacement_chains_collapses_transitive_edges() {
        use std::collections::HashMap;
        let mut m: HashMap<u32, u32> = HashMap::new();
        // Build the chain `1 -> 2 -> 3 -> 4` and a standalone edge `5 -> 6`.
        m.insert(1, 2);
        m.insert(2, 3);
        m.insert(3, 4);
        m.insert(5, 6);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[&1], 4);
        assert_eq!(m[&2], 4);
        assert_eq!(m[&3], 4);
        assert_eq!(m[&5], 6);
    }

    #[test]
    fn flatten_replacement_chains_is_noop_on_direct_edges() {
        use std::collections::HashMap;
        let mut m: HashMap<u32, u32> = HashMap::new();
        m.insert(1, 10);
        m.insert(2, 20);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[&1], 10);
        assert_eq!(m[&2], 20);
    }
}
