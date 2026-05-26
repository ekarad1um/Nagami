//! Centralised classifiers and walkers for [`naga::Expression`] and
//! [`naga::Statement`] shapes.  Every helper uses an exhaustive
//! `match` with no `_` arm so a future naga variant trips the build
//! here, forcing a deliberate classification decision at the single
//! point of truth rather than letting passes drift on private
//! deny-lists.
//!
//! Consumers:
//! * `inlining` - template clone gate (`is_disallowed_inline_expression`)
//!   and statement-handle remapper (`try_map_expression_handles_in_place`).
//! * `dead_branch` - `expression_needs_emit` for the short-circuit
//!   re-sugar invariant (`val_a` must be declarative).
//! * `load_dedup` - `needs_pre_emit` for the scope-leak filter; reuses
//!   the emit-range rebuilder.
//! * `cse` - `flatten_replacement_chains`,
//!   `try_map_expression_handles_in_place`, and the post-CSE
//!   statement walker.
//! * `const_fold` - `rebuild_emit_ranges_after_removal`, plus a local
//!   `is_pure_to_clone` that is a *tightened derivative* of
//!   `is_disallowed_inline_expression` (additionally rejects Load and
//!   image / derivative reads because const_fold CLONES expression
//!   content into sibling arena slots rather than referencing it).
//!
//! `coalescing` deliberately keeps its own hand-rolled exhaustive
//! statement walker and does NOT consume this module.

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

/// Read-only counterpart to [`try_map_expression_handles_in_place`]:
/// invoke `visit` for every child-expression handle of `expression`
/// in naga's IR-exposure order.  Declarative and result variants
/// have no children and are skipped.
///
/// Exhaustive match - load-bearing for downstream ref-counting
/// (`const_fold`'s identity gate) and liveness (`dead_param`'s root
/// collector).  A missed variant would understate counts, letting
/// the identity gate green-light an unsafe clone or dead-param
/// elimination drop a live argument.
pub fn visit_expression_children(
    expression: &naga::Expression,
    mut visit: impl FnMut(naga::Handle<naga::Expression>),
) {
    use naga::Expression as E;
    match expression {
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
                visit(c);
            }
        }
        E::Access { base, index } => {
            visit(*base);
            visit(*index);
        }
        E::AccessIndex { base, .. } => visit(*base),
        E::Splat { value, .. } => visit(*value),
        E::Swizzle { vector, .. } => visit(*vector),
        E::Load { pointer } => visit(*pointer),
        E::Unary { expr: e, .. } => visit(*e),
        E::Binary { left, right, .. } => {
            visit(*left);
            visit(*right);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            visit(*condition);
            visit(*accept);
            visit(*reject);
        }
        E::Derivative { expr: e, .. } => visit(*e),
        E::Relational { argument, .. } => visit(*argument),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            visit(*arg);
            if let Some(a) = arg1 {
                visit(*a);
            }
            if let Some(a) = arg2 {
                visit(*a);
            }
            if let Some(a) = arg3 {
                visit(*a);
            }
        }
        E::As { expr: e, .. } => visit(*e),
        E::ArrayLength(e) => visit(*e),
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
            visit(*image);
            visit(*sampler);
            visit(*coordinate);
            if let Some(ai) = array_index {
                visit(*ai);
            }
            if let Some(o) = offset {
                visit(*o);
            }
            match level {
                naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                naga::SampleLevel::Exact(h) | naga::SampleLevel::Bias(h) => visit(*h),
                naga::SampleLevel::Gradient { x, y } => {
                    visit(*x);
                    visit(*y);
                }
            }
            if let Some(d) = depth_ref {
                visit(*d);
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            visit(*image);
            visit(*coordinate);
            if let Some(ai) = array_index {
                visit(*ai);
            }
            if let Some(s) = sample {
                visit(*s);
            }
            if let Some(l) = level {
                visit(*l);
            }
        }
        E::ImageQuery { image, query } => {
            visit(*image);
            match query {
                naga::ImageQuery::Size { level: Some(l) } => visit(*l),
                naga::ImageQuery::Size { level: None }
                | naga::ImageQuery::NumLevels
                | naga::ImageQuery::NumLayers
                | naga::ImageQuery::NumSamples => {}
            }
        }
        E::RayQueryVertexPositions { query, .. } => visit(*query),
        E::RayQueryGetIntersection { query, .. } => visit(*query),
        E::CooperativeLoad { data, .. } => {
            visit(data.pointer);
            visit(data.stride);
        }
        E::CooperativeMultiplyAdd { a, b, c } => {
            visit(*a);
            visit(*b);
            visit(*c);
        }
    }
}

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
            // Exhaustive match over ImageQuery variants so a future
            // naga release that adds a handle-bearing variant breaks
            // the build here instead of silently bypassing the
            // remap walk (matches the contract documented in this
            // file's module header).
            match query {
                naga::ImageQuery::Size { level: Some(level) } => {
                    *level = remap(*level)?;
                }
                naga::ImageQuery::Size { level: None }
                | naga::ImageQuery::NumLevels
                | naga::ImageQuery::NumLayers
                | naga::ImageQuery::NumSamples => {}
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

/// Remap every expression handle referenced by `statement`.
/// Exhaustive match - a future naga variant breaks the build here.
///
/// Per-statement only: callers walk nested blocks themselves.
/// Read-only recursive counterpart is
/// [`visit_block_expression_handles`]; any new handle-bearing
/// `Statement` variant must be added to both.
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

/// Visit every expression handle referenced by `block`, recursing
/// through nested control flow.  Fires `visit` once per direct
/// statement-field reference (`Store.value`, `If.condition`, each
/// `Call.arguments` element, etc.).
///
/// `include_emit_handles` selects between two semantics:
/// * `true` - also fires `visit` once per handle inside each
///   `Statement::Emit` range.  Use for analyses that treat Emit'd
///   expressions as reachable (liveness, where they are let-bound
///   names visible to downstream consumers).
/// * `false` - suppresses Emit visits.  Use for data-flow reference
///   counting; Emit is sequencing, not a use, and conflating the two
///   would push every Emit'd expression to refcount `>= 1` and defeat
///   any unique-owner gate.
///
/// Exhaustive match (no `_` arm), in lockstep with
/// [`remap_statement_handles`]; any new handle-bearing statement
/// variant must be added to both.
pub fn visit_block_expression_handles<F>(
    block: &naga::Block,
    include_emit_handles: bool,
    visit: &mut F,
) where
    F: FnMut(naga::Handle<naga::Expression>),
{
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(range) => {
                if include_emit_handles {
                    for h in range.clone() {
                        visit(h);
                    }
                }
            }
            naga::Statement::Block(inner) => {
                visit_block_expression_handles(inner, include_emit_handles, visit);
            }
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                visit(*condition);
                visit_block_expression_handles(accept, include_emit_handles, visit);
                visit_block_expression_handles(reject, include_emit_handles, visit);
            }
            naga::Statement::Switch { selector, cases } => {
                visit(*selector);
                for case in cases {
                    visit_block_expression_handles(&case.body, include_emit_handles, visit);
                }
            }
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                visit_block_expression_handles(body, include_emit_handles, visit);
                visit_block_expression_handles(continuing, include_emit_handles, visit);
                if let Some(handle) = break_if {
                    visit(*handle);
                }
            }
            naga::Statement::Return { value } => {
                if let Some(handle) = value {
                    visit(*handle);
                }
            }
            naga::Statement::Store { pointer, value } => {
                visit(*pointer);
                visit(*value);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                visit(*image);
                visit(*coordinate);
                if let Some(index) = array_index {
                    visit(*index);
                }
                visit(*value);
            }
            naga::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                visit(*pointer);
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    visit(*c);
                }
                visit(*value);
                if let Some(handle) = result {
                    visit(*handle);
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                visit(*image);
                visit(*coordinate);
                if let Some(index) = array_index {
                    visit(*index);
                }
                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                    visit(*c);
                }
                visit(*value);
            }
            naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                visit(*pointer);
                visit(*result);
            }
            naga::Statement::Call {
                arguments, result, ..
            } => {
                for &argument in arguments {
                    visit(argument);
                }
                if let Some(handle) = result {
                    visit(*handle);
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                visit(*query);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        visit(*acceleration_structure);
                        visit(*descriptor);
                    }
                    naga::RayQueryFunction::Proceed { result } => visit(*result),
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => visit(*hit_t),
                    naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } = fun;
                visit(*acceleration_structure);
                visit(*descriptor);
                visit(*payload);
            }
            naga::Statement::SubgroupBallot { result, predicate } => {
                visit(*result);
                if let Some(handle) = predicate {
                    visit(*handle);
                }
            }
            naga::Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                visit(*argument);
                visit(*result);
                match mode {
                    naga::GatherMode::Broadcast(handle)
                    | naga::GatherMode::Shuffle(handle)
                    | naga::GatherMode::ShuffleDown(handle)
                    | naga::GatherMode::ShuffleUp(handle)
                    | naga::GatherMode::ShuffleXor(handle)
                    | naga::GatherMode::QuadBroadcast(handle) => visit(*handle),
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                }
            }
            naga::Statement::SubgroupCollectiveOperation {
                argument, result, ..
            } => {
                visit(*argument);
                visit(*result);
            }
            naga::Statement::CooperativeStore { target, data } => {
                visit(*target);
                visit(data.pointer);
                visit(data.stride);
            }
            // Terminators / barriers reference no handles.
            naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_) => {}
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
            // Leaf statements - only `Emit` ranges and nested blocks
            // need rebuilding.  Enumerated explicitly so a future
            // naga release adding a new block-bearing variant breaks
            // the build here instead of silently bypassing recursion.
            naga::Statement::Store { .. }
            | naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Return { .. }
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. }
            | naga::Statement::Call { .. }
            | naga::Statement::Atomic { .. }
            | naga::Statement::RayQuery { .. }
            | naga::Statement::RayPipelineFunction(_)
            | naga::Statement::WorkGroupUniformLoad { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. }
            | naga::Statement::CooperativeStore { .. } => {}
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
/// # Cycles
///
/// Every current caller produces an acyclic map by construction.  As a
/// defence in depth against a future caller that violates that
/// invariant, the inner walk is bounded by the map size: at most `N`
/// hops can occur in an acyclic chain over `N` entries, so a longer walk
/// proves a cycle.  On detection the function debug-asserts and exits
/// the chain at the entry where the cycle was reached, leaving the rest
/// of the map untouched - a debug build fails loudly, release builds
/// degrade to a single-level resolution rather than hanging the
/// pipeline.
pub fn flatten_replacement_chains<H>(replacements: &mut std::collections::HashMap<H, H>)
where
    H: Copy + Eq + std::hash::Hash,
{
    let keys: Vec<H> = replacements.keys().copied().collect();
    let max_hops = replacements.len();
    for key in keys {
        let mut target = replacements[&key];
        let mut hops = 0usize;
        while let Some(&next) = replacements.get(&target) {
            if hops >= max_hops {
                debug_assert!(
                    false,
                    "flatten_replacement_chains: cycle detected \
                     (chain exceeded {max_hops} hops); upstream pass produced a cyclic \
                     replacement map"
                );
                break;
            }
            target = next;
            hops += 1;
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

    #[test]
    fn flatten_replacement_chains_terminates_on_cycles() {
        // Defensive guard: every current caller produces an acyclic
        // map by construction, but if a future caller violates that
        // invariant the pipeline must not hang.  In debug builds the
        // bounded walk also debug-asserts; we use the release branch
        // semantics here so the test passes in both profiles.
        //
        // The map below encodes the cycle `1 -> 2 -> 1`.  Without the
        // bound the inner `while let Some(&next) = ...` would loop
        // forever; with it, the function returns in at most `len`
        // iterations per key and leaves the map in a consistent state.
        let mut m: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        m.insert(1, 2);
        m.insert(2, 1);

        // Release builds skip the debug_assert; in debug builds the
        // assert triggers a panic.  Both outcomes are acceptable, but
        // the function must NOT hang.  Catch any debug panic so the
        // test passes in both profiles.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            flatten_replacement_chains(&mut m);
        }));
        let _ = result;
    }

    #[test]
    fn flatten_replacement_chains_terminates_on_three_node_cycle() {
        // Same defence in depth as the two-node cycle above, but with
        // `1 -> 2 -> 3 -> 1` so the hop count must reach 3 before the
        // guard kicks in.
        let mut m: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        m.insert(1, 2);
        m.insert(2, 3);
        m.insert(3, 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            flatten_replacement_chains(&mut m);
        }));
        let _ = result;
    }
}
