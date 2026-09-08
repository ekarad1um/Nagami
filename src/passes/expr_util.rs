//! Classifiers and walkers for [`naga::Expression`] and [`naga::Statement`]
//! shapes, shared by the passes and the generator.  Every match here is
//! exhaustive with no `_` arm, so a new naga variant fails the build at this
//! single point of truth instead of drifting silently through consumers'
//! private deny-lists.  (`coalescing` keeps its own exhaustive statement
//! walker.)

use crate::handle_set::{HandleMap, HandleSet};

// MARK: Module traversal

/// Every function body: free functions, then entry points - the order the
/// generator's per-function caches are indexed by, and one walk so a pass
/// cannot optimise one arena and silently skip the other.
pub(crate) fn all_functions(module: &naga::Module) -> impl Iterator<Item = &naga::Function> {
    module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function))
}

/// Mutable [`all_functions`].  A callback, not an iterator: a `Chain` leaves
/// one call site per pass, which tips LLVM into inlining each per-function
/// worker into the pass body (+6,080 bytes of text, measured).  Takes the
/// arenas rather than the module so a caller can hold `&module.types` across
/// the walk.
pub(crate) fn for_each_function_mut(
    functions: &mut naga::Arena<naga::Function>,
    entry_points: &mut [naga::EntryPoint],
    visit: &mut dyn FnMut(&mut naga::Function),
) {
    for (_, function) in functions.iter_mut() {
        visit(function);
    }
    for entry in entry_points.iter_mut() {
        visit(&mut entry.function);
    }
}

// MARK: Expression classifiers

/// Whether `expression` must sit inside an `Emit` range: declarative
/// references and statement-attached results are produced implicitly, every
/// computation is not.
pub fn expression_needs_emit(expression: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expression {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_) => false,
        E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => false,
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

/// Whether `expression` cannot be cloned into a caller during inlining.
/// `LocalVariable` names a function-scoped slot; statement-attached results
/// (`CallResult`, `AtomicResult`, ...) exist only at their statement; and
/// the ray-query / cooperative-matrix reads, although Emit'd, depend on a
/// mutable cursor or lane state that does not relocate.  `GlobalVariable`
/// remaps 1-to-1 and `FunctionArgument` is substituted from the call site,
/// so both are allowed.
pub fn is_disallowed_inline_expression(expression: &naga::Expression) -> bool {
    use naga::Expression as E;
    match expression {
        E::LocalVariable(_) => true,
        E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. }
        | E::RayQueryVertexPositions { .. }
        | E::RayQueryGetIntersection { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => true,
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

/// Invoke `visit` for every child handle of `expression` in naga's IR order.
/// Reference counting and liveness rely on it being complete: a missed child
/// understates a count and green-lights an unsafe clone or a live-argument
/// drop.
#[inline]
pub fn visit_expression_children(
    expression: &naga::Expression,
    mut visit: impl FnMut(naga::Handle<naga::Expression>),
) {
    visit_expression_children_dyn(expression, &mut visit)
}

/// A `dyn` callback keeps this match from being instantiated per call-site
/// closure.
fn visit_expression_children_dyn(
    expression: &naga::Expression,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
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

/// Remap every child handle of `expression` through `remap`, or `None` as
/// soon as `remap` declines, leaving `expression` partially remapped
/// (callers abandon it).
pub fn try_map_expression_handles_in_place(
    expression: &mut naga::Expression,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> Option<naga::Handle<naga::Expression>>,
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

/// Remap the compare-exchange operand, the only handle an
/// [`naga::AtomicFunction`] carries.
pub fn map_atomic_function_handles(
    fun: &mut naga::AtomicFunction,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    match fun {
        naga::AtomicFunction::Exchange {
            compare: Some(compare),
        } => {
            *compare = remap(*compare);
        }
        naga::AtomicFunction::Exchange { compare: None }
        | naga::AtomicFunction::Add
        | naga::AtomicFunction::Subtract
        | naga::AtomicFunction::And
        | naga::AtomicFunction::ExclusiveOr
        | naga::AtomicFunction::InclusiveOr
        | naga::AtomicFunction::Min
        | naga::AtomicFunction::Max => {}
    }
}

/// Read-only [`map_atomic_function_handles`].
pub fn visit_atomic_function_handles(
    fun: &naga::AtomicFunction,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    match fun {
        naga::AtomicFunction::Exchange {
            compare: Some(compare),
        } => visit(*compare),
        naga::AtomicFunction::Exchange { compare: None }
        | naga::AtomicFunction::Add
        | naga::AtomicFunction::Subtract
        | naga::AtomicFunction::And
        | naga::AtomicFunction::ExclusiveOr
        | naga::AtomicFunction::InclusiveOr
        | naga::AtomicFunction::Min
        | naga::AtomicFunction::Max => {}
    }
}

/// Remap the per-lane operand of a subgroup gather mode.
pub fn map_gather_mode_handles(
    mode: &mut naga::GatherMode,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
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

/// Remap every operand handle of a [`naga::RayQueryFunction`].
pub fn map_ray_query_function_handles(
    fun: &mut naga::RayQueryFunction,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
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

/// Remap every operand handle of a [`naga::RayPipelineFunction`].
pub fn map_ray_pipeline_function_handles(
    fun: &mut naga::RayPipelineFunction,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
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

/// Remap the `pointer` and `stride` operands of a cooperative-matrix load /
/// store.
pub fn map_cooperative_data_handles(
    data: &mut naga::CooperativeData,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    data.pointer = remap(data.pointer);
    data.stride = remap(data.stride);
}

// MARK: Statement walkers

/// Remap the expression handles in `statement`'s own fields; nested blocks
/// are the caller's.  `Emit` is deliberately a no-op: its range members are
/// not remapped, because callers that renumber the arena rebuild Emit ranges
/// themselves and callers that filter in place never renumber.  Kept in
/// lockstep with [`visit_statement_expression_handles`].
pub fn remap_statement_handles(
    statement: &mut naga::Statement,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
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

/// Every expression handle `block` references, nested blocks included:
/// operands and results, plus each `Emit` range handle when
/// `include_emit_handles` (semantics on [`visit_statement_operands`]).
pub fn visit_block_expression_handles(
    block: &naga::Block,
    include_emit_handles: bool,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    for stmt in block.iter() {
        visit_statement_expression_handles(stmt, include_emit_handles, visit);
    }
}

/// Per-statement counterpart of [`visit_block_expression_handles`]: the
/// statement's own fields, results included, then its nested blocks.
pub fn visit_statement_expression_handles(
    stmt: &naga::Statement,
    include_emit_handles: bool,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    visit_statement_fields(stmt, include_emit_handles, true, visit);
    for nested in nested_blocks(stmt) {
        visit_block_expression_handles(nested, include_emit_handles, visit);
    }
}

/// The handles `stmt` reads directly: operands, not the results it defines
/// or its nested blocks.  `include_emit_handles` also visits each `Emit`
/// range handle: right for liveness (an Emit'd expression is a reachable
/// let-bound name), wrong for reference counting (Emit is sequencing, not a
/// use; counting it gives every emitted expression a count >= 1 and defeats
/// unique-owner gates).
pub fn visit_statement_operands(
    stmt: &naga::Statement,
    include_emit_handles: bool,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    visit_statement_fields(stmt, include_emit_handles, false, visit);
}

/// `include_results` adds the results a statement defines, which only
/// whole-function reference counts want.
fn visit_statement_fields(
    stmt: &naga::Statement,
    include_emit_handles: bool,
    include_results: bool,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    let result = |h: naga::Handle<naga::Expression>, visit: &mut dyn FnMut(_)| {
        if include_results {
            visit(h);
        }
    };
    match stmt {
        naga::Statement::Emit(range) => {
            if include_emit_handles {
                for h in range.clone() {
                    visit(h);
                }
            }
        }
        naga::Statement::If { condition, .. } => visit(*condition),
        naga::Statement::Switch { selector, .. } => visit(*selector),
        naga::Statement::Loop { break_if, .. } => {
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
            result: res,
        } => {
            visit(*pointer);
            visit_atomic_function_handles(fun, visit);
            visit(*value);
            if let Some(handle) = res {
                result(*handle, visit);
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
            visit_atomic_function_handles(fun, visit);
            visit(*value);
        }
        naga::Statement::WorkGroupUniformLoad {
            pointer,
            result: res,
        } => {
            visit(*pointer);
            result(*res, visit);
        }
        naga::Statement::Call {
            arguments,
            result: res,
            ..
        } => {
            for &argument in arguments {
                visit(argument);
            }
            if let Some(handle) = res {
                result(*handle, visit);
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
                naga::RayQueryFunction::Proceed { result: res } => result(*res, visit),
                naga::RayQueryFunction::GenerateIntersection { hit_t } => visit(*hit_t),
                naga::RayQueryFunction::ConfirmIntersection | naga::RayQueryFunction::Terminate => {
                }
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
        naga::Statement::SubgroupBallot {
            result: res,
            predicate,
        } => {
            result(*res, visit);
            if let Some(handle) = predicate {
                visit(*handle);
            }
        }
        naga::Statement::SubgroupGather {
            mode,
            argument,
            result: res,
        } => {
            visit(*argument);
            result(*res, visit);
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
            argument,
            result: res,
            ..
        } => {
            visit(*argument);
            result(*res, visit);
        }
        naga::Statement::CooperativeStore { target, data } => {
            visit(*target);
            visit(data.pointer);
            visit(data.stride);
        }
        naga::Statement::Block(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_) => {}
    }
}

/// Pointer operands `stmt` may WRITE through: `Store` / `Atomic` pointers,
/// every `Call` argument (a `ptr<function>` parameter lets the callee write
/// the pointee), `traceRay`'s payload, a cooperative store's destination,
/// and the ray-query object.
pub fn visit_statement_write_pointers(
    stmt: &naga::Statement,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    match stmt {
        naga::Statement::Store { pointer, .. } | naga::Statement::Atomic { pointer, .. } => {
            visit(*pointer)
        }
        naga::Statement::Call { arguments, .. } => {
            for &argument in arguments {
                visit(argument);
            }
        }
        naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            payload,
            ..
        }) => visit(*payload),
        naga::Statement::CooperativeStore { data, .. } => visit(data.pointer),
        naga::Statement::RayQuery { query, .. } => visit(*query),
        naga::Statement::Emit(_)
        | naga::Statement::Block(_)
        | naga::Statement::If { .. }
        | naga::Statement::Switch { .. }
        | naga::Statement::Loop { .. }
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Return { .. }
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::ImageStore { .. }
        | naga::Statement::ImageAtomic { .. }
        | naga::Statement::WorkGroupUniformLoad { .. }
        | naga::Statement::SubgroupBallot { .. }
        | naga::Statement::SubgroupGather { .. }
        | naga::Statement::SubgroupCollectiveOperation { .. } => {}
    }
}

/// Pre-order, syntactic-order walk of `block` and its nested blocks.
pub fn for_each_statement(block: &naga::Block, f: &mut dyn FnMut(&naga::Statement)) {
    for stmt in block.iter() {
        f(stmt);
        for nested in nested_blocks(stmt) {
            for_each_statement(nested, f);
        }
    }
}

// MARK: Nested-block traversal

/// The blocks nested directly inside one statement, in syntactic order;
/// from [`nested_blocks`].
pub enum NestedBlocks<'a> {
    /// Block-free statement.
    None,
    /// Up to two fixed blocks (`Block` yields one; `If` / `Loop` two).
    Pair(Option<&'a naga::Block>, Option<&'a naga::Block>),
    /// One body per switch case, in declaration order.
    Cases(std::slice::Iter<'a, naga::SwitchCase>),
}

impl<'a> Iterator for NestedBlocks<'a> {
    type Item = &'a naga::Block;

    fn next(&mut self) -> Option<&'a naga::Block> {
        match self {
            NestedBlocks::None => None,
            NestedBlocks::Pair(first, second) => first.take().or_else(|| second.take()),
            NestedBlocks::Cases(cases) => cases.next().map(|case| &case.body),
        }
    }
}

/// Mutable [`NestedBlocks`]; from [`nested_blocks_mut`].
pub enum NestedBlocksMut<'a> {
    /// Block-free statement.
    None,
    /// Up to two fixed blocks (`Block` yields one; `If` / `Loop` two).
    Pair(Option<&'a mut naga::Block>, Option<&'a mut naga::Block>),
    /// One body per switch case, in declaration order.
    Cases(std::slice::IterMut<'a, naga::SwitchCase>),
}

impl<'a> Iterator for NestedBlocksMut<'a> {
    type Item = &'a mut naga::Block;

    fn next(&mut self) -> Option<&'a mut naga::Block> {
        match self {
            NestedBlocksMut::None => None,
            NestedBlocksMut::Pair(first, second) => first.take().or_else(|| second.take()),
            NestedBlocksMut::Cases(cases) => cases.next().map(|case| &mut case.body),
        }
    }
}

/// Every block nested directly inside `stmt`, in syntactic order (`If`:
/// accept then reject; `Switch`: cases in declaration order; `Loop`: body
/// then continuing).  The order is part of the contract: order-sensitive
/// accumulation (naming, first-appearance ranking) relies on it matching
/// source order.
///
/// This match is THE crate-wide answer to which statement variants carry
/// blocks: recursive walkers delegate their descent here, so a new naga
/// variant fails the build once instead of being silently treated as a leaf
/// by hand-written walkers, a miscompile.  A walker-local `_` arm is then
/// safe for recursion; a variant needing special treatment still needs its
/// own arm.
pub fn nested_blocks(stmt: &naga::Statement) -> NestedBlocks<'_> {
    match stmt {
        naga::Statement::Block(inner) => NestedBlocks::Pair(Some(inner), None),
        naga::Statement::If { accept, reject, .. } => {
            NestedBlocks::Pair(Some(accept), Some(reject))
        }
        naga::Statement::Switch { cases, .. } => NestedBlocks::Cases(cases.iter()),
        naga::Statement::Loop {
            body, continuing, ..
        } => NestedBlocks::Pair(Some(body), Some(continuing)),
        naga::Statement::Emit(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::Return { .. }
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::Store { .. }
        | naga::Statement::ImageStore { .. }
        | naga::Statement::Atomic { .. }
        | naga::Statement::ImageAtomic { .. }
        | naga::Statement::WorkGroupUniformLoad { .. }
        | naga::Statement::Call { .. }
        | naga::Statement::RayQuery { .. }
        | naga::Statement::RayPipelineFunction(_)
        | naga::Statement::SubgroupBallot { .. }
        | naga::Statement::SubgroupGather { .. }
        | naga::Statement::SubgroupCollectiveOperation { .. }
        | naga::Statement::CooperativeStore { .. } => NestedBlocks::None,
    }
}

/// Mutable [`nested_blocks`]; same order contract, kept in lockstep.
pub fn nested_blocks_mut(stmt: &mut naga::Statement) -> NestedBlocksMut<'_> {
    match stmt {
        naga::Statement::Block(inner) => NestedBlocksMut::Pair(Some(inner), None),
        naga::Statement::If { accept, reject, .. } => {
            NestedBlocksMut::Pair(Some(accept), Some(reject))
        }
        naga::Statement::Switch { cases, .. } => NestedBlocksMut::Cases(cases.iter_mut()),
        naga::Statement::Loop {
            body, continuing, ..
        } => NestedBlocksMut::Pair(Some(body), Some(continuing)),
        naga::Statement::Emit(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::Return { .. }
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::Store { .. }
        | naga::Statement::ImageStore { .. }
        | naga::Statement::Atomic { .. }
        | naga::Statement::ImageAtomic { .. }
        | naga::Statement::WorkGroupUniformLoad { .. }
        | naga::Statement::Call { .. }
        | naga::Statement::RayQuery { .. }
        | naga::Statement::RayPipelineFunction(_)
        | naga::Statement::SubgroupBallot { .. }
        | naga::Statement::SubgroupGather { .. }
        | naga::Statement::SubgroupCollectiveOperation { .. }
        | naga::Statement::CooperativeStore { .. } => NestedBlocksMut::None,
    }
}

// MARK: Emit-range surgery

/// Push `surviving` as `Emit` statements, one per contiguous run.  Shared by
/// every pass that drops handles out of a range, so the run-splitting exists
/// once.
pub(crate) fn push_emit_runs(
    block: &mut naga::Block,
    surviving: &[naga::Handle<naga::Expression>],
    span: naga::Span,
) {
    let Some(&first) = surviving.first() else {
        return;
    };
    let mut start = first;
    let mut end = first;
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
}

/// Drop every handle in `removed` from every `Emit` range in `block`
/// (nested control flow included), rebuilding contiguous sub-ranges around
/// the survivors and discarding `Emit` statements left empty.  One
/// implementation for every pass that rewrites handles in place (CSE's
/// canonical replacements, folded literals), so a new control-flow
/// statement is handled once.  Crate-private: `HandleSet` is.
pub(crate) fn rebuild_emit_ranges_after_removal(
    block: &mut naga::Block,
    removed: &HandleSet<naga::Expression>,
) {
    let original = std::mem::replace(block, naga::Block::new());
    for (mut statement, span) in original.span_into_iter() {
        match &mut statement {
            naga::Statement::Emit(range) => {
                // An emit that lost every handle disappears.
                let surviving: Vec<_> = range.clone().filter(|h| !removed.contains(h)).collect();
                push_emit_runs(block, &surviving, span);
                continue;
            }
            _ => {
                for nested in nested_blocks_mut(&mut statement) {
                    rebuild_emit_ranges_after_removal(nested, removed);
                }
            }
        }
        block.push(statement, span);
    }
}

// MARK: Literal predicates

/// Convert a width-8 literal (`F64` / `U64` / `I64`) to `target` (f32 / i32 /
/// u32 / bool), `None` otherwise - f16 / f64 / i64 / u64 targets included,
/// since naga accepts those forms.  naga's frontend refuses to const-fold these casts
/// (f64 / u64 / i64 are non-standard WGSL), so the `As` node survives and an
/// emitted `f32(<F64 literal>)` is rejected on re-parse.  Semantics mirror
/// `naga::proc::ConstantEvaluator::cast`: a float source rounds to nearest
/// (declined when non-finite) and CLAMPS to integer range; an integer source
/// WRAPS (`i64(-1) -> u32` is `4294967295u`); `-> bool` is `v != 0`.
///
/// Shared because both sides of the round-trip need it: `const_fold` folds
/// the scalar `As`, and the generator converts the components of a vector
/// narrowing `materialize_vector` cannot, needing each converted literal as
/// an arena handle.
pub(crate) fn cast_width8_to(src: naga::Literal, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    if let L::F64(v) = src {
        return match (target.kind, target.width) {
            (K::Float, 4) => {
                let r = v as f32;
                r.is_finite().then_some(L::F32(r))
            }
            (K::Sint, 4) => Some(L::I32(v.clamp(i32::MIN as f64, i32::MAX as f64) as i32)),
            (K::Uint, 4) => Some(L::U32(v.clamp(u32::MIN as f64, u32::MAX as f64) as u32)),
            (K::Bool, _) => Some(L::Bool(v != 0.0)),
            _ => None,
        };
    }
    let v: i128 = match src {
        L::U64(v) => v as i128,
        L::I64(v) => v as i128,
        _ => return None,
    };
    match (target.kind, target.width) {
        (K::Float, 4) => Some(L::F32(v as f32)),
        (K::Sint, 4) => Some(L::I32(v as i32)),
        (K::Uint, 4) => Some(L::U32(v as u32)),
        (K::Bool, _) => Some(L::Bool(v != 0)),
        _ => None,
    }
}

pub(crate) fn is_bool_true(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

pub(crate) fn is_bool_false(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Const-ness of `expr` itself as a WGSL const-expression: `Some` decides,
/// `None` defers to the operands.  Children precede parents in a naga arena,
/// so one forward pass makes it per-handle.  Exhaustive because no default is
/// safe: `load_dedup` declines only where this says const and `const_fold`
/// only where it says runtime, so either guess stops a decline that matters.
pub(crate) fn const_expression_leaf(expr: &naga::Expression) -> Option<bool> {
    use naga::Expression as E;
    match expr {
        // `Override` is not const, but a slot made of one moves the error
        // to pipeline creation, which the runtime input did not have.
        E::Literal(_) | E::Constant(_) | E::Override(_) | E::ZeroValue(_) => Some(true),

        E::Compose { .. }
        | E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. } => None,

        E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. }
        | E::Load { .. }
        | E::Derivative { .. }
        | E::ArrayLength(_)
        | E::ImageSample { .. }
        | E::ImageLoad { .. }
        | E::ImageQuery { .. }
        | E::RayQueryVertexPositions { .. }
        | E::RayQueryGetIntersection { .. }
        | E::CooperativeLoad { .. }
        | E::CooperativeMultiplyAdd { .. } => Some(false),
    }
}

/// `true` when a `ZeroValue` of `ty` is a FLOAT zero - the only kind a
/// `-x` / `x * y` / `x / y` can re-sign.  An integer `vec4i()` must answer
/// false or the untyped sign-sensitive roots stop being inert and the guard
/// declines integer slots that have no signed zero at all.  Anything that is
/// not a numeric scalar / vector / matrix cannot be such an operand, so the
/// conservative answer costs nothing.
pub(crate) fn zero_value_is_float(
    types: &naga::UniqueArena<naga::Type>,
    ty: naga::Handle<naga::Type>,
) -> bool {
    match types[ty].inner {
        naga::TypeInner::Scalar(scalar) | naga::TypeInner::Vector { scalar, .. } => matches!(
            scalar.kind,
            naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat
        ),
        naga::TypeInner::Matrix { .. } => true,
        _ => true,
    }
}

/// A float zero of either sign: what an enclosing `-x` / `x * y` / `x / y`
/// is free to re-sign, whatever this literal's own sign is.
pub(crate) fn is_float_zero_literal(lit: &naga::Literal) -> bool {
    match *lit {
        naga::Literal::F32(v) => v == 0.0,
        naga::Literal::F64(v) | naga::Literal::AbstractFloat(v) => v == 0.0,
        naga::Literal::F16(v) => v.to_f32() == 0.0,
        _ => false,
    }
}

/// Any float literal, of any width.
pub(crate) fn is_float_literal(lit: &naga::Literal) -> bool {
    matches!(
        *lit,
        naga::Literal::F32(_)
            | naga::Literal::F64(_)
            | naga::Literal::AbstractFloat(_)
            | naga::Literal::F16(_)
    )
}

/// A literal `-0.0` of any float width.  WGSL leaves a zero's sign to the
/// implementation, and Dawn on Metal flushes a LITERAL `-0.0` to `+0.0` while
/// keeping the sign of a runtime negation: the asymmetry every `-0.0` guard
/// in the passes exists to respect.
pub(crate) fn is_negative_zero_literal(lit: &naga::Literal) -> bool {
    match *lit {
        naga::Literal::F32(v) => v == 0.0 && v.is_sign_negative(),
        naga::Literal::F64(v) | naga::Literal::AbstractFloat(v) => v == 0.0 && v.is_sign_negative(),
        naga::Literal::F16(v) => v.to_bits() == 0x8000,
        _ => false,
    }
}

/// `true` when `h` may carry a `-0.0` only a const-evaluator sees: a literal,
/// a splat / compose of one, or a module-scope value this arena cannot
/// resolve.  Substituting one for a runtime read (forwarding, inlining) flips
/// the sign the input shipped.  `Constant` / `Override` resolve through
/// `module.global_expressions`, which the mutating passes do not hold, and an
/// override has no fixed value at all - both decline.
pub fn has_negative_zero_leaf(
    arena: &naga::Arena<naga::Expression>,
    h: naga::Handle<naga::Expression>,
) -> bool {
    match &arena[h] {
        naga::Expression::Literal(lit) => is_negative_zero_literal(lit),
        naga::Expression::Constant(_) | naga::Expression::Override(_) => true,
        naga::Expression::Splat { value, .. } => has_negative_zero_leaf(arena, *value),
        naga::Expression::Compose { components, .. } => {
            components.iter().any(|&c| has_negative_zero_leaf(arena, c))
        }
        _ => false,
    }
}

/// Bitwise literal equality: `-0.0 != 0.0` and NaN equals itself, so a
/// fold that treats two literals as the same value never merges IEEE
/// values a shader can tell apart.
pub fn literal_bit_eq(a: &naga::Literal, b: &naga::Literal) -> bool {
    use naga::Literal as L;
    match (a, b) {
        (L::F64(x), L::F64(y)) | (L::AbstractFloat(x), L::AbstractFloat(y)) => {
            x.to_bits() == y.to_bits()
        }
        (L::F32(x), L::F32(y)) => x.to_bits() == y.to_bits(),
        (L::F16(x), L::F16(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Per-handle reference counts plus the liveness bitmap, in one walk.  Live
/// is "materialised by an `Emit` range", so dead code's children score zero
/// and never claim a short name.  A statement RESULT is not a use - it stays
/// at 0 until something consumes it, which single-use call inlining and
/// dead-`let` elision key on - nor is an `Emit` handle, emission being
/// sequencing.
///
/// The generator prices its output by these counts and `rename` spends short
/// names by them; the two MUST agree, or a name the generator inlines away
/// takes the shortest identifier with it.
pub fn live_expression_ref_counts(function: &naga::Function) -> (Vec<usize>, Vec<bool>) {
    let len = function.expressions.len();

    let mut live = vec![false; len];
    for_each_statement(&function.body, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                live[h.index()] = true;
            }
        }
    });

    let mut counts = vec![0usize; len];
    for (h, expr) in function.expressions.iter() {
        if live[h.index()] {
            visit_expression_children(expr, |child| counts[child.index()] += 1);
        }
    }
    for_each_statement(&function.body, &mut |stmt| {
        visit_statement_operands(
            stmt,
            /*include_emit_handles=*/ false,
            &mut |h| counts[h.index()] += 1,
        );
    });

    (counts, live)
}

/// A literal that turns a legal RUNTIME `/` `%` into a WGSL shader-creation
/// error once it lands in the divisor.  Forwarding and folding both decline
/// on it: either would build IR naga rejects, costing the driver's rollback
/// a whole pass run.  Only INTEGER division qualifies - float `x / 0.0` is a
/// defined `inf` / `nan`.
pub(crate) fn is_integer_zero_literal(lit: &naga::Literal) -> bool {
    matches!(
        lit,
        naga::Literal::I16(0)
            | naga::Literal::U16(0)
            | naga::Literal::I32(0)
            | naga::Literal::U32(0)
            | naga::Literal::I64(0)
            | naga::Literal::U64(0)
            | naga::Literal::AbstractInt(0)
    )
}

/// [`is_integer_zero_literal`]'s counterpart: a shift by `>= 32` overruns the
/// ubiquitous 32-bit operand.  The operand width is out of reach here, so a
/// 16-bit one shifted by `[16, 32)` slips through to the whole-module
/// rollback and a 64-bit one is over-declined; declining only ever costs a
/// missed optimization.
pub(crate) fn shift_amount_is_static_error(lit: &naga::Literal) -> bool {
    let amount: i128 = match lit {
        naga::Literal::U32(v) => i128::from(*v),
        naga::Literal::U16(v) => i128::from(*v),
        naga::Literal::I32(v) => i128::from(*v),
        naga::Literal::I16(v) => i128::from(*v),
        naga::Literal::U64(v) => i128::from(*v),
        naga::Literal::I64(v) => i128::from(*v),
        naga::Literal::AbstractInt(v) => i128::from(*v),
        _ => return false,
    };
    amount >= 32
}

/// Root local of a pointer expression: the `LocalVariable` an
/// `Access` / `AccessIndex` chain bottoms out in, `None` for any other root
/// (a global, a function-argument pointer).  The one definition: liveness,
/// dead-store elimination and the generator's deferred-variable analysis
/// must agree on which local a pointer names, or one drops a store the
/// other still reads.
pub fn root_local_var(
    pointer: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match &expressions[pointer] {
        naga::Expression::LocalVariable(local) => Some(*local),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            root_local_var(*base, expressions)
        }
        _ => None,
    }
}

/// The value of an integer `Literal` index; a `u64` past `i64::MAX`
/// saturates, which is out of bounds for any composite.
pub fn const_index_value(
    handle: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<i64> {
    match arena[handle] {
        naga::Expression::Literal(naga::Literal::I32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::U32(v)) => Some(v as i64),
        naga::Expression::Literal(naga::Literal::I64(v)) => Some(v),
        naga::Expression::Literal(naga::Literal::U64(v)) => {
            Some(i64::try_from(v).unwrap_or(i64::MAX))
        }
        naga::Expression::Literal(naga::Literal::AbstractInt(v)) => Some(v),
        _ => None,
    }
}

// MARK: Replacement chain flattening

/// Collapse transitive chains in a replacement map so every key points at
/// its terminal target.  Passes accumulate `A -> B` entries and can produce
/// `A -> B -> C` (CSE picks `B` canonical before `C` supplants it;
/// load_dedup forwards a load to a load that was itself forwarded), and one
/// [`try_map_expression_handles_in_place`] walk resolves a single level, so
/// callers MUST flatten before applying the map or dangling references
/// survive.
///
/// Every caller builds an acyclic map; as defence in depth the walk is
/// bounded by the map size (an acyclic chain over `N` entries has at most
/// `N` hops) and a longer one debug-asserts and stops at that entry, so a
/// debug build fails loudly and a release build degrades to single-level
/// resolution instead of hanging.
pub(crate) fn flatten_replacement_chains(
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let keys: Vec<_> = replacements.keys().copied().collect();
    let max_hops = replacements.len();
    for key in keys {
        let mut target = replacements[&key];
        let mut hops = 0usize;
        while let Some(&next) = replacements.get(target) {
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

// MARK: Arena rebuild

/// Rebuild `function.expressions` in emission order: only expressions
/// reachable from the body survive, each appended after its operands and
/// after everything emitted before it, with the body, the locals' inits,
/// and the named expressions remapped (names of dropped expressions go
/// too).  Callers rely on the ORDER as much as on the garbage collection: a
/// pass that appends a synthesized expression puts it at the END of the
/// arena, behind consumers that already exist, and a later pass forwarding
/// it into one of them would create a forward reference naga rejects.
pub fn rebuild_function_expressions(function: &mut naga::Function) {
    let old_expressions = std::mem::take(&mut function.expressions);
    let mut new_expressions = naga::Arena::new();
    let mut handle_map = HandleMap::default();

    // A declarative init (`var c = BG;`) sits in no `Emit` range, so the
    // body walk never reaches it; cloned after the body it would land
    // BEHIND every consumer of the local, and a later store-to-load forward
    // of the init value would read as a forward reference and be declined.
    // An emitted init (`var x = OV * 3.0;`) is cloned by the body walk at
    // its own `Emit`, ahead of the local's loads; cloning it here would
    // leave the init on an un-emitted duplicate, which a forward then hands
    // to a statement and naga rejects as out of scope.
    let mut emitted_inits = Vec::new();
    for (lh, local) in function.local_variables.iter_mut() {
        if let Some(init) = &mut local.init {
            if expression_needs_emit(&old_expressions[*init]) {
                emitted_inits.push(lh);
            } else {
                *init = clone_expression_handle(
                    *init,
                    &old_expressions,
                    &mut new_expressions,
                    &mut handle_map,
                );
            }
        }
    }

    rebuild_block_expressions(
        &mut function.body,
        &old_expressions,
        &mut new_expressions,
        &mut handle_map,
    );

    for lh in emitted_inits {
        if let Some(init) = &mut function.local_variables[lh].init {
            *init = clone_expression_handle(
                *init,
                &old_expressions,
                &mut new_expressions,
                &mut handle_map,
            );
        }
    }

    let named = std::mem::take(&mut function.named_expressions);
    function.named_expressions = named
        .into_iter()
        .filter_map(|(h, name)| handle_map.get(h).map(|&m| (m, name)))
        .collect();

    function.expressions = new_expressions;
}

fn rebuild_block_expressions(
    block: &mut naga::Block,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        // Cloning a child may append a non-emittable dependency between two
        // emitted handles, so the range is split around it.
        if let naga::Statement::Emit(ref range) = statement {
            let mut mapped_handles = Vec::new();
            for handle in range.clone() {
                // A clone from an earlier walk would leave this copy dead
                // and the map on the wrong one.
                debug_assert!(!handle_map.contains_key(handle));
                let mut expression = old_expressions[handle].clone();
                let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
                    Some(clone_expression_handle(
                        child,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    ))
                });
                let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
                handle_map.insert(handle, mapped);
                mapped_handles.push(mapped);
            }

            push_emit_runs(&mut rebuilt, &mapped_handles, span);
            continue;
        }

        // A loop's `break_if` is emitted inside `body` / `continuing`, so
        // those are rebuilt first and the `break_if` clone memo-hits the copy
        // its owning Emit produced; the generic remap would clone it first,
        // appending an un-emitted duplicate that sits in no Emit range.
        if matches!(statement, naga::Statement::Loop { .. }) {
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = &mut statement
            {
                rebuild_block_expressions(body, old_expressions, new_expressions, handle_map);
                rebuild_block_expressions(continuing, old_expressions, new_expressions, handle_map);
                if let Some(handle) = break_if {
                    *handle = clone_expression_handle(
                        *handle,
                        old_expressions,
                        new_expressions,
                        handle_map,
                    );
                }
            }
            rebuilt.push(statement, span);
            continue;
        }

        remap_statement_handles(&mut statement, &mut |h| {
            clone_expression_handle(h, old_expressions, new_expressions, handle_map)
        });

        debug_assert!(!matches!(statement, naga::Statement::Loop { .. }));
        for nested in nested_blocks_mut(&mut statement) {
            rebuild_block_expressions(nested, old_expressions, new_expressions, handle_map);
        }

        rebuilt.push(statement, span);
    }

    *block = rebuilt;
}

fn clone_expression_handle(
    handle: naga::Handle<naga::Expression>,
    old_expressions: &naga::Arena<naga::Expression>,
    new_expressions: &mut naga::Arena<naga::Expression>,
    handle_map: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) -> naga::Handle<naga::Expression> {
    if let Some(mapped) = handle_map.get(handle).copied() {
        return mapped;
    }

    let mut expression = old_expressions[handle].clone();
    let _ = try_map_expression_handles_in_place(&mut expression, &mut |child| {
        Some(clone_expression_handle(
            child,
            old_expressions,
            new_expressions,
            handle_map,
        ))
    });

    let mapped = new_expressions.append(expression, old_expressions.get_span(handle));
    handle_map.insert(handle, mapped);
    mapped
}

// MARK: naga variant tripwire

/// Compile-time tripwire: none of naga's IR enums are `#[non_exhaustive]`,
/// so a wildcard-free match makes any new variant a build error HERE instead
/// of a construct that existing walkers' `_ => {}` arms swallow and liveness
/// / effect analysis never sees (a miscompile).  When it breaks on a naga
/// upgrade: add the variant here, audit every walker with a wildcard arm
/// (`_ =>` near `Statement` / `Expression` matches in `src/passes` and
/// `src/generator`, the shared walkers in this file first) for handles,
/// effects, or nested blocks it must handle, and extend the generator if the
/// variant reaches emission.  Never called.
#[allow(dead_code)]
fn naga_variant_tripwire(
    statement: &naga::Statement,
    expression: &naga::Expression,
    type_inner: &naga::TypeInner,
    literal: &naga::Literal,
) {
    use naga::{Expression as E, Literal as L, Statement as S, TypeInner as T};
    match statement {
        S::Emit(_) | S::Block(_) | S::If { .. } | S::Switch { .. } | S::Loop { .. } => {}
        S::Break | S::Continue | S::Return { .. } | S::Kill => {}
        S::ControlBarrier(_) | S::MemoryBarrier(_) => {}
        S::Store { .. } | S::ImageStore { .. } | S::Atomic { .. } | S::ImageAtomic { .. } => {}
        S::WorkGroupUniformLoad { .. } | S::Call { .. } => {}
        S::RayQuery { .. } | S::RayPipelineFunction { .. } => {}
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => {}
        S::CooperativeStore { .. } => {}
    }
    match expression {
        E::Literal(_) | E::Constant(_) | E::Override(_) | E::ZeroValue(_) => {}
        E::Compose { .. } | E::Access { .. } | E::AccessIndex { .. } => {}
        E::Splat { .. } | E::Swizzle { .. } => {}
        E::FunctionArgument(_) | E::GlobalVariable(_) | E::LocalVariable(_) | E::Load { .. } => {}
        E::ImageSample { .. } | E::ImageLoad { .. } | E::ImageQuery { .. } => {}
        E::Unary { .. } | E::Binary { .. } | E::Select { .. } => {}
        E::Derivative { .. } | E::Relational { .. } | E::Math { .. } | E::As { .. } => {}
        E::CallResult(_) | E::AtomicResult { .. } | E::WorkGroupUniformLoadResult { .. } => {}
        E::ArrayLength(_) => {}
        E::RayQueryVertexPositions { .. }
        | E::RayQueryProceedResult
        | E::RayQueryGetIntersection { .. } => {}
        E::SubgroupBallotResult | E::SubgroupOperationResult { .. } => {}
        E::CooperativeLoad { .. } | E::CooperativeMultiplyAdd { .. } => {}
    }
    match type_inner {
        T::Scalar(_) | T::Vector { .. } | T::Matrix { .. } | T::CooperativeMatrix { .. } => {}
        T::Atomic(_) | T::Pointer { .. } | T::ValuePointer { .. } => {}
        T::Array { .. } | T::Struct { .. } => {}
        T::Image { .. } | T::Sampler { .. } => {}
        T::AccelerationStructure { .. } | T::RayQuery { .. } => {}
        T::BindingArray { .. } => {}
    }
    match literal {
        L::F64(_) | L::F32(_) | L::F16(_) => {}
        L::U16(_) | L::I16(_) | L::U32(_) | L::I32(_) | L::U64(_) | L::I64(_) => {}
        L::Bool(_) => {}
        L::AbstractInt(_) | L::AbstractFloat(_) => {}
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// A real handle for variants whose fields the classifiers never inspect.
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

    /// `n` expression handles, so replacement maps in tests use real keys.
    fn expr_handles(n: usize) -> Vec<naga::Handle<naga::Expression>> {
        let mut arena = naga::Arena::new();
        (0..n)
            .map(|i| {
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(i as u32)),
                    naga::Span::UNDEFINED,
                )
            })
            .collect()
    }

    #[test]
    fn flatten_replacement_chains_collapses_transitive_edges() {
        let h = expr_handles(7);
        let mut m = HandleMap::default();
        m.insert(h[1], h[2]);
        m.insert(h[2], h[3]);
        m.insert(h[3], h[4]);
        m.insert(h[5], h[6]);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[h[1]], h[4]);
        assert_eq!(m[h[2]], h[4]);
        assert_eq!(m[h[3]], h[4]);
        assert_eq!(m[h[5]], h[6]);
    }

    #[test]
    fn flatten_replacement_chains_is_noop_on_direct_edges() {
        let h = expr_handles(4);
        let mut m = HandleMap::default();
        m.insert(h[0], h[2]);
        m.insert(h[1], h[3]);
        flatten_replacement_chains(&mut m);
        assert_eq!(m[h[0]], h[2]);
        assert_eq!(m[h[1]], h[3]);
    }

    /// Every caller builds an acyclic map; a cyclic one must still
    /// terminate (debug builds may assert, release builds return).
    #[test]
    fn flatten_replacement_chains_terminates_on_cycles() {
        for cycle in [2usize, 3] {
            let h = expr_handles(cycle);
            let mut m = HandleMap::default();
            for i in 0..cycle {
                m.insert(h[i], h[(i + 1) % cycle]);
            }
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                flatten_replacement_chains(&mut m);
            }));
        }
    }

    /// A block holding `n` `Break` statements, so nested-block iteration
    /// order is observable through `len()` alone.
    fn block_of_breaks(n: usize) -> naga::Block {
        naga::Block::from_vec(vec![naga::Statement::Break; n])
    }

    #[test]
    fn nested_blocks_yields_if_arms_in_syntactic_order() {
        let (_, h) = dummy_handles();
        let stmt = naga::Statement::If {
            condition: h,
            accept: block_of_breaks(1),
            reject: block_of_breaks(2),
        };
        let lens: Vec<usize> = nested_blocks(&stmt).map(|b| b.len()).collect();
        assert_eq!(lens, [1, 2]);
    }

    #[test]
    fn nested_blocks_yields_loop_body_before_continuing() {
        let stmt = naga::Statement::Loop {
            body: block_of_breaks(3),
            continuing: block_of_breaks(1),
            break_if: None,
        };
        let lens: Vec<usize> = nested_blocks(&stmt).map(|b| b.len()).collect();
        assert_eq!(lens, [3, 1]);
    }

    #[test]
    fn nested_blocks_yields_switch_cases_in_declaration_order() {
        let (_, h) = dummy_handles();
        let case = |n: usize, value| naga::SwitchCase {
            value,
            body: block_of_breaks(n),
            fall_through: false,
        };
        let stmt = naga::Statement::Switch {
            selector: h,
            cases: vec![
                case(2, naga::SwitchValue::I32(0)),
                case(0, naga::SwitchValue::I32(1)),
                case(1, naga::SwitchValue::Default),
            ],
        };
        let lens: Vec<usize> = nested_blocks(&stmt).map(|b| b.len()).collect();
        assert_eq!(lens, [2, 0, 1]);
    }

    #[test]
    fn nested_blocks_treats_leaf_statements_as_empty() {
        let (_, h) = dummy_handles();
        assert_eq!(nested_blocks(&naga::Statement::Break).count(), 0);
        assert_eq!(nested_blocks(&naga::Statement::Kill).count(), 0);
        assert_eq!(
            nested_blocks(&naga::Statement::Store {
                pointer: h,
                value: h
            })
            .count(),
            0
        );
        assert_eq!(
            nested_blocks(&naga::Statement::Return { value: None }).count(),
            0
        );
    }

    #[test]
    fn nested_blocks_mut_reaches_every_block() {
        let (_, h) = dummy_handles();
        let mut stmt = naga::Statement::If {
            condition: h,
            accept: block_of_breaks(1),
            reject: naga::Block::new(),
        };
        for block in nested_blocks_mut(&mut stmt) {
            block.push(naga::Statement::Continue, naga::Span::UNDEFINED);
        }
        let lens: Vec<usize> = nested_blocks(&stmt).map(|b| b.len()).collect();
        assert_eq!(lens, [2, 1]);
    }

    #[test]
    fn rebuild_keeps_one_copy_of_an_emitted_initializer() {
        // `x`'s init is emitted by the body, `c`'s is a declarative reference
        // no `Emit` covers: `x` must keep the emitted copy (an un-emitted
        // second one is what a later init forward hands to statements naga
        // rejects) and `c`'s must precede the local's loads so that forward
        // is not declined.
        let src = r#"
const BG: f32 = 2.0;
override OV: f32 = 3.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() {
    var x = OV * 3.0;
    var c = BG;
    out[0] = x + c;
}
"#;
        let mut module = naga::front::wgsl::parse_str(src).expect("source should parse");
        rebuild_function_expressions(&mut module.entry_points[0].function);
        let function = &module.entry_points[0].function;
        let products = function
            .expressions
            .iter()
            .filter(|(_, e)| {
                matches!(
                    e,
                    naga::Expression::Binary {
                        op: naga::BinaryOperator::Multiply,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(products, 1, "an emitted init is cloned once");
        let init_of = |name: &str| {
            function
                .local_variables
                .iter()
                .find(|(_, l)| l.name.as_deref() == Some(name))
                .and_then(|(_, l)| l.init)
                .expect("local keeps its init")
        };
        let x_init = init_of("x");
        let emitted = function.body.iter().any(
            |s| matches!(s, naga::Statement::Emit(range) if range.clone().any(|h| h == x_init)),
        );
        assert!(emitted, "the init is the copy an `Emit` covers");
        let first_load = function
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, naga::Expression::Load { .. }))
            .map(|(h, _)| h)
            .expect("the body loads its locals");
        assert!(
            init_of("c").index() < first_load.index(),
            "a declarative init precedes the loads"
        );
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module stays valid after the rebuild");
    }
}
