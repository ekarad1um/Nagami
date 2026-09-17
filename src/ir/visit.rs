//! Read-only and in-place walks over naga's IR: the children of an
//! expression, the handles a statement reads or writes through, the blocks a
//! statement nests, and the functions a module holds.  Every match here is
//! exhaustive with no `_` arm, so a new naga variant fails the build at this
//! single point of truth instead of drifting silently through consumers'
//! private deny-lists.
//!
//! A pass that only needs "every statement", "every handle" or "every
//! statement, edited in place" reads those off [`walk_block`] and its
//! adapters; a walk stays hand-written where its order or its state is the
//! analysis itself: the dead-branch eliminator (a `Break` binds to the loop
//! or the switch case it sits in), the tail-return rewrites (they split a
//! block), the per-arm dataflow walks (`must_bind::LoadAnalysis`,
//! `find_inlineable_calls_in_block`, `collect_redundant_loads`,
//! `remove_dead_stores_in_block`, coalescing's `scan_block_usage`) and the
//! tint-behaviour classifiers, which all carry a state that changes at
//! block entry and exit in a way no hook order reproduces.

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
/// worker into the pass body.  Takes the arenas rather than the module so a
/// caller can hold `&module.types` across the walk.
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

/// [`for_each_function_mut`] with the module readable beside the function:
/// each body is taken out of the module while `visit` rewrites it, so a
/// pass can size its tables against the module at the point of use (the
/// typifier resolves a call's result through `functions`, and no pass here
/// changes a signature, so the taken-out body's own slot - a default
/// function - is never the one asked).
pub(crate) fn for_each_function_taken(
    module: &mut naga::Module,
    visit: &mut dyn FnMut(usize, &mut naga::Function, &naga::Module),
) {
    let handles: Vec<_> = module.functions.iter().map(|(h, _)| h).collect();
    for (body, h) in handles.into_iter().enumerate() {
        let mut function = std::mem::take(&mut module.functions[h]);
        visit(body, &mut function, module);
        module.functions[h] = function;
    }
    for i in 0..module.entry_points.len() {
        let mut function = std::mem::take(&mut module.entry_points[i].function);
        visit(module.functions.len() + i, &mut function, module);
        module.entry_points[i].function = function;
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
/// closure, and `inline(never)` keeps LLVM from copying it into its callers
/// anyway: with the attribute off, both targets held a copy at most of the
/// forty-odd call sites.
#[inline(never)]
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
/// (callers abandon it).  One copy, as [`visit_expression_children_dyn`].
#[inline(never)]
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
fn map_atomic_function_handles(
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
fn map_gather_mode_handles(
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
fn map_ray_query_function_handles(
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
fn map_ray_pipeline_function_handles(
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
fn map_cooperative_data_handles(
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
/// lockstep with [`walk_statement_fields`].
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

/// What a handle is to the statement that holds it.  A pointer a statement
/// may write through is reported twice, as an `Operand` and then as a
/// `WritePointer`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot {
    /// Covered by an `Emit` range: sequencing, not a use.
    Emitted,
    /// Read by the statement.
    Operand,
    /// Defined by the statement (`CallResult`, `AtomicResult`, ...): only
    /// a whole-function reference count wants these.
    Result,
    /// A pointer the statement may WRITE through: `Store` / `Atomic`
    /// pointers, every `Call` argument (a `ptr<function>` parameter lets
    /// the callee write the pointee), `traceRay`'s payload, a cooperative
    /// store's destination, and the ray-query object.
    WritePointer,
}

/// The hooks of a read-only walk over a block ([`walk_block`]).  Default
/// hooks do nothing, so a consumer implements the ones it reads; `&mut dyn`
/// keeps the walk one copy.
pub(crate) trait Visitor {
    /// Each statement, before its fields and its nested blocks; `false`
    /// skips both.
    fn stmt(&mut self, _statement: &naga::Statement, _scope: Scope) -> bool {
        true
    }
    /// Each handle in the statement's own fields, in naga's field order.
    fn handle(&mut self, _handle: naga::Handle<naga::Expression>, _slot: Slot) {}
    /// A block, before its statements.
    fn enter_block(&mut self, _block: &naga::Block, _scope: Scope) {}
    /// A block, after its statements.
    fn exit_block(&mut self, _block: &naga::Block, _scope: Scope) {}
}

/// Pre-order, syntactic-order walk of `block` and its nested blocks:
/// per statement, `stmt`, then its fields, then its nested blocks
/// ([`nested_blocks`]' order) one loop deeper under a `Loop`.
pub(crate) fn walk_block(block: &naga::Block, scope: Scope, visitor: &mut dyn Visitor) {
    visitor.enter_block(block, scope);
    for statement in block.iter() {
        if !visitor.stmt(statement, scope) {
            continue;
        }
        walk_statement_fields(statement, &mut |h, slot| visitor.handle(h, slot));
        let inner = scope.inner(statement);
        for nested in nested_blocks(statement) {
            walk_block(nested, inner, visitor);
        }
    }
    visitor.exit_block(block, scope);
}

/// The handles in `statement`'s own fields, in naga's field order, each
/// with its [`Slot`]; nested blocks are the caller's.  THE exhaustive
/// read-only statement match: kept in lockstep with
/// [`remap_statement_handles`].  A `dyn` callback, one copy.
fn walk_statement_fields(
    statement: &naga::Statement,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>, Slot),
) {
    let mut v = |h, slot| visit(h, slot);
    match statement {
        naga::Statement::Emit(range) => {
            for h in range.clone() {
                v(h, Slot::Emitted);
            }
        }
        naga::Statement::If { condition, .. } => v(*condition, Slot::Operand),
        naga::Statement::Switch { selector, .. } => v(*selector, Slot::Operand),
        naga::Statement::Loop { break_if, .. } => {
            if let Some(handle) = break_if {
                v(*handle, Slot::Operand);
            }
        }
        naga::Statement::Return { value } => {
            if let Some(handle) = value {
                v(*handle, Slot::Operand);
            }
        }
        naga::Statement::Store { pointer, value } => {
            v(*pointer, Slot::Operand);
            v(*pointer, Slot::WritePointer);
            v(*value, Slot::Operand);
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            v(*image, Slot::Operand);
            v(*coordinate, Slot::Operand);
            if let Some(index) = array_index {
                v(*index, Slot::Operand);
            }
            v(*value, Slot::Operand);
        }
        naga::Statement::Atomic {
            pointer,
            fun,
            value,
            result,
        } => {
            v(*pointer, Slot::Operand);
            v(*pointer, Slot::WritePointer);
            visit_atomic_function_handles(fun, &mut |h| v(h, Slot::Operand));
            v(*value, Slot::Operand);
            if let Some(handle) = result {
                v(*handle, Slot::Result);
            }
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            fun,
            value,
        } => {
            v(*image, Slot::Operand);
            v(*coordinate, Slot::Operand);
            if let Some(index) = array_index {
                v(*index, Slot::Operand);
            }
            visit_atomic_function_handles(fun, &mut |h| v(h, Slot::Operand));
            v(*value, Slot::Operand);
        }
        naga::Statement::WorkGroupUniformLoad { pointer, result } => {
            v(*pointer, Slot::Operand);
            v(*result, Slot::Result);
        }
        naga::Statement::Call {
            arguments, result, ..
        } => {
            for &argument in arguments {
                v(argument, Slot::Operand);
                v(argument, Slot::WritePointer);
            }
            if let Some(handle) = result {
                v(*handle, Slot::Result);
            }
        }
        naga::Statement::RayQuery { query, fun } => {
            v(*query, Slot::Operand);
            v(*query, Slot::WritePointer);
            match fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => {
                    v(*acceleration_structure, Slot::Operand);
                    v(*descriptor, Slot::Operand);
                }
                naga::RayQueryFunction::Proceed { result } => v(*result, Slot::Result),
                naga::RayQueryFunction::GenerateIntersection { hit_t } => v(*hit_t, Slot::Operand),
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
            v(*acceleration_structure, Slot::Operand);
            v(*descriptor, Slot::Operand);
            v(*payload, Slot::Operand);
            v(*payload, Slot::WritePointer);
        }
        naga::Statement::SubgroupBallot { result, predicate } => {
            v(*result, Slot::Result);
            if let Some(handle) = predicate {
                v(*handle, Slot::Operand);
            }
        }
        naga::Statement::SubgroupGather {
            mode,
            argument,
            result,
        } => {
            v(*argument, Slot::Operand);
            v(*result, Slot::Result);
            match mode {
                naga::GatherMode::Broadcast(handle)
                | naga::GatherMode::Shuffle(handle)
                | naga::GatherMode::ShuffleDown(handle)
                | naga::GatherMode::ShuffleUp(handle)
                | naga::GatherMode::ShuffleXor(handle)
                | naga::GatherMode::QuadBroadcast(handle) => v(*handle, Slot::Operand),
                naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
            }
        }
        naga::Statement::SubgroupCollectiveOperation {
            argument, result, ..
        } => {
            v(*argument, Slot::Operand);
            v(*result, Slot::Result);
        }
        naga::Statement::CooperativeStore { target, data } => {
            v(*target, Slot::Operand);
            v(data.pointer, Slot::Operand);
            v(data.pointer, Slot::WritePointer);
            v(data.stride, Slot::Operand);
        }
        naga::Statement::Block(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_) => {}
    }
}

/// The [`Slot`]s a handle callback reads: `Emitted` when `emits` (right for
/// liveness - an emitted expression is a reachable let-bound name; wrong for
/// reference counting, where counting the `Emit` gives every emitted
/// expression a count >= 1 and defeats unique-owner gates), `Result` when
/// `results`, `Operand` always, `WritePointer` never (the same handle was
/// just reported as an operand).
fn reads(slot: Slot, emits: bool, results: bool) -> bool {
    match slot {
        Slot::Operand => true,
        Slot::Emitted => emits,
        Slot::Result => results,
        Slot::WritePointer => false,
    }
}

/// Every expression handle `block` references, nested blocks included:
/// operands and results, plus each `Emit` range handle when
/// `include_emit_handles` (semantics on [`reads`]).  Direct recursion with
/// [`visit_statement_expression_handles`] rather than a [`walk_block`]
/// adapter: it is asked per statement on the generator's hot path, where
/// the adapter's second indirection per handle shows in the profile.
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
    walk_statement_fields(stmt, &mut |h, slot| {
        if reads(slot, include_emit_handles, true) {
            visit(h);
        }
    });
    for nested in nested_blocks(stmt) {
        visit_block_expression_handles(nested, include_emit_handles, visit);
    }
}

/// The handles `stmt` reads directly: operands, not the results it defines
/// or its nested blocks; `include_emit_handles` also visits each `Emit`
/// range handle (semantics on [`reads`]).
pub fn visit_statement_operands(
    stmt: &naga::Statement,
    include_emit_handles: bool,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    walk_statement_fields(stmt, &mut |h, slot| {
        if reads(slot, include_emit_handles, false) {
            visit(h);
        }
    });
}

/// Pointer operands `stmt` may WRITE through ([`Slot::WritePointer`]).
pub fn visit_statement_write_pointers(
    stmt: &naga::Statement,
    visit: &mut dyn FnMut(naga::Handle<naga::Expression>),
) {
    walk_statement_fields(stmt, &mut |h, slot| {
        if slot == Slot::WritePointer {
            visit(h);
        }
    });
}

/// A `Return` at ANY depth of `statements`, nested loops and switches
/// included: it exits the function, hence every enclosing loop.  `Kill` is not
/// one: `discard` demotes the invocation and falls through.
pub(crate) fn contains_return(statements: &[naga::Statement]) -> bool {
    statements.iter().any(|stmt| {
        matches!(stmt, naga::Statement::Return { .. })
            || nested_blocks(stmt).any(|block| contains_return(block))
    })
}

/// Mutable [`for_each_statement`]: `f` edits each statement in place before
/// the walk descends into its nested blocks, so a statement `f` replaces is
/// descended into as replaced.
pub(crate) fn for_each_statement_mut(
    block: &mut naga::Block,
    f: &mut dyn FnMut(&mut naga::Statement),
) {
    for statement in block.iter_mut() {
        f(statement);
        for nested in nested_blocks_mut(statement) {
            for_each_statement_mut(nested, f);
        }
    }
}

/// [`remap_statement_handles`] over `block` and every block nested in it.
pub(crate) fn remap_block_handles(
    block: &mut naga::Block,
    remap: &mut dyn FnMut(naga::Handle<naga::Expression>) -> naga::Handle<naga::Expression>,
) {
    for_each_statement_mut(block, &mut |statement| {
        remap_statement_handles(statement, remap)
    });
}

/// Pre-order, syntactic-order walk of `block` and its nested blocks: the
/// [`walk_block`] of a [`Visitor`] with only `stmt`, spelled out because it
/// is the hottest walk (every pass asks it every sweep) and the field walk
/// its hooks would ignore shows in the profile.
pub fn for_each_statement(block: &naga::Block, f: &mut dyn FnMut(&naga::Statement)) {
    for stmt in block.iter() {
        f(stmt);
        for nested in nested_blocks(stmt) {
            for_each_statement(nested, f);
        }
    }
}

// MARK: Scope

/// Where in a function body a walk stands, as far as the passes and the
/// generator ask: how many `Loop`s enclose the statement.  A `for` body, its
/// `continuing` and its `break_if` are all inside the loop; a switch case or
/// an `if` arm adds nothing.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Scope {
    pub(crate) loop_depth: u16,
}

impl Scope {
    /// The scope of the blocks nested directly inside `statement`.
    pub(crate) fn inner(self, statement: &naga::Statement) -> Scope {
        Scope {
            loop_depth: self.loop_depth
                + u16::from(matches!(statement, naga::Statement::Loop { .. })),
        }
    }

    /// Inside at least one loop: the statement may run more than once per
    /// invocation of the function.
    pub(crate) fn in_loop(self) -> bool {
        self.loop_depth > 0
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

// MARK: naga variant tripwire

/// Compile-time tripwire: none of naga's IR enums are `#[non_exhaustive]`,
/// so a wildcard-free match makes any new variant a build error HERE instead
/// of a construct that existing walkers' `_ => {}` arms swallow and liveness
/// / effect analysis never sees (a miscompile).  `Expression` is covered by
/// [`crate::analysis::ExprClass::node`], which every classifier reads.  When
/// it breaks on a naga upgrade: add the variant here (or its class there),
/// audit every wildcard arm on `Statement` / `Expression` across the passes
/// and the generator (the shared walkers in this file first) for handles,
/// effects, or nested blocks it must handle, and extend the generator if the
/// variant reaches emission.  Never called.
#[allow(dead_code)]
fn naga_variant_tripwire(
    statement: &naga::Statement,
    type_inner: &naga::TypeInner,
    literal: &naga::Literal,
) {
    use naga::{Literal as L, Statement as S, TypeInner as T};
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

    /// A block holding `n` `Break` statements, so nested-block iteration
    /// order is observable through `len()` alone.
    fn block_of_breaks(n: usize) -> naga::Block {
        naga::Block::from_vec(vec![naga::Statement::Break; n])
    }

    /// The walk's trace, one token per hook call.
    #[derive(Default)]
    struct Trace(Vec<String>);
    impl Visitor for Trace {
        fn stmt(&mut self, statement: &naga::Statement, scope: Scope) -> bool {
            let name = match statement {
                naga::Statement::Emit(_) => "emit",
                naga::Statement::If { .. } => "if",
                naga::Statement::Loop { .. } => "loop",
                naga::Statement::Store { .. } => "store",
                naga::Statement::Call { .. } => "call",
                naga::Statement::Break => "break",
                naga::Statement::Continue => "continue",
                _ => "other",
            };
            self.0.push(format!("{name}@{}", scope.loop_depth));
            name != "if" || scope.loop_depth == 0
        }
        fn handle(&mut self, h: naga::Handle<naga::Expression>, slot: Slot) {
            let tag = match slot {
                Slot::Emitted => 'e',
                Slot::Operand => 'o',
                Slot::Result => 'r',
                Slot::WritePointer => 'w',
            };
            self.0.push(format!("{tag}{}", h.index()));
        }
        fn enter_block(&mut self, block: &naga::Block, _: Scope) {
            self.0.push(format!("[{}", block.len()));
        }
        fn exit_block(&mut self, block: &naga::Block, _: Scope) {
            self.0.push(format!("]{}", block.len()));
        }
    }

    #[test]
    fn walk_block_visits_fields_before_blocks_and_prunes_on_false() {
        let mut arena = naga::Arena::<naga::Expression>::new();
        let h: Vec<_> = (0..4)
            .map(|i| {
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(i)),
                    naga::Span::UNDEFINED,
                )
            })
            .collect();
        let inner_if = naga::Statement::If {
            condition: h[2],
            accept: naga::Block::from_vec(vec![naga::Statement::Break]),
            reject: naga::Block::new(),
        };
        let body = naga::Block::from_vec(vec![
            naga::Statement::Emit(naga::Range::new_from_bounds(h[0], h[1])),
            naga::Statement::Store {
                pointer: h[0],
                value: h[1],
            },
            inner_if,
        ]);
        let block = naga::Block::from_vec(vec![
            naga::Statement::Loop {
                body,
                continuing: naga::Block::from_vec(vec![naga::Statement::Continue]),
                break_if: Some(h[3]),
            },
            naga::Statement::If {
                condition: h[3],
                accept: naga::Block::new(),
                reject: naga::Block::from_vec(vec![naga::Statement::Break]),
            },
        ]);
        let mut trace = Trace::default();
        walk_block(&block, Scope::default(), &mut trace);
        let expected = [
            "[2",
            "loop@0",
            "o3",
            "[3",
            "emit@1",
            "e0",
            "e1",
            "store@1",
            "o0",
            "w0",
            "o1",
            "if@1",
            "]3",
            "[1",
            "continue@1",
            "]1",
            "if@0",
            "o3",
            "[0",
            "]0",
            "[1",
            "break@0",
            "]1",
            "]2",
        ];
        assert_eq!(
            trace.0, expected,
            "a pruned `if` shows neither its condition nor its arms"
        );
    }

    #[test]
    fn write_pointers_are_the_store_atomic_call_and_query_operands() {
        let (arena, h) = dummy_handles();
        let _ = arena;
        let mut seen = Vec::new();
        for stmt in [
            naga::Statement::Store {
                pointer: h,
                value: h,
            },
            naga::Statement::Call {
                function: naga::Arena::<naga::Function>::new()
                    .append(naga::Function::default(), naga::Span::UNDEFINED),
                arguments: vec![h, h],
                result: None,
            },
            naga::Statement::ImageStore {
                image: h,
                coordinate: h,
                array_index: None,
                value: h,
            },
            naga::Statement::Return { value: Some(h) },
        ] {
            let mut n = 0;
            visit_statement_write_pointers(&stmt, &mut |_| n += 1);
            seen.push(n);
        }
        assert_eq!(seen, [1, 2, 0, 0]);
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
}
