//! Load / store dataflow cleanup.  Four phases run per function on
//! every sweep:
//!
//! 1. Dead-store removal: whole-variable `Store`s overwritten before any
//!    read in the same block, or still pending at a terminator.
//! 2. Load deduplication: repeated `Load`s forward to the most recent
//!    dominating stored (or init) value through a `ScopedMap`; aliasing
//!    stores and calls invalidate, and a loop starts from an empty
//!    cache because its iteration count is unknown.
//! 3. Write-only-local elimination: every store, whole or partial, to a
//!    local nothing ever observes.
//! 4. Dead-init removal: inits overwritten before any surviving read,
//!    plus zero inits, which WGSL supplies implicitly.
//!
//! The order is load-bearing: phase 1 keeps never-read stores out of
//! the dominance map, and phase 4 relies on the seeded loads phase 2
//! has already forwarded and dropped.

use rustc_hash::{FxHashMap, FxHashSet};
use std::marker::PhantomData;

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

use super::expr_util::has_negative_zero_leaf;
use super::expr_util::{
    flatten_replacement_chains, for_each_statement, nested_blocks, nested_blocks_mut,
    remap_statement_handles, try_map_expression_handles_in_place, visit_expression_children,
    visit_statement_write_pointers,
};
use super::scoped_map::ScopedMap;
use crate::handle_set::{HandleMap, HandleSet};

/// Four-phase load / store dataflow cleanup.
#[derive(Debug, Default)]
pub struct LoadDedupPass;

impl Pass for LoadDedupPass {
    fn name(&self) -> &'static str {
        "load_dedup"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for (_, function) in module.functions.iter_mut() {
            changed |= remove_dead_stores_in_function(function);
            changed |= dedup_loads_in_function(function);
            changed |= eliminate_write_only_locals(function);
            changed |= remove_dead_inits(function);
        }
        for entry in module.entry_points.iter_mut() {
            changed |= remove_dead_stores_in_function(&mut entry.function);
            changed |= dedup_loads_in_function(&mut entry.function);
            changed |= eliminate_write_only_locals(&mut entry.function);
            changed |= remove_dead_inits(&mut entry.function);
        }
        Ok(changed)
    }
}

/// Identity of a Store statement: `(pointer_handle, value_handle)`.
type StoreId = (
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
);

/// `StoreId` plus whether the Store sits inside a loop.
type StoreInfo = (
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
    bool,
);

/// Forwardable location within a local: the whole variable, one
/// `AccessIndex` field, or one `Access` element.
#[derive(Hash, Eq, PartialEq, Clone)]
enum PointerKey {
    Local(naga::Handle<naga::LocalVariable>),
    LocalField(naga::Handle<naga::LocalVariable>, u32),
    LocalDynamic(
        naga::Handle<naga::LocalVariable>,
        naga::Handle<naga::Expression>,
    ),
}

// MARK: Expression scope index

/// Per-function DFS position index, built once before load forwarding.
/// Answers in O(1) whether handle `h` is emitted inside block `B`'s
/// subtree (the scope-carry filter at every Block / If / Switch) and
/// whether a live load sits at a later execution position than a Store
/// (the last-store-inlining gate); per-probe subtree walks would
/// otherwise multiply across up to 16 sweeps.
///
/// One DFS assigns every statement a monotonic `u32` position:
///
/// * `handle_pos[h.index()]`: every handle in an `Emit` range shares the
///   Emit's position (the range materialises atomically), and the
///   statement-bound results (`Call`, `Atomic`, `WorkGroupUniformLoad`,
///   `SubgroupBallot`, `SubgroupGather`, `SubgroupCollectiveOperation`,
///   `RayQueryFunction::Proceed`) take their statement's position: naga
///   `let`-binds them at the statement site, so they must join the
///   subtree check or store-scope leaks reappear.  A dense `Vec` because
///   arena handles are dense.  Pre-emit expressions (`Literal`,
///   `Constant`, `Override`, `ZeroValue`, `FunctionArgument`,
///   `GlobalVariable`, `LocalVariable`) have no position, so
///   `is_in_subtree` answers `false`; callers that must keep them
///   through a scope-narrowing filter pair the probe with
///   `needs_pre_emit()`.
/// * `store_pos[(ptr, val)]`: the MINIMUM position when two Stores share
///   an identity, so a load between them is never misread as "no later
///   live load" (over-keeping is the conservative direction).
/// * `block_interval[B] = [enter, exit)`: the positions of every
///   descendant statement; `h` is in `B`'s subtree iff its position lies
///   inside.
///
/// # Safety: `*const naga::Block` keys
///
/// Blocks have no intrinsic id, so they are keyed by address.  Every
/// block is owned by `Function::body`, the `PhantomData<&'body Block>`
/// holds an immutable borrow of that body for the index's lifetime, and
/// the producer drops the index before its mutation phase.  Without the
/// borrow a moved block would miss the map and the fallback would
/// mis-classify the handle - sound but cache-poisoning.
struct ExpressionScopeIndex<'body> {
    handle_pos: Vec<Option<u32>>,
    store_pos: FxHashMap<StoreId, u32>,
    block_interval: FxHashMap<*const naga::Block, (u32, u32)>,
    _phantom: PhantomData<&'body naga::Block>,
}

impl<'body> ExpressionScopeIndex<'body> {
    /// `expressions` only sizes `handle_pos`; the arena is not retained.
    fn build(body: &'body naga::Block, expressions: &naga::Arena<naga::Expression>) -> Self {
        let mut idx = ExpressionScopeIndex {
            handle_pos: vec![None; expressions.len()],
            store_pos: Default::default(),
            block_interval: Default::default(),
            _phantom: PhantomData,
        };
        let mut pos = 0u32;
        idx.walk(body, &mut pos);
        idx
    }

    fn walk(&mut self, block: &'body naga::Block, pos: &mut u32) {
        let enter = *pos;
        for stmt in block.iter() {
            let here = *pos;
            // Panics in release too: a wrapped position would corrupt
            // both interval membership and store / load ordering, and
            // 2^32 statements is a bug, not a workload.
            *pos = pos
                .checked_add(1)
                .expect("ExpressionScopeIndex: more than 2^32 statements in a single function");
            match stmt {
                naga::Statement::Emit(range) => {
                    for h in range.clone() {
                        self.handle_pos[h.index()] = Some(here);
                    }
                }
                naga::Statement::Store { pointer, value } => {
                    self.store_pos
                        .entry((*pointer, *value))
                        .and_modify(|p| {
                            if here < *p {
                                *p = here;
                            }
                        })
                        .or_insert(here);
                }
                naga::Statement::Call {
                    result: Some(r), ..
                }
                | naga::Statement::Atomic {
                    result: Some(r), ..
                } => {
                    self.handle_pos[r.index()] = Some(here);
                }
                naga::Statement::WorkGroupUniformLoad { result, .. }
                | naga::Statement::SubgroupBallot { result, .. }
                | naga::Statement::SubgroupGather { result, .. }
                | naga::Statement::SubgroupCollectiveOperation { result, .. } => {
                    self.handle_pos[result.index()] = Some(here);
                }
                naga::Statement::RayQuery { fun, .. } => {
                    if let naga::RayQueryFunction::Proceed { result } = fun {
                        self.handle_pos[result.index()] = Some(here);
                    }
                }
                // No handles introduced.  Enumerated rather than `_` so a
                // new naga variant carrying a result handle trips the
                // build; `nested_blocks` gives the same guarantee for
                // descent.
                naga::Statement::Block(_)
                | naga::Statement::If { .. }
                | naga::Statement::Switch { .. }
                | naga::Statement::Loop { .. }
                | naga::Statement::Call { result: None, .. }
                | naga::Statement::Atomic { result: None, .. }
                | naga::Statement::Break
                | naga::Statement::Continue
                | naga::Statement::Return { .. }
                | naga::Statement::Kill
                | naga::Statement::ControlBarrier(_)
                | naga::Statement::MemoryBarrier(_)
                | naga::Statement::ImageStore { .. }
                | naga::Statement::ImageAtomic { .. }
                | naga::Statement::RayPipelineFunction(_)
                | naga::Statement::CooperativeStore { .. } => {}
            }
            for nested in nested_blocks(stmt) {
                self.walk(nested, pos);
            }
        }
        self.block_interval
            .insert(block as *const naga::Block, (enter, *pos));
    }

    /// `None` for pre-emit handles and for handles no statement reached.
    fn handle_position(&self, h: naga::Handle<naga::Expression>) -> Option<u32> {
        self.handle_pos.get(h.index()).copied().flatten()
    }

    /// Earliest position of a Store with identity `id`.
    fn store_position(&self, id: StoreId) -> Option<u32> {
        self.store_pos.get(&id).copied()
    }

    /// `false` for pre-emit handles (in scope everywhere) and for dead
    /// handles no statement reached.
    fn is_in_subtree(&self, block: &naga::Block, h: naga::Handle<naga::Expression>) -> bool {
        let key = block as *const naga::Block;
        let Some(&(enter, exit)) = self.block_interval.get(&key) else {
            // Unreachable unless the body was mutated under the index.
            // `true` is the safe answer: callers use this as a drop
            // filter, and `false` would forward a handle whose `let`
            // exits with the block.
            debug_assert!(
                false,
                "ExpressionScopeIndex: block address not in index - \
                 body mutated during query?"
            );
            return true;
        };
        self.handle_position(h)
            .is_some_and(|p| p >= enter && p < exit)
    }
}

// MARK: Dead-init removal

/// Drop zero inits (WGSL zero-initialises locals) and inits overwritten
/// before any surviving read.  Runs after load forwarding has dropped
/// init-seeded loads, so the sequential scan meets the overwriting
/// `Store` first.
fn remove_dead_inits(function: &mut naga::Function) -> bool {
    let mut changed = false;

    for (_, lvar) in function.local_variables.iter_mut() {
        if let Some(init) = lvar.init
            && is_zero_init(&function.expressions, init)
        {
            lvar.init = None;
            changed = true;
        }
    }

    let dead = find_dead_inits(
        &function.body,
        &function.expressions,
        &function.local_variables,
    );
    for lh in dead {
        if function.local_variables[lh].init.is_some() {
            function.local_variables[lh].init = None;
            changed = true;
        }
    }

    changed
}

/// All-zero value of any type, through `Compose` / `Splat`.
pub(crate) fn is_zero_init(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> bool {
    match &expressions[handle] {
        naga::Expression::ZeroValue(_) => true,
        naga::Expression::Literal(lit) => is_zero_literal(lit),
        naga::Expression::Compose { components, .. } => {
            components.iter().all(|&c| is_zero_init(expressions, c))
        }
        naga::Expression::Splat { value, .. } => is_zero_init(expressions, *value),
        _ => false,
    }
}

/// Bit-exact: `-0.0` is not the implicit zero init.
pub(crate) fn is_zero_literal(lit: &naga::Literal) -> bool {
    match lit {
        naga::Literal::Bool(false) => true,
        naga::Literal::I16(0) | naga::Literal::U16(0) => true,
        naga::Literal::I32(0) | naga::Literal::U32(0) => true,
        naga::Literal::I64(0) | naga::Literal::U64(0) => true,
        naga::Literal::AbstractInt(0) => true,
        naga::Literal::F32(v) => v.to_bits() == 0,
        naga::Literal::F64(v) => v.to_bits() == 0,
        naga::Literal::F16(v) => v.to_bits() == 0,
        naga::Literal::AbstractFloat(v) => v.to_bits() == 0,
        _ => false,
    }
}

/// Inits overwritten before any read on the top-level sequential path.
/// A local touched inside any nested block leaves tracking, since that
/// block may not run.
fn find_dead_inits(
    body: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    local_variables: &naga::Arena<naga::LocalVariable>,
) -> HandleSet<naga::LocalVariable> {
    let mut pending: HandleSet<naga::LocalVariable> = local_variables
        .iter()
        .filter(|(_, lvar)| lvar.init.is_some())
        .map(|(h, _)| h)
        .collect();

    if pending.is_empty() {
        return Default::default();
    }

    let mut dead = HandleSet::default();

    for stmt in body.iter() {
        if pending.is_empty() {
            break;
        }
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = &expressions[h]
                        && let Some(local) = get_stored_local(expressions, *pointer)
                    {
                        pending.remove(local);
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    if pending.remove(lh) {
                        dead.insert(lh);
                    }
                } else if let Some(local) = get_stored_local(expressions, *pointer) {
                    // A partial store reads the old value.
                    pending.remove(local);
                }
            }
            // A callee / atomic / ray / cooperative write may read or
            // overwrite the init.
            other => visit_statement_write_pointers(other, &mut |p| {
                if let Some(local) = get_stored_local(expressions, p) {
                    pending.remove(local);
                }
            }),
        }
        invalidate_involved(stmt, expressions, &mut pending);
    }
    dead
}

fn invalidate_involved(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut HandleSet<naga::LocalVariable>,
) {
    let mut involved = Default::default();
    for block in nested_blocks(stmt) {
        collect_touched_locals(block, expressions, &mut involved);
    }
    for lh in &involved {
        pending.remove(lh);
    }
}

/// Reads count too - a merely-read init is still observed - which is
/// why this is not `collect_modified_locals`.
fn collect_touched_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    touched: &mut HandleSet<naga::LocalVariable>,
) {
    for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            for h in range.clone() {
                if let naga::Expression::Load { pointer } = &expressions[h]
                    && let Some(local) = get_stored_local(expressions, *pointer)
                {
                    touched.insert(local);
                }
            }
        }
        visit_statement_write_pointers(stmt, &mut |p| {
            if let Some(local) = get_stored_local(expressions, p) {
                touched.insert(local);
            }
        });
    });
}

// MARK: Dead-store removal

fn remove_dead_stores_in_function(function: &mut naga::Function) -> bool {
    remove_dead_stores_in_block(&mut function.body, &function.expressions)
}

/// Drop every store, whole or partial, to a local nothing observes.
/// Sound because a local qualifies only when its every reference is a
/// `Store` pointer: any `Load` / `CooperativeLoad` rooted at it, any
/// pointer handed to a callee, and any atomic / workgroup-uniform-load
/// pointer marks it used.  Side effects that produced the stored values
/// live in their own statements and survive; later DCE drops the
/// declaration.
fn eliminate_write_only_locals(function: &mut naga::Function) -> bool {
    let exprs = &function.expressions;
    let mut used: HandleSet<naga::LocalVariable> = Default::default();

    // `Load` and `CooperativeLoad` are the only pointer-reading expressions
    // (image reads address globals); omitting one would strip the stores of
    // a local read solely through it.
    for (_, expr) in exprs.iter() {
        let read_ptr = match expr {
            naga::Expression::Load { pointer } => Some(*pointer),
            naga::Expression::CooperativeLoad { data, .. } => Some(data.pointer),
            _ => None,
        };
        if let Some(ptr) = read_ptr
            && let Some(local) = get_stored_local(exprs, ptr)
        {
            used.insert(local);
        }
    }
    // The partial-store set is irrelevant here: with no loads and no
    // escapes the local is dead whether its stores are whole or partial.
    let mut partially_stored_unused = Default::default();
    collect_escaped_and_partially_stored(
        &function.body,
        exprs,
        &mut used,
        &mut partially_stored_unused,
    );
    collect_nonstore_pointer_locals(&function.body, exprs, &mut used);

    if used.len() == function.local_variables.len() {
        return false;
    }

    remove_stores_to_dead_locals(&mut function.body, exprs, &used)
}

fn collect_nonstore_pointer_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    used: &mut HandleSet<naga::LocalVariable>,
) {
    for_each_statement(block, &mut |stmt| {
        let pointer = match stmt {
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => *pointer,
            // `Store` is the write being proven dead; call / ray / cooperative
            // escapes belong to the escape walk.  No `_` arm: a new
            // pointer-reading variant must trip the build, not ship a dropped
            // store.
            naga::Statement::Emit(_)
            | naga::Statement::Block(_)
            | naga::Statement::If { .. }
            | naga::Statement::Switch { .. }
            | naga::Statement::Loop { .. }
            | naga::Statement::Store { .. }
            | naga::Statement::Call { .. }
            | naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Return { .. }
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. }
            | naga::Statement::RayPipelineFunction(_)
            | naga::Statement::RayQuery { .. }
            | naga::Statement::CooperativeStore { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. } => return,
        };
        if let Some(local) = get_stored_local(expressions, pointer) {
            used.insert(local);
        }
    });
}

fn remove_stores_to_dead_locals(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    used: &HandleSet<naga::LocalVariable>,
) -> bool {
    let mut changed = false;
    let original = std::mem::replace(block, naga::Block::new());
    for (mut stmt, span) in original.span_into_iter() {
        if let naga::Statement::Store { pointer, .. } = &stmt
            && let Some(local) = get_stored_local(expressions, *pointer)
            && !used.contains(local)
        {
            changed = true;
            continue;
        }
        for nested in nested_blocks_mut(&mut stmt) {
            changed |= remove_stores_to_dead_locals(nested, expressions, used);
        }
        block.push(stmt, span);
    }
    changed
}

/// A whole-variable Store overwritten by another before any Load of the
/// local, or still pending at a terminator, is dead.
fn remove_dead_stores_in_block(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut changed = false;

    for stmt in block.iter_mut() {
        for nested in nested_blocks_mut(stmt) {
            changed |= remove_dead_stores_in_block(nested, expressions);
        }
    }

    // Latest whole-variable Store per local not yet followed by a Load of it.
    let mut pending_store: HandleMap<naga::LocalVariable, usize> = Default::default();
    let mut dead_indices: Vec<usize> = Vec::new();

    for (idx, stmt) in block.iter().enumerate() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = &expressions[h]
                        && let Some(local) = get_stored_local(expressions, *pointer)
                    {
                        pending_store.remove(local);
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    if let Some(prev_idx) = pending_store.insert(lh, idx) {
                        dead_indices.push(prev_idx);
                    }
                } else if let Some(local) = get_stored_local(expressions, *pointer) {
                    // A partial store reads the old value.
                    pending_store.remove(local);
                }
            }
            // Nothing observes a local after Return / Kill.
            naga::Statement::Return { .. } | naga::Statement::Kill => {
                for (_, prev_idx) in pending_store.drain() {
                    dead_indices.push(prev_idx);
                }
            }
            // A boundary or jump may join paths that read the local.
            naga::Statement::If { .. }
            | naga::Statement::Switch { .. }
            | naga::Statement::Loop { .. }
            | naga::Statement::Block(_)
            | naga::Statement::Break
            | naga::Statement::Continue => pending_store.clear(),
            // A callee / atomic / ray / cooperative write may be partial and
            // observe the unwritten bytes: keep the pending store live.
            other => visit_statement_write_pointers(other, &mut |p| {
                if let Some(local) = get_stored_local(expressions, p) {
                    pending_store.remove(local);
                }
            }),
        }
    }

    if !dead_indices.is_empty() {
        // Descending so earlier indices stay valid; contiguous runs cull as
        // one `Vec::drain` (each is an O(N) tail shift), and the common
        // `x=a; x=b;` pattern is a single run.
        dead_indices.sort_unstable_by(|a, b| b.cmp(a));
        let mut iter = dead_indices.iter().copied();
        let mut hi = iter.next().expect("non-empty checked above");
        let mut lo = hi;
        for idx in iter {
            if idx + 1 == lo {
                lo = idx;
            } else {
                block.cull(lo..=hi);
                hi = idx;
                lo = idx;
            }
        }
        block.cull(lo..=hi);
        changed = true;
    }

    changed
}

// MARK: Load deduplication

/// Only INTEGER `/` `%` by zero is a WGSL shader-creation error; float
/// `x / 0.0` is a defined `inf` / `nan` and must stay forwardable.
fn is_integer_zero_literal(lit: &naga::Literal) -> bool {
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

/// A shift by `>= 32` is a shader-creation error for the ubiquitous
/// 32-bit operand.  A 16-bit operand shifted by `[16, 32)` is left to
/// the whole-module rollback (fail-safe); a 64-bit one is over-declined
/// harmlessly.
fn shift_amount_is_static_error(lit: &naga::Literal) -> bool {
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

/// Literal at the end of `start`'s forward chain; the budget is a cycle
/// safety net (chains are acyclic by construction).
fn resolve_forward_literal(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    start: naga::Handle<naga::Expression>,
) -> Option<naga::Literal> {
    let mut handle = start;
    let mut budget = expressions.len() + 1;
    while let Some(&next) = replacements.get(handle) {
        handle = next;
        budget -= 1;
        if budget == 0 {
            return None;
        }
    }
    match &expressions[handle] {
        naga::Expression::Literal(lit) => Some(*lit),
        _ => None,
    }
}

/// Drop forwards that would put a constant into the RHS of an integer
/// `/` `%` `<<` `>>` and turn a legal RUNTIME operation into a
/// shader-creation error (integer divide / modulo by zero, shift `>=`
/// bit width): naga rejects the const form and post-pass validation
/// rolls back the WHOLE module, discarding every other forward.  MUST
/// precede the dead-local scan - a declined load stays live, so its
/// store must not be classed dead.  Declining only ever keeps more live.
fn decline_static_error_forwards(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
) {
    let mut to_decline: Vec<naga::Handle<naga::Expression>> = Vec::new();
    for (_, expr) in expressions.iter() {
        let naga::Expression::Binary { op, right, .. } = expr else {
            continue;
        };
        match op {
            naga::BinaryOperator::Divide | naga::BinaryOperator::Modulo => {
                collect_static_error_leaves(
                    expressions,
                    replacements,
                    *right,
                    &is_integer_zero_literal,
                    &mut to_decline,
                );
            }
            naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight => {
                collect_static_error_leaves(
                    expressions,
                    replacements,
                    *right,
                    &shift_amount_is_static_error,
                    &mut to_decline,
                );
            }
            _ => {}
        }
    }
    for handle in to_decline {
        replacements.remove(handle);
    }
}

/// Forward-chain leaves under a divisor / shift amount that resolve to a
/// literal `is_dangerous` accepts, through `Splat` / `Compose` because
/// ANY offending const lane makes a componentwise op a shader-creation
/// error.  A literal already present in the input is naga's to reject.
fn collect_static_error_leaves<F: Fn(&naga::Literal) -> bool>(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    handle: naga::Handle<naga::Expression>,
    is_dangerous: &F,
    out: &mut Vec<naga::Handle<naga::Expression>>,
) {
    if replacements.contains_key(handle) {
        if resolve_forward_literal(expressions, replacements, handle)
            .as_ref()
            .is_some_and(is_dangerous)
        {
            out.push(handle);
        }
        return;
    }
    match &expressions[handle] {
        naga::Expression::Splat { value, .. } => {
            collect_static_error_leaves(expressions, replacements, *value, is_dangerous, out);
        }
        naga::Expression::Compose { components, .. } => {
            for &component in components {
                collect_static_error_leaves(
                    expressions,
                    replacements,
                    component,
                    is_dangerous,
                    out,
                );
            }
        }
        _ => {}
    }
}

/// Phase 2 driver: collect forwards, prune the unsafe and unprofitable
/// ones, then rewrite.
fn dedup_loads_in_function(function: &mut naga::Function) -> bool {
    let mut replacements = Default::default();
    let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
    let mut all_loads: HandleMap<naga::LocalVariable, Vec<naga::Handle<naga::Expression>>> =
        Default::default();
    let mut seeded_by_store: HandleMap<naga::Expression, StoreInfo> = Default::default();

    // Inits are const-evaluable (runtime inits lower to Stores), so loads
    // can forward to them.  Compose inits are skipped: a renamed variable
    // is shorter than a repeated constructor.
    for (lh, lvar) in function.local_variables.iter() {
        if let Some(init) = lvar.init
            && !matches!(function.expressions[init], naga::Expression::Compose { .. })
            && !has_negative_zero_leaf(&function.expressions, init)
        {
            cache.insert(PointerKey::Local(lh), init);
        }
    }

    let scope_idx = ExpressionScopeIndex::build(&function.body, &function.expressions);

    collect_redundant_loads(
        &function.body,
        &function.expressions,
        &scope_idx,
        &mut cache,
        &mut replacements,
        &mut all_loads,
        &mut seeded_by_store,
        false,
        &mut Default::default(),
    );

    if replacements.is_empty() {
        return false;
    }

    // Ordering: a declined load stays live, so this must precede the
    // dead-local scan.
    decline_static_error_forwards(&function.expressions, &mut replacements);
    if replacements.is_empty() {
        return false;
    }

    let mut escaped: HandleSet<naga::LocalVariable> = Default::default();
    let mut partially_stored: HandleSet<naga::LocalVariable> = Default::default();
    collect_escaped_and_partially_stored(
        &function.body,
        &function.expressions,
        &mut escaped,
        &mut partially_stored,
    );

    // A local is dead when every Load rooted at it (whole or partial) has
    // an EFFECTIVE replacement, `r < load`: the arena walk blocks forward
    // references, so a target appended later (e.g. by short-circuit
    // re-sugaring) leaves the Load live and its Store must survive.
    // `escaped` locals are excluded (the callee reads through the pointer
    // at any time); `partially_stored` needs no exclusion because an
    // un-forwarded partial Load already fails the all-loads gate.
    let dead_locals: HandleSet<naga::LocalVariable> = all_loads
        .iter()
        .filter(|(lh, loads)| {
            !escaped.contains(*lh)
                && loads
                    .iter()
                    .all(|l| replacements.get(l).is_some_and(|&r| r < *l))
        })
        .map(|(lh, _)| *lh)
        .collect();

    // Pricing: forwarding a complex value into a surviving local's loads
    // shares an Emit and inflates output, so such forwards are undone
    // unless they retire something - the whole local (dead), or by
    // last-store inlining the seeding Store, once every load it seeded
    // has an effective replacement even though earlier loads keep the
    // local.  Simple values (`is_simple_for_forwarding`) never inflate.

    // Loop-internal Stores are never removal candidates: the back edge
    // reads them.
    let mut loads_per_store: FxHashMap<StoreId, Vec<naga::Handle<naga::Expression>>> =
        Default::default();
    for (&load_h, &(ptr, val, store_in_loop)) in &seeded_by_store {
        if store_in_loop {
            continue;
        }
        loads_per_store.entry((ptr, val)).or_default().push(load_h);
    }

    let undo_candidates: Vec<_> = replacements
        .iter()
        .filter_map(|(&load_h, &replacement_h)| {
            if let naga::Expression::Load { pointer } = &function.expressions[load_h] {
                let local = get_stored_local(&function.expressions, *pointer)?;
                if dead_locals.contains(local) {
                    return None;
                }
                if is_simple_for_forwarding(&function.expressions[replacement_h]) {
                    return None;
                }
                Some(load_h)
            } else {
                None
            }
        })
        .collect();

    // A Store retires when it is whole-variable, its local has no partial
    // store (that reads the whole value) and never escapes, every load it
    // seeded has an effective replacement, and no live load follows it.

    let max_live_load_pos = build_max_live_load_positions(&all_loads, &replacements, &scope_idx);
    let mut dead_store_ids: FxHashSet<StoreId> = Default::default();
    for (&store_id, seeded_loads) in &loads_per_store {
        let (store_ptr, _) = store_id;
        if !matches!(
            function.expressions[store_ptr],
            naga::Expression::LocalVariable(_)
        ) {
            continue;
        }
        let local = match get_stored_local(&function.expressions, store_ptr) {
            Some(l) => l,
            None => continue,
        };
        if partially_stored.contains(local) {
            continue;
        }
        if escaped.contains(local) {
            continue;
        }
        let all_valid = seeded_loads
            .iter()
            .all(|&load_h| replacements.get(load_h).is_some_and(|&r| r < load_h));
        if !all_valid {
            continue;
        }
        // A later live (un-replaced) Load may observe this Store on a path
        // the seeded-load cache never saw, e.g. a loop body after the
        // cache was cleared.  Execution order is the statement DFS
        // position (arena order is unreliable once `inlining` appends to
        // the tail); liveness is `r >= load_h` in arena order, exact
        // because the apply walk iterates the arena and blocks forward
        // references.  INVARIANT, the other half being the cache-promotion
        // gate: a canonical that is itself a Load is never rebound, so
        // producer-Loads stay out of `replacements` and keep their Store
        // alive here.  A Store with no recorded position is kept.
        let Some(store_pos) = scope_idx.store_position(store_id) else {
            continue;
        };
        let has_later_live_load = max_live_load_pos
            .get(local)
            .is_some_and(|&max_pos| max_pos > store_pos);
        if has_later_live_load {
            continue;
        }
        dead_store_ids.insert(store_id);
    }

    for h in undo_candidates {
        if let Some(&(ptr, val, _in_loop)) = seeded_by_store.get(h)
            && dead_store_ids.contains(&(ptr, val))
        {
            continue;
        }
        replacements.remove(h);
    }

    // `dead_store_ids` was judged against the pre-undo replacements; an
    // undone forward resurrects a load, and removing its Store would make
    // that load read the zero-init.  Re-judge against the post-undo set
    // (undo only ever adds live loads, so the maxima must be rebuilt).
    let max_live_load_pos = build_max_live_load_positions(&all_loads, &replacements, &scope_idx);
    dead_store_ids.retain(|&store_id| {
        let (store_ptr, _) = store_id;
        // A lookup miss keeps the Store (`false` drops it from the dead
        // set), the safe direction.
        let Some(local) = get_stored_local(&function.expressions, store_ptr) else {
            return false;
        };
        let Some(store_pos) = scope_idx.store_position(store_id) else {
            return false;
        };
        let regained_live_load = max_live_load_pos
            .get(local)
            .is_some_and(|&max_pos| max_pos > store_pos);
        !regained_live_load
    });

    // Release the body borrow before mutation.
    drop(scope_idx);

    // Undo may have emptied the map; reporting a change would block
    // convergence.
    if replacements.is_empty() {
        return false;
    }

    // Load-to-Load dedup on a store-forwarded canonical creates chains;
    // the arena walk resolves one level.
    flatten_replacement_chains(&mut replacements);

    // Keep only forwards effective at EVERY use.  The arena walk rewrites
    // a use only where `target < use`, and a use always outranks its
    // load, so `target < load` is the exact condition.  A target that
    // trails the load (an init `inlining` appended after it) applies
    // nowhere: keeping it would drop the load's `let` while its uses still
    // read the bare, possibly since-mutated load (a miscompile past a
    // write to the place, e.g. a loop `break_if`), and would report a
    // change every sweep.
    replacements.retain(|load, target| *target < *load);
    if replacements.is_empty() {
        return false;
    }

    // `r < handle` blocks illegal forward references; `dead_locals`
    // mirrors the same guard.
    for (handle, expr) in function.expressions.iter_mut() {
        let _ = try_map_expression_handles_in_place(expr, &mut |h| match replacements.get(h) {
            Some(&r) if r < handle => Some(r),
            _ => Some(h),
        });
    }

    // A dead local with an init would still be declared.
    for &lh in &dead_locals {
        function.local_variables[lh].init = None;
    }

    apply_to_block(
        &mut function.body,
        &replacements,
        &dead_locals,
        &dead_store_ids,
        &function.expressions,
        false,
    );

    // A name on a replaced handle would become a dangling `let`.
    function
        .named_expressions
        .retain(|h, _| !replacements.contains_key(h));

    true
}

/// `None` for anything but a local or a depth-1 `AccessIndex` / `Access`
/// on one; deeper chains and non-local roots are never forwarded.
fn get_pointer_key(
    expressions: &naga::Arena<naga::Expression>,
    pointer_handle: naga::Handle<naga::Expression>,
) -> Option<PointerKey> {
    match &expressions[pointer_handle] {
        naga::Expression::LocalVariable(local) => Some(PointerKey::Local(*local)),
        naga::Expression::AccessIndex { base, index } => {
            if let naga::Expression::LocalVariable(local) = &expressions[*base] {
                Some(PointerKey::LocalField(*local, *index))
            } else {
                None
            }
        }
        naga::Expression::Access { base, index } => {
            if let naga::Expression::LocalVariable(local) = &expressions[*base] {
                Some(PointerKey::LocalDynamic(*local, *index))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Per local, the maximum DFS position of a live load, so every Store's
/// later-live-load guard is O(1) rather than O(loads) - quadratic on
/// machine-generated single-accumulator functions.  Live means no
/// effective replacement (`r < load_h`); loads without a position
/// contribute nothing.
fn build_max_live_load_positions(
    all_loads: &HandleMap<naga::LocalVariable, Vec<naga::Handle<naga::Expression>>>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    scope_idx: &ExpressionScopeIndex<'_>,
) -> HandleMap<naga::LocalVariable, u32> {
    let mut max_pos = HandleMap::default();
    for (&local, loads) in all_loads {
        for &load_h in loads {
            let live = replacements.get(load_h).is_none_or(|&r| r >= load_h);
            if live && let Some(pos) = scope_idx.handle_position(load_h) {
                max_pos
                    .entry(local)
                    .and_modify(|m: &mut u32| *m = (*m).max(pos))
                    .or_insert(pos);
            }
        }
    }
    max_pos
}

/// Root local of an `Access` / `AccessIndex` chain, if any.
pub(crate) fn get_stored_local(
    expressions: &naga::Arena<naga::Expression>,
    pointer_handle: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match &expressions[pointer_handle] {
        naga::Expression::LocalVariable(local) => Some(*local),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            get_stored_local(expressions, *base)
        }
        _ => None,
    }
}

/// Cheap to reference from many sites: pre-emit declaratives never get a
/// `let`, and a `Load` forwarded to N sites is N references to one
/// already-emitted handle, not N memory reads - unlike `const_fold`'s
/// `is_pure_to_clone`, which rejects `Load` because it clones expression
/// CONTENT into a new arena slot and would emit a second read.
/// Sign-of-zero is not filtered here: the seeds this chooses among already
/// passed `has_negative_zero_leaf`.
fn is_simple_for_forwarding(expr: &naga::Expression) -> bool {
    matches!(
        expr,
        naga::Expression::Literal(_)
            | naga::Expression::Constant(_)
            | naga::Expression::Override(_)
            | naga::Expression::ZeroValue(_)
            | naga::Expression::FunctionArgument(_)
            | naga::Expression::GlobalVariable(_)
            | naga::Expression::LocalVariable(_)
            | naga::Expression::Load { .. }
    )
}

/// Ceiling on a store-seeded value's effective (post-forwarding) tree
/// depth.  Forwarding a value that absorbed the previous store's tree
/// grows flat reassignment chains without bound (`a = a*2.0+1.0;` x9000
/// -> an 18k-deep tree) and overflows every recursive consumer:
/// `render_depth`, naga's writer, and wasm's ~1 MB stack that the CLI's
/// big-stack worker does not cover - and the tree ships in the OUTPUT,
/// so every re-minification pays again.  128 is far above hand-written
/// depth and far below any consumer's budget.
const SUBSTITUTION_DEPTH_CAP: u32 = 128;

/// Effective depth of `value` reading through `replacements`: a
/// forwarded `Load` counts as its replacement's tree, an un-forwarded one
/// as its pointer chain, whose subscripts can absorb a previous
/// iteration's tree (`x = a[x & 3] + 1;`).  BFS saturating at `cap + 1`.
/// The shared visited set yields SHORTEST-path depth, under-counting a
/// diamond against the longest path recursive consumers stack; that is
/// unreachable here because forwarding never synthesises sharing
/// (re-absorption grows both paths in lockstep) and the naga parser
/// bounds pre-existing input.
fn effective_forwarded_depth(
    value: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    cap: u32,
) -> u32 {
    let mut frontier = vec![value];
    let mut next = Vec::new();
    let mut seen = HandleSet::default();
    let mut budget = 8192usize;
    let mut depth = 0u32;
    while !frontier.is_empty() {
        depth += 1;
        if depth > cap {
            return depth;
        }
        for handle in frontier.drain(..) {
            if !seen.insert(handle) {
                continue;
            }
            budget -= 1;
            if budget == 0 {
                return cap + 1;
            }
            match &expressions[handle] {
                naga::Expression::Load { pointer } => {
                    if let Some(&replacement) = replacements.get(handle) {
                        next.push(replacement);
                    } else {
                        next.push(*pointer);
                    }
                }
                expr => visit_expression_children(expr, |child| next.push(child)),
            }
        }
        std::mem::swap(&mut frontier, &mut next);
    }
    depth
}

/// Locals whose pointer reaches a callee (`escaped`) and locals with
/// field / index writes (`partially_stored`), in one walk.
fn collect_escaped_and_partially_stored(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    escaped: &mut HandleSet<naga::LocalVariable>,
    partially_stored: &mut HandleSet<naga::LocalVariable>,
) {
    for_each_statement(block, &mut |stmt| match stmt {
        naga::Statement::Store { pointer, .. } => {
            if !matches!(expressions[*pointer], naga::Expression::LocalVariable(_))
                && let Some(local) = get_stored_local(expressions, *pointer)
            {
                partially_stored.insert(local);
            }
        }
        // A cooperative write covers only part of the local; `target` is
        // the source value.
        naga::Statement::CooperativeStore { data, .. } => {
            if let Some(local) = get_stored_local(expressions, data.pointer) {
                partially_stored.insert(local);
            }
        }
        // A callee (`Call`, `traceRay` payload) or the ray-query runtime
        // may read or write through the pointer later; an atomic pointer
        // lands here too but never roots at a function local.
        other => visit_statement_write_pointers(other, &mut |p| {
            if let Some(local) = get_stored_local(expressions, p) {
                escaped.insert(local);
            }
        }),
    });
}

#[cfg(test)]
fn count_local_stores(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> HandleMap<naga::LocalVariable, usize> {
    let mut counts = Default::default();
    count_stores_recursive(block, expressions, &mut counts);
    counts
}

#[cfg(test)]
fn count_stores_recursive(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    counts: &mut HandleMap<naga::LocalVariable, usize>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Store { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    *counts.entry(lh).or_insert(0) += 1;
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                // Read-modify-write counts as a store.
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    *counts.entry(lh).or_insert(0) += 1;
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // Initialize / Proceed mutate through the query pointer.
                if let Some(lh) = get_stored_local(expressions, *query) {
                    *counts.entry(lh).or_insert(0) += 1;
                }
            }
            _ => {}
        }
        for nested in nested_blocks(stmt) {
            count_stores_recursive(nested, expressions, counts);
        }
    }
}

/// Dominance-aware walker: threads `cache` (`PointerKey` -> canonical
/// value) through the statement tree, recording each forwardable `Load`
/// in `replacements` and every load rooted at a local in `all_loads`.
///
/// `seeded_by_store` maps a forwarded Load to the `(pointer, value)`
/// identity of the Store that seeded its canonical, so the pricing step
/// can retire a Store once every load it seeded is covered.
///
/// Every control-flow boundary is a checkpoint: each arm runs from the
/// pre-boundary state and rolls back, then entries for every local
/// written in ANY arm are invalidated (logged, so an outer rollback
/// undoes that too).  A loop's body and continuing each start from an
/// empty cache - an iteration cannot rely on pre-loop values, nor
/// post-loop code on in-loop ones - via a logged drain the outer
/// rollback reverses.
///
/// `modified_out` receives every local `block` (recursively) may write;
/// each arm collects its own set during the same walk, so the meet and
/// the invalidation need no separate `collect_modified_locals` pass.
#[allow(clippy::too_many_arguments)]
fn collect_redundant_loads<'body>(
    block: &'body naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    scope_idx: &ExpressionScopeIndex<'body>,
    cache: &mut ScopedMap<PointerKey, naga::Handle<naga::Expression>>,
    replacements: &mut HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    all_loads: &mut HandleMap<naga::LocalVariable, Vec<naga::Handle<naga::Expression>>>,
    seeded_by_store: &mut HandleMap<naga::Expression, StoreInfo>,
    in_loop: bool,
    modified_out: &mut HandleSet<naga::LocalVariable>,
) {
    // Store-seeded keys only.  Block-local scratch that never crosses a
    // branch, hence a plain map rather than a `ScopedMap`.
    let mut store_source: FxHashMap<PointerKey, StoreInfo> = Default::default();
    for statement in block {
        match statement {
            naga::Statement::Emit(range) => {
                for handle in range.clone() {
                    let naga::Expression::Load { pointer } = &expressions[handle] else {
                        continue;
                    };
                    // Every load rooted at a local feeds the liveness sets,
                    // even the depth>=2 chains (`e.a.x`) `get_pointer_key`
                    // cannot forward: otherwise their local could be judged
                    // dead, its Store dropped, and the surviving load left
                    // reading the zero-default.
                    if let Some(local) = get_stored_local(expressions, *pointer) {
                        all_loads.entry(local).or_default().push(handle);
                    }
                    if let Some(key) = get_pointer_key(expressions, *pointer) {
                        if let Some(&canonical) = cache.get(&key) {
                            replacements.insert(handle, canonical);
                            if let Some(&store_id) = store_source.get(&key) {
                                seeded_by_store.insert(handle, store_id);
                            }
                            // Promote the cache to this Load so later
                            // Load-to-Load forwarding chains through a
                            // surviving Emit - unless the canonical IS a
                            // Load.  INVARIANT, the other half being the
                            // later-live-load guard: a producer-Load stays
                            // the canonical, hence out of `replacements`,
                            // or its Store would be judged dead across a
                            // loop / branch boundary where the cache was
                            // cleared.
                            if !matches!(expressions[canonical], naga::Expression::Load { .. }) {
                                cache.insert(key, handle);
                            }
                        } else {
                            cache.insert(key, handle);
                        }
                    }
                }
            }
            naga::Statement::Store { pointer, value } => {
                if let Some(local) = get_stored_local(expressions, *pointer) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                    // Seed with the stored value; the pricing step later
                    // undoes complex forwards on surviving locals.  A value
                    // deeper than `SUBSTITUTION_DEPTH_CAP` stays unseeded
                    // (the stale entry is already gone, so later loads keep
                    // the variable read), or reassignment chains grow
                    // without bound.
                    if let Some(key) = get_pointer_key(expressions, *pointer) {
                        let mut resolved = *value;
                        while let Some(&next) = replacements.get(resolved) {
                            resolved = next;
                        }
                        if effective_forwarded_depth(
                            resolved,
                            expressions,
                            replacements,
                            SUBSTITUTION_DEPTH_CAP,
                        ) <= SUBSTITUTION_DEPTH_CAP
                            && !has_negative_zero_leaf(expressions, resolved)
                        {
                            cache.insert(key.clone(), resolved);
                            store_source.insert(key, (*pointer, *value, in_loop));
                        }
                    }
                }
            }
            naga::Statement::Block(inner) => {
                // A nested `{ }` is a lexical scope in the output: a
                // handle `let`-bound inside it (Emit or statement result)
                // is out of scope afterwards, so a post-block read
                // forwarded to it fails the round-trip and the pass
                // silently rolls back - hence the `is_in_subtree` filter
                // on `carry` (pre-emit values are in scope everywhere and
                // exempt).  An arena-index highwater would be a no-op:
                // this pass never appends expressions, so position-based
                // membership is the only check that fires.  Rollback also
                // resurrects pre-block entries, including init seeds
                // (`Local(F) -> false`) the block made stale; those are
                // invalidated from `block_modified` after the rollback,
                // else a later `Load(F)` forwards to the init and drops
                // F's accumulated state (`F = false | X` for `F |= X`).
                let cp_pre_block = cache.checkpoint();

                // Own set: the invalidation must cover only writes inside
                // the block, not siblings' writes already in `modified_out`.
                let mut block_modified = Default::default();
                collect_redundant_loads(
                    inner,
                    expressions,
                    scope_idx,
                    cache,
                    replacements,
                    all_loads,
                    seeded_by_store,
                    in_loop,
                    &mut block_modified,
                );

                let carry: Vec<(PointerKey, naga::Handle<naga::Expression>)> = cache
                    .as_map()
                    .iter()
                    .filter(|(_, v)| {
                        !scope_idx.is_in_subtree(inner, **v) || expressions[**v].needs_pre_emit()
                    })
                    .map(|(k, &v)| (k.clone(), v))
                    .collect();
                cache.rollback_to(cp_pre_block);

                for local in &block_modified {
                    invalidate_cache_for_local(cache, *local);
                }
                for (k, v) in carry {
                    cache.insert(k, v);
                }

                modified_out.extend(block_modified);
            }
            naga::Statement::If { accept, reject, .. } => {
                let cp_pre_if = cache.checkpoint();

                // A handle `let`-bound inside a branch is out of scope after
                // the if; pre-emit values are in scope everywhere.
                let in_branches = |v: naga::Handle<naga::Expression>| {
                    scope_idx.is_in_subtree(accept, v) || scope_idx.is_in_subtree(reject, v)
                };

                let mut accept_modified = Default::default();
                collect_redundant_loads(
                    accept,
                    expressions,
                    scope_idx,
                    cache,
                    replacements,
                    all_loads,
                    seeded_by_store,
                    in_loop,
                    &mut accept_modified,
                );
                // Meet candidates: keys `accept` modified, at accept's final
                // value.  A key `accept` left alone but `reject` re-stored
                // to the pre-if value is not carried (it merely cache-misses
                // later); one branch's set avoids a second full walk.
                let mut meet: FxHashMap<PointerKey, naga::Handle<naga::Expression>> = cache
                    .as_map()
                    .iter()
                    .filter(|(k, _)| pointer_key_involves_any_local(k, &accept_modified))
                    .filter(|(_, v)| !in_branches(**v) || expressions[**v].needs_pre_emit())
                    .map(|(k, &v)| (k.clone(), v))
                    .collect();
                cache.rollback_to(cp_pre_if);

                let mut reject_modified = Default::default();
                collect_redundant_loads(
                    reject,
                    expressions,
                    scope_idx,
                    cache,
                    replacements,
                    all_loads,
                    seeded_by_store,
                    in_loop,
                    &mut reject_modified,
                );
                meet.retain(|k, v| cache.get(k) == Some(v));
                cache.rollback_to(cp_pre_if);

                // `union`, not `chain`: each invalidation is a full cache
                // scan.
                for local in accept_modified.union(&reject_modified) {
                    invalidate_cache_for_local(cache, *local);
                }

                // Meet: `if/else` is total, so an entry both arms end with
                // under the same canonical is sound after the if.  An arm
                // that exits early may end with entries never reached
                // post-if; harmless, since the meet only introduces agreed
                // entries and those are read solely on paths that fall
                // through the whole if.
                for (k, v) in meet {
                    cache.insert(k, v);
                }

                modified_out.extend(accept_modified);
                modified_out.extend(reject_modified);
            }
            naga::Statement::Switch { cases, .. } => {
                let cp_pre_switch = cache.checkpoint();
                // Same scope rule as `If`: a handle `let`-bound inside a case
                // is out of scope after the switch.
                let in_cases = |v: naga::Handle<naga::Expression>| {
                    cases
                        .iter()
                        .any(|case| scope_idx.is_in_subtree(&case.body, v))
                };

                // The meet needs exactly one case per path and fall-off as
                // the only exit: a `Default` (total), no `fall_through` (a
                // case's final state would include later cases' writes),
                // and no bare `break`, which reaches post-switch code with
                // the PRE-break state - the SPIR-V structurizer idiom
                // `switch(0u){default:{ if(c){x=1; break;} x=2; }}` exits
                // with x=1, which a fall-end meet (x=2) never sees;
                // forwarding it deletes live stores.
                let has_default = cases
                    .iter()
                    .any(|c| matches!(c.value, naga::SwitchValue::Default));
                let any_fallthrough = cases.iter().any(|c| c.fall_through);
                let any_switch_break = cases
                    .iter()
                    .any(|c| crate::passes::dead_branch::contains_bare_break(&c.body));
                let meet_applicable = has_default && !any_fallthrough && !any_switch_break;

                let mut total_modified: HandleSet<naga::LocalVariable> = Default::default();
                let mut meet: Option<FxHashMap<PointerKey, naga::Handle<naga::Expression>>> = None;
                for case in cases {
                    let mut case_modified = Default::default();
                    collect_redundant_loads(
                        &case.body,
                        expressions,
                        scope_idx,
                        cache,
                        replacements,
                        all_loads,
                        seeded_by_store,
                        in_loop,
                        &mut case_modified,
                    );
                    if meet_applicable {
                        match meet.as_mut() {
                            None => {
                                // First case: candidates as in `If`; later
                                // cases intersect in place.
                                let initial: FxHashMap<_, _> = cache
                                    .as_map()
                                    .iter()
                                    .filter(|(k, _)| {
                                        pointer_key_involves_any_local(k, &case_modified)
                                    })
                                    .filter(|(_, v)| {
                                        !in_cases(**v) || expressions[**v].needs_pre_emit()
                                    })
                                    .map(|(k, &v)| (k.clone(), v))
                                    .collect();
                                meet = Some(initial);
                            }
                            Some(m) => {
                                // No scope filter: survivors of the first-case
                                // filter already live outside every case body.
                                m.retain(|k, v| cache.get(k) == Some(v));
                            }
                        }
                    }
                    cache.rollback_to(cp_pre_switch);
                    total_modified.extend(case_modified);
                }

                for local in &total_modified {
                    invalidate_cache_for_local(cache, *local);
                }

                if let Some(m) = meet {
                    // With bare breaks excluded, the remaining early exits
                    // (Return / Kill / Continue) never reach post-switch
                    // code, so every path that does fell off a case end and
                    // agrees with the meet.
                    for (k, v) in m {
                        cache.insert(k, v);
                    }
                }

                modified_out.extend(total_modified);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                let cp_pre_loop = cache.checkpoint();
                cache.drain_logged();
                let cp_empty = cache.checkpoint();

                let mut loop_modified = Default::default();
                collect_redundant_loads(
                    body,
                    expressions,
                    scope_idx,
                    cache,
                    replacements,
                    all_loads,
                    seeded_by_store,
                    true,
                    &mut loop_modified,
                );
                cache.rollback_to(cp_empty);

                collect_redundant_loads(
                    continuing,
                    expressions,
                    scope_idx,
                    cache,
                    replacements,
                    all_loads,
                    seeded_by_store,
                    true,
                    &mut loop_modified,
                );
                cache.rollback_to(cp_pre_loop);

                for local in &loop_modified {
                    invalidate_cache_for_local(cache, *local);
                }

                modified_out.extend(loop_modified);
            }
            // A callee / atomic / ray / cooperative write invalidates `cache`
            // and `store_source` together: a stale `store_source` entry
            // outliving a cache miss mis-classifies its Store's forwarded
            // loads.
            other => visit_statement_write_pointers(other, &mut |p| {
                if let Some(local) = get_stored_local(expressions, p) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                }
            }),
        }
    }
}

fn invalidate_cache_for_local(
    cache: &mut ScopedMap<PointerKey, naga::Handle<naga::Expression>>,
    local: naga::Handle<naga::LocalVariable>,
) {
    cache.retain_logged(|k, _| match k {
        PointerKey::Local(l) | PointerKey::LocalField(l, _) | PointerKey::LocalDynamic(l, _) => {
            *l != local
        }
    });
}

/// Must stay in lockstep with `invalidate_cache_for_local`; the
/// exhaustive match makes a new `PointerKey` variant trip every site.
fn invalidate_store_source_for_local(
    store_source: &mut FxHashMap<PointerKey, StoreInfo>,
    local: naga::Handle<naga::LocalVariable>,
) {
    store_source.retain(|key, _| match key {
        PointerKey::Local(l) | PointerKey::LocalField(l, _) | PointerKey::LocalDynamic(l, _) => {
            *l != local
        }
    });
}

fn pointer_key_involves_any_local(
    key: &PointerKey,
    locals: &HandleSet<naga::LocalVariable>,
) -> bool {
    match key {
        PointerKey::Local(l) | PointerKey::LocalField(l, _) | PointerKey::LocalDynamic(l, _) => {
            locals.contains(l)
        }
    }
}

#[cfg(test)]
fn locals_passed_by_pointer(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> HandleSet<naga::LocalVariable> {
    let mut escaped = Default::default();
    collect_escaped_locals(block, expressions, &mut escaped);
    escaped
}

#[cfg(test)]
fn collect_escaped_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    escaped: &mut HandleSet<naga::LocalVariable>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(local) = get_stored_local(expressions, arg) {
                        escaped.insert(local);
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = get_stored_local(expressions, *payload) {
                    escaped.insert(local);
                }
            }
            naga::Statement::CooperativeStore { data, .. } => {
                // `data.pointer` is the write side.
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    escaped.insert(local);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // The runtime mutates through the query pointer.
                if let Some(local) = get_stored_local(expressions, *query) {
                    escaped.insert(local);
                }
            }
            _ => {}
        }
        for nested in nested_blocks(stmt) {
            collect_escaped_locals(nested, expressions, escaped);
        }
    }
}

/// Locals any statement in `block` may write; `dead_branch` shares the
/// predicate.
pub(crate) fn collect_modified_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    modified: &mut HandleSet<naga::LocalVariable>,
) {
    for_each_statement(block, &mut |stmt| {
        visit_statement_write_pointers(stmt, &mut |p| {
            if let Some(lh) = get_stored_local(expressions, p) {
                modified.insert(lh);
            }
        });
    });
}

// MARK: Replacement application

/// Rewrite statement operands, rebuild `Emit` ranges around survivors,
/// and drop retired Stores, in one walk.
fn apply_to_block(
    block: &mut naga::Block,
    replacements: &HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    dead_locals: &HandleSet<naga::LocalVariable>,
    dead_store_ids: &FxHashSet<StoreId>,
    expressions: &naga::Arena<naga::Expression>,
    // Set once inside a `Loop`.  `dead_store_ids` keys by `(pointer, value)`
    // alone, and an in-loop Store can share that identity with the retired
    // out-of-loop one (`x = e` before and inside the loop); in-loop Stores
    // are never candidates, so a match there is a collision the back edge
    // still reads.
    in_loop: bool,
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        match &statement {
            naga::Statement::Emit(range) => {
                // Every replacement is effective at all uses (the driver
                // kept only `target < load`), so the binding can go.
                let surviving: Vec<_> = range
                    .clone()
                    .filter(|h| !replacements.contains_key(h))
                    .collect();

                if surviving.is_empty() {
                    continue;
                }

                let mut start = surviving[0];
                let mut end = surviving[0];
                for &h in &surviving[1..] {
                    if h.index() == end.index() + 1 {
                        end = h;
                    } else {
                        rebuilt.push(
                            naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                            span,
                        );
                        start = h;
                        end = h;
                    }
                }
                rebuilt.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                    span,
                );
                continue;
            }
            naga::Statement::Store { pointer, value } => {
                if let Some(lh) = get_stored_local(expressions, *pointer)
                    && dead_locals.contains(lh)
                {
                    continue;
                }
                if !in_loop && dead_store_ids.contains(&(*pointer, *value)) {
                    continue;
                }
            }
            _ => {}
        }

        let enter_loop = in_loop || matches!(statement, naga::Statement::Loop { .. });
        for nested in nested_blocks_mut(&mut statement) {
            apply_to_block(
                nested,
                replacements,
                dead_locals,
                dead_store_ids,
                expressions,
                enter_loop,
            );
        }

        let mut remap = |h: naga::Handle<naga::Expression>| -> naga::Handle<naga::Expression> {
            replacements.get(h).copied().unwrap_or(h)
        };
        remap_statement_handles(&mut statement, &mut remap);

        rebuilt.push(statement, span);
    }

    *block = rebuilt;
}

// MARK: Tests

#[cfg(test)]
#[path = "load_dedup_tests.rs"]
mod tests;
