//! Load / store dataflow cleanup.  Four phases run per function on
//! every sweep:
//!
//! 1. **Dead-store removal.**  Discards `Store` statements whose
//!    written value is never read, including the last store to a
//!    local if its value is trivially dead at function exit.
//! 2. **Load deduplication.**  Forwards repeated `Load` expressions
//!    to the value most recently stored (or seeded from an init)
//!    using a dominance-scoped `ScopedMap`.  Stores and calls that
//!    might alias a cached load invalidate it; loop handlers drain
//!    the cache before the body runs because iteration counts are
//!    unknown at pass time.
//! 3. **Write-only-local elimination.**  A whole-function scan drops
//!    every `Store` (whole or partial) to a local that is never
//!    observed - stronger than phase 1's per-block, whole-var reach.
//! 4. **Dead-init removal.**  After phase 2 has forwarded seeded
//!    loads, init values that are overwritten before any surviving
//!    read can be dropped.  Zero inits (`T(0)`) are also removed
//!    because WGSL already zero-initialises uninitialised locals.
//!
//! The phases are load-bearing in order: dropping dead stores before
//! load dedup keeps the dominance map free of stores that will never
//! influence a downstream read, and dead-init removal relies on the
//! forwarded-and-dropped emits the dedup phase leaves behind.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

use super::expr_util::{
    flatten_replacement_chains, nested_blocks, nested_blocks_mut, remap_statement_handles,
    try_map_expression_handles_in_place, visit_expression_children,
};
use super::scoped_map::ScopedMap;

/// Pass object for the four-phase load / store cleanup.  See the
/// module-level doc for phase ordering.
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

/// Store identity with a flag indicating whether the Store is inside a loop.
type StoreInfo = (
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
    bool,
);

/// Cache key identifying a unique memory location within a local variable.
#[derive(Hash, Eq, PartialEq, Clone)]
enum PointerKey {
    /// Whole local variable, e.g. `e`
    Local(naga::Handle<naga::LocalVariable>),
    /// Single field of a local variable, e.g. `e.x` (`AccessIndex`)
    LocalField(naga::Handle<naga::LocalVariable>, u32),
    /// Dynamic-indexed element of a local variable, e.g. `e[i]` (Access)
    LocalDynamic(
        naga::Handle<naga::LocalVariable>,
        naga::Handle<naga::Expression>,
    ),
}

// MARK: Expression scope index

/// Per-function precomputed scope index built once by
/// [`dedup_loads_in_function`].  Answers two recurring questions in O(1):
///
/// 1. **"Is expression handle `h` emitted inside block `B`'s subtree?"**
///    Consumed by [`collect_redundant_loads`] at every Block / If /
///    Switch boundary to drop cache entries whose value is `let`-bound
///    inside the closing brace.  Implemented as an interval check
///    (`enter <= handle_pos < exit`) so each probe is O(1); deeply
///    nested IR would otherwise pay a per-probe subtree DFS, multiplied
///    by up to 16 pipeline sweeps per minification.
///
/// 2. **"Is the live-load handle `h` at a later execution position than
///    Store `s`?"**  Consumed by the `has_later_live_load` gate in
///    [`dedup_loads_in_function`] to decide whether last-store inlining
///    is safe.  The position numbering is shared with the
///    subtree-membership view above so the build walk amortises both
///    answers across one DFS.
///
/// # Layout
///
/// A single depth-first walk of `body` assigns every statement a
/// monotonically-increasing `u32` position.  Three views derive from
/// that numbering:
///
/// * `handle_pos[h.index()] = Some(pos)` for every expression handle
///   introduced by a statement.  Two sources contribute:
///
///   - **Emit ranges**: every handle in a `Statement::Emit` range
///     shares the position of the Emit statement (the whole range
///     materialises atomically at execution time, so per-handle
///     ordering inside the range is meaningless).
///   - **Statement-bound result handles**: `Call.result`,
///     `Atomic.result`, `WorkGroupUniformLoad.result`,
///     `SubgroupBallot.result`, `SubgroupGather.result`,
///     `SubgroupCollectiveOperation.result`, and
///     `RayQueryFunction::Proceed.result`.  These are NOT in any
///     Emit range but are still `let`-bound at the statement site by
///     naga's WGSL writer, so they must participate in the
///     subtree-membership check or the store-scope leak regressions
///     reappear (see `block_scope_leak_regression_call_result_in_block`
///     and the atomic counterpart).
///
///   `Vec<Option<u32>>` indexed by `handle.index()` avoids the per-probe
///   hash on the hot membership check.  Arena handles are dense
///   integers in `[0, expressions.len())`, so the indexing is exact.
///
///   Pre-emit expressions (`Literal`, `Constant`, `Override`,
///   `ZeroValue`, `FunctionArgument`, `GlobalVariable`,
///   `LocalVariable`) are NOT in this map - `handle_position` returns
///   `None` for them, and [`is_in_subtree`](Self::is_in_subtree)
///   returns `false` (membership predicate: "no recorded position
///   inside the subtree's interval").  Callers that want pre-emit
///   handles to survive a scope-narrowing filter combine the call
///   with `expressions[h].needs_pre_emit()` so the pair correctly
///   models "in scope everywhere" without conflating it with the
///   in-subtree answer.
///
/// * `store_pos[(ptr, val)] = pos` for every `Store` statement.  When
///   two distinct Stores share an identity (same pointer + same value
///   handle, possible e.g. from store-seeding in different basic
///   blocks), the MINIMUM position is kept - a Load between the
///   earlier Store and the later one would otherwise fall through
///   the gap and be misclassified as "no later live Load",
///   incorrectly killing the earlier Store.  Min-aggregation makes
///   `has_later_live_load` over-keep on identity collisions, which
///   is the conservative direction.
///
/// * `block_interval[B] = (enter, exit)` is the half-open
///   `[enter, exit)` range of positions consumed by every descendant
///   statement of block `B` (the block's own statements plus all
///   nested control-flow subtrees).  A handle `h` is "inside `B`'s
///   subtree" iff `handle_pos[h.index()] in [enter, exit)`.  An
///   empty block has `enter == exit`; no handle is in it.
///
/// # Safety: `*const naga::Block` as map key
///
/// `block_interval` is keyed by block address because blocks have no
/// intrinsic numeric identifier in naga's IR.  Address stability is
/// guaranteed by three invariants enforced jointly:
///
/// 1. All blocks live inside `naga::Function::body` and are reached
///    via owning references - their addresses do not change unless
///    the body is mutated.
/// 2. The struct's `PhantomData<&'body naga::Block>` lifetime
///    parameter ties its existence to an immutable borrow of `body`;
///    the borrow checker rejects any concurrent mutation of the body
///    for as long as the index is live.
/// 3. The single production caller ([`dedup_loads_in_function`])
///    drops the index before its mutation phase ([`apply_to_block`]
///    etc.) begins.
///
/// Without invariant (2) a HashMap lookup against a moved block would
/// silently return `None` (different address) and the `is_in_subtree`
/// fallback would mis-classify the handle - sound but cache-poisoning.
/// The PhantomData lifetime prevents that statically.
struct ExpressionScopeIndex<'body> {
    handle_pos: Vec<Option<u32>>,
    store_pos: HashMap<StoreId, u32>,
    block_interval: HashMap<*const naga::Block, (u32, u32)>,
    _phantom: PhantomData<&'body naga::Block>,
}

impl<'body> ExpressionScopeIndex<'body> {
    /// Build the index from a function body in a single DFS walk.
    /// `expressions` is used solely to size the dense `handle_pos`
    /// vector; the arena itself is not stored.
    fn build(body: &'body naga::Block, expressions: &naga::Arena<naga::Expression>) -> Self {
        let mut idx = ExpressionScopeIndex {
            handle_pos: vec![None; expressions.len()],
            store_pos: HashMap::new(),
            block_interval: HashMap::new(),
            _phantom: PhantomData,
        };
        let mut pos = 0u32;
        idx.walk(body, &mut pos);
        idx
    }

    /// Recursive DFS worker.  Records each statement's position,
    /// populates `handle_pos` / `store_pos` for the variants that
    /// contribute, and records the block's `[enter, exit)` interval
    /// on exit.  The match is exhaustive (no `_ => {}` catch-all) so
    /// a future naga release adding a new emit-handle-bearing
    /// statement variant trips the build here instead of silently
    /// slipping past this index's coverage; block descent is
    /// delegated to [`nested_blocks`] for the same guarantee on the
    /// recursion axis.
    fn walk(&mut self, block: &'body naga::Block, pos: &mut u32) {
        let enter = *pos;
        for stmt in block.iter() {
            let here = *pos;
            // `checked_add` so a release build still panics on
            // overflow - wrapping back to 0 would re-issue used
            // positions and corrupt both subtree-membership
            // (`enter <= pos < exit`) and `store_pos < load_pos`
            // ordering.  2^32 statements per function is a bug
            // signal, not a workload cap.
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
                // Statements that introduce no expression handles (no
                // Emit range, no result field, no Store identity).
                // Enumerated explicitly - a future naga variant that
                // adds emit-handle semantics would otherwise silently
                // bypass this index.  Nested blocks recurse below.
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

    /// DFS position of expression handle `h`, or `None` if `h` is a
    /// pre-emit expression (`Literal`, `Constant`, `Override`,
    /// `ZeroValue`, `FunctionArgument`, `GlobalVariable`,
    /// `LocalVariable`) and therefore not bound by any statement.
    fn handle_position(&self, h: naga::Handle<naga::Expression>) -> Option<u32> {
        self.handle_pos.get(h.index()).copied().flatten()
    }

    /// Earliest DFS position of any `Store` with identity `id`, or
    /// `None` if no such Store was encountered during the build walk.
    fn store_position(&self, id: StoreId) -> Option<u32> {
        self.store_pos.get(&id).copied()
    }

    /// `true` when handle `h` is introduced by a statement inside
    /// `block`'s subtree (the block itself plus all nested children).
    ///
    /// Returns `false` for:
    /// * Pre-emit handles (not bound by any statement; in scope
    ///   everywhere - their `handle_position` is `None`).  Callers
    ///   that need pre-emit handles to survive a scope-narrowing
    ///   filter pair this with a separate `needs_pre_emit()` clause.
    /// * Handles in the arena but not reached by the build walk
    ///   (e.g. dead expressions left behind by an earlier pass).
    fn is_in_subtree(&self, block: &naga::Block, h: naga::Handle<naga::Expression>) -> bool {
        let key = block as *const naga::Block;
        let Some(&(enter, exit)) = self.block_interval.get(&key) else {
            // Block address not in the index - only possible if the
            // body was mutated during the query, which the pass
            // contract forbids.  Callers combine this with
            // `needs_pre_emit()` as a KEEP filter, so `true` drops
            // the cache entry (safe direction): `false` would
            // forward a handle whose Emit range exits with the
            // block and trip the validator on rollback.
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

/// Drop redundant local initializers.
///
/// Sub-phase 1 - zero inits: WGSL already zero-initialises uninit
/// locals, so `var x: T = T(0)` is strictly redundant.
///
/// Sub-phase 2 - dead inits: any init value whose whole-variable
/// `Store` overrides it before a surviving read on every path can be
/// removed.  By the time this phase runs, `dedup_loads_in_function`
/// has already forwarded init-seeded loads and dropped their `Emit`
/// ranges, so the sequential scan naturally sees the overwriting
/// `Store` first.
fn remove_dead_inits(function: &mut naga::Function) -> bool {
    let mut changed = false;

    // Sub-phase 1: zero inits are always redundant in WGSL.
    for (_, lvar) in function.local_variables.iter_mut() {
        if let Some(init) = lvar.init
            && is_zero_init(&function.expressions, init)
        {
            lvar.init = None;
            changed = true;
        }
    }

    // Sub-phase 2: non-zero inits overwritten before first load.
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

/// `true` when `handle` evaluates to the all-zero value for its type.
/// Recursively walks `Compose` and `Splat` so `vec3f(0.0, 0.0, 0.0)`
/// and `vec3f(0.0)` are both recognised as zero.
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

/// `true` when `lit` is the zero value of its scalar type.  Used by
/// [`is_zero_init`] and by `dead_branch` when detecting redundant
/// `var x: T = T(0)` patterns.
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

/// Locals whose init is provably overwritten before any read on the
/// sequential top-level path.  Conservative for sub-blocks: any local
/// loaded or modified inside an If / Switch / Loop / Block is dropped
/// from tracking, since the branch may not run.
fn find_dead_inits(
    body: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    local_variables: &naga::Arena<naga::LocalVariable>,
) -> HashSet<naga::Handle<naga::LocalVariable>> {
    let mut pending: HashSet<naga::Handle<naga::LocalVariable>> = local_variables
        .iter()
        .filter(|(_, lvar)| lvar.init.is_some())
        .map(|(h, _)| h)
        .collect();

    if pending.is_empty() {
        return HashSet::new();
    }

    let mut dead = HashSet::new();

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
                        pending.remove(&local); // init is read
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    // Whole-variable store overwrites whatever init held.
                    if pending.remove(&lh) {
                        dead.insert(lh);
                    }
                } else if let Some(local) = get_stored_local(expressions, *pointer) {
                    // Partial store (field / index) reads the old value.
                    pending.remove(&local);
                }
            }
            // Statements that may modify a single local through a pointer.
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(local) = get_stored_local(expressions, arg) {
                        pending.remove(&local);
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                if let Some(local) = get_stored_local(expressions, *pointer) {
                    pending.remove(&local);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                if let Some(local) = get_stored_local(expressions, *query) {
                    pending.remove(&local);
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = get_stored_local(expressions, *payload) {
                    pending.remove(&local);
                }
            }
            naga::Statement::CooperativeStore { data, .. } => {
                // `data.pointer` is the destination of the matrix write
                // (validator rejects non-`STORE`-space pointers).  `target`
                // is the source matrix VALUE - reads through it are
                // already tracked by the Emit/Load walking elsewhere.
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    pending.remove(&local);
                }
            }
            // Statements that neither touch local-variable inits
            // directly nor modify one through a pointer channel -
            // enumerated explicitly so a future naga release adding a
            // new pointer-bearing statement breaks the build here
            // instead of silently letting an init survive past a
            // Store-through-it.  Nested blocks (control flow) carry no
            // direct action; the post-match sweep invalidates every
            // local they touch.
            naga::Statement::Block(_)
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
        invalidate_involved(stmt, expressions, &mut pending);
    }
    dead
}

/// Remove `pending` entries for any local that is loaded OR modified
/// inside `stmt`'s nested blocks.  Used by `find_dead_inits` to drop
/// init-tracking for any local that might be observed (read) or
/// mutated (write) on a path through a sub-block - either case means
/// the init cannot be proved dead from a top-level scan.  No-op for
/// block-free statements.
fn invalidate_involved(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    let mut involved = HashSet::new();
    for block in nested_blocks(stmt) {
        collect_touched_locals(block, expressions, &mut involved);
    }
    for lh in &involved {
        pending.remove(lh);
    }
}

/// Every local that is read OR written inside `block` (recursively),
/// in a single walk.  Used by [`invalidate_involved`] where the union
/// of reads and writes is what disqualifies a local from init-death
/// tracking.  Distinct from [`collect_modified_locals`], which
/// `dead_branch` calls for write-only analysis (its else-store
/// equality check tolerates pure reads).
fn collect_touched_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    touched: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    for stmt in block {
        match stmt {
            // Reads: Emit ranges carry Load expressions whose pointer
            // chains back to a local.  The Emit range itself is
            // visited once - the inner loop scans every handle in the
            // range for a Load that targets a local.
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = &expressions[h]
                        && let Some(local) = get_stored_local(expressions, *pointer)
                    {
                        touched.insert(local);
                    }
                }
            }
            // Writes - same arm shape and pointer-root resolution as
            // `collect_modified_locals`, kept in lockstep so a future
            // naga variant adding a new pointer-writer triggers
            // exhaustive-match failures in both places.
            naga::Statement::Store { pointer, .. } | naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    touched.insert(lh);
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = get_stored_local(expressions, arg) {
                        touched.insert(lh);
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(lh) = get_stored_local(expressions, *payload) {
                    touched.insert(lh);
                }
            }
            naga::Statement::CooperativeStore { data, .. } => {
                // Write side is `data.pointer`; `target` is the matrix
                // VALUE (a read), already counted via Emit walking.
                if let Some(lh) = get_stored_local(expressions, data.pointer) {
                    touched.insert(lh);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                if let Some(lh) = get_stored_local(expressions, *query) {
                    touched.insert(lh);
                }
            }
            // Statements that neither read a local through `Load` nor
            // write a local through any pointer-bearing field.
            // Enumerated explicitly so a future naga variant breaks
            // the build here rather than silently mis-classifying a
            // touch.  Nested blocks recurse below.
            naga::Statement::Block(_)
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
        for nested in nested_blocks(stmt) {
            collect_touched_locals(nested, expressions, touched);
        }
    }
}

// MARK: Dead-store removal

/// Drop `Store` statements whose written value never reaches a read.
/// Entry point for phase 1; dispatches to
/// [`remove_dead_stores_in_block`] which walks the control-flow tree.
fn remove_dead_stores_in_function(function: &mut naga::Function) -> bool {
    remove_dead_stores_in_block(&mut function.body, &function.expressions)
}

/// Eliminate WRITE-ONLY locals: a local that is stored to but whose value is
/// never observed (no `Load`, never passed by pointer, never an atomic/barrier
/// target) is dead in its entirety, so every `Store` to it - whole-variable AND
/// partial (`a.x = ...`, `a[i] = ...`) - can be dropped.
///
/// This is strictly more powerful than [`remove_dead_stores_in_block`] (which
/// only kills a whole-var store overwritten before any read in the SAME block):
/// it reasons over the whole function and removes the var entirely once its
/// stores are gone (later DCE drops the now-unused declaration and the dead
/// store-value expressions).
///
/// Soundness: a local is a candidate only when EVERY reference to it is the
/// pointer of a `Store`.  Any `Load` or `CooperativeLoad` rooted at it, any
/// pointer passed to a callee (the escape walk - the callee can read it), and
/// any atomic / workgroup-uniform-load pointer mark it used.  Removing a `Store`
/// keeps the call/atomic/etc. statements that produce its value's side effects
/// (those are separate statements); only the dead write disappears.
fn eliminate_write_only_locals(function: &mut naga::Function) -> bool {
    let exprs = &function.expressions;
    let mut used: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();

    // Reads: an expression that reads memory through a pointer rooting at a
    // local marks it used.  `Load` and `CooperativeLoad` are the only such
    // expressions (image reads address globals, not local pointers); both must
    // be mirrored from `expr_util`'s read classification or a local read solely
    // by the omitted one would be misjudged write-only and lose its stores.
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
    // Escapes (callee may read through the pointer) and other pointer uses.
    // The shared helper also reports partially-stored locals, but that output is
    // intentionally unused here: a local with zero `Load`s and zero escapes is
    // dead whether its stores are whole or partial, so every store is removed
    // regardless.  (The set is consumed only by the load-forwarding path.)
    let mut partially_stored_unused = HashSet::new();
    collect_escaped_and_partially_stored(
        &function.body,
        exprs,
        &mut used,
        &mut partially_stored_unused,
    );
    collect_nonstore_pointer_locals(&function.body, exprs, &mut used);

    if used.len() == function.local_variables.len() {
        // Every local is observed; nothing to eliminate (fast path).
        return false;
    }

    remove_stores_to_dead_locals(&mut function.body, exprs, &used)
}

/// Mark locals referenced as the pointer of any statement OTHER than `Store`
/// (atomics, workgroup-uniform-load, ...) as used.  `Store` is excluded - that
/// is precisely the write [`eliminate_write_only_locals`] removes.  Call /
/// ray / cooperative escapes are handled by the escape walk.
fn collect_nonstore_pointer_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    used: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    let mark = |ptr: naga::Handle<naga::Expression>,
                used: &mut HashSet<naga::Handle<naga::LocalVariable>>| {
        if let Some(local) = get_stored_local(expressions, ptr) {
            used.insert(local);
        }
    };
    for stmt in block {
        match stmt {
            naga::Statement::Atomic { pointer, .. } => mark(*pointer, used),
            naga::Statement::WorkGroupUniformLoad { pointer, .. } => mark(*pointer, used),
            // Statements that do not observe a local through a pointer.
            // `Store` is excluded by design (it is the write this analysis is
            // proving dead); callee/atomic/cooperative pointer escapes are
            // covered by `collect_escaped_and_partially_stored`.  Enumerated
            // (no `_`) per this file's forward-compat contract, so a future
            // pointer-bearing variant trips the build here.  Nested blocks
            // recurse below.
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
            | naga::Statement::SubgroupCollectiveOperation { .. } => {}
        }
        for nested in nested_blocks(stmt) {
            collect_nonstore_pointer_locals(nested, expressions, used);
        }
    }
}

/// Drop every `Store` whose pointer roots at a local NOT in `used` (recursing
/// into nested blocks).  Returns whether any statement was removed.
fn remove_stores_to_dead_locals(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    used: &HashSet<naga::Handle<naga::LocalVariable>>,
) -> bool {
    let mut changed = false;
    let original = std::mem::replace(block, naga::Block::new());
    for (mut stmt, span) in original.span_into_iter() {
        if let naga::Statement::Store { pointer, .. } = &stmt
            && let Some(local) = get_stored_local(expressions, *pointer)
            && !used.contains(&local)
        {
            changed = true;
            continue; // drop the dead store
        }
        for nested in nested_blocks_mut(&mut stmt) {
            changed |= remove_stores_to_dead_locals(nested, expressions, used);
        }
        block.push(stmt, span);
    }
    changed
}

/// Remove dead stores: a whole-variable Store to a local that is overwritten
/// by another whole-variable Store before any Load of that local is dead.
fn remove_dead_stores_in_block(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    let mut changed = false;

    // Recurse into sub-blocks first.
    for stmt in block.iter_mut() {
        for nested in nested_blocks_mut(stmt) {
            changed |= remove_dead_stores_in_block(nested, expressions);
        }
    }

    // For each local, track the index of the most recent whole-variable Store
    // that has NOT been followed by any Load of that local.
    let mut pending_store: HashMap<naga::Handle<naga::LocalVariable>, usize> = HashMap::new();
    let mut dead_indices: Vec<usize> = Vec::new();

    for (idx, stmt) in block.iter().enumerate() {
        match stmt {
            naga::Statement::Emit(range) => {
                // A Load from a local makes its pending Store live (not dead).
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = &expressions[h]
                        && let Some(local) = get_stored_local(expressions, *pointer)
                    {
                        pending_store.remove(&local);
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    // Whole-variable Store: if the previous Store to the same
                    // local was never loaded, it is dead.
                    if let Some(prev_idx) = pending_store.insert(lh, idx) {
                        dead_indices.push(prev_idx);
                    }
                } else if let Some(local) = get_stored_local(expressions, *pointer) {
                    // Field / index Store - the pending whole-variable Store
                    // might still be needed (partial overwrite reads old value).
                    pending_store.remove(&local);
                }
            }
            // Terminators: any pending Store cannot be observed after the
            // function returns or the invocation is killed, so it is dead.
            naga::Statement::Return { .. } | naga::Statement::Kill => {
                for (_, prev_idx) in pending_store.drain() {
                    dead_indices.push(prev_idx);
                }
            }
            // Statements that cannot reference a function-local pointer
            // (atomics target storage/workgroup, image ops target globals,
            // workgroup-uniform loads target workgroup, subgroup ops take
            // value arguments only).  These cannot make any pending
            // function-local Store live, so they leave `pending_store`
            // unchanged - more precise than a blanket clear.
            naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::WorkGroupUniformLoad { .. }
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. } => {}
            // Atomic read-modify-writes through `pointer`.  WGSL forbids
            // `atomic<T>` in function memory so a chain rooted at a
            // function-local would be IR-malformed, but the precise-by-
            // root-local invalidation keeps this site in lockstep with
            // the cache walker - the two must agree on which locals a
            // statement disturbs.
            naga::Statement::Atomic { pointer, .. } => {
                if let Some(local) = get_stored_local(expressions, *pointer) {
                    pending_store.remove(&local);
                }
            }
            // CooperativeStore: `data.pointer` is the write destination
            // (validator-required STORE-space pointer).  When it roots at
            // a function-local, the matrix write may be partial (matrix
            // < local's full type), so a prior whole-variable pending
            // Store is NOT cleanly overwritten - the unwritten bytes
            // still observe the pending value, keeping it live.
            // Conservatively remove the local from pending.
            naga::Statement::CooperativeStore { data, .. } => {
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    pending_store.remove(&local);
                }
            }
            // Call: the only channel by which a function-local Store can be
            // observed across the call boundary is a `ptr<function, T>`
            // argument, which in naga IR is necessarily a chain of
            // LocalVariable / Access / AccessIndex (`get_stored_local`
            // recovers the root local).  Invalidate pending only for those
            // locals; non-pointer arguments cannot alias.
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(local) = get_stored_local(expressions, arg) {
                        pending_store.remove(&local);
                    }
                }
            }
            // RayQuery: the `query` field is a `ptr<function, ray_query>`
            // (or a chain rooted at a function local).  Invalidate that
            // local only.
            naga::Statement::RayQuery { query, .. } => {
                if let Some(local) = get_stored_local(expressions, *query) {
                    pending_store.remove(&local);
                }
            }
            // TraceRay's payload is the only pointer reachable through
            // the call boundary, so only the local rooting `payload`
            // can be observed.  Precise invalidation matches the cache
            // walker; a blanket clear would needlessly kill unrelated
            // pending Stores.
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = get_stored_local(expressions, *payload) {
                    pending_store.remove(&local);
                }
            }
            // Structural statements: If/Switch/Loop/Block were
            // already recursed above; arriving here on the pending
            // walk means we're at the top-level scan and need to
            // conservatively clear pending across the control-flow
            // boundary (the boundary may join paths that read F).
            naga::Statement::If { .. }
            | naga::Statement::Switch { .. }
            | naga::Statement::Loop { .. }
            | naga::Statement::Block(_) => {
                pending_store.clear();
            }
            // Break / Continue: clear pending - the control transfer
            // ends the current basic block; a pending store followed
            // by Break could still be observable by the Break target.
            naga::Statement::Break | naga::Statement::Continue => {
                pending_store.clear();
            } // Statements explicitly enumerated above (Emit / Store /
              // Atomic / RayQuery / Call / Return / Kill / barriers /
              // image-store / image-atomic / workgroup-uniform-load /
              // Subgroup{Ballot, Gather, CollectiveOperation} /
              // CooperativeStore) are handled by their dedicated arms.
              // This file enumerates every Statement variant explicitly
              // (no `_ =>` catch-all) per the forward-compat contract
              // established in 311f6a4: a future naga release adding a
              // new variant trips the build here instead of silently
              // taking the conservative-clear default.
        }
    }

    if !dead_indices.is_empty() {
        // Remove in descending order so earlier indices stay valid, and
        // coalesce contiguous runs into a single `cull(lo..=hi)`.  Each
        // `cull` is a `Vec::drain` (O(N) tail shift), so culling one index at
        // a time was O(D*N); the natural dead-store pattern (`var x; x=a;
        // x=b;`) produces a single contiguous run, making this O(N).
        dead_indices.sort_unstable_by(|a, b| b.cmp(a));
        let mut iter = dead_indices.iter().copied();
        let mut hi = iter.next().expect("non-empty checked above");
        let mut lo = hi;
        for idx in iter {
            if idx + 1 == lo {
                // Extends the current descending run downward.
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

/// `true` when `lit` is an integer-typed zero.  Narrower than
/// [`is_zero_literal`] (which also matches `0.0`): only INTEGER divide /
/// modulo by zero is a WGSL shader-creation error - float `x / 0.0` yields a
/// defined `inf`/`nan` and must stay foldable, so it must NOT count as a
/// dangerous divisor.
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

/// `true` when a shift by the constant `lit` is a WGSL shader-creation error
/// for a 32-bit operand (`e2 >= bit width of e1`).  The shift amount is a
/// `u32` in WGSL, and `>= 32` covers the ubiquitous 32-bit `e1`.  A 16-bit
/// `e1` with an amount in `[16, 32)` is deliberately NOT caught (the pass's
/// existing rollback still covers that rarer case, fail-safe); a 64-bit `e1`
/// is over-declined harmlessly (one legal forward skipped, never a miscompile).
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

/// Resolve `start` through the forward chain in `replacements` to the literal
/// it would be rewritten to, or `None` when the final target is not a literal.
/// The budget guards against a malformed cycle (chains are acyclic by
/// construction, so it is a safety net, not an expected path).
fn resolve_forward_literal(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    start: naga::Handle<naga::Expression>,
) -> Option<naga::Literal> {
    let mut handle = start;
    let mut budget = expressions.len() + 1;
    while let Some(&next) = replacements.get(&handle) {
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

/// Drop store-to-load forwards that would substitute a constant into the RHS
/// of an integer `/`, `%`, `<<`, or `>>` and thereby turn a legal RUNTIME
/// operation into a const-expression WGSL rejects at shader-creation time
/// (integer divide/modulo by zero; shift amount `>=` the bit width).  Left
/// in place, naga rejects the const form and the post-pass validation rolls
/// the WHOLE module back - discarding every other valid forward the pass made
/// and printing a warning - even though the pre-forward runtime read was
/// valid.  Declining just these forwards keeps that read and lets the rest of
/// the pass stand.
///
/// MUST run before the dead-local scan: dropping a load's replacement keeps
/// the load live, so its store must not be classed dead.  Removal is always
/// the safe direction (a declined forward only ever keeps more state live).
fn decline_static_error_forwards(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
) {
    let mut to_decline: Vec<naga::Handle<naga::Expression>> = Vec::new();
    for (_, expr) in expressions.iter() {
        let naga::Expression::Binary { op, right, .. } = expr else {
            continue;
        };
        // The RHS may be the operand directly, or - for a componentwise
        // `vecN <op> scalar` - a `Splat`/`Compose` naga wraps around the
        // forwarded leaf; recurse to the leaves and decline any that turn a
        // lane into a static error (WGSL componentwise `/` `%` `<<` `>>` make
        // it a shader-creation error if ANY const lane is offending).
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
        replacements.remove(&handle);
    }
}

/// Walk a divisor / shift-amount operand collecting the forward-chain leaves
/// (`replacements` keys) that would resolve to a literal `is_dangerous`
/// accepts.  Recurses `Splat`/`Compose` so a scalar-broadcast divisor
/// (`a / vec3(b)`) and an element-wise one (`a / vec3(x, y, b)`) are both
/// covered.  A non-forwarded literal is never collected - an offending
/// constant already present in the input is naga's to reject, not ours.
fn collect_static_error_leaves<F: Fn(&naga::Literal) -> bool>(
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    handle: naga::Handle<naga::Expression>,
    is_dangerous: &F,
    out: &mut Vec<naga::Handle<naga::Expression>>,
) {
    if replacements.contains_key(&handle) {
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

/// Run phase 2: forward repeated `Load` expressions to the most
/// recent dominating stored value, using a dominance-scoped
/// [`ScopedMap`] to track per-pointer-key cache state across control
/// flow.  Returns `true` when at least one replacement fired.
fn dedup_loads_in_function(function: &mut naga::Function) -> bool {
    let mut replacements = HashMap::new();
    let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
    let mut all_loads: HashMap<
        naga::Handle<naga::LocalVariable>,
        Vec<naga::Handle<naga::Expression>>,
    > = HashMap::new();
    let mut seeded_by_store: HashMap<naga::Handle<naga::Expression>, StoreInfo> = HashMap::new();

    // Seed cache with local variable initializers so that loads
    // following `var x = <const_expr>` can be forwarded to the
    // init expression directly (init is only set for const-evaluatable
    // expressions; runtime inits are lowered to Store statements).
    //
    // Skip Compose inits: after rename, variable references (1-2 chars)
    // are shorter than inline vector/matrix constructors, so forwarding
    // a Compose to multiple Load sites inflates the output.
    for (lh, lvar) in function.local_variables.iter() {
        if let Some(init) = lvar.init
            && !matches!(function.expressions[init], naga::Expression::Compose { .. })
        {
            cache.insert(PointerKey::Local(lh), init);
        }
    }

    // Build the per-function scope index once.  It powers both the
    // O(1) "is handle emitted inside block subtree" probes consumed
    // by `collect_redundant_loads` at every Block/If/Switch scope
    // and the execution-position lookups consumed by the
    // `has_later_live_load` gate further down.  See the doc-comment
    // on `ExpressionScopeIndex` for layout and safety.
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
        &mut HashSet::new(),
    );

    if replacements.is_empty() {
        return false;
    }

    // Decline forwards that would make an integer `/ % << >>` a static
    // shader-creation error.  MUST precede the dead-local scan so a declined
    // load, left live, keeps its backing store from being culled.
    decline_static_error_forwards(&function.expressions, &mut replacements);
    if replacements.is_empty() {
        return false;
    }

    // Identify dead locals: variables whose every live Load was replaced.
    // Exclude variables whose pointer escapes via Call arguments.
    //
    // A replacement `load_handle -> replacement_handle` is only effective when
    // `replacement_handle < load_handle`.  When the replacement target was
    // appended at the end of the expression arena (e.g. by short-circuit
    // re-sugaring), it has a higher index than all existing uses.  The
    // forward-reference guard in the expression-arena walk below will
    // block such replacements, so any local whose loads depend on them
    // must NOT be considered dead - otherwise we remove the Store while the
    // Load remains, leaving the variable uninitialised.
    //
    // We also need the partially-stored-local set a bit further down;
    // computing both in a single traversal halves the tail-end scan
    // cost on large functions.
    let mut escaped: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();
    let mut partially_stored: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();
    collect_escaped_and_partially_stored(
        &function.body,
        &function.expressions,
        &mut escaped,
        &mut partially_stored,
    );

    // A local is "dead" when every Load (whole or partial) has been
    // forwarded with a valid replacement; the surviving Stores are then
    // dead writes whose value no reader observes.  Excluding `escaped`
    // is mandatory (callee can read through the leaked ptr at any
    // time); `partially_stored` does NOT need exclusion because the
    // all-loads-replaced gate already covers every partial Load - any
    // un-forwarded partial Load would block the entry here.
    let dead_locals: HashSet<naga::Handle<naga::LocalVariable>> = all_loads
        .iter()
        .filter(|(lh, loads)| {
            !escaped.contains(lh)
                && loads
                    .iter()
                    .all(|l| replacements.get(l).is_some_and(|&r| r < *l))
        })
        .map(|(lh, _)| *lh)
        .collect();

    // Undo store-to-load forwarding for variables that persist (non-dead)
    // when the forwarded expression is "complex" (needs an Emit and would
    // create a `let` binding if shared).  For dead variables the forwarding
    // is kept because it eliminates the variable entirely, which is a net
    // size win even if the expression becomes shared.
    //
    // "Simple" expressions (Literal, Constant, Load, FunctionArgument, etc.)
    // are cheap to duplicate and never cause let-binding inflation, so they
    // are always forwarded regardless of variable liveness.
    //
    // **Last-store inlining**: when ALL loads seeded by a single Store have
    // valid replacements (target < load), the Store becomes dead even though
    // the variable itself is not dead (earlier loads keep it alive).  In
    // this case we keep the forwarding and record the Store for removal.

    // Group store-seeded loads by their source Store identity (pointer, value).
    // Skip stores inside loops - loop back-edges make it unsafe to remove them.
    let mut loads_per_store: HashMap<StoreId, Vec<naga::Handle<naga::Expression>>> = HashMap::new();
    for (&load_h, &(ptr, val, store_in_loop)) in &seeded_by_store {
        if store_in_loop {
            continue; // Never consider loop-internal Stores for removal
        }
        loads_per_store.entry((ptr, val)).or_default().push(load_h);
    }

    // First pass: identify candidate undo keys (complex forwarding on non-dead locals).
    let undo_candidates: Vec<_> = replacements
        .iter()
        .filter_map(|(&load_h, &replacement_h)| {
            if let naga::Expression::Load { pointer } = &function.expressions[load_h] {
                let local = get_stored_local(&function.expressions, *pointer)?;
                if dead_locals.contains(&local) {
                    return None; // Variable is dead - forwarding is beneficial
                }
                if is_simple_for_forwarding(&function.expressions[replacement_h]) {
                    return None; // Cheap expression - safe to share
                }
                Some(load_h)
            } else {
                None
            }
        })
        .collect();

    // Check which Stores have ALL their seeded loads with valid forward-ref replacements.
    // Those Stores can be removed, so we keep the forwarding.
    // Exclude variables that have partial (field/index) stores - those
    // implicitly read the variable's full value, so the preceding
    // whole-variable Store must be preserved.  (Both `partially_stored`
    // and `escaped` were populated by the combined walk above.)

    let max_live_load_pos = build_max_live_load_positions(&all_loads, &replacements, &scope_idx);
    let mut dead_store_ids: HashSet<StoreId> = HashSet::new();
    for (&store_id, seeded_loads) in &loads_per_store {
        let (store_ptr, _) = store_id;
        // The store must be a whole-variable store (LocalVariable pointer).
        if !matches!(
            function.expressions[store_ptr],
            naga::Expression::LocalVariable(_)
        ) {
            continue;
        }
        // The variable must not have partial stores.
        let local = match get_stored_local(&function.expressions, store_ptr) {
            Some(l) => l,
            None => continue,
        };
        if partially_stored.contains(&local) {
            continue;
        }
        if escaped.contains(&local) {
            continue;
        }
        let all_valid = seeded_loads
            .iter()
            .all(|&load_h| replacements.get(&load_h).is_some_and(|&r| r < load_h));
        if !all_valid {
            continue;
        }
        // Guard: a Store is only dead when no un-replaced Load of the
        // variable appears at a later statement position.  Such a
        // Load could observe this Store through paths not captured by
        // the seeded-load cache (e.g. a loop body's Load reaching the
        // cleared cache).
        //
        // INVARIANT (paired with the cache-promotion gate at
        // `Emit(Load)` in `collect_redundant_loads`): when the cache's
        // canonical for a key IS a `Load`, the promotion must NOT
        // rebind to the newer Load handle.  Producer-Loads must stay
        // out of `replacements` so their Stores stay alive against
        // this guard; rebinding would drag them in and silently mark
        // the producer Store dead, re-introducing the meet-over-branches
        // miscompile this guard prevents.
        //
        // Two orderings combine:
        // 1. `load_pos > store_pos` (statement DFS positions, source
        //    of truth for execution order; arena order is unreliable
        //    after `inlining` appends to the tail).
        // 2. `r >= load_h` (arena index): an ineffective replacement
        //    - the arena-walk apply step uses `r < handle` to block
        //    forward references, so any `r >= load_h` leaves the Load
        //    live regardless of program position.  Arena order is
        //    correct for this comparison because the apply step
        //    iterates in that order.
        //
        // Defensive: if the Store's position is missing (its
        // `(ptr, val)` pair wasn't seen by the index build), keep it.
        let Some(store_pos) = scope_idx.store_position(store_id) else {
            continue;
        };
        let has_later_live_load = max_live_load_pos
            .get(&local)
            .is_some_and(|&max_pos| max_pos > store_pos);
        if has_later_live_load {
            continue;
        }
        dead_store_ids.insert(store_id);
    }

    // Undo: only undo candidates whose Store is NOT fully dead.
    for h in undo_candidates {
        if let Some(&(ptr, val, _in_loop)) = seeded_by_store.get(&h)
            && dead_store_ids.contains(&(ptr, val))
        {
            continue; // Store is dead - keep forwarding
        }
        replacements.remove(&h);
    }

    // `dead_store_ids` was computed against the PRE-undo replacements.  The undo
    // loop just removed replacements for non-dead complex forwards, so a Store
    // counted dead can regain a load whose forwarding was undone and is now
    // live.  Removing that Store while the load survives makes the load read the
    // variable's zero-init (a silent miscompile).  Re-run the later-live-load
    // guard against the post-undo replacements - undoing only ever resurrects
    // loads, so the max must be rebuilt - and drop any Store that is no longer
    // dead.
    let max_live_load_pos = build_max_live_load_positions(&all_loads, &replacements, &scope_idx);
    dead_store_ids.retain(|&store_id| {
        let (store_ptr, _) = store_id;
        // On any lookup miss, conservatively KEEP the Store (drop it from the
        // dead set, i.e. `retain` returns false) - matching the `continue` in the
        // guard that populated the set.  Removing a Store we can no longer analyze
        // is the unsafe direction (a surviving load would read the zero-init).
        let Some(local) = get_stored_local(&function.expressions, store_ptr) else {
            return false;
        };
        let Some(store_pos) = scope_idx.store_position(store_id) else {
            return false;
        };
        let regained_live_load = max_live_load_pos
            .get(&local)
            .is_some_and(|&max_pos| max_pos > store_pos);
        !regained_live_load
    });

    // The collection/undo phases are complete; release the immutable borrow on
    // `function.body` so `apply_to_block` below can take its `&mut`.
    drop(scope_idx);

    // After undoing complex forwarding for non-dead variables, all
    // detected store-to-load opportunities may have been reverted.
    // Without this check the pass falsely reports `changed = true`
    // every sweep, preventing pipeline convergence.
    if replacements.is_empty() {
        return false;
    }

    // Resolve replacement chains: Load_a -> Load_b -> expr chains can arise
    // when a store-forwarded entry's Load is used as a canonical for later
    // Load-to-Load dedup.  The single-pass apply below only resolves one
    // level, so we flatten chains here to avoid dangling references.
    flatten_replacement_chains(&mut replacements);

    // Keep only replacements that take effect at EVERY use.  The arena walk
    // below rewrites a load to its target only where `target < use`, and a use
    // always outranks the load it reads, so `target < load` is exactly the
    // condition for the rewrite to reach every use.  A target that does not
    // precede the load - e.g. an init expression `inlining` appended to the
    // arena after the load - applies nowhere; retaining it would drop the
    // load's `let` while its uses still read the bare, possibly since-mutated
    // load (a miscompile when a use sits past a write to the place, e.g. a
    // loop `break_if`), and would also make this pass report a change every
    // sweep and never converge.  This is the same effectiveness test
    // `dead_locals` and `dead_store_ids` already apply.
    replacements.retain(|load, target| *target < *load);
    if replacements.is_empty() {
        return false;
    }

    // Apply replacements to all expression children in the arena.
    // Guard: skip a replacement if the target handle is >= the current
    // expression's handle, which would create an illegal forward reference.
    // The dead_locals computation above mirrors this guard so that locals
    // whose replacements would be blocked are never marked dead.
    for (handle, expr) in function.expressions.iter_mut() {
        let _ = try_map_expression_handles_in_place(expr, &mut |h| match replacements.get(&h) {
            Some(&r) if r < handle => Some(r),
            _ => Some(h),
        });
    }

    // Clear init on dead locals so the generator doesn't emit
    // a dangling declaration for a fully-eliminated variable.
    for &lh in &dead_locals {
        function.local_variables[lh].init = None;
    }

    // Apply replacements to statement handles, fix Emit ranges,
    // and remove dead stores.
    apply_to_block(
        &mut function.body,
        &replacements,
        &dead_locals,
        &dead_store_ids,
        &function.expressions,
        false,
    );

    // Remove named expressions that point to replaced handles.
    function
        .named_expressions
        .retain(|h, _| !replacements.contains_key(h));

    true
}

/// Lower a pointer expression to a [`PointerKey`] by walking through
/// `AccessIndex` and `Access` wrappers.  Returns `None` for pointers
/// whose root is not a local (function-argument pointers, globals,
/// etc.); such pointers are excluded from forwarding.
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

/// Maximum statement-DFS position of a LIVE load, per local.  A Store at
/// `store_pos` has a later live load iff this maximum exceeds `store_pos`,
/// so one O(loads) build answers every store's dead-store guard in O(1)
/// where the previous per-store scan over all of the local's loads was
/// O(stores x loads) - quadratic on machine-generated single-accumulator
/// functions.  "Live" mirrors the guard exactly: a load is live unless its
/// replacement precedes it in arena order (`r < load_h`; anything else
/// leaves the Load in place because the apply walk blocks forward
/// references).  Positions missing from the index contribute nothing,
/// matching the old scan's `is_some_and`.
fn build_max_live_load_positions(
    all_loads: &HashMap<naga::Handle<naga::LocalVariable>, Vec<naga::Handle<naga::Expression>>>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    scope_idx: &ExpressionScopeIndex<'_>,
) -> HashMap<naga::Handle<naga::LocalVariable>, u32> {
    let mut max_pos = HashMap::with_capacity(all_loads.len());
    for (&local, loads) in all_loads {
        for &load_h in loads {
            let live = replacements.get(&load_h).is_none_or(|&r| r >= load_h);
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

/// Resolve `pointer` to the root local it ultimately refers to, or
/// `None` when the root is not a local.  Re-exported so `dead_branch`
/// agrees with this pass on the "stored the whole local" predicate.
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

/// `true` when `expr` is cheap to duplicate across multiple reference
/// sites: declarative expressions (Literal / Constant / Override /
/// ZeroValue / FunctionArgument / GlobalVariable / LocalVariable)
/// and `Load`.
///
/// The declarative half never creates a `let` binding in the
/// generator regardless of reference count.  `Load` is included
/// because the surrounding undo-keep mechanism here is concerned
/// with output size rather than re-execution: a `Load` forwarded
/// to N sites is N references to the same already-Emit'd Load
/// handle, not N new memory reads - it would only become N reads if
/// the undo-keep logic eliminated the original Load's Emit, which it
/// does not.  (Contrast with `const_fold::is_pure_to_clone` which
/// rejects `Load` because the const-fold rewrite clones EXPRESSION
/// CONTENT into a sibling arena slot, producing a second `Emit` for
/// the same Load and thus a second memory access.)
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

/// Ceiling on the effective (post-forwarding) expression-tree depth a
/// store-seeded cache entry may have.  Store-forwarding a value that
/// itself absorbed the previous store's tree grows the IR without bound on
/// flat reassignment chains (`a = a*2.0+1.0;` x9000 -> an 18k-deep tree),
/// which overflows every RECURSIVE consumer downstream: `render_depth`'s
/// own measurement, naga's writer, and wasm's ~1 MB caller stack that the
/// CLI's big-stack worker does not cover - and the deep tree ships in the
/// OUTPUT, so every re-minification pays it again.  128 is far above any
/// human-written statement's depth and far below the shallowest consumer
/// stack budget.
const SUBSTITUTION_DEPTH_CAP: u32 = 128;

/// Approximate effective depth of `value`'s expression tree, reading
/// through recorded load `replacements` (a forwarded `Load` counts as its
/// replacement's tree, an un-forwarded one as its POINTER chain - subscript
/// expressions are real rendered tree content and can absorb a previous
/// iteration's forwarded tree, e.g. `x = a[x & 3] + 1;` chains).
/// Level-order (BFS) walk, saturating at `cap + 1`.  The shared visited set
/// makes this the SHORTEST-path depth, so a node reachable via both a short
/// and a long path is under-counted versus the longest-path depth the
/// recursive consumers (`render_depth`, naga's writer) actually stack; the
/// gap is bounded by the `budget` node cap, not one diamond.  This pass never
/// synthesizes such sharing (forwarding only substitutes handles into existing
/// nodes, and re-absorption chains grow both paths in lockstep), and the naga
/// parser recursion limit caps any pre-existing input, so the under-count is
/// unreachable here in practice.
fn effective_forwarded_depth(
    value: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    cap: u32,
) -> u32 {
    let mut frontier = vec![value];
    let mut next = Vec::new();
    let mut seen = HashSet::new();
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
                    if let Some(&replacement) = replacements.get(&handle) {
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

/// Populate `escaped` (locals whose pointer is passed to a callee) and
/// `partially_stored` (locals receiving field- or index-level writes)
/// in a single statement-tree walk.  Locals in either set are
/// disqualified from full-variable forwarding because a later access
/// could observe an unmodelled side-effect.
fn collect_escaped_and_partially_stored(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    escaped: &mut HashSet<naga::Handle<naga::LocalVariable>>,
    partially_stored: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Store { pointer, .. } => {
                if !matches!(expressions[*pointer], naga::Expression::LocalVariable(_))
                    && let Some(local) = get_stored_local(expressions, *pointer)
                {
                    partially_stored.insert(local);
                }
            }
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
                // `data.pointer` is the destination of the matrix write
                // (validator requires STORE access).  When it roots at
                // a function-local, the write is partial in general
                // (matrix may not cover the local's full type), so the
                // local must be classified as partially-stored: any
                // preceding whole-variable Store has to be preserved.
                // `target` is the source matrix VALUE; reads are
                // tracked elsewhere via Emit/Load walking.
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    partially_stored.insert(local);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                if let Some(local) = get_stored_local(expressions, *query) {
                    escaped.insert(local);
                }
            }
            // Statements that do not escape a local through a pointer -
            // enumerated explicitly so a future naga release adding a
            // new pointer-bearing variant breaks the build here instead
            // of silently leaving a local mis-classified as un-escaped.
            // Nested blocks recurse below.
            naga::Statement::Emit(_)
            | naga::Statement::Block(_)
            | naga::Statement::If { .. }
            | naga::Statement::Switch { .. }
            | naga::Statement::Loop { .. }
            | naga::Statement::Atomic { .. }
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
        for nested in nested_blocks(stmt) {
            collect_escaped_and_partially_stored(nested, expressions, escaped, partially_stored);
        }
    }
}

/// Count how many `Store` statements target each local across the
/// entire function body.  Test-only helper - the production
/// last-store-inlining path drives off `seeded_by_store` /
/// `loads_per_store` from `dedup_loads_in_function`, not this count.
/// Kept for direct assertions in `tests::*` cases that need to
/// verify the post-pass IR's Store distribution.
#[cfg(test)]
fn count_local_stores(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> HashMap<naga::Handle<naga::LocalVariable>, usize> {
    let mut counts = HashMap::new();
    count_stores_recursive(block, expressions, &mut counts);
    counts
}

#[cfg(test)]
fn count_stores_recursive(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    counts: &mut HashMap<naga::Handle<naga::LocalVariable>, usize>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Store { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    *counts.entry(lh).or_insert(0) += 1;
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                // Atomics perform a read-modify-write - count as a store.
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    *counts.entry(lh).or_insert(0) += 1;
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // RayQuery mutates state through the query pointer
                // (Initialize / Proceed) - count as a store.
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

/// Core dominance-aware walker.  Identifies redundant `Load`
/// expressions that can be forwarded to an earlier `Load` from the
/// same memory location, and populates `replacements` with the
/// `(duplicate_load_handle -> replacement_handle)` map by threading a
/// `ScopedMap<PointerKey, cached_value>` through the statement tree,
/// honouring branch dominance and invalidating entries on aliasing
/// stores, calls, atomics, and similar.  Loop handlers drain the
/// cache before the body runs because the iteration count is unknown
/// at pass time.
///
/// `seeded_by_store` records which Load handles received store-forwarded
/// replacements.  The key is the Load handle; the value is the Store's
/// `(pointer, value)` identity that seeded the cache entry.  This lets the
/// undo phase decide per-Store whether all seeded loads are covered,
/// enabling "last-store inlining".
///
/// `scope_idx` answers "is handle `h` emitted inside block `B`'s
/// subtree" in O(1) at every Block / If / Switch scope-carry filter.
/// See [`ExpressionScopeIndex`] for the layout and safety story.
///
/// ## Scope-state persistence
///
/// The forwarding `cache` is a [`ScopedMap`] driven by checkpoint +
/// rollback at every control-flow boundary (If / Switch / Loop).  This
/// matches the policy used by the `cse` and `dead_branch` passes and
/// replaces the earlier `HashMap::clone` per branch: each scope now costs
/// O(in-scope writes) instead of O(entire table).
///
/// Per-branch semantics are preserved by:
/// 1. Entering the branch at the current `cache` state (checkpoint captured).
/// 2. Recursing into the branch - writes and invalidations are logged.
/// 3. Rolling back to the checkpoint, restoring pre-branch `cache`.
/// 4. After all branches of a control-flow construct have been processed,
///    permanently invalidating entries for any local written in *any*
///    branch (via `invalidate_cache_for_local`, which itself logs so an
///    outer scope can undo it on rollback).
///
/// For `Loop`, body and continuing each see a *fresh empty* `cache`
/// (matching the original semantics: an iteration cannot rely on
/// values forwarded before the loop, and post-loop code cannot rely on
/// values forwarded inside).  This is implemented by draining the map
/// before recursion and rolling back to the empty state between body and
/// continuing; the pre-loop state is restored by rolling further back
/// to the outer checkpoint.
///
/// `modified_out` is populated inline with every local variable
/// potentially mutated within `block` (or any of its recursive children).
/// Callers in `If` / `Switch` / `Loop` arms allocate a fresh
/// `HashSet` per scope, pass it down to the recursive call, and union
/// the resulting per-scope set with their own `modified_out` and use
/// it locally for the conservative invalidation step - replacing the
/// previous separate full-tree walks via `collect_modified_locals`
/// with a single combined pass per branch.
#[allow(clippy::too_many_arguments)]
fn collect_redundant_loads<'body>(
    block: &'body naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    scope_idx: &ExpressionScopeIndex<'body>,
    cache: &mut ScopedMap<PointerKey, naga::Handle<naga::Expression>>,
    replacements: &mut HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    all_loads: &mut HashMap<naga::Handle<naga::LocalVariable>, Vec<naga::Handle<naga::Expression>>>,
    seeded_by_store: &mut HashMap<naga::Handle<naga::Expression>, StoreInfo>,
    in_loop: bool,
    modified_out: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    // Track which cache entries were seeded by a Store (not by a Load or init).
    // Maps PointerKey -> Store's (pointer, value, in_loop) identity.  This is
    // block-local scratch - it is reset on every `collect_redundant_loads`
    // call and never crosses branch boundaries, so it stays a plain HashMap.
    let mut store_source: HashMap<PointerKey, StoreInfo> = HashMap::new();
    for statement in block {
        match statement {
            naga::Statement::Emit(range) => {
                for handle in range.clone() {
                    let naga::Expression::Load { pointer } = &expressions[handle] else {
                        continue;
                    };
                    // Track EVERY load rooted at a local for the liveness
                    // sets - independent of `get_pointer_key`.  A depth>=2
                    // nested chain (`e.a.x`) is not forwardable (its base is
                    // itself an Access), so `get_pointer_key` returns `None`;
                    // but `get_stored_local` still resolves the root local.
                    // Recording it keeps the local out of `dead_locals` (its
                    // load is never replaced) and keeps its backing Store out
                    // of `dead_store_ids`, so the surviving nested load is not
                    // left reading the zero-default after the Store is dropped.
                    if let Some(local) = get_stored_local(expressions, *pointer) {
                        all_loads.entry(local).or_default().push(handle);
                    }
                    // Forwarding / dedup only applies to the depth-1 pointer
                    // shapes `get_pointer_key` can describe as a cache key.
                    if let Some(key) = get_pointer_key(expressions, *pointer) {
                        if let Some(&canonical) = cache.get(&key) {
                            replacements.insert(handle, canonical);
                            // Track whether this replacement was seeded by a Store.
                            if let Some(&store_id) = store_source.get(&key) {
                                seeded_by_store.insert(handle, store_id);
                            }
                            // Cache-promotion gate.  When the canonical
                            // came from store-seeding or an init,
                            // re-bind to this Load handle so subsequent
                            // Load-to-Load forwarding chains through a
                            // surviving Emit.
                            //
                            // INVARIANT (paired with `has_later_live_load`
                            // in `dedup_loads_in_function`): when the
                            // canonical IS already a `Load`, do NOT
                            // rebind - the producer-Load's handle must
                            // stay in the cache so it remains out of
                            // `replacements`, keeping its producing
                            // Store alive against the dead-store check.
                            // Rebinding would drag the producer-Load
                            // into `replacements` and silently kill
                            // perfectly-live Stores across loop / branch
                            // boundaries where the cache was cleared.
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
                    // Invalidate ALL cache entries involving this local.
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                    // Seed cache with the stored value so subsequent loads
                    // can be forwarded to the value expression directly.
                    // The undo phase in dedup_loads_in_function will later
                    // revert forwarding for non-dead variables when the
                    // forwarded expression is complex (would inflate output).
                    //
                    // Depth gate: an effectively-deep value stays UNSEEDED
                    // (the invalidation above already removed the stale
                    // entry, so later loads simply keep the variable read) -
                    // otherwise flat reassignment chains grow the tree
                    // without bound; see `SUBSTITUTION_DEPTH_CAP`.
                    if let Some(key) = get_pointer_key(expressions, *pointer) {
                        let mut resolved = *value;
                        while let Some(&next) = replacements.get(&resolved) {
                            resolved = next;
                        }
                        if effective_forwarded_depth(
                            resolved,
                            expressions,
                            replacements,
                            SUBSTITUTION_DEPTH_CAP,
                        ) <= SUBSTITUTION_DEPTH_CAP
                        {
                            cache.insert(key.clone(), resolved);
                            store_source.insert(key, (*pointer, *value, in_loop));
                        }
                    }
                }
            }
            naga::Statement::Block(inner) => {
                // `Statement::Block` introduces a nested WGSL `{ ... }`
                // scope.  Two correctness concerns:
                //
                // 1. **Scope leak.**  Any expression bound by an
                //    `Emit` (or statement-bound result handle) inside
                //    `inner` is `let`-bound inside that scope by
                //    naga's WGSL writer and is out of lexical scope
                //    outside the closing brace.  Forwarding a
                //    post-block read to such a handle would emit
                //    `let _outer = _inner;` where `_inner` is out of
                //    scope - the backend rejects the round-trip and
                //    the pass silently rolls back.  Solved by the
                //    `is_in_subtree` filter on `carry` below.  Pre-emit
                //    values (`needs_pre_emit`: Literal, Constant,
                //    Override, ZeroValue, FunctionArgument,
                //    GlobalVariable, LocalVariable) are exempted -
                //    naga considers them in scope everywhere.
                //
                // 2. **Stale-init carry.**  Rolling the cache back to
                //    the pre-block checkpoint restores any entry that
                //    existed before the block, including entries
                //    seeded by an `init` value (e.g.
                //    `Local(F) -> Literal(false)` from `var F = false`).
                //    If `inner` writes to F, the pre-block init is
                //    stale post-block: a subsequent `Load(F)` forwarded
                //    to `Literal(false)` produces output that
                //    overwrites F's accumulated state (e.g.
                //    `F = false | X` instead of the source's
                //    `F |= X`, dropping every earlier accumulation).
                //    Solved by capturing modified-locals into a fresh
                //    `block_modified` set and invalidating cache entries
                //    for those locals AFTER rollback - mirrors the
                //    If/Switch/Loop arms' invalidation step.
                //
                // `scope_idx.is_in_subtree` is O(1) per probe.  An
                // arena-index highwater (`v.index() < arena_len_pre`)
                // would be sound but a no-op: this pass never appends
                // to the expression arena, so every handle trivially
                // satisfies it.  Position-based membership is the only
                // working check.
                let cp_pre_block = cache.checkpoint();

                // Fresh modified-locals set for THIS block, separate
                // from `modified_out` so the invalidation step below
                // only drops cache entries written inside the block
                // (not entries written by sibling statements that
                // share `modified_out` with us).
                let mut block_modified = HashSet::new();
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

                // Drop pre-block entries for any local written inside
                // `inner`.  Without this, an init-seeded entry (e.g.
                // `Local(F) -> Literal(false)`) would survive the
                // rollback and incorrectly forward a post-block
                // `Load(F)` to the init value, dropping F's
                // accumulated runtime state.
                for local in &block_modified {
                    invalidate_cache_for_local(cache, *local);
                }
                for (k, v) in carry {
                    cache.insert(k, v);
                }

                // Propagate block's writes up to the enclosing scope.
                modified_out.extend(block_modified);
            }
            naga::Statement::If { accept, reject, .. } => {
                // Snapshot pre-if state; each branch is explored and then
                // rolled back so the outer scope sees only the permanent
                // invalidations we apply after both branches finish.
                let cp_pre_if = cache.checkpoint();

                // Filter at the meet step: forwarding to a handle
                // emitted only inside a branch is unsound post-if -
                // naga's WGSL writer `let`-binds inside the branch's
                // brace, so the binding is out of scope on the other
                // path and after the if.  `needs_pre_emit` expressions
                // (Literal, Constant, Override, ZeroValue,
                // FunctionArgument, GlobalVariable, LocalVariable)
                // are not bound by Emit and remain in scope
                // everywhere, so they bypass the filter at the use
                // site.  Same arena-highwater reasoning as the
                // `Block` arm above: position-based membership is
                // the only check that actually fires.
                let in_branches = |v: naga::Handle<naga::Expression>| {
                    scope_idx.is_in_subtree(accept, v) || scope_idx.is_in_subtree(reject, v)
                };

                // Each branch's recursion populates its own modified set.
                // Combining them upfront via separate `collect_modified_locals`
                // walks would cost an extra full traversal per branch; the
                // inline collection lets the meet capture / invalidation
                // share the same single walk.
                let mut accept_modified = HashSet::new();
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
                // Capture accept's final-cache snapshot restricted to
                // keys whose local is modified by `accept`.  This is a
                // slight under-restriction relative to the previous
                // policy (which used the union of both branches'
                // modified sets); the only observable difference is
                // for keys unmodified in `accept` but re-stored to the
                // *same* pre-if value in `reject`, which previously
                // would survive via the meet but now drop.  That edge
                // case is rare and never affects correctness - the
                // entry simply isn't carried forward, so subsequent
                // loads cache-miss instead of being forwarded.  The
                // size win from skipping a full-tree walk per branch
                // is judged worth the trade.
                let mut meet: HashMap<PointerKey, naga::Handle<naga::Expression>> = cache
                    .as_map()
                    .iter()
                    .filter(|(k, _)| pointer_key_involves_any_local(k, &accept_modified))
                    .filter(|(_, v)| !in_branches(**v) || expressions[**v].needs_pre_emit())
                    .map(|(k, &v)| (k.clone(), v))
                    .collect();
                cache.rollback_to(cp_pre_if);

                let mut reject_modified = HashSet::new();
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
                // Running intersection: drop any candidate that reject's
                // final cache disagrees with.  Reads against the live
                // cache are O(1); no clone of reject's state is needed.
                meet.retain(|k, v| cache.get(k) == Some(v));
                cache.rollback_to(cp_pre_if);

                // Conservatively drop entries for any local written in
                // either branch.  Removals are logged so an outer-scope
                // rollback can undo them.  Use `union` rather than
                // `chain` so a local modified in both branches isn't
                // invalidated twice (each call is a full cache scan).
                for local in accept_modified.union(&reject_modified) {
                    invalidate_cache_for_local(cache, *local);
                }

                // Meet-over-branches: an `if/else` is total - exactly one
                // branch executes - so any forwarding present with the
                // *same* canonical handle at the end of both branches is
                // safe post-if.
                //
                // Early-exit soundness: a branch that terminates early
                // (Return / Break / Continue / Kill / ControlBarrier
                // followed by no writes, etc.) may leave its final
                // snapshot holding forwarding entries that are never
                // actually reached post-if.  That is still sound: the
                // meet can only *introduce* an entry when both
                // snapshots agree on a canonical handle, and the
                // resulting entry is only read on control-flow paths
                // that fall through the entire if.  Paths that exit
                // the function / loop from inside a branch never read
                // the meet entry, so false positives in a terminating
                // branch's snapshot are unobservable.
                for (k, v) in meet {
                    cache.insert(k, v);
                }

                // Propagate the union of both branches to the caller's
                // `modified_out` so an enclosing scope can apply its
                // own invalidation step.
                modified_out.extend(accept_modified);
                modified_out.extend(reject_modified);
            }
            naga::Statement::Switch { cases, .. } => {
                let cp_pre_switch = cache.checkpoint();
                // Forwarding to a handle emitted only inside a case
                // is unsound post-switch because naga's WGSL writer
                // `let`-binds inside the case's brace.  Per-case O(1)
                // membership against `scope_idx`; total per-probe cost
                // is O(K) where K = number of cases.  Same scope and
                // arena-highwater reasoning as the `If` arm above.
                let in_cases = |v: naga::Handle<naga::Expression>| {
                    cases
                        .iter()
                        .any(|case| scope_idx.is_in_subtree(&case.body, v))
                };

                // Meet-over-branches is only sound for switches that
                // execute *exactly one* case on every path AND whose only
                // route to post-switch code is falling off a case's end.
                // That requires (a) a `Default` case so the switch is
                // total, (b) no `fall_through` between cases (otherwise a
                // case's "final state" includes later cases' writes and
                // the pre-meet snapshot is meaningless), and (c) no bare
                // `break` in any case: a switch-binding break reaches
                // post-switch code carrying the PRE-break state - e.g. the
                // SPIR-V-structurizer phi idiom
                // `switch(0u){default:{ if(c){x=1; break;} x=2; }}` exits
                // with x=1 on the break path, which the fall-end meet
                // (x=2) never sees; forwarding it deletes live stores.
                let has_default = cases
                    .iter()
                    .any(|c| matches!(c.value, naga::SwitchValue::Default));
                let any_fallthrough = cases.iter().any(|c| c.fall_through);
                let any_switch_break = cases
                    .iter()
                    .any(|c| crate::passes::dead_branch::contains_bare_break(&c.body));
                let meet_applicable = has_default && !any_fallthrough && !any_switch_break;

                let mut total_modified: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();
                let mut meet: Option<HashMap<PointerKey, naga::Handle<naga::Expression>>> = None;
                for case in cases {
                    let mut case_modified = HashSet::new();
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
                                // First case: capture entries whose
                                // local is modified by this case
                                // (analogous under-restriction to the
                                // `If` arm; later cases narrow further
                                // via in-place retain).  Filter on
                                // the in-cases scope + `needs_pre_emit`
                                // gate so the meet only carries values
                                // reachable on every post-switch path.
                                let initial: HashMap<_, _> = cache
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
                                // Subsequent cases: intersect in place.
                                // The scope filter is omitted here
                                // because every entry surviving the
                                // first-case filter is already known
                                // not to live inside any case body, and
                                // case-subtree membership is a function
                                // of the cases collection alone (the
                                // `scope_idx` is immutable across the
                                // walk) - it cannot change per case.
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
                    // With bare breaks excluded by `meet_applicable`, the
                    // only early exits left in a case are Return / Kill
                    // (leave the function) and Continue (jumps to an
                    // enclosing loop's continuing block) - none reach
                    // post-switch code, so every path that DOES reach it
                    // fell off a case end and agrees with the meet.
                    for (k, v) in m {
                        cache.insert(k, v);
                    }
                }

                modified_out.extend(total_modified);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Loop body can execute multiple times; body + continuing
                // must each see a fresh empty cache.  Drain pre-loop
                // entries (logged), then recurse with an empty map, then
                // rollback to restore pre-loop state, then permanently
                // invalidate modified locals.
                let cp_pre_loop = cache.checkpoint();
                cache.drain_logged();
                let cp_empty = cache.checkpoint();

                let mut loop_modified = HashSet::new();
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
            // Pointer-bearing statements that may mutate a local
            // through an aliased pointer.  Each arm invalidates BOTH
            // `cache` AND `store_source` for the affected local; the
            // two must stay in lockstep because a stale `store_source`
            // entry left after a cache miss would mis-classify the
            // producing Store as having no surviving forwarded loads.
            naga::Statement::Call { arguments, .. } => {
                // Invalidate cache entries for any local whose pointer is
                // passed as an argument - the callee may write through it.
                for &arg in arguments {
                    if let Some(local) = get_stored_local(expressions, arg) {
                        modified_out.insert(local);
                        invalidate_cache_for_local(cache, local);
                        invalidate_store_source_for_local(&mut store_source, local);
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                // TraceRay passes a payload pointer that the callee can
                // read/write through - invalidate cache for that local.
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = get_stored_local(expressions, *payload) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                }
            }
            naga::Statement::CooperativeStore { data, .. } => {
                // CooperativeStore writes the matrix value (carried in
                // the unused `target` field) through `data.pointer`.
                // When `data.pointer` roots at a function-local, that
                // local's slot is mutated and must be invalidated;
                // `target` is the source read and is already tracked
                // via the Emit/Load walking.
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                // Atomic performs a read-modify-write - invalidate cache
                // for the target local to prevent stale load dedup.
                if let Some(local) = get_stored_local(expressions, *pointer) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // `query` is the only function-local pointer this
                // statement carries (typed `ptr<function, ray_query>`).
                // The `fun`-level operands (acceleration_structure,
                // descriptor, hit_t) are typed values, not pointers;
                // any `Load` they wrap is caught by the `Emit` arm
                // above.
                if let Some(local) = get_stored_local(expressions, *query) {
                    modified_out.insert(local);
                    invalidate_cache_for_local(cache, local);
                    invalidate_store_source_for_local(&mut store_source, local);
                }
            }
            // Statements that do not touch function-local pointers and
            // do not contain nested blocks - enumerated explicitly
            // (no `_ => {}`) so a future naga release adding a new
            // pointer-bearing statement breaks the build here instead
            // of silently bypassing this walker's load-forwarding /
            // dead-store analysis.
            naga::Statement::Break
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
}

/// Remove every `cache` entry whose `PointerKey` names `local`, logging
/// each removal in the `ScopedMap` undo log so the enclosing scope can
/// roll back if needed.
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

/// Remove every `store_source` entry whose `PointerKey` names `local`.
/// Mirrors [`invalidate_cache_for_local`]'s predicate so the two
/// invalidation steps stay in lockstep; the same five statement arms
/// (Call / Atomic / RayPipelineFunction / RayQuery / CooperativeStore)
/// that clear `cache` for a touched local also clear `store_source`.
/// If a new `PointerKey` variant is added, the exhaustive `match` here
/// (and in `invalidate_cache_for_local` /
/// `pointer_key_involves_any_local`) breaks the build, forcing every
/// site to be updated in lockstep.
fn invalidate_store_source_for_local(
    store_source: &mut HashMap<PointerKey, StoreInfo>,
    local: naga::Handle<naga::LocalVariable>,
) {
    store_source.retain(|key, _| match key {
        PointerKey::Local(l) | PointerKey::LocalField(l, _) | PointerKey::LocalDynamic(l, _) => {
            *l != local
        }
    });
}

/// Return `true` when a cache key names any local in `locals`.
fn pointer_key_involves_any_local(
    key: &PointerKey,
    locals: &HashSet<naga::Handle<naga::LocalVariable>>,
) -> bool {
    match key {
        PointerKey::Local(l) | PointerKey::LocalField(l, _) | PointerKey::LocalDynamic(l, _) => {
            locals.contains(l)
        }
    }
}

/// Collect local variables whose pointer is passed as a Call argument,
/// meaning the callee can read/write through it.
///
/// Kept as a stand-alone helper for tests that assert the escape set in
/// isolation; production callers inside this module use
/// [`collect_escaped_and_partially_stored`] to fold this scan with the
/// partial-store discovery, avoiding a second full walk of the
/// statement tree.
#[cfg(test)]
fn locals_passed_by_pointer(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
) -> HashSet<naga::Handle<naga::LocalVariable>> {
    let mut escaped = HashSet::new();
    collect_escaped_locals(block, expressions, &mut escaped);
    escaped
}

#[cfg(test)]
fn collect_escaped_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    escaped: &mut HashSet<naga::Handle<naga::LocalVariable>>,
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
                // Test-only mirror of the production walker:
                // `data.pointer` is the write side.
                if let Some(local) = get_stored_local(expressions, data.pointer) {
                    escaped.insert(local);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // The runtime mutates state through the query pointer.
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

/// Every local whose value can be mutated by the statements inside
/// `block` (recursively).  Shared with `dead_branch`; both passes rely
/// on the same "might change" predicate to invalidate cached or
/// hoisted load values across control-flow joins.
pub(crate) fn collect_modified_locals(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    modified: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Store { pointer, .. } | naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    modified.insert(lh);
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = get_stored_local(expressions, arg) {
                        modified.insert(lh);
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(lh) = get_stored_local(expressions, *payload) {
                    modified.insert(lh);
                }
            }
            naga::Statement::CooperativeStore { data, .. } => {
                // Write side is `data.pointer`; `target` is the matrix
                // VALUE (a CooperativeMatrix-typed read), so it cannot
                // root at a LocalVariable per naga's validator.
                if let Some(lh) = get_stored_local(expressions, data.pointer) {
                    modified.insert(lh);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                if let Some(lh) = get_stored_local(expressions, *query) {
                    modified.insert(lh);
                }
            }
            // Statements that do not modify a function-local through a
            // pointer - enumerated explicitly so a future naga release
            // adding a new pointer-bearing variant breaks the build
            // here instead of silently leaving a local out of the
            // modified set (and thereby letting the caller forward a
            // stale value).  Nested blocks recurse below.
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
        for nested in nested_blocks(stmt) {
            collect_modified_locals(nested, expressions, modified);
        }
    }
}

// MARK: Replacement application

/// Apply the redundancy map to every statement: remap statement-level
/// handles onto their replacements, rebuild `Emit` ranges around the
/// surviving handles, and drop any `Store` that targets a local in
/// `dead_locals` or whose `(pointer, value)` pair is in
/// `dead_store_ids`.  Mirrors the fused traversal in
/// `cse::apply_and_rebuild`.
fn apply_to_block(
    block: &mut naga::Block,
    replacements: &HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Expression>>,
    dead_locals: &HashSet<naga::Handle<naga::LocalVariable>>,
    dead_store_ids: &HashSet<StoreId>,
    expressions: &naga::Arena<naga::Expression>,
    // `true` once recursion has entered a `Loop` body/continuing.  `dead_store_ids`
    // keys a dead Store by `(pointer, value)` identity alone, and a loop-internal
    // Store can share that identity with the out-of-loop Store the inlining path
    // retired (e.g. `x = e` written both before and inside a loop).  Such in-loop
    // Stores are never removal candidates, so one matching a dead id is a
    // collision victim the back edge still reads: never remove it by identity.
    in_loop: bool,
) {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());

    for (mut statement, span) in original.span_into_iter() {
        match &statement {
            naga::Statement::Emit(range) => {
                // Every entry in `replacements` is effective (target precedes
                // the load; see the retain in `dedup_loads_in_function`), so a
                // load that is a replacement key is rewritten at all its uses
                // and its binding can be dropped.
                let surviving: Vec<_> = range
                    .clone()
                    .filter(|h| !replacements.contains_key(h))
                    .collect();

                if surviving.is_empty() {
                    continue;
                }

                // Split into contiguous sub-ranges.
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
                // Skip stores to dead locals (all loads already replaced).
                if let Some(lh) = get_stored_local(expressions, *pointer)
                    && dead_locals.contains(&lh)
                {
                    continue;
                }
                // Remove a dead Store only outside loops; inside, its
                // (pointer, value) identity can collide with a live loop-carried
                // Store (see the `in_loop` param doc).
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
            replacements.get(&h).copied().unwrap_or(h)
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
