//! Load / store dataflow cleanup.  Three phases run per function on
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
//! 3. **Dead-init removal.**  After phase 2 has forwarded seeded
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
    flatten_replacement_chains, remap_statement_handles, try_map_expression_handles_in_place,
};
use super::scoped_map::ScopedMap;

/// Pass object for the three-phase load / store cleanup.  See the
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
    /// a future naga release adding a new emit-handle-bearing or
    /// block-bearing statement variant trips the build here instead
    /// of silently slipping past this index's coverage.
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
                naga::Statement::Block(inner) => {
                    self.walk(inner, pos);
                }
                naga::Statement::If { accept, reject, .. } => {
                    self.walk(accept, pos);
                    self.walk(reject, pos);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.walk(&case.body, pos);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    self.walk(body, pos);
                    self.walk(continuing, pos);
                }
                // Statements that introduce no expression handles
                // (no Emit range, no result field, no Store identity)
                // and contain no nested blocks.  Enumerated explicitly
                // - a future naga variant that adds emit-handle
                // semantics or carries a nested block would otherwise
                // silently bypass this index.
                naga::Statement::Call { result: None, .. }
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

    // Phase 1: zero inits are always redundant in WGSL.
    for (_, lvar) in function.local_variables.iter_mut() {
        if let Some(init) = lvar.init
            && is_zero_init(&function.expressions, init)
        {
            lvar.init = None;
            changed = true;
        }
    }

    // Phase 2: non-zero inits overwritten before first load.
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
            // Control-flow: stop tracking any local involved inside.
            naga::Statement::If { accept, reject, .. } => {
                invalidate_involved(&[accept, reject], expressions, &mut pending);
            }
            naga::Statement::Switch { cases, .. } => {
                let blocks: Vec<&naga::Block> = cases.iter().map(|c| &c.body).collect();
                invalidate_involved(&blocks, expressions, &mut pending);
            }
            naga::Statement::Loop {
                body: lb,
                continuing,
                ..
            } => {
                invalidate_involved(&[lb, continuing], expressions, &mut pending);
            }
            naga::Statement::Block(inner) => {
                invalidate_involved(&[inner], expressions, &mut pending);
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
            // Statements that do not touch local-variable inits and
            // do not contain nested blocks - enumerated explicitly
            // so a future naga release adding a new pointer-bearing
            // statement breaks the build here instead of silently
            // letting an init survive past a Store-through-it.
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
    dead
}

/// Remove `pending` entries for any local that is loaded OR modified
/// inside the given sub-blocks.  Used by `find_dead_inits` to drop
/// init-tracking for any local that might be observed (read) or
/// mutated (write) on a path through a sub-block - either case means
/// the init cannot be proved dead from a top-level scan.
fn invalidate_involved(
    blocks: &[&naga::Block],
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut HashSet<naga::Handle<naga::LocalVariable>>,
) {
    let mut involved = HashSet::new();
    for block in blocks {
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
            // Recurse into nested blocks.
            naga::Statement::If { accept, reject, .. } => {
                collect_touched_locals(accept, expressions, touched);
                collect_touched_locals(reject, expressions, touched);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_touched_locals(&case.body, expressions, touched);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_touched_locals(body, expressions, touched);
                collect_touched_locals(continuing, expressions, touched);
            }
            naga::Statement::Block(inner) => {
                collect_touched_locals(inner, expressions, touched);
            }
            // Statements that neither read a local through `Load`
            // nor write a local through any pointer-bearing field,
            // and contain no nested blocks.  Enumerated explicitly
            // so a future naga variant breaks the build here rather
            // than silently mis-classifying a touch.
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
            naga::Statement::If { accept, reject, .. } => {
                collect_nonstore_pointer_locals(accept, expressions, used);
                collect_nonstore_pointer_locals(reject, expressions, used);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_nonstore_pointer_locals(&case.body, expressions, used);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_nonstore_pointer_locals(body, expressions, used);
                collect_nonstore_pointer_locals(continuing, expressions, used);
            }
            naga::Statement::Block(inner) => {
                collect_nonstore_pointer_locals(inner, expressions, used)
            }
            // Statements that neither observe a local through a pointer nor
            // carry a nested block.  `Store` is excluded by design (it is the
            // write this analysis is proving dead); callee/atomic/cooperative
            // pointer escapes are covered by `collect_escaped_and_partially_stored`.
            // Enumerated (no `_`) per this file's forward-compat contract, so a
            // future block- or pointer-bearing variant trips the build here.
            naga::Statement::Emit(_)
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
        match &mut stmt {
            naga::Statement::Store { pointer, .. } => {
                if let Some(local) = get_stored_local(expressions, *pointer)
                    && !used.contains(&local)
                {
                    changed = true;
                    continue; // drop the dead store
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                changed |= remove_stores_to_dead_locals(accept, expressions, used);
                changed |= remove_stores_to_dead_locals(reject, expressions, used);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    changed |= remove_stores_to_dead_locals(&mut case.body, expressions, used);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed |= remove_stores_to_dead_locals(body, expressions, used);
                changed |= remove_stores_to_dead_locals(continuing, expressions, used);
            }
            naga::Statement::Block(inner) => {
                changed |= remove_stores_to_dead_locals(inner, expressions, used);
            }
            // Non-`Store`, non-block statements are passed through untouched.
            // Enumerated (no `_`) per this file's forward-compat contract, so a
            // future block-bearing variant trips the build rather than silently
            // skipping recursion into it.
            naga::Statement::Emit(_)
            | naga::Statement::Call { .. }
            | naga::Statement::Atomic { .. }
            | naga::Statement::WorkGroupUniformLoad { .. }
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
        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                changed |= remove_dead_stores_in_block(accept, expressions);
                changed |= remove_dead_stores_in_block(reject, expressions);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    changed |= remove_dead_stores_in_block(&mut case.body, expressions);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed |= remove_dead_stores_in_block(body, expressions);
                changed |= remove_dead_stores_in_block(continuing, expressions);
            }
            naga::Statement::Block(inner) => {
                changed |= remove_dead_stores_in_block(inner, expressions);
            }
            // Leaf statements - the dead-Store linear scan below
            // handles all non-block statements; recursion only fires
            // for block-bearing variants.  Enumerated explicitly so a
            // future naga release adding a new block-bearing variant
            // breaks the build here instead of silently bypassing
            // recursion.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
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
        let has_later_live_load = all_loads.get(&local).is_some_and(|loads| {
            loads.iter().any(|&load_h| {
                scope_idx
                    .handle_position(load_h)
                    .is_some_and(|lp| lp > store_pos)
                    && replacements.get(&load_h).is_none_or(|&r| r >= load_h)
            })
        });
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
    // guard against the post-undo replacements and drop any Store that is no
    // longer dead.
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
        let regained_live_load = all_loads.get(&local).is_some_and(|loads| {
            loads.iter().any(|&load_h| {
                scope_idx
                    .handle_position(load_h)
                    .is_some_and(|lp| lp > store_pos)
                    && replacements.get(&load_h).is_none_or(|&r| r >= load_h)
            })
        });
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
            naga::Statement::Block(inner) => {
                collect_escaped_and_partially_stored(inner, expressions, escaped, partially_stored);
            }
            naga::Statement::If { accept, reject, .. } => {
                collect_escaped_and_partially_stored(
                    accept,
                    expressions,
                    escaped,
                    partially_stored,
                );
                collect_escaped_and_partially_stored(
                    reject,
                    expressions,
                    escaped,
                    partially_stored,
                );
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_escaped_and_partially_stored(
                        &case.body,
                        expressions,
                        escaped,
                        partially_stored,
                    );
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_escaped_and_partially_stored(body, expressions, escaped, partially_stored);
                collect_escaped_and_partially_stored(
                    continuing,
                    expressions,
                    escaped,
                    partially_stored,
                );
            }
            // Statements that neither escape a local through a pointer
            // nor contain nested blocks - enumerated explicitly so a
            // future naga release adding a new pointer-bearing variant
            // breaks the build here instead of silently leaving a
            // local mis-classified as un-escaped.
            naga::Statement::Emit(_)
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
            naga::Statement::If { accept, reject, .. } => {
                count_stores_recursive(accept, expressions, counts);
                count_stores_recursive(reject, expressions, counts);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                count_stores_recursive(body, expressions, counts);
                count_stores_recursive(continuing, expressions, counts);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    count_stores_recursive(&case.body, expressions, counts);
                }
            }
            naga::Statement::Block(inner) => {
                count_stores_recursive(inner, expressions, counts);
            }
            _ => {}
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
                    if let Some(key) = get_pointer_key(expressions, *pointer) {
                        let mut resolved = *value;
                        while let Some(&next) = replacements.get(&resolved) {
                            resolved = next;
                        }
                        cache.insert(key.clone(), resolved);
                        store_source.insert(key, (*pointer, *value, in_loop));
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
                // execute *exactly one* case on every path.  That
                // requires (a) a `Default` case so the switch is total,
                // and (b) no `fall_through` between cases (otherwise a
                // case's "final state" includes later cases' writes and
                // the pre-meet snapshot is meaningless).
                let has_default = cases
                    .iter()
                    .any(|c| matches!(c.value, naga::SwitchValue::Default));
                let any_fallthrough = cases.iter().any(|c| c.fall_through);
                let meet_applicable = has_default && !any_fallthrough;

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
                    // Same early-exit soundness reasoning as the `If`
                    // arm above: with `has_default && !any_fallthrough`
                    // the switch executes exactly one case, so the
                    // meet can only introduce forwarding that is
                    // consistent across all cases.  Entries surviving
                    // in a case that terminates early (Return / Break
                    // out of an enclosing loop / etc.) are sound
                    // because the meet entry is only read on paths
                    // that fall through the switch.
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
            naga::Statement::If { accept, reject, .. } => {
                collect_escaped_locals(accept, expressions, escaped);
                collect_escaped_locals(reject, expressions, escaped);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_escaped_locals(&case.body, expressions, escaped);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_escaped_locals(body, expressions, escaped);
                collect_escaped_locals(continuing, expressions, escaped);
            }
            naga::Statement::Block(inner) => {
                collect_escaped_locals(inner, expressions, escaped);
            }
            _ => {}
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
            naga::Statement::If { accept, reject, .. } => {
                collect_modified_locals(accept, expressions, modified);
                collect_modified_locals(reject, expressions, modified);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_modified_locals(&case.body, expressions, modified);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_modified_locals(body, expressions, modified);
                collect_modified_locals(continuing, expressions, modified);
            }
            naga::Statement::Block(inner) => {
                collect_modified_locals(inner, expressions, modified);
            }
            // Statements that neither modify a function-local through
            // a pointer nor contain nested blocks - enumerated
            // explicitly so a future naga release adding a new
            // pointer-bearing variant breaks the build here instead
            // of silently leaving a local out of the modified set
            // (and thereby letting the caller forward a stale value).
            naga::Statement::Emit(_)
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
        match &mut statement {
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
            naga::Statement::Block(inner) => {
                apply_to_block(
                    inner,
                    replacements,
                    dead_locals,
                    dead_store_ids,
                    expressions,
                    in_loop,
                );
            }
            naga::Statement::If { accept, reject, .. } => {
                apply_to_block(
                    accept,
                    replacements,
                    dead_locals,
                    dead_store_ids,
                    expressions,
                    in_loop,
                );
                apply_to_block(
                    reject,
                    replacements,
                    dead_locals,
                    dead_store_ids,
                    expressions,
                    in_loop,
                );
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    apply_to_block(
                        &mut case.body,
                        replacements,
                        dead_locals,
                        dead_store_ids,
                        expressions,
                        in_loop,
                    );
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                apply_to_block(
                    body,
                    replacements,
                    dead_locals,
                    dead_store_ids,
                    expressions,
                    true,
                );
                apply_to_block(
                    continuing,
                    replacements,
                    dead_locals,
                    dead_store_ids,
                    expressions,
                    true,
                );
            }
            // Leaf statements - apply_to_block only specializes Emit
            // (range rebuild) and Store (dead-Store filter); every
            // other variant is preserved verbatim.  Enumerated
            // explicitly so a future naga release adding a new
            // block-bearing variant breaks the build here instead
            // of silently skipping recursion through it.
            naga::Statement::Break
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
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = LoadDedupPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let changed = pass
            .run(&mut module, &ctx)
            .expect("load dedup pass should run");
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        (changed, module)
    }

    fn count_loads_from_local(function: &naga::Function) -> usize {
        function
            .expressions
            .iter()
            .filter(|(handle, expr)| {
                if let naga::Expression::Load { pointer } = expr
                    && let naga::Expression::LocalVariable(_) = &function.expressions[*pointer]
                {
                    // Check if this handle is actually referenced (in an Emit range)
                    return is_handle_in_any_emit(&function.body, *handle);
                }
                false
            })
            .count()
    }

    fn is_handle_in_any_emit(block: &naga::Block, target: naga::Handle<naga::Expression>) -> bool {
        block.iter().any(|statement| match statement {
            naga::Statement::Emit(range) => range.clone().any(|h| h == target),
            naga::Statement::Block(inner) => is_handle_in_any_emit(inner, target),
            naga::Statement::If { accept, reject, .. } => {
                is_handle_in_any_emit(accept, target) || is_handle_in_any_emit(reject, target)
            }
            naga::Statement::Switch { cases, .. } => cases
                .iter()
                .any(|case| is_handle_in_any_emit(&case.body, target)),
            naga::Statement::Loop {
                body, continuing, ..
            } => is_handle_in_any_emit(body, target) || is_handle_in_any_emit(continuing, target),
            _ => false,
        })
    }

    #[test]
    fn nested_load_keeps_backing_store_alive() {
        // A local read through a depth>=2 nested pointer chain (`e.a.x`) must
        // keep its backing Store alive.  The liveness scan previously ignored
        // such loads (`get_pointer_key` can't forward them), so `e` was wrongly
        // marked dead and `e = s` removed - leaving the nested load reading the
        // WGSL zero-default instead of the stored value (silent miscompile).
        let source = r#"
struct Inner { x: f32, y: f32 }
struct Outer { a: Inner, b: f32 }
@fragment
fn fs(@location(0) k: f32) -> @location(0) vec4f {
    var e: Outer;
    var s: Outer;
    s.a.x = k;
    s.a.y = k;
    s.b = k;
    e = s;
    let whole = e;
    let nested = e.a.x;
    return vec4f(whole.b, nested, 0.0, 1.0);
}
"#;
        let (_changed, module) = run_pass(source);
        let func = &module.entry_points[0].function;
        let e = func
            .local_variables
            .iter()
            .find(|(_, lv)| lv.name.as_deref() == Some("e"))
            .map(|(h, _)| h)
            .expect("local `e` must exist");
        assert!(
            count_stores_to_local(func, e) >= 1,
            "the `e = s` store must survive: `e` is still read via the nested `e.a.x`"
        );
    }

    #[test]
    fn deduplicates_consecutive_loads_from_same_local() {
        let source = r#"
fn hash(p: vec2<f32>) -> f32 {
    var v: vec3<f32>;
    v = fract(vec3<f32>(p.x, p.y, p.x) * 0.1313);
    let a = v;
    let b = v;
    let c = v;
    v = a + vec3(dot(b, c.yzx + vec3(3.333)));
    let x = v.x;
    let y = v.y;
    let z = v.z;
    return fract((x + y) * z);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let h = hash(vec2(1.0, 2.0));
    return vec4f(h);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "should detect redundant loads");

        // The hash function should have fewer emitted whole-var loads
        // after dedup: 3 consecutive loads of v become 1.
        let hash_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("hash"))
            .map(|(_, f)| f)
            .expect("hash function should exist");

        let active_loads = count_loads_from_local(hash_fn);
        assert!(
            active_loads < 3,
            "expected fewer than 3 active whole-var loads, got {}",
            active_loads
        );
    }

    #[test]
    fn forwards_multi_store_variable_in_straight_line() {
        // Variable `a` has 2 stores.  Both loads are forwarded to the
        // stored values because each store seeds the cache.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    a = x + 1.0;
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "pass should forward multi-store loads");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "both loads should be forwarded (multi-store variable in straight-line code)"
        );
    }

    #[test]
    fn forwards_single_store_copy_variable() {
        // Variable `tmp` has exactly 1 store and is only used to carry
        // a value to `result`. The forwarding should replace Load(tmp)
        // with the stored value and eliminate tmp.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var tmp: f32;
    tmp = x * 2.0;
    let y = tmp + 1.0;
    return y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "pass should report changes");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "single-store variable loads should be forwarded"
        );
    }

    #[test]
    fn handles_no_local_variables() {
        let source = r#"
fn pure_fn(x: f32) -> f32 {
    return x * 2.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(pure_fn(1.0));
}
"#;

        let (changed, _) = run_pass(source);
        assert!(!changed, "no locals means nothing to deduplicate");
    }

    #[test]
    fn dead_store_removed_for_forwarded_variable() {
        // After forwarding, the Store to `tmp` becomes dead and should be
        // removed.  The local variable itself has no live references.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var tmp: f32;
    tmp = x;
    return tmp;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "pass should report changes");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Count Store statements remaining in the function body.
        fn count_stores(block: &naga::Block) -> usize {
            let mut count = 0;
            for stmt in block {
                match stmt {
                    naga::Statement::Store { .. } => count += 1,
                    naga::Statement::Block(inner) => count += count_stores(inner),
                    naga::Statement::If { accept, reject, .. } => {
                        count += count_stores(accept) + count_stores(reject);
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => {
                        count += count_stores(body) + count_stores(continuing);
                    }
                    _ => {}
                }
            }
            count
        }
        assert_eq!(
            count_stores(&test_fn.body),
            0,
            "dead store to forwarded variable should be removed"
        );
    }

    #[test]
    fn var_with_init_not_forwarded() {
        // Variable has init (not None) - should NOT be eligible for
        // store forwarding even if it has 1 store later.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 0.0;
    a = x;
    let v = a;
    return v;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (_, module) = run_pass(source);
        // The key point is the module remains valid after the pass.
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
    }

    #[test]
    fn cache_invalidated_after_if_branches() {
        // After an if/else that stores to `a`, the cache should be cleared.
        // A subsequent load of `a` must NOT be replaced with a stale value.
        let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    if c {
        a = x + 1.0;
    } else {
        a = x + 2.0;
    }
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, true));
}
"#;

        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        // v2 must NOT be replaced with x (from the store before if), because
        // the if branches modify a. The module being valid confirms this.
    }

    #[test]
    fn chained_copy_variables_forwarded() {
        // a = expr; b = a; use b -> b should resolve to expr through the chain.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x * 2.0;
    var b: f32;
    b = a;
    return b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "pass should report changes");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "chained copy variables should all be forwarded"
        );
    }

    #[test]
    fn dynamic_index_store_invalidates_cache() {
        // A store through a runtime-indexed pointer (Access) must invalidate
        // the cache for that local, not silently leave stale entries.
        // Use whole-variable loads so count_loads_from_local picks them up.
        let source = r#"
fn test_fn(idx: i32) -> vec3<f32> {
    var v: vec3<f32>;
    v = vec3(1.0, 2.0, 3.0);
    let a = v;
    v[idx] = 99.0;
    let b = v;
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let r = test_fn(0);
    return vec4f(r, 1.0);
}
"#;

        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");

        // `a` would normally be forwarded to vec3(1,2,3) via store
        // seeding, but the undo phase reverts this: `v` is not dead
        // (b still loads it), and the Compose expression is complex,
        // so forwarding is reverted to keep the cheap variable reference.
        // `b` is never forwarded because v[idx] invalidates the cache.
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 2,
            "both loads preserved: store forwarding reverted for non-dead variable with complex expression, got {} active loads",
            active_loads
        );
    }

    #[test]
    fn trace_ray_payload_invalidates_cache_and_marks_escaped() {
        // Construct IR where a local `payload_var` is stored to, loaded,
        // then passed as the payload of TraceRay, then loaded again.
        // The second load must NOT be deduplicated with the first because
        // TraceRay can modify the payload through the pointer.

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

        let local_payload = function.local_variables.append(
            naga::LocalVariable {
                name: Some("payload_var".into()),
                ty: f32_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        let ptr_payload = function.expressions.append(
            naga::Expression::LocalVariable(local_payload),
            naga::Span::UNDEFINED,
        );
        let load1 = function.expressions.append(
            naga::Expression::Load {
                pointer: ptr_payload,
            },
            naga::Span::UNDEFINED,
        );
        let load2 = function.expressions.append(
            naga::Expression::Load {
                pointer: ptr_payload,
            },
            naga::Span::UNDEFINED,
        );
        let lit_val = function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            naga::Span::UNDEFINED,
        );

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

        // Block:
        //   Store(payload_var, 1.0)
        //   Emit(load1)          -> cache: {payload_var: load1}
        //   TraceRay(payload=ptr_payload) -> should INVALIDATE cache
        //   Emit(load2)          -> must NOT be replaced with load1
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_payload,
                value: lit_val,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
                acceleration_structure: accel_expr,
                descriptor: desc_expr,
                payload: ptr_payload,
            }),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        // Run the core redundant-load collection.
        let mut replacements = HashMap::new();
        let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
        let mut all_loads = HashMap::new();
        let mut seeded_by_store = HashMap::new();
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

        // load2 must NOT be in replacements - TraceRay should have cleared
        // the cache so the second load is treated as a fresh access.
        assert!(
            !replacements.contains_key(&load2),
            "load after TraceRay must not be deduplicated (cache should be invalidated)"
        );

        // Verify escaped-local tracking: payload_var should be marked escaped.
        let escaped = locals_passed_by_pointer(&function.body, &function.expressions);
        assert!(
            escaped.contains(&local_payload),
            "local passed as TraceRay payload should be marked as escaped"
        );
    }

    #[test]
    fn cooperative_store_through_data_pointer_invalidates_cache() {
        // Construct realistic IR (matching what naga's WGSL frontend
        // produces for `coopStore(matrix_value, &mat_var, stride)`):
        //
        //   target = FunctionArgument(0)   (CooperativeMatrix VALUE - read)
        //   data.pointer = LocalVariable(mat_var)  (destination - WRITE)
        //
        // A cache entry for `mat_var` seeded before the CooperativeStore
        // must be invalidated by data.pointer's root, even though
        // `target` has CooperativeMatrix type and therefore cannot itself
        // resolve to a LocalVariable per naga's validator.

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

        let mut function = naga::Function::default();
        function.arguments.push(naga::FunctionArgument {
            name: Some("mat_in".into()),
            ty: coop_ty,
            binding: None,
        });

        let local_mat = function.local_variables.append(
            naga::LocalVariable {
                name: Some("mat_var".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        let ptr_mat = function.expressions.append(
            naga::Expression::LocalVariable(local_mat),
            naga::Span::UNDEFINED,
        );
        let matrix_arg = function
            .expressions
            .append(naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED);
        let load1 = function.expressions.append(
            naga::Expression::Load { pointer: ptr_mat },
            naga::Span::UNDEFINED,
        );
        let load2 = function.expressions.append(
            naga::Expression::Load { pointer: ptr_mat },
            naga::Span::UNDEFINED,
        );
        let stride = function.expressions.append(
            naga::Expression::Literal(naga::Literal::U32(16)),
            naga::Span::UNDEFINED,
        );

        // Block:
        //   Store(&mat_var, mat_in)         (seed: mat_var := mat_in)
        //   Emit(load1)                     (cache forwards to mat_in)
        //   coopStore(mat_in, &mat_var, 16) (writes into mat_var via data.pointer)
        //   Emit(load2)                     (must NOT dedup with load1)
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_mat,
                value: matrix_arg,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::CooperativeStore {
                target: matrix_arg,
                data: naga::CooperativeData {
                    pointer: ptr_mat,
                    stride,
                    row_major: false,
                },
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        // Run the core redundant-load collection.
        let mut replacements = HashMap::new();
        let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
        let mut all_loads = HashMap::new();
        let mut seeded_by_store = HashMap::new();
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

        // load2 must NOT be in replacements: the CooperativeStore wrote
        // to mat_var through `data.pointer`, so the cache entry forwarded
        // by load1 is stale.
        assert!(
            !replacements.contains_key(&load2),
            "load after CooperativeStore must not be deduplicated \
             (cache must be invalidated via data.pointer's root local)"
        );

        // Verify partial-store tracking: `mat_var` should be flagged as
        // partially-stored because the matrix write through
        // `data.pointer` may not cover the local's full type.
        let mut escaped = HashSet::new();
        let mut partially_stored = HashSet::new();
        collect_escaped_and_partially_stored(
            &function.body,
            &function.expressions,
            &mut escaped,
            &mut partially_stored,
        );
        assert!(
            partially_stored.contains(&local_mat),
            "local written by CooperativeStore must be flagged as partially_stored \
             (matrix write through data.pointer may not cover the full local)"
        );
    }

    #[test]
    fn atomic_invalidates_cache_and_counts_as_store() {
        // Construct IR where a local is stored to, loaded, then modified
        // by an Atomic operation, then loaded again.  The second load must
        // NOT be deduplicated with the first.

        let mut module = naga::Module::default();

        let u32_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Atomic(naga::Scalar::U32),
            },
            naga::Span::UNDEFINED,
        );

        let mut function = naga::Function::default();

        let local_var = function.local_variables.append(
            naga::LocalVariable {
                name: Some("atom_var".into()),
                ty: u32_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        let ptr_var = function.expressions.append(
            naga::Expression::LocalVariable(local_var),
            naga::Span::UNDEFINED,
        );
        let load1 = function.expressions.append(
            naga::Expression::Load { pointer: ptr_var },
            naga::Span::UNDEFINED,
        );
        let load2 = function.expressions.append(
            naga::Expression::Load { pointer: ptr_var },
            naga::Span::UNDEFINED,
        );
        let lit_val = function.expressions.append(
            naga::Expression::Literal(naga::Literal::U32(1)),
            naga::Span::UNDEFINED,
        );
        let atomic_result = function.expressions.append(
            naga::Expression::AtomicResult {
                ty: u32_ty,
                comparison: false,
            },
            naga::Span::UNDEFINED,
        );

        // Block:
        //   Store(atom_var, 1)
        //   Emit(load1)            -> cache: {atom_var: load1}
        //   Atomic(atom_var, Add)  -> should INVALIDATE cache
        //   Emit(load2)            -> must NOT be replaced with load1
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_var,
                value: lit_val,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load1, load1)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Atomic {
                pointer: ptr_var,
                fun: naga::AtomicFunction::Add,
                value: lit_val,
                result: Some(atomic_result),
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load2, load2)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        // Verify Atomic counts as a store.
        let store_counts = count_local_stores(&function.body, &function.expressions);
        assert_eq!(
            store_counts.get(&local_var).copied().unwrap_or(0),
            2,
            "Store + Atomic should count as 2 stores"
        );

        // Verify the load cache is invalidated by Atomic.
        let mut replacements = HashMap::new();
        let mut cache: ScopedMap<PointerKey, naga::Handle<naga::Expression>> = ScopedMap::new();
        let mut all_loads = HashMap::new();
        let mut seeded_by_store = HashMap::new();
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

        assert!(
            !replacements.contains_key(&load2),
            "load after Atomic must not be deduplicated (cache should be invalidated)"
        );
    }

    #[test]
    fn deduplicates_loads_through_same_dynamic_index() {
        // Two consecutive loads through the same Access (dynamic index) expression
        // should be deduplicated.
        let source = r#"
fn test_fn(idx: i32) -> f32 {
    var v: vec3<f32>;
    v = vec3(1.0, 2.0, 3.0);
    let a = v[idx];
    let b = v[idx];
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let r = test_fn(0);
    return vec4f(r, 0.0, 0.0, 1.0);
}
"#;

        let (changed, module) = run_pass(source);
        assert!(
            changed,
            "should detect redundant loads through same dynamic index"
        );

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Count emitted Load expressions that reference a local through Access.
        let dynamic_load_count = test_fn
            .expressions
            .iter()
            .filter(|(handle, expr)| {
                if let naga::Expression::Load { pointer } = expr
                    && let naga::Expression::Access { base, .. } = &test_fn.expressions[*pointer]
                    && matches!(
                        test_fn.expressions[*base],
                        naga::Expression::LocalVariable(_)
                    )
                {
                    return is_handle_in_any_emit(&test_fn.body, *handle);
                }
                false
            })
            .count();

        assert!(
            dynamic_load_count < 2,
            "expected at most 1 active dynamic-indexed load, got {}",
            dynamic_load_count
        );
    }

    #[test]
    fn init_seeded_load_forwarded_and_dead_local_removed() {
        // `var a: f32 = 0.0;` has init=Some(literal(0.0)).
        // A subsequent Load(a) should be forwarded to the init expression,
        // and the dead local should have its stores removed.
        let source = r#"
fn test_fn() -> f32 {
    var a: f32 = 0.0;
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn());
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "init-seeded load should be forwarded");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "load of init-seeded variable should be forwarded"
        );
    }

    #[test]
    fn init_seeded_then_overwritten_uses_store_value() {
        // If a variable has an init but is overwritten before reading,
        // the load should use the stored value, not the init.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 0.0;
    a = x;
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(5.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "load should be forwarded to store value");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "load after overwrite should forward to store value, not init"
        );
    }

    #[test]
    fn forwarding_survives_loop_for_unmodified_local() {
        // A variable stored before a loop and loaded after should be forwarded
        // when the loop body does not modify it.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = x;
    var sum: f32 = 0.0;
    var i: i32 = 0;
    loop {
        if i >= 4 { break; }
        sum += 1.0;
        continuing {
            i += 1;
        }
    }
    return a + sum;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "load of a should be forwarded across the loop");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // `a` is not modified in the loop, so Load(a) after the loop should
        // be forwarded, leaving 0 active loads of `a`.
        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let a_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("a"))
            .map(|(h, _)| h)
            .expect("variable a should exist");
        assert_eq!(
            store_count.get(&a_handle).copied().unwrap_or(0),
            0,
            "stores to a should be eliminated (dead local after forwarding)"
        );
    }

    #[test]
    fn forwarding_survives_if_for_unmodified_local() {
        // A variable stored before an if and loaded after should be forwarded
        // when neither branch modifies it.
        let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = x;
    var b: f32 = 0.0;
    if c {
        b = 1.0;
    } else {
        b = 2.0;
    }
    return a + b;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, true));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "load of a should be forwarded across the if");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let a_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("a"))
            .map(|(h, _)| h)
            .expect("variable a should exist");
        assert_eq!(
            store_count.get(&a_handle).copied().unwrap_or(0),
            0,
            "stores to a should be eliminated (dead local after forwarding across if)"
        );
    }

    #[test]
    fn no_forwarding_across_loop_when_modified() {
        // If the loop modifies the variable, forwarding should NOT happen.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = x;
    var i: i32 = 0;
    loop {
        if i >= 4 { break; }
        a += 1.0;
        continuing {
            i += 1;
        }
    }
    return a;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (_, module) = run_pass(source);

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // `a` is modified in the loop, so it must NOT be eliminated.
        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let a_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("a"))
            .map(|(h, _)| h)
            .expect("variable a should exist");
        assert!(
            store_count.get(&a_handle).copied().unwrap_or(0) > 0,
            "stores to a should NOT be eliminated when loop modifies it"
        );
    }

    #[test]
    fn dead_store_eliminated_when_overwritten_before_load() {
        // Two consecutive whole-variable Stores with no Load in between:
        // the first Store is dead and should be removed.
        let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var p: vec3f;
    p = x;
    p = x + vec3f(1.0);
    return p.x + p.y + p.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "should detect dead store");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Only one Store to p should remain (the second one).
        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let p_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("p"))
            .map(|(h, _)| h)
            .expect("variable p should exist");
        assert_eq!(
            store_count.get(&p_handle).copied().unwrap_or(0),
            1,
            "first dead store to p should be removed, leaving only one"
        );
    }

    #[test]
    fn dead_store_not_eliminated_when_loaded_between_stores() {
        // A Load between two Stores means the first Store is NOT dead.
        let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var p: vec3f;
    p = x;
    let a = p.x;
    p = x + vec3f(1.0);
    return a + p.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

        let (_, module) = run_pass(source);

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Both Stores should remain because the first is loaded before
        // the second.
        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let p_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("p"))
            .map(|(h, _)| h)
            .expect("variable p should exist");
        assert_eq!(
            store_count.get(&p_handle).copied().unwrap_or(0),
            2,
            "both stores to p should remain when first is loaded before second"
        );
    }

    #[test]
    fn dead_store_chain_eliminates_all_but_last() {
        // Three consecutive whole-variable Stores with no Loads: only the last
        // survives.  Use vec3f + field access so load-dedup cannot forward the
        // whole-variable Store, keeping exactly one Store alive.
        let source = r#"
fn test_fn(x: vec3f) -> f32 {
    var a: vec3f;
    a = x;
    a = x + vec3f(1.0);
    a = x + vec3f(2.0);
    return a.x + a.y + a.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3f(1.0)));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "should detect dead stores in chain");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        let store_count = count_local_stores(&test_fn.body, &test_fn.expressions);
        let a_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("a"))
            .map(|(h, _)| h)
            .expect("variable a should exist");
        assert_eq!(
            store_count.get(&a_handle).copied().unwrap_or(0),
            1,
            "only the last store in a chain should survive"
        );
    }

    #[test]
    fn undo_phase_reverts_complex_forwarding_for_non_dead_variable() {
        // Store seeds `v -> (x + vec3(1))` and load `a` gets forwarded.
        // But a partial store `v.x = 0` invalidates the cache, so load `b`
        // is fresh.  `v` is NOT dead (b's load is not replaced).
        // The undo phase should revert `a -> (x + vec3(1))` because v is
        // non-dead and the forwarded expr is complex (Binary).
        let source = r#"
fn test_fn(x: vec3<f32>) -> f32 {
    var v: vec3<f32>;
    v = x + vec3<f32>(1.0, 2.0, 3.0);
    let a = v;
    v.x = 0.0;
    let b = v;
    return a.x + b.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(1.0)));
}
"#;

        let (_, module) = run_pass(source);
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Both loads should remain: the undo phase reverts the complex
        // store-to-load forwarding for non-dead v, and b was never forwarded.
        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 2,
            "undo phase should revert complex forwarding for non-dead variable, got {} active loads",
            active_loads
        );
    }

    #[test]
    fn undo_phase_keeps_simple_forwarding_for_non_dead_variable() {
        // Store seeds `a` with a simple expression (FunctionArgument),
        // and `a` is not dead (has loads across a cache-invalidating store).
        // The undo phase should keep the simple forwarding.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32;
    a = x;
    let v1 = a;
    a = x + 1.0;
    let v2 = a;
    return v1 + v2;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0));
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "pass should forward loads");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // Both loads forwarded: v1 -> x (simple, kept), v2 -> x+1 (a is dead
        // because both loads replaced, so kept regardless).
        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 0,
            "all loads should be forwarded (simple and dead-local cases)"
        );
    }

    #[test]
    fn compose_init_not_seeded_for_multi_load_variable() {
        // `var v: vec3f = vec3(1,2,3)` has a Compose init.
        // The Compose init should NOT be seeded into the cache, so
        // loads go through Load-to-Load dedup instead of Compose forwarding.
        // This prevents output inflation from duplicating constructors.
        let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    let a = v;
    let b = v;
    return a.x + b.y;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn());
}
"#;

        let (changed, module) = run_pass(source);
        assert!(changed, "should deduplicate consecutive loads");
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // With Compose init skip, the first Load becomes canonical and the
        // second Load deduplicates to the first (Load-to-Load).
        // The variable is NOT dead because the first Load is NOT replaced.
        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 1,
            "Compose init should not be forwarded; Load-to-Load dedup keeps 1 active load, got {}",
            active_loads
        );

        // Variable v should still have stores (it's alive).
        let v_handle = test_fn
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("v"))
            .map(|(h, _)| h)
            .expect("variable v should exist");
        assert!(
            test_fn.local_variables[v_handle].init.is_some(),
            "variable v should retain its init (not marked dead)"
        );
    }

    #[test]
    fn chain_resolution_flattens_load_chains_after_undo() {
        // Scenario: store seeds `a -> expr`, load1 -> expr (+ re-register a -> load1),
        // load2 -> load1 (Load-to-Load). If load1->expr is undone, load2->load1 must
        // still be valid.  This tests chain resolution handles the post-undo state.
        let source = r#"
fn test_fn(x: vec3<f32>) -> f32 {
    var a: vec3<f32>;
    a = x + vec3<f32>(1.0, 2.0, 3.0);
    let v1 = a;
    let v2 = a;
    a.x = 0.0;
    let v3 = a;
    return v1.x + v2.y + v3.z;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(1.0)));
}
"#;

        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module)
            .expect("module should remain valid after chain resolution");

        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn function should exist");

        // v1 and v2 are both loads before the partial store.
        // v1 -> complex expr (undone because a is not dead), v2 -> v1 (Load-to-Load).
        // After undo of v1->complex, chain resolution makes v2->v1 (stays as-is since v1 is
        // no longer in replacements).  v3 is after cache invalidation, fresh.
        // Result: 2 active loads (v1=canonical, v3=fresh; v2 deduplicated to v1).
        let active_loads = count_loads_from_local(test_fn);
        assert_eq!(
            active_loads, 2,
            "chain resolution should preserve Load-to-Load dedup after undo, got {} active loads",
            active_loads
        );
    }

    // Dead init removal (Phase 1: zero inits, Phase 2: dead non-zero inits)

    /// Helper: check whether a local variable has an init expression.
    fn local_has_init(function: &naga::Function, name: &str) -> bool {
        function
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .map(|(_, v)| v.init.is_some())
            .unwrap_or(false)
    }

    #[test]
    fn zero_init_f32_removed() {
        let source = r#"
fn test_fn() -> f32 {
    var a: f32 = 0.0;
    a = 1.0;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let f = module.functions.iter().next().unwrap().1;
        assert!(!local_has_init(f, "a"), "zero f32 init should be removed");
    }

    #[test]
    fn zero_init_i32_removed() {
        let source = r#"
fn test_fn() -> i32 {
    var a: i32 = 0i;
    a = 1i;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(f32(test_fn())); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let f = module.functions.iter().next().unwrap().1;
        assert!(!local_has_init(f, "a"), "zero i32 init should be removed");
    }

    #[test]
    fn zero_init_bool_removed() {
        let source = r#"
fn test_fn() -> bool {
    var a: bool = false;
    a = true;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(f32(test_fn())); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let f = module.functions.iter().next().unwrap().1;
        assert!(!local_has_init(f, "a"), "false bool init should be removed");
    }

    #[test]
    fn non_zero_init_preserved_when_loaded() {
        // Compose init with non-zero value - Compose inits are NOT seeded
        // into the forwarding cache, so the first Load reads the init.
        // The variable stays alive and the init must be preserved.
        let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    return v.x + v.y + v.z;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        let f = module.functions.iter().next().unwrap().1;
        let v_init = f
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("v"))
            .map(|(_, v)| v.init.is_some());
        assert_eq!(
            v_init,
            Some(true),
            "non-zero Compose init should be preserved"
        );
    }

    #[test]
    fn zero_init_vec3_compose_removed() {
        // vec3(0.0, 0.0, 0.0) is a Compose of zero literals.
        let source = r#"
fn test_fn() -> vec3<f32> {
    var v: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    v = vec3<f32>(1.0, 2.0, 3.0);
    return v;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(), 1.0); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let f = module.functions.iter().next().unwrap().1;
        assert!(
            !local_has_init(f, "v"),
            "all-zero Compose init should be removed"
        );
    }

    #[test]
    fn partially_non_zero_compose_preserved() {
        // vec3(0.0, 1.0, 0.0) has a non-zero component.
        let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    return v.y;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
        let (_, module) = run_pass(source);
        let f = module.functions.iter().next().unwrap().1;
        // Variable may be eliminated by forwarding; if it survives, init must be kept.
        let v_init = f
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("v"))
            .map(|(_, v)| v.init.is_some());
        if let Some(has_init) = v_init {
            assert!(
                has_init,
                "partially non-zero Compose init should be preserved"
            );
        }
    }

    // Phase 2: dead non-zero init removal

    #[test]
    fn dead_non_zero_scalar_init_removed() {
        // `var a = 5.0; a = x;` - init is overwritten before any load.
        // After dedup_loads forwards the load to x, the Emit is gone,
        // and find_dead_inits sees Store(a, x) as the first reference.
        let source = r#"
fn test_fn(x: f32) -> f32 {
    var a: f32 = 5.0;
    a = x;
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0)); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
    }

    #[test]
    fn dead_non_zero_compose_init_removed() {
        // Compose init overwritten before any load.
        let source = r#"
fn test_fn(x: vec3<f32>) -> vec3<f32> {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    v = x;
    return v;
}
@fragment fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(vec3(4.0, 5.0, 6.0)), 1.0);
}
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let f = module.functions.iter().next().unwrap().1;
        assert!(
            !local_has_init(f, "v"),
            "dead Compose init should be removed"
        );
    }

    #[test]
    fn dead_init_across_unrelated_control_flow() {
        // An If that does NOT involve `a` should not block dead-init
        // detection.  `a = x` is the first reference to `a`.
        let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = 5.0;
    var b: f32;
    if c { b = 1.0; } else { b = 2.0; }
    a = x;
    return a + b;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0, true)); }
"#;
        let (changed, module) = run_pass(source);
        assert!(changed);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
    }

    #[test]
    fn init_preserved_when_read_through_control_flow() {
        // `a` is loaded inside the if - in the else path, the init value
        // is the one that reaches `return a`.  Init must be preserved.
        let source = r#"
fn test_fn(x: f32, c: bool) -> f32 {
    var a: f32 = 5.0;
    if c { a = x; }
    return a;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn(1.0, true)); }
"#;
        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        let f = module.functions.iter().next().unwrap().1;
        // a is modified in one branch - find_dead_inits stops tracking.
        // The init is NOT zero, so it must be preserved.
        let a_init = f
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("a"))
            .map(|(_, v)| v.init.is_some());
        assert_eq!(
            a_init,
            Some(true),
            "init must be preserved when the variable is read through control flow"
        );
    }

    #[test]
    fn dead_init_partial_store_prevents_removal() {
        // A partial (field) store reads the old value, so the init IS used.
        let source = r#"
fn test_fn() -> f32 {
    var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    v.x = 99.0;
    return v.x + v.y + v.z;
}
@fragment fn fs_main() -> @location(0) vec4f { return vec4f(test_fn()); }
"#;
        let (_, module) = run_pass(source);
        let _ = crate::io::validate_module(&module).expect("module should remain valid");
        let f = module.functions.iter().next().unwrap().1;
        let v_init = f
            .local_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("v"))
            .map(|(_, v)| v.init.is_some());
        if let Some(has_init) = v_init {
            assert!(
                has_init,
                "partial store reads old value - init must be preserved"
            );
        }
    }

    /// Helper: run dead_branch then load_dedup, validating after each.
    fn run_dead_branch_then_load_dedup(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        let mut db = crate::passes::dead_branch::DeadBranchPass;
        let _ = db.run(&mut module, &ctx).expect("dead_branch should run");
        let _ = crate::io::validate_module(&module).expect("valid after dead_branch");

        let mut ld = LoadDedupPass;
        let changed = ld.run(&mut module, &ctx).expect("load_dedup should run");
        let _ = crate::io::validate_module(&module).expect("valid after load_dedup");

        (changed, module)
    }

    /// Verify the module round-trips through WGSL emission and re-parsing.
    fn assert_wgsl_round_trips(module: &naga::Module) {
        let info = crate::io::validate_module(module).expect("module should validate");
        let wgsl =
            naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
                .expect("WGSL emission should succeed");
        let reparsed = naga::front::wgsl::parse_str(&wgsl);
        assert!(
            reparsed.is_ok(),
            "emitted WGSL should re-parse, got error: {:?}\nemitted WGSL:\n{}",
            reparsed.err(),
            wgsl
        );
    }

    #[test]
    fn forward_ref_replacement_preserves_short_circuit_local() {
        // Regression: short-circuit re-sugaring can create forward
        // references that break dead-local detection.
        //
        // naga lowers `a && b` into:
        //   var local: bool;
        //   if (a) { local = b; } else { local = false; }
        //   let hit = local;
        //
        // desugar_short_circuit folds this back into a Binary(LogicalAnd)
        // appended at the END of the expression arena:
        //   local = (a && b);   // Binary handle > original Load handle
        //   let hit = local;
        //
        // load_dedup's store-forwarding detects Load(local) -> Binary_h,
        // but Binary_h > Load_h (forward reference). The expression-arena
        // apply loop guards against forward references, so this replacement
        // is never applied.  If load_dedup still marks `local` as dead it
        // removes the Store while the Load persists, leaving the variable
        // uninitialised.
        let source = r#"
fn test_fn(a: f32, b: f32, c: f32) -> f32 {
    let hit = a > 0.0 && b > 0.0 && c > 0.0;
    return select(0.0, 1.0, hit);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, 2.0, 3.0));
}
"#;

        let (_, module) = run_dead_branch_then_load_dedup(source);
        assert_wgsl_round_trips(&module);

        // Extra safety: verify no local has loads but no stores or init
        // (which would indicate the bug).
        let test_fn = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("test_fn"))
            .map(|(_, f)| f)
            .expect("test_fn should exist");

        for (lh, lvar) in test_fn.local_variables.iter() {
            let ptr_h = test_fn
                .expressions
                .iter()
                .find(|(_, e)| matches!(e, naga::Expression::LocalVariable(l) if *l == lh))
                .map(|(h, _)| h);
            let Some(ptr_h) = ptr_h else { continue };

            let has_load = test_fn
                .expressions
                .iter()
                .any(|(_, e)| matches!(e, naga::Expression::Load { pointer } if *pointer == ptr_h));
            if !has_load {
                continue;
            }

            let has_store = has_store_to(&test_fn.body, ptr_h);
            let has_init = lvar.init.is_some();
            assert!(
                has_store || has_init,
                "local {:?} has loads but no store/init - forward-ref replacement bug",
                lvar.name
            );
        }
    }

    fn has_store_to(block: &naga::Block, ptr_h: naga::Handle<naga::Expression>) -> bool {
        block.iter().any(|stmt| match stmt {
            naga::Statement::Store { pointer, .. } => *pointer == ptr_h,
            naga::Statement::Block(inner) => has_store_to(inner, ptr_h),
            naga::Statement::If { accept, reject, .. } => {
                has_store_to(accept, ptr_h) || has_store_to(reject, ptr_h)
            }
            naga::Statement::Switch { cases, .. } => {
                cases.iter().any(|case| has_store_to(&case.body, ptr_h))
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => has_store_to(body, ptr_h) || has_store_to(continuing, ptr_h),
            _ => false,
        })
    }

    #[test]
    fn forward_ref_chained_short_circuit_preserves_locals() {
        // Chained short-circuit (a && b && c && d) creates multiple locals,
        // each with a forward-reference Binary replacement.  This is common
        // in ray-tracing shaders that chain intersection tests.
        let source = r#"
fn test_fn(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let hit = a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0;
    return select(0.0, 1.0, hit);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(test_fn(1.0, 2.0, 3.0, 4.0));
}
"#;

        let (_, module) = run_dead_branch_then_load_dedup(source);
        assert_wgsl_round_trips(&module);
    }

    // Regression tests for `remove_dead_stores_in_block` per-local invalidation
    //
    // These tests pin the precision of `remove_dead_stores_in_block`'s
    // pending-store invalidation: a non-aliasing statement (Barrier,
    // Atomic on a global, ImageStore, etc.) must not save a dead Store
    // from removal, and a Return / Kill terminator must mark all still-
    // pending Stores as dead since they cannot be observed afterwards.

    fn count_stores_to_local(
        function: &naga::Function,
        local: naga::Handle<naga::LocalVariable>,
    ) -> usize {
        fn walk(
            block: &naga::Block,
            expressions: &naga::Arena<naga::Expression>,
            local: naga::Handle<naga::LocalVariable>,
            count: &mut usize,
        ) {
            for stmt in block {
                match stmt {
                    naga::Statement::Store { pointer, .. } => {
                        if let naga::Expression::LocalVariable(lh) = expressions[*pointer]
                            && lh == local
                        {
                            *count += 1;
                        }
                    }
                    naga::Statement::Block(inner) => walk(inner, expressions, local, count),
                    naga::Statement::If { accept, reject, .. } => {
                        walk(accept, expressions, local, count);
                        walk(reject, expressions, local, count);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        for case in cases {
                            walk(&case.body, expressions, local, count);
                        }
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => {
                        walk(body, expressions, local, count);
                        walk(continuing, expressions, local, count);
                    }
                    _ => {}
                }
            }
        }
        let mut n = 0;
        walk(&function.body, &function.expressions, local, &mut n);
        n
    }

    fn local_handle_by_name(
        function: &naga::Function,
        name: &str,
    ) -> naga::Handle<naga::LocalVariable> {
        function
            .local_variables
            .iter()
            .find(|(_, lv)| lv.name.as_deref() == Some(name))
            .map(|(h, _)| h)
            .unwrap_or_else(|| panic!("local {name} not found"))
    }

    #[test]
    fn dead_store_removed_across_atomic_on_global() {
        // `x = 1; atomicAdd(&g, 1); x = 2; if(c) { x = 3; } return x;`
        // The atomic targets a storage-bound atomic (cannot alias the
        // function-local `x`), so `x = 1` is overwritten before it can be
        // observed and must be collapsed.  The trailing If keeps the load
        // dependency live so dedup_loads cannot trim the later stores -
        // the only path that removes `x = 1` is the new per-local
        // invalidation in `remove_dead_stores_in_block`.
        let source = r#"
@group(0) @binding(0) var<storage, read_write> g: atomic<i32>;
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    atomicAdd(&g, 1);
    x = 2;
    if (c) { x = 3; }
    return x;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
        let (_, module) = run_pass(source);
        let f = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .map(|(_, f)| f)
            .expect("f exists");
        let x = local_handle_by_name(f, "x");
        assert_eq!(
            count_stores_to_local(f, x),
            2,
            "dead Store before atomic should be removed; remaining = x=2 + x=3"
        );
    }

    #[test]
    fn dead_store_removed_across_barrier() {
        // Same shape as above, with a workgroup barrier in place of the
        // atomic.  Barriers do not reference function-local pointers, so
        // the first store is unconditionally dead.
        let source = r#"
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    workgroupBarrier();
    x = 2;
    if (c) { x = 3; }
    return x;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
        let (_, module) = run_pass(source);
        let f = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .map(|(_, f)| f)
            .expect("f exists");
        let x = local_handle_by_name(f, "x");
        assert_eq!(
            count_stores_to_local(f, x),
            2,
            "dead Store before barrier should be removed"
        );
    }

    #[test]
    fn dead_trailing_store_removed_before_return_unit() {
        // Direct unit test of `remove_dead_stores_in_function` (NOT the
        // whole pass): the Return terminator drains `pending_store`
        // marking trailing stores dead.  At the full-pass level
        // `dead_store_ids` in `dedup_loads_in_function` reaches the same
        // result via a different path; this test pins the standalone
        // semantic so future refactors of either side don't drop it.
        let source = r#"
fn f() -> i32 {
    var x: i32;
    var y: i32;
    y = 7;
    x = 1;
    return y;
}
@compute @workgroup_size(1) fn main() { _ = f(); }
"#;
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let f_handle = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .map(|(h, _)| h)
            .expect("f exists");
        let f = module.functions.get_mut(f_handle);
        let x = local_handle_by_name(f, "x");
        let y = local_handle_by_name(f, "y");
        let changed = remove_dead_stores_in_function(f);
        assert!(changed, "should mark trailing Store dead");
        assert_eq!(
            count_stores_to_local(f, x),
            0,
            "trailing Store of `x` before Return should be removed by terminator drain"
        );
        assert_eq!(
            count_stores_to_local(f, y),
            1,
            "Store of `y` is read by `return y` and must be preserved"
        );
    }

    #[test]
    fn dead_store_kept_across_call_with_pointer_arg() {
        // `x = 1; g(&x); x = 2; if(c){x=3;} return x + y;` - the callee
        // may read the pending Store through the `ptr<function, i32>`
        // argument, so per-arg invalidation must drop `x` from
        // `pending_store`, keeping the first Store live.
        let source = r#"
fn g(p: ptr<function, i32>) -> i32 {
    return *p;
}
fn f(c: bool) -> i32 {
    var x: i32;
    x = 1;
    let y = g(&x);
    x = 2;
    if (c) { x = 3; }
    return x + y;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
        let (_, module) = run_pass(source);
        let f = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .map(|(_, f)| f)
            .expect("f exists");
        let x = local_handle_by_name(f, "x");
        assert_eq!(
            count_stores_to_local(f, x),
            3,
            "Store observable through ptr arg must NOT be removed"
        );
    }

    #[test]
    fn dead_store_removed_across_call_without_pointer_arg() {
        // `x = 1; g(&y); x = 2; if(c){x=3;} return x;` - the call cannot
        // observe `x` (its `ptr<function, T>` arg points to `y`).  The
        // first Store is dead under per-arg invalidation; the trailing If
        // prevents dedup_loads from trimming `x = 2` / `x = 3`.
        let source = r#"
fn g(p: ptr<function, i32>) -> i32 {
    return *p;
}
fn f(c: bool) -> i32 {
    var x: i32;
    var y: i32;
    x = 1;
    let r = g(&y);
    x = 2;
    if (c) { x = 3; }
    return x + r;
}
@compute @workgroup_size(1) fn main() { _ = f(true); }
"#;
        let (_, module) = run_pass(source);
        let f = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .map(|(_, f)| f)
            .expect("f exists");
        let x = local_handle_by_name(f, "x");
        assert_eq!(
            count_stores_to_local(f, x),
            2,
            "dead Store across call with non-aliasing ptr arg should be removed"
        );
    }

    // MARK: Block / If / Switch scope-leak regressions
    //
    // Each of these shaders sets up a value that is bound by an `Emit`
    // inside a `Statement::Block` / `Statement::If` / `Statement::Switch`,
    // then reads the variable that received it from outside the block.  The
    // pass must NOT forward the post-block load to the in-block value handle:
    // that produces IR the validator rejects ("expression used outside its
    // scope").  `run_pass`'s post-pass `validate_module` call asserts this
    // never happens.

    #[test]
    fn block_scope_leak_regression_statement_block() {
        // The inner brace forces the Compose's `let temp` binding to be
        // lexically scoped to the block; the post-block load of `x` would be
        // forwarded to that binding without the scope-aware filter.
        let source = r#"
fn f(a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    {
        let temp = vec3<f32>(a, b, c);
        x = temp;
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(1.0, 2.0, 3.0); }
"#;
        let (_, _module) = run_pass(source);
    }

    #[test]
    fn block_scope_leak_regression_if_branch() {
        // The Compose is bound inside the accept branch.  A naive
        // forward into the post-if read of `x` would land on an
        // out-of-scope handle on the reject path.
        let source = r#"
fn f(c: bool, a: f32, b: f32, d: f32) -> vec3<f32> {
    var x: vec3<f32>;
    if c {
        let temp = vec3<f32>(a, b, d);
        x = temp;
    } else {
        x = vec3<f32>(0.0);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(true, 1.0, 2.0, 3.0); }
"#;
        let (_, _module) = run_pass(source);
    }

    // MARK: ExpressionScopeIndex regressions

    #[test]
    fn scope_index_assigns_monotonic_positions_in_emit_order() {
        // Build a tiny function body by hand:
        //   Emit(e0, e1)   <- position 0
        //   Store(p, v)    <- position 1
        //   Emit(e2)       <- position 2
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        // Fake pointer + value; we never resolve them to real types,
        // we just need stable Expression handles.
        let p = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            naga::Span::UNDEFINED,
        );
        let v = arena.append(
            naga::Expression::Literal(naga::Literal::U32(1)),
            naga::Span::UNDEFINED,
        );
        let e0 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(2)),
            naga::Span::UNDEFINED,
        );
        let e1 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(3)),
            naga::Span::UNDEFINED,
        );
        let e2 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(4)),
            naga::Span::UNDEFINED,
        );
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(e0, e1)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: p,
                value: v,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(e2, e2)),
            naga::Span::UNDEFINED,
        );

        let idx = ExpressionScopeIndex::build(&body, &arena);

        // All handles in the first Emit share position 0.
        assert_eq!(idx.handle_position(e0), Some(0));
        assert_eq!(idx.handle_position(e1), Some(0));
        // Store gets position 1.
        assert_eq!(idx.store_position((p, v)), Some(1));
        // Second Emit gets position 2.
        assert_eq!(idx.handle_position(e2), Some(2));
        // The top-level body's interval covers all 3 positions.
        assert!(idx.is_in_subtree(&body, e0));
        assert!(idx.is_in_subtree(&body, e2));
    }

    #[test]
    fn scope_index_recurses_into_control_flow() {
        // Body: Store(p,v)               pos 0
        //       If {                     pos 1
        //         accept: Emit(a)        pos 2
        //         reject: Emit(b)        pos 3
        //       }
        //       Store(p2, v2)            pos 4
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let p = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            naga::Span::UNDEFINED,
        );
        let v = arena.append(
            naga::Expression::Literal(naga::Literal::U32(1)),
            naga::Span::UNDEFINED,
        );
        let cond = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            naga::Span::UNDEFINED,
        );
        let a = arena.append(
            naga::Expression::Literal(naga::Literal::U32(2)),
            naga::Span::UNDEFINED,
        );
        let b = arena.append(
            naga::Expression::Literal(naga::Literal::U32(3)),
            naga::Span::UNDEFINED,
        );
        let p2 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(4)),
            naga::Span::UNDEFINED,
        );
        let v2 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(5)),
            naga::Span::UNDEFINED,
        );

        let mut accept = naga::Block::new();
        accept.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(a, a)),
            naga::Span::UNDEFINED,
        );
        let mut reject = naga::Block::new();
        reject.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(b, b)),
            naga::Span::UNDEFINED,
        );

        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: p,
                value: v,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::If {
                condition: cond,
                accept,
                reject,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: p2,
                value: v2,
            },
            naga::Span::UNDEFINED,
        );

        let idx = ExpressionScopeIndex::build(&body, &arena);

        assert_eq!(idx.store_position((p, v)), Some(0));
        assert_eq!(idx.handle_position(a), Some(2));
        assert_eq!(idx.handle_position(b), Some(3));
        // Second Store comes AFTER both branches, so position 4.
        assert_eq!(idx.store_position((p2, v2)), Some(4));
        // And the key invariant: second-store position > both
        // branch-Emit positions.
        assert!(idx.store_position((p2, v2)).unwrap() > idx.handle_position(a).unwrap());
        assert!(idx.store_position((p2, v2)).unwrap() > idx.handle_position(b).unwrap());

        // Subtree membership: `a` is in `accept` only, `b` is in
        // `reject` only.  Re-borrow the branches through the body so
        // the assertions exercise the same block addresses that
        // production callers see.
        let (accept_ref, reject_ref) = match &body[1] {
            naga::Statement::If { accept, reject, .. } => (accept, reject),
            _ => unreachable!(),
        };
        assert!(idx.is_in_subtree(accept_ref, a));
        assert!(!idx.is_in_subtree(accept_ref, b));
        assert!(idx.is_in_subtree(reject_ref, b));
        assert!(!idx.is_in_subtree(reject_ref, a));
    }

    #[test]
    fn scope_index_collides_stores_to_minimum() {
        // Two distinct Store statements with the same (ptr, val)
        // identity: the index records the MINIMUM position so a
        // Load between them is correctly classified as "after" the
        // earliest Store (conservative: over-keeps).
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let p = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            naga::Span::UNDEFINED,
        );
        let v = arena.append(
            naga::Expression::Literal(naga::Literal::U32(1)),
            naga::Span::UNDEFINED,
        );
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: p,
                value: v,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: p,
                value: v,
            },
            naga::Span::UNDEFINED,
        );

        let idx = ExpressionScopeIndex::build(&body, &arena);

        // Two distinct stores share identity (p, v); the recorded
        // position is the earliest (0), not the latest (1).
        assert_eq!(idx.store_position((p, v)), Some(0));
    }

    /// Statement-bound result handles (Call.result, Atomic.result,
    /// WorkGroupUniformLoad.result, Subgroup*.result,
    /// RayQueryFunction::Proceed.result) are NOT in any Emit range.
    /// The pre-fix `compute_statement_positions` only walked Emit
    /// ranges and Stores, so result handles had no recorded position -
    /// any has_later_live_load check against a result handle would
    /// fail and the producing Store would be wrongly classified as
    /// having no later live Load.  The merged `ExpressionScopeIndex`
    /// must record result-handle positions so the scope-leak filter
    /// and the dead-store gate both see consistent positions.
    ///
    /// Atomic exercises the `Option<Handle>`-bearing arm; the same
    /// codepath in the index walker covers all other result-bearing
    /// statement variants (WorkGroupUniformLoad, Call, Subgroup*,
    /// RayQueryFunction::Proceed).
    #[test]
    fn scope_index_records_statement_bound_result_handles() {
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let ptr = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            naga::Span::UNDEFINED,
        );
        let val = arena.append(
            naga::Expression::Literal(naga::Literal::U32(1)),
            naga::Span::UNDEFINED,
        );
        // Using a Literal placeholder for the result handle keeps the
        // test self-contained; the index doesn't validate expression
        // shape, only handle identity.
        let atomic_result = arena.append(
            naga::Expression::Literal(naga::Literal::U32(2)),
            naga::Span::UNDEFINED,
        );
        let wg_result = arena.append(
            naga::Expression::Literal(naga::Literal::U32(3)),
            naga::Span::UNDEFINED,
        );

        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Atomic {
                pointer: ptr,
                fun: naga::AtomicFunction::Add,
                value: val,
                result: Some(atomic_result),
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::WorkGroupUniformLoad {
                pointer: ptr,
                result: wg_result,
            },
            naga::Span::UNDEFINED,
        );

        let idx = ExpressionScopeIndex::build(&body, &arena);

        // Statement-bound result handles get the position of their
        // emitting statement.  Atomic is at position 0,
        // WorkGroupUniformLoad at 1.
        assert_eq!(idx.handle_position(atomic_result), Some(0));
        assert_eq!(idx.handle_position(wg_result), Some(1));

        // Both result handles are inside the body's subtree.
        assert!(idx.is_in_subtree(&body, atomic_result));
        assert!(idx.is_in_subtree(&body, wg_result));
    }

    /// The empty-block interval `[enter, enter)` must reject every
    /// handle (no handle is "inside" an empty block).  Loop
    /// continuing blocks are routinely empty in WGSL output, so the
    /// scope-leak filter's `is_in_subtree` would mis-classify
    /// handles if the empty case were wrong.
    #[test]
    fn scope_index_empty_block_subtree_membership() {
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let e0 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            naga::Span::UNDEFINED,
        );
        // Body: Emit(e0)                  pos 0
        //       Block { /* empty */ }     pos 1 (no descendants)
        let inner = naga::Block::new();
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(e0, e0)),
            naga::Span::UNDEFINED,
        );
        body.push(naga::Statement::Block(inner), naga::Span::UNDEFINED);

        let idx = ExpressionScopeIndex::build(&body, &arena);

        // e0 is in body's subtree but NOT in the empty inner block.
        let inner_ref = match &body[1] {
            naga::Statement::Block(b) => b,
            _ => unreachable!(),
        };
        assert!(idx.is_in_subtree(&body, e0));
        assert!(!idx.is_in_subtree(inner_ref, e0));
    }

    /// Pre-emit expressions (Literal, Constant, LocalVariable, ...)
    /// are not bound by any statement and are in scope everywhere in
    /// the function.  `handle_position` must return `None` for them,
    /// and the subtree-membership check correctly treats them as
    /// outside (the filter sites `||` with `needs_pre_emit` to keep
    /// them).
    #[test]
    fn scope_index_pre_emit_handles_have_no_position() {
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        // A literal that is NEVER emitted - it could be used as a
        // pointer/value expression but no Emit statement covers it.
        let unused_literal = arena.append(
            naga::Expression::Literal(naga::Literal::U32(42)),
            naga::Span::UNDEFINED,
        );
        let body = naga::Block::new(); // empty body
        let idx = ExpressionScopeIndex::build(&body, &arena);
        assert_eq!(idx.handle_position(unused_literal), None);
        assert!(!idx.is_in_subtree(&body, unused_literal));
    }

    #[test]
    fn block_scope_leak_regression_switch_case() {
        // Same shape inside a switch case body.  `has_default` and no
        // fall-through make the switch eligible for meet-over-branches,
        // so the post-switch cache would otherwise have carried the
        // in-case value forward.
        let source = r#"
fn f(sel: u32, a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    switch sel {
        case 0u: {
            let temp = vec3<f32>(a, b, c);
            x = temp;
        }
        default: {
            x = vec3<f32>(0.0);
        }
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(0u, 1.0, 2.0, 3.0); }
"#;
        let (_, _module) = run_pass(source);
    }

    /// Statement-bound result handles (CallResult, AtomicResult,
    /// WorkGroupUniformLoadResult, SubgroupBallotResult,
    /// SubgroupOperationResult, RayQueryProceedResult) are NOT bound
    /// inside `Emit` ranges - they are let-bound by naga's WGSL writer
    /// at the statement's containing block.  If `Statement::Block { x =
    /// helper(); }` stores a `CallResult` into a local, the cache entry
    /// `Local(x) -> CallResult_handle` must be filtered out at the
    /// closing brace just like Emit'd values.  Pre-fix
    /// `collect_emitted_handles_in_block` only walked `Emit` ranges
    /// and missed every statement-bound result, leaking forwarding
    /// targets past the brace.
    #[test]
    fn block_scope_leak_regression_call_result_in_block() {
        let source = r#"
fn helper(a: f32, b: f32, c: f32) -> vec3<f32> { return vec3<f32>(a, b, c); }
fn f(a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    {
        x = helper(a, b, c);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(1.0, 2.0, 3.0); }
"#;
        let (_, _module) = run_pass(source);
    }

    /// Same scope-leak shape but exercising `Statement::If` branch
    /// containing the CallResult-seeded Store.
    #[test]
    fn block_scope_leak_regression_call_result_in_if_branch() {
        let source = r#"
fn helper(a: f32, b: f32, c: f32) -> vec3<f32> { return vec3<f32>(a, b, c); }
fn f(cond: bool, a: f32, b: f32, c: f32) -> vec3<f32> {
    var x: vec3<f32>;
    if cond {
        x = helper(a, b, c);
    } else {
        x = vec3<f32>(0.0);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(true, 1.0, 2.0, 3.0); }
"#;
        let (_, _module) = run_pass(source);
    }

    /// Same scope-leak shape exercising `Statement::Atomic` whose
    /// `result` is bound at the statement site (not in an Emit).
    #[test]
    fn block_scope_leak_regression_atomic_result_in_block() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
fn f() -> u32 {
    var x: u32;
    {
        x = atomicAdd(&counter, 1u);
    }
    let post = x;
    return post;
}
@compute @workgroup_size(1) fn main() { _ = f(); }
"#;
        let (_, _module) = run_pass(source);
    }

    /// Regression for the Block-arm cache-staleness bug.
    ///
    /// Setup: a local `var failed = false;` followed by `failed |= X;`
    /// chains, where SOME `failed |= X;` statements live inside nested
    /// `{ ... }` blocks.
    ///
    /// Pre-fix bug: on exit from a nested `Statement::Block` that wrote
    /// to `failed`, the cache rollback restored the pre-block init
    /// entry (`Local(failed) -> Literal(false)`) without invalidating
    /// it.  A subsequent `Load(failed)` inside another nested block
    /// would cache-hit on the stale init and be forwarded to
    /// `Literal(false)`, producing `failed = false | X` in the
    /// generator output - structurally valid WGSL that OVERWRITES
    /// `failed`'s accumulated state with `X`, dropping every earlier
    /// `failed |= ...`.  Naga's validator does not catch this because
    /// it checks structural validity, not semantic equivalence with
    /// the source.
    ///
    /// Fix: after rollback, drop cache entries for any local that was
    /// modified inside the block - mirrors the existing invalidation
    /// step in the `If`, `Switch`, and `Loop` arms.
    ///
    /// This `var failed = false; { failed |= ...; }` shape occurs in real
    /// shaders, where pre-fix output dropped accumulator state at several of
    /// the nested-block writes.
    ///
    /// Signal: emit the optimised IR back to WGSL and check that the
    /// pattern `failed=false|` does NOT appear in writes inside nested
    /// blocks.  An init-forwarded first write is acceptable (init IS
    /// `false`), but subsequent writes inside nested blocks must read
    /// `failed`'s current value, not the pre-block init.
    #[test]
    fn block_arm_does_not_forward_stale_init_after_inner_block_writes() {
        let source = r#"
fn test_fn(a: bool, b: bool, c: bool) -> bool {
    var failed = false;
    {
        failed |= a;
    }
    {
        failed |= b;
    }
    {
        failed |= c;
    }
    return failed;
}

@compute @workgroup_size(1) fn main() {
    _ = test_fn(true, false, false);
}
"#;
        let (_, module) = run_pass(source);

        // Emit the optimised IR back to WGSL.  Bypass naga's
        // capability gate for f16-only features; this test uses only
        // booleans.
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module should validate");
        let mut out = String::new();
        let mut writer =
            naga::back::wgsl::Writer::new(&mut out, naga::back::wgsl::WriterFlags::empty());
        writer.write(&module, &info).expect("should emit WGSL");

        // Strip whitespace for robust pattern matching across naga
        // formatting tweaks.
        let stripped: String = out.chars().filter(|c| !c.is_whitespace()).collect();

        // Pre-fix bug signature: writes inside nested blocks would
        // emit as `failed = false | X` (or some form where the LHS of
        // the OR is the literal `false` instead of `Load(failed)`).
        // With the fix, the first write may still forward the init
        // (`failed = false | a`) - that one is correct because the
        // init IS `false`.  Subsequent writes must NOT carry that
        // pattern.  We check that the count of `failed=false|`
        // occurrences is at most one (the init-forwarded first
        // write).
        let bad_pattern_count = stripped.matches("failed=false|").count();
        assert!(
            bad_pattern_count <= 1,
            "expected at most 1 init-forwarded `failed=false|` (the first \
             write), found {} occurrences.  Pre-fix bug forwards \
             Load(failed) inside nested blocks to the stale init, \
             producing additional `failed=false|X` writes that drop \
             accumulator state.  Output:\n{}",
            bad_pattern_count,
            out,
        );
    }
}
