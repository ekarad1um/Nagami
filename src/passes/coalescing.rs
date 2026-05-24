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
//!
//! On top of the live-range check, a per-block first-touch gate
//! (`coalesce_safe`, set by [`mark_block_first`]) refuses to coalesce
//! any local whose first observed action in some control-flow scope
//! is a non-`Store` operation.  Such a local reads the slot's prior
//! contents on at least one runtime path (either via zero-init on the
//! first execution of the scope, or via a loop back-edge on
//! subsequent iterations), and aliasing it would substitute another
//! local's last-written value for those expected bytes.  The gate
//! is over-conservative on shapes like `x = 1.0; if c { let y = x; }`
//! where the outer Store dominates the inner Load - some valid
//! coalesces are lost - but it is correct without per-path dataflow
//! and never silently miscompiles.
//!
//! # Partial writes to aggregate locals
//!
//! Aggregate locals (`vec`, `mat`, `array`, `struct`) can be
//! partially written via `Access` / `AccessIndex` chains -
//! `v.x = ...;` writes one component, `arr[0] = ...;` writes one
//! element, leaving the rest at WGSL's promised zero-init.  Without
//! per-element tracking, coalescing such a local with a prior local
//! L would leak L's residue into the unwritten bytes.  The
//! [`ElementInit`] state attached to each local tracks which
//! elements have been written by which kind of Store; the gate
//! at every Load site then refuses to coalesce locals reading
//! elements that are not provably written on every reaching path.
//! Two patterns interplay:
//!
//! - **Fully covered by partial Stores** (safe): `v.x=...; v.y=...;
//!   v.z=...; let p=v;` - three bits set, full coverage, Load safe.
//! - **Partial Store + uncovered Read** (unsafe): `arr[0]=...; let
//!   v=arr[2];` - bit 0 set, reading bit 2 - Load fails coverage,
//!   `coalesce_safe = false`.
//!
//! Control flow merges by intersection (`If` arms only contribute
//! guarantees that hold on BOTH paths).  `Switch` and `Loop` stay
//! conservative - their writes never propagate, accepting some
//! missed optimisations to avoid mis-classifying case-fallthrough
//! or early-`break` paths.

use std::collections::{HashMap, HashSet};

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
///
/// `coalesce_safe` reflects a per-control-flow-scope correctness
/// gate: it stays `true` iff in *every* block scope where the local
/// is touched, the first touch within that scope is a write
/// (`Statement::Store` or `CooperativeStore` destination), AND every
/// Read of the local resolves to an element provably written before
/// that Read on every reaching path.  Either gate failing means at
/// least one runtime execution reads the slot's pre-coalesce
/// contents - the prior local's residue instead of either
/// zero-init or the local's own writes - so the local cannot be
/// safely coalesced.  Element-coverage tracking (see [`ElementInit`]
/// and [`ElementSpec`]) provides the second gate; without it, the
/// pre-existing partial-write hazard for aggregates - `arr[0] = 1;
/// let v = arr[2];` - would slip through.
#[derive(Debug, Clone, Copy)]
struct LocalUse {
    ty: naga::Handle<naga::Type>,
    first: usize,
    last: usize,
    used: bool,
    init_is_none: bool,
    coalesce_safe: bool,
}

/// Upper bound on the number of elements / members / columns we
/// track per aggregate local.  Aggregates larger than this are
/// treated as "untrackable" - any partial Store of them still
/// updates `ElementInit::any_indeterminate_write`, and any
/// subsequent Read flags `coalesce_safe = false`.  Sixty-four
/// elements covers every WGSL vector, matrix, the typical struct,
/// and arrays up to length 64; longer arrays fall back to the
/// conservative path.
const MAX_TRACKED_ELEMENTS: u32 = 64;

/// Which part of an aggregate local a pointer-expression chain
/// touches.  Computed by [`resolve_local_and_element`].
#[derive(Debug, Clone, Copy)]
enum ElementSpec {
    /// Direct pointer to the LocalVariable expression - the
    /// statement touches the entire local.  For a Store this is a
    /// full overwrite; for a Load it reads every byte.
    Full,
    /// `AccessIndex(LocalVariable(L), i)` with a one-level chain -
    /// touches element / member / column `i`.  Indices `>=`
    /// [`MAX_TRACKED_ELEMENTS`] degrade to [`ElementSpec::Dynamic`]
    /// inside [`ElementInit`]'s update logic.
    Index(u32),
    /// Either an `Access` with a runtime-computed index, or a
    /// nested AccessIndex into a sub-aggregate (which we don't
    /// resolve to a single element id).  Treated conservatively:
    /// reads require full coverage; writes don't contribute to
    /// per-element coverage.
    Dynamic,
}

/// Per-aggregate element-initialization summary.  Used to detect
/// partial-Store-then-uncovered-Read patterns the per-block first-
/// touch gate alone cannot see (because the first touch in scope
/// IS a Store - just a partial one - so the Load that follows is
/// not the "first touch" yet still observes uninitialised bytes).
#[derive(Debug, Clone, Copy)]
struct ElementInit {
    /// Number of distinct elements / members / columns in the
    /// local's type; `0` when the type is untrackable (runtime-
    /// sized array, override-sized array, opaque type, aggregate
    /// larger than [`MAX_TRACKED_ELEMENTS`], etc.).  An
    /// `element_count` of `0` makes [`Self::is_fully_covered`]
    /// require an explicit full Store - matching the conservative
    /// behaviour we want for shapes we cannot prove coverage for.
    element_count: u32,
    /// Bitset of element indices explicitly written so far, where
    /// bit `i` corresponds to a top-level `Store` whose pointer was
    /// `AccessIndex(LocalVariable(L), i)` for `i < MAX_TRACKED_ELEMENTS`.
    elements_written: u64,
    /// `true` once a direct `Store(LocalVariable(L), value)` has
    /// fired - the entire slot is unconditionally written from this
    /// point.  Subsumes all per-element bits.
    fully_written: bool,
}

impl ElementInit {
    fn new(element_count: u32) -> Self {
        Self {
            element_count,
            elements_written: 0,
            fully_written: false,
        }
    }

    /// `true` iff every byte of the local is provably written.
    /// Used for full Loads and dynamic-index Loads.
    fn is_fully_covered(&self) -> bool {
        if self.fully_written {
            return true;
        }
        if self.element_count == 0 || self.element_count > MAX_TRACKED_ELEMENTS {
            return false;
        }
        let mask = if self.element_count == 64 {
            !0u64
        } else {
            (1u64 << self.element_count) - 1
        };
        (self.elements_written & mask) == mask
    }

    /// `true` iff element `idx` is provably written.  Used for
    /// constant-index Loads.  Out-of-range indices fall back to
    /// requiring full coverage.
    fn covers_element(&self, idx: u32) -> bool {
        if self.fully_written {
            return true;
        }
        if idx >= MAX_TRACKED_ELEMENTS {
            return false;
        }
        (self.elements_written & (1u64 << idx)) != 0
    }

    /// Merge by intersection - only bits / flags set in BOTH
    /// inputs survive.  Used when joining the two arms of an `If`
    /// at the merge point: a local is provably initialised post-If
    /// only if both branches wrote it.
    fn intersect(self, other: Self) -> Self {
        Self {
            element_count: self.element_count.max(other.element_count),
            elements_written: self.elements_written & other.elements_written,
            fully_written: self.fully_written && other.fully_written,
        }
    }

    /// `true` iff this state carries any information worth
    /// persisting in the parent block's `local_init` map.  Used to
    /// avoid storing empty entries.
    fn is_empty(self) -> bool {
        !self.fully_written && self.elements_written == 0
    }
}

/// Number of elements / members / columns in `ty`, capped at
/// `MAX_TRACKED_ELEMENTS + 1` (so `is_fully_covered` correctly
/// rejects larger arrays).  Returns `0` for types whose internal
/// structure we cannot enumerate (pointers, images, samplers,
/// runtime- / override-sized arrays, atomic types, etc.).
fn element_count_for_type(ty: naga::Handle<naga::Type>, types: &naga::UniqueArena<naga::Type>) -> u32 {
    match types[ty].inner {
        naga::TypeInner::Scalar(_) => 1,
        naga::TypeInner::Vector { size, .. } => size as u32,
        naga::TypeInner::Matrix { columns, rows, .. } => (columns as u32) * (rows as u32),
        naga::TypeInner::Array {
            size: naga::ArraySize::Constant(n),
            ..
        } => {
            let n = n.get();
            if n > MAX_TRACKED_ELEMENTS {
                MAX_TRACKED_ELEMENTS + 1
            } else {
                n
            }
        }
        naga::TypeInner::Struct { ref members, .. } => {
            let n = members.len() as u32;
            if n > MAX_TRACKED_ELEMENTS {
                MAX_TRACKED_ELEMENTS + 1
            } else {
                n
            }
        }
        _ => 0,
    }
}

/// Walk a pointer-expression chain back to its root `LocalVariable`
/// AND classify which part of the local the chain targets.  This is
/// the element-aware sibling of [`resolve_ptr_to_local`].
///
/// One-level constant chains (`AccessIndex(LocalVariable(L), i)`)
/// give [`ElementSpec::Index`]; runtime-indexed `Access` chains and
/// nested aggregate paths collapse to [`ElementSpec::Dynamic`] -
/// the analysis treats them as "we don't know which element", so
/// they don't contribute to per-element coverage and any subsequent
/// Read requires full coverage.
fn resolve_local_and_element(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<(naga::Handle<naga::LocalVariable>, ElementSpec)> {
    match expressions[expr] {
        naga::Expression::LocalVariable(lh) => Some((lh, ElementSpec::Full)),
        naga::Expression::AccessIndex { base, index } => {
            let (local, parent) = resolve_local_and_element(base, expressions)?;
            // Only a single-level `AccessIndex` directly off a
            // `LocalVariable` can be tracked as a specific element
            // index.  Anything further (`AccessIndex(AccessIndex(L,
            // 0), 1)`) is a nested aggregate; we cannot describe
            // "L's slot, element-of-element-0, sub-element-1" with
            // our flat bitset, so collapse to `Dynamic`.
            let spec = if matches!(parent, ElementSpec::Full) {
                ElementSpec::Index(index)
            } else {
                ElementSpec::Dynamic
            };
            Some((local, spec))
        }
        naga::Expression::Access { base, .. } => {
            let (local, _) = resolve_local_and_element(base, expressions)?;
            Some((local, ElementSpec::Dynamic))
        }
        _ => None,
    }
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
            changed += coalesce_function_locals(function, &module.types);
        }
        for entry in module.entry_points.iter_mut() {
            changed += coalesce_function_locals(&mut entry.function, &module.types);
        }

        Ok(changed > 0)
    }
}

/// Rewrite every `LocalVariable` expression in `function` to use the
/// coalesced representative.  Returns the number of expression
/// references rewritten (used by the caller as a change flag).
fn coalesce_function_locals(
    function: &mut naga::Function,
    types: &naga::UniqueArena<naga::Type>,
) -> usize {
    if function.local_variables.is_empty() {
        return 0;
    }

    let usage = collect_local_usage(function, types);
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

/// Build the per-local [`LocalUse`] table.  Pre-resolves every
/// `Load` expression to `(root_local, element_spec)` so the DFS
/// below never re-walks pointer chains, and pre-computes each
/// local's element count from `types` so the element-coverage
/// analysis has constant-time access during the scan.
fn collect_local_usage(
    function: &naga::Function,
    types: &naga::UniqueArena<naga::Type>,
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
                    // Default true; clamped to `false` the first
                    // time any block scope first-touches this
                    // local as a non-Store action OR a Load is
                    // observed for an element that has not been
                    // provably written on every reaching path.
                    coalesce_safe: true,
                },
            )
        })
        .collect::<HashMap<_, _>>();

    // Pre-resolve every `Load` to `(root_local, element_spec)` so
    // the DFS attributes reads to the correct handle without
    // repeating the pointer walk, AND knows which element of an
    // aggregate the read targets so element-coverage analysis
    // can fire at the use site.
    let mut load_to_local_and_element: HashMap<
        naga::Handle<naga::Expression>,
        (naga::Handle<naga::LocalVariable>, ElementSpec),
    > = HashMap::new();
    for (eh, expr) in function.expressions.iter() {
        if let naga::Expression::Load { pointer } = *expr
            && let Some(pair) = resolve_local_and_element(pointer, &function.expressions)
        {
            load_to_local_and_element.insert(eh, pair);
        }
    }

    // Element counts indexed by local handle so the scan does not
    // re-walk type arenas per touch.
    let mut local_element_count: HashMap<naga::Handle<naga::LocalVariable>, u32> = HashMap::new();
    for (handle, local) in function.local_variables.iter() {
        local_element_count.insert(handle, element_count_for_type(local.ty, types));
    }

    // DFS the statement tree with monotonic positions; each local
    // records its minimum `first` and maximum `last` across every
    // access to approximate its live range.  `local_init` is the
    // entry-state for element-coverage tracking; we start empty
    // at the function body (no writes yet) and pass `&mut` so
    // each recursive call can update it.  The returned writes /
    // post-state are unused at function-body scope (there is no
    // enclosing scope to propagate to).
    let mut pos = 0usize;
    let mut local_init: HashMap<naga::Handle<naga::LocalVariable>, ElementInit> = HashMap::new();
    let _ = scan_block_usage(
        &function.body,
        &function.expressions,
        &load_to_local_and_element,
        &local_element_count,
        &mut pos,
        &mut usage,
        &mut local_init,
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
/// lane packer can skip locals that never appear.  The
/// `coalesce_safe` flag is updated *separately* by [`mark_block_first`]
/// so per-block first-touch tracking is independent of the source-
/// order monotonic position used for live ranges.
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

/// Record a touch within the current block scope and, if this is the
/// first touch in this scope AND the action is not a Store, clear the
/// `coalesce_safe` flag on the local.  The `block_seen` set is fresh
/// per recursive `scan_block_usage` call so that each If arm, Switch
/// case body, Loop body, Loop continuing, and nested `Statement::Block`
/// gets its own first-touch ledger.
///
/// This is the workhorse that closes both the loop-carried hazard
/// (loop body's first touch reads the back-edge value) and the
/// if/else hazard (one arm reads zero-init while the other writes).
fn mark_block_first(
    usage: &mut HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
    block_seen: &mut HashSet<naga::Handle<naga::LocalVariable>>,
    local: naga::Handle<naga::LocalVariable>,
    is_store: bool,
) {
    if block_seen.insert(local) && !is_store {
        if let Some(info) = usage.get_mut(&local) {
            info.coalesce_safe = false;
        }
    }
}

/// DFS traversal that attributes every read, store, call argument,
/// and pointer-flavoured statement operand to the root local it
/// ultimately touches, widening that local's live range AND
/// maintaining per-block first-touch tracking via the freshly-allocated
/// `block_seen` set (each recursive call gets its own, so nested
/// control-flow scopes are independent).
///
/// Returns the set of locals this block *unconditionally writes
/// before exiting* - i.e., every runtime control-flow path through
/// the block performs at least one top-level `Store` to the local
/// before reaching the block's end.  The caller uses this to treat a
/// nested `If` whose both arms unconditionally write a local (or a
/// nested `Block` that unconditionally writes one) as a Store-first
/// touch in the parent scope, so a subsequent Load that reads the
/// guaranteed-written value is not mis-flagged as the parent's
/// first touch.  Without this propagation, perfectly safe patterns
/// like `if c { x = a; } else { x = b; } let y = x;` would have
/// coalescing refused on `x` because the parent scope's syntactic
/// first touch of `x` is the post-If Load.
fn scan_block_usage(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    load_to_local_and_element: &HashMap<
        naga::Handle<naga::Expression>,
        (naga::Handle<naga::LocalVariable>, ElementSpec),
    >,
    local_element_count: &HashMap<naga::Handle<naga::LocalVariable>, u32>,
    pos: &mut usize,
    usage: &mut HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
    local_init: &mut HashMap<naga::Handle<naga::LocalVariable>, ElementInit>,
) -> HashSet<naga::Handle<naga::LocalVariable>> {
    // Per-block first-touch ledger.  Reset (by virtue of being a fresh
    // local on every recursive entry) at every control-flow scope:
    // function body, each If arm, each Switch case, Loop body and
    // continuing, and each nested `Statement::Block`.  Within this
    // scope, the *first* touch of a given local decides whether the
    // local stays `coalesce_safe`; later touches in the same scope do
    // not relax the verdict.
    let mut block_seen: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();
    // Per-block "unconditional writes" set.  Membership means every
    // runtime path through this block (so far) performs at least one
    // top-level `Store` to the local before the block exits.  A
    // direct `Store` adds the local; an `If` whose both arms
    // unconditionally write contributes the intersection of the two
    // arms' write sets; a nested `Block` whose body unconditionally
    // writes contributes its returned set verbatim.  `Switch` and
    // `Loop` are intentionally not propagated (Switch needs default-
    // case + fall-through + per-case write-set analysis; a `Loop`
    // body's writes only count when the loop is provably executed
    // and contains no early `break`/`Return`).
    let mut block_writes: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();

    // Helper closure-equivalents are expanded inline below to keep
    // the borrow checker happy with `local_init`'s `&mut`.

    for stmt in block {
        let current = *pos;
        *pos += 1;
        match stmt {
            naga::Statement::Emit(range) => {
                // `Emit` carries `Load` expressions, which are reads;
                // a Load is never the kind of write that overrides the
                // shared slot's prior value.  In addition to the
                // first-touch gate, check element coverage: a Load of
                // an element that hasn't been provably written on
                // every reaching path observes the slot's
                // pre-coalesce contents (zero-init in the user's
                // mental model, prior local's residue after a
                // coalesce).
                for h in range.clone() {
                    if let Some(&(local, spec)) = load_to_local_and_element.get(&h) {
                        mark_used(usage, local, current);
                        mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                        // Coverage check; uncovered Reads make
                        // coalescing observably wrong.
                        if !load_covers(local_init.get(&local), spec) {
                            if let Some(info) = usage.get_mut(&local) {
                                info.coalesce_safe = false;
                            }
                        }
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                // Top-level `Store` to a local.  Update the
                // element-coverage state; only a direct
                // `LocalVariable` pointer fully overrides the slot
                // (and thereby contributes to the unconditional-
                // writes set and to the first-touch gate as a Store).
                // Access-chained Stores write one element / member /
                // column at a time; they leave the other bytes with
                // whatever the slot held, so a coalesce with a prior
                // local would leak that local's residue into the
                // unwritten parts.
                if let Some((local, spec)) = resolve_local_and_element(*pointer, expressions) {
                    mark_used(usage, local, current);
                    let element_count = local_element_count.get(&local).copied().unwrap_or(0);
                    let init = local_init
                        .entry(local)
                        .or_insert_with(|| ElementInit::new(element_count));
                    let is_full_store = update_init_for_store(init, spec);
                    // The first-touch gate gets `is_store = true`
                    // for BOTH full and partial Stores: a partial
                    // Store still ACTIVELY touches the slot (it is
                    // not the "first action in scope is a Load that
                    // sees zero-init / loop-carry" case the gate
                    // exists to catch).  Whether the partial Store
                    // is safe is decided by the per-element coverage
                    // check at the subsequent Load - that is the
                    // gate that detects "read of an unwritten
                    // element after a partial Store".  Marking the
                    // first-touch as Load here would over-restrict
                    // the very common pattern of fully initialising
                    // a vector via three swizzle Stores
                    // (`v.x=...; v.y=...; v.z=...;`) before reading.
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    // Only full Stores contribute to the
                    // unconditional-writes propagation: a partial
                    // Store doesn't unconditionally claim the slot.
                    if is_full_store {
                        block_writes.insert(local);
                    }
                }
            }
            naga::Statement::Call { arguments, .. } => {
                // The callee may load through the pointer before
                // storing (or never store at all); the slot's prior
                // value can reach the callee.  Conservative: treat as
                // a read.  This is independent of element tracking
                // (the callee can read or write any bytes).
                for &arg in arguments {
                    if let Some(local) = resolve_ptr_to_local(arg, expressions) {
                        mark_used(usage, local, current);
                        mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                // Atomic ops are read-modify-write; the prior value is
                // observed.  Conservative: treat as a read.
                if let Some(local) = resolve_ptr_to_local(*pointer, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // The ray-query state object's prior bytes can matter
                // for subsequent ops; conservative read.
                if let Some(local) = resolve_ptr_to_local(*query, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                // `TraceRay` reads the payload pointer as input and may
                // write it; conservative read (the prior value reaches
                // the callee).
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(local) = resolve_ptr_to_local(*payload, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                }
            }
            naga::Statement::CooperativeStore { target, data } => {
                // Both operands of a cooperative store can resolve to
                // function-locals and BOTH must be tracked for
                // correctness, with opposite read/write semantics:
                //
                // - `target` is the source matrix VALUE.  Per naga's
                //   validator (`naga/valid/function.rs:1660`) it must
                //   resolve to `CooperativeMatrix`; if the IR routes
                //   that through a `LocalVariable(_)` expression, the
                //   matrix is loaded out of that local - a READ.
                //   Check element coverage just like an Emit'd Load.
                // - `data.pointer` is the destination POINTER.  Per
                //   `naga/valid/function.rs:1681` its address space
                //   must include `STORE`, and `AddressSpace::Function`
                //   does (LOAD | STORE per `proc/mod.rs:214`), so a
                //   function-local is a legitimate destination.  The
                //   matrix value is WRITTEN into that local's slot;
                //   update element-coverage like a regular Store.
                if let Some((local, spec)) = resolve_local_and_element(*target, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    if !load_covers(local_init.get(&local), spec) {
                        if let Some(info) = usage.get_mut(&local) {
                            info.coalesce_safe = false;
                        }
                    }
                }
                if let Some((local, spec)) = resolve_local_and_element(data.pointer, expressions) {
                    mark_used(usage, local, current);
                    let element_count = local_element_count.get(&local).copied().unwrap_or(0);
                    let init = local_init
                        .entry(local)
                        .or_insert_with(|| ElementInit::new(element_count));
                    let is_full_store = update_init_for_store(init, spec);
                    // Same `is_store = true` rationale as the
                    // regular `Store` arm: the first-touch gate
                    // treats this as a Store touch (no flag);
                    // element-coverage at the subsequent Load is
                    // what fires for partial-write hazards.
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    if is_full_store {
                        block_writes.insert(local);
                    }
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                // Snapshot the entry-state of `local_init` so we can
                // run each arm against a fresh copy, then merge the
                // two arms' post-states by intersection.  Only
                // initialisation guarantees that hold on BOTH paths
                // are post-If safe; if accept writes `v.x` and reject
                // writes `v.y`, neither bit survives the intersection
                // and a post-If `let p = v` is correctly flagged
                // uncovered.
                let entry_init = local_init.clone();
                let accept_writes = scan_block_usage(
                    accept,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                let accept_init = std::mem::replace(local_init, entry_init.clone());
                let reject_writes = scan_block_usage(
                    reject,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                let reject_init = std::mem::replace(local_init, entry_init);

                let newly_fully_covered =
                    merge_inits_into(local_init, accept_init, reject_init);

                // Locals written on every path through the If
                // (intersection of both arms) are unconditionally
                // written by the If as a whole.  Propagate them into
                // the parent scope's first-touch ledger as a Store
                // touch so a post-If `Load` is not mis-flagged.  Also
                // add them to `block_writes` so an enclosing block
                // can propagate further up.
                for &local in accept_writes.intersection(&reject_writes) {
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    block_writes.insert(local);
                }
                // Aggregates whose partial Stores on EVERY arm
                // happened to cover all elements (and thus became
                // post-If fully-covered via merge) are also
                // unconditional-writes from the parent's view: their
                // slot is fully claimed by the If.  Treat them
                // identically to direct full Stores for the
                // first-touch gate, otherwise the parent's
                // subsequent Load would be mis-flagged as
                // "first-touch is Load".
                for local in newly_fully_covered {
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    block_writes.insert(local);
                }
            }
            naga::Statement::Switch { cases, .. } => {
                // Switch deliberately stays conservative: each case
                // runs against the entry-state, results are discarded
                // (no merge, no propagation).  A precise analysis
                // would intersect every case's post-state (and require
                // `Default` + no fall-through), but the risk of
                // misclassifying an arm outweighs the missed
                // optimisation.
                let entry_init = local_init.clone();
                for case in cases {
                    *local_init = entry_init.clone();
                    let _ = scan_block_usage(
                        &case.body,
                        expressions,
                        load_to_local_and_element,
                        local_element_count,
                        pos,
                        usage,
                        local_init,
                    );
                }
                *local_init = entry_init;
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Loop body's writes do NOT propagate to the parent
                // scope's init state.  Even when reachable, an early
                // `break` / `Return` inside the body bypasses later
                // writes; without terminator-aware analysis we cannot
                // claim post-loop coverage.  Restore entry-state
                // after each recursion to keep the parent's view
                // unchanged.
                let entry_init = local_init.clone();
                let _ = scan_block_usage(
                    body,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                *local_init = entry_init.clone();
                let _ = scan_block_usage(
                    continuing,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                *local_init = entry_init;
            }
            naga::Statement::Block(inner) => {
                // A plain `Statement::Block` is a flat passthrough at
                // the IR level: writes inside happen on the parent
                // path and their element-coverage carries forward.
                // Pass `local_init` straight through (no snapshot, no
                // merge) so coverage flows naturally.  An
                // unconditional Store inside the inner block remains
                // unconditional from the parent's perspective.
                let inner_writes = scan_block_usage(
                    inner,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                for &local in &inner_writes {
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    block_writes.insert(local);
                }
            }
            _ => {}
        }
    }

    block_writes
}

/// Apply an [`ElementSpec`]-classified Store to a local's
/// [`ElementInit`].  Returns `true` iff the Store is a direct
/// `LocalVariable` (full overwrite) so the caller can route
/// first-touch / unconditional-write bookkeeping accordingly.
fn update_init_for_store(init: &mut ElementInit, spec: ElementSpec) -> bool {
    match spec {
        ElementSpec::Full => {
            init.fully_written = true;
            true
        }
        ElementSpec::Index(i) if i < MAX_TRACKED_ELEMENTS => {
            init.elements_written |= 1u64 << i;
            false
        }
        // Out-of-range constant index or runtime / nested Access:
        // we cannot localise the write to a specific bit, so it
        // contributes nothing to coverage and is treated as a
        // partial Store from the gate's perspective.
        ElementSpec::Index(_) | ElementSpec::Dynamic => false,
    }
}

/// Coverage check for a Load given the local's running
/// [`ElementInit`].  `None` means the local was never written in
/// this scope - any Load is uncovered.
fn load_covers(init: Option<&ElementInit>, spec: ElementSpec) -> bool {
    let Some(init) = init else { return false };
    match spec {
        ElementSpec::Full => init.is_fully_covered(),
        ElementSpec::Index(idx) => init.covers_element(idx),
        // Dynamic-index Read could touch any element; require full
        // coverage to be sure.
        ElementSpec::Dynamic => init.is_fully_covered(),
    }
}

/// Merge two arm post-states into the parent's `local_init` at the
/// If's merge point: a local is provably initialised post-If only
/// if the same guarantees hold on BOTH paths.  Locals touched on
/// only one path drop out of the parent's init view (they keep
/// whatever entry-state was in `local_init` BEFORE this merge call,
/// since the caller already reset to entry-state by `mem::replace`).
///
/// Returns the set of locals that became fully-covered via this
/// merge but were NOT fully-covered in the pre-If entry state.
/// The caller treats those as Store-first touches in the parent's
/// `block_seen` ledger, so a subsequent post-If Load is not
/// mis-flagged as the parent's first touch.
fn merge_inits_into(
    local_init: &mut HashMap<naga::Handle<naga::LocalVariable>, ElementInit>,
    accept: HashMap<naga::Handle<naga::LocalVariable>, ElementInit>,
    reject: HashMap<naga::Handle<naga::LocalVariable>, ElementInit>,
) -> HashSet<naga::Handle<naga::LocalVariable>> {
    let mut newly_fully_covered: HashSet<naga::Handle<naga::LocalVariable>> = HashSet::new();
    // Only locals appearing in BOTH arms can contribute new
    // guarantees; a write on one arm only is not a post-If
    // guarantee, so we skip those locals entirely (their parent
    // state is whatever was there before the If, which is the
    // correct conservative answer).
    for (local, a) in accept {
        if let Some(b) = reject.get(&local).copied() {
            let merged = a.intersect(b);
            if merged.is_empty() {
                // Intersection produced no new knowledge; keep the
                // parent's pre-If entry as-is.
                continue;
            }
            let was_fully_covered = local_init
                .get(&local)
                .map(|i| i.is_fully_covered())
                .unwrap_or(false);
            // Combine with parent's pre-If entry: the post-If state
            // is at least as informative as the pre-If state, with
            // the intersection added.  Use union-of-bits to combine.
            local_init
                .entry(local)
                .and_modify(|existing| {
                    existing.elements_written |= merged.elements_written;
                    existing.fully_written |= merged.fully_written;
                    existing.element_count = existing.element_count.max(merged.element_count);
                })
                .or_insert(merged);
            if !was_fully_covered
                && local_init
                    .get(&local)
                    .map(|i| i.is_fully_covered())
                    .unwrap_or(false)
            {
                newly_fully_covered.insert(local);
            }
        }
    }
    newly_fully_covered
}

/// Pack disjoint live ranges into type-keyed lanes and emit an alias
/// map from each coalesced local onto its lane representative.
///
/// Only `used && init_is_none && coalesce_safe` locals participate.
/// The combined gate ensures the local's first observed value comes
/// from a write that immediately overrides whatever the shared slot
/// held, so neither (a) explicit initialisers nor (b) zero-init
/// reads of a stale slot can leak across coalesced boundaries.
/// `coalesce_safe` (set by per-block first-touch tracking; see
/// [`mark_block_first`]) closes two distinct hazards the prior
/// DFS-order ranges missed:
///   - Loop body: the first iteration reads the slot's value at
///     loop-entry; subsequent iterations read the previous
///     iteration's writes via the back-edge.  Either way the slot
///     must hold the local's own value, not a coalesced predecessor's.
///   - If/else: one arm may write the local while the other arm
///     reads it (zero-init or pre-if value); coalescing breaks the
///     read arm even though DFS source order saw the write first.
///
/// Within each `ty` bucket, locals are assigned to the lane whose
/// latest `last` is still strictly earlier than the candidate's
/// `first`, which greedily maximises reuse of already-hot lanes
/// while still respecting non-overlap.
fn build_alias_map(
    usage: &HashMap<naga::Handle<naga::LocalVariable>, LocalUse>,
) -> HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::LocalVariable>> {
    let mut locals = usage
        .iter()
        .filter_map(|(&handle, info)| {
            (info.used && info.init_is_none && info.coalesce_safe).then_some(LocalSpan {
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

    /// Count how many distinct LocalVariable handles still appear in
    /// the entry point's expression arena.  Coalescing rewrites
    /// `Expression::LocalVariable(victim)` references to point at the
    /// representative, so a successful coalesce reduces this count by
    /// the number of victims.  Used by the loop-carried regression to
    /// catch the case where the buggy pass coalesces two locals into
    /// one even though one of them is loop-carried.
    fn distinct_local_handles_referenced(module: &naga::Module) -> usize {
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
    fn no_coalesce_loop_carried_local_with_inner_only_local() {
        // Regression for the loop-carried hazard.  `carried` is read
        // at the *top* of every loop iteration - the very first
        // touch within the loop body is therefore a `Load`, served
        // by either the zero-init slot (iteration 1) or the previous
        // iteration's `Store` (iteration 2+).  `inner` is written
        // later in the same body and is funneled into an init'd
        // accumulator that survives across the loop, so `carried`
        // is NOT used after the loop exits.  DFS-order positions
        // would put `inner.first` strictly after `carried.last` and
        // the greedy packer would coalesce them - on iteration 2
        // the first read of `carried` would then see the previous
        // iteration's `inner` value left in the shared slot.
        //
        // The fix is per-block first-touch tracking in
        // `scan_block_usage`: each control-flow scope gets its own
        // `block_seen` ledger and, when the *first* touch of a local
        // in a scope is not a Store, the local is flagged
        // `coalesce_safe = false`.  Inside the loop body the first
        // touch of `carried` is the `Load` for `let v = carried`,
        // which trips the flag and makes `carried` ineligible for
        // coalescing - regardless of what its DFS source-order
        // first touch was elsewhere in the function.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var carried: f32;
    var inner: f32;
    var output: f32 = 0.0;
    loop {
        let v = carried;
        carried = v + 1.0;
        inner = v * 100.0;
        output = output + inner;
        if v > 100.0 { break; }
    }
    return vec4f(output, 0.0, 0.0, 1.0);
}
"#;

        let (_, module) = run_pass(source);

        // Both `carried` and `inner` must retain a distinct
        // LocalVariable handle in the expression arena.  A buggy
        // coalesce that aliased one into the other would rewrite every
        // `Expression::LocalVariable(victim)` to the representative,
        // dropping the distinct-handle count from 2 to 1.  `output`
        // has an initializer and is therefore ineligible for
        // coalescing regardless of the back-edge analysis.
        assert_eq!(
            distinct_local_handles_referenced(&module),
            3,
            "loop-carried local `carried` must not share a slot with `inner`, \
             which is written inside the same loop body (miscompile risk)"
        );
    }

    #[test]
    fn no_coalesce_when_branch_arm_reads_before_any_outer_store() {
        // Companion regression to the loop-carried case: the *same*
        // class of hazard manifests through control-flow branches.
        // `b` is written in the `accept` arm and read in the `reject`
        // arm of an `If` - DFS source order visits the `Store` (in
        // accept) before the `Load` (in reject), so the prior DFS-
        // first-action gate would have marked `b` as Store-first
        // and eligible for coalescing.  But at runtime the cond=false
        // path reads `b`'s zero-init slot value, and aliasing `b`
        // with `a` (which gets `1.0` written to its slot before the
        // `If`) would silently substitute `1.0` for that zero.
        //
        // Per-block first-touch tracking catches this: the reject
        // arm's `block_seen` ledger sees `b`'s first touch as a
        // `Load`, flipping `b.coalesce_safe` to `false`.  After the
        // fix, `b` retains a distinct LocalVariable handle in the
        // expression arena.
        let source = r#"
@fragment
fn fs_main(@location(0) cond: f32) -> @location(0) vec4f {
    var a: f32;
    var b: f32;
    a = 1.0;
    let x = a;
    if cond > 0.5 {
        b = 5.0;
    } else {
        let y = b;
        _ = y;
    }
    return vec4f(x, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            2,
            "branch-arm-reads-before-write hazard: `b`, which the reject \
             arm reads via zero-init, must not share a slot with `a`, \
             whose pre-if `Store` would otherwise leak into that read"
        );
    }

    #[test]
    fn coalesces_local_written_unconditionally_in_both_if_arms() {
        // Companion to `no_coalesce_when_branch_arm_reads_before_any_outer_store`.
        // When BOTH arms of an `If` unconditionally write `b` before
        // the `If` exits, the post-`If` Load of `b` reads one of those
        // writes - never the slot's pre-`If` value - so the per-block
        // first-touch gate should NOT mark `b` unsafe just because
        // the parent scope's syntactic first touch (the post-`If` Load)
        // is a Load.  The unconditional-writes propagation in
        // `scan_block_usage` treats the `If` as a Store touch in the
        // parent ledger.
        //
        // Without that propagation, this shader regressed by ~90 bytes
        // versus baseline on real shader corpora (e.g.
        // data/extra-test4/bug/tint/913.wgsl) because none of `b` /
        // similar locals could share slots.
        let source = r#"
@fragment
fn fs_main(@location(0) cond: f32) -> @location(0) vec4f {
    var a: f32;
    var b: f32;
    a = 1.0;
    let x = a;
    if cond > 0.5 {
        b = 5.0;
    } else {
        b = 7.0;
    }
    let y = b;
    return vec4f(x + y, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        // `a` and `b` have disjoint live ranges - `a` ends at `let x`
        // before the `If`, `b` starts inside the `If` and is read
        // afterward.  With unconditional-writes propagation the
        // post-`If` Load is not the parent's first touch of `b`, so
        // `b` stays `coalesce_safe` and shares `a`'s slot - one
        // LocalVariable handle in the expression arena.
        assert_eq!(
            distinct_local_handles_referenced(&module),
            1,
            "both-arms-write should be safely coalescable: when every \
             control-flow path through an `If` stores into `b`, the \
             post-`If` Load reads a guaranteed-written value and the \
             pre-`If` slot contents are dead - `b` can share a slot \
             with `a` whose range ended before the `If`"
        );
    }

    #[test]
    fn no_coalesce_partial_write_followed_by_uncovered_element_read() {
        // Regression for the partial-write-to-aggregate hazard.  An
        // aggregate local that is partially written (e.g.
        // `arr[0] = 1.0;`) leaves the other elements at WGSL's
        // promised zero-init.  Coalescing such a local with a prior
        // local L that wrote actual bytes into the shared slot would
        // substitute L's residue for those zero-init reads - a silent
        // miscompile.  The element-coverage analysis in
        // `scan_block_usage` flags the local `coalesce_safe = false`
        // the moment a Load of an uncovered element appears.
        //
        // L (fully overwritten) precedes arr (only one element
        // written).  Reading `arr[2]` afterwards expects 0.0 per
        // WGSL spec; if `arr` were coalesced into L's slot it would
        // read L[2]'s last value instead.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var l_full: array<f32, 4>;
    l_full = array<f32, 4>(10.0, 20.0, 30.0, 40.0);
    let l_sum = l_full[0] + l_full[1] + l_full[2] + l_full[3];

    var arr: array<f32, 4>;
    arr[0] = 1.0;
    let v = arr[2];
    return vec4f(l_sum + v, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        // `arr` must retain its own distinct LocalVariable handle:
        // it was only partially written (element 0), so reading
        // `arr[2]` requires the WGSL zero-init guarantee that
        // coalescing with `l_full`'s slot would silently break.
        assert_eq!(
            distinct_local_handles_referenced(&module),
            2,
            "aggregate `arr` partially-written via `arr[0] = ...` then \
             read via `arr[2]` must not share a slot with `l_full`, \
             whose fully-written bytes would leak into the supposed \
             zero-init read"
        );
    }

    #[test]
    fn coalesces_aggregate_fully_initialised_via_partial_writes() {
        // Companion positive: when every element of an aggregate is
        // explicitly written before any Read, the slot is fully
        // claimed by the local and coalescing with a same-typed
        // prior local is safe even though the writes were partial.
        // Three partial Stores (`arr[0]`, `arr[1]`, `arr[2]`) plus
        // a direct `arr[3]` cover all four bits in `elements_written`;
        // `is_fully_covered` then returns true and the Load is not
        // flagged.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var first: array<f32, 4>;
    first[0] = 10.0;
    first[1] = 20.0;
    first[2] = 30.0;
    first[3] = 40.0;
    let s1 = first[0] + first[1] + first[2] + first[3];

    var second: array<f32, 4>;
    second[0] = 1.0;
    second[1] = 2.0;
    second[2] = 3.0;
    second[3] = 4.0;
    let s2 = second[0] + second[1] + second[2] + second[3];

    return vec4f(s1 + s2, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        // `first` and `second` have disjoint live ranges and both
        // are FULLY covered by their respective four element-writes.
        // The coalescer should merge them into one slot, leaving
        // exactly one distinct LocalVariable handle referenced.
        assert_eq!(
            distinct_local_handles_referenced(&module),
            1,
            "aggregate fully covered by per-element Stores should be \
             treated as fully initialised - element-coverage analysis \
             must allow coalescing of `first` and `second` here"
        );
    }

    #[test]
    fn no_coalesce_partial_write_in_only_one_if_arm_then_full_read() {
        // CF variant of the partial-write hazard.  If `accept`
        // writes `arr[0]` and `reject` writes `arr[1]`, the post-If
        // element-coverage is the intersection of `{0}` and `{1}` -
        // the empty set.  A subsequent full Load of `arr` is then
        // uncovered (no element provably written on every path),
        // and coalescing with a prior fully-written local would
        // leak that local's residue into the uncovered elements.
        let source = r#"
@fragment
fn fs_main(@location(0) cond: f32) -> @location(0) vec4f {
    var l_full: vec2<f32>;
    l_full = vec2<f32>(10.0, 20.0);
    let l_sum = l_full.x + l_full.y;

    var v: vec2<f32>;
    if cond > 0.5 {
        v.x = 1.0;
    } else {
        v.y = 2.0;
    }
    let p = v.x + v.y;
    return vec4f(l_sum + p, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            2,
            "If with heterogeneous partial-writes in each arm must not \
             allow coalescing of `v` with `l_full`: the intersection \
             of `{{x}}` and `{{y}}` is empty, so the post-If read of \
             `v` is uncovered and would observe leaked bytes"
        );
    }

    #[test]
    fn coalesces_aggregate_fully_initialised_in_both_if_arms() {
        // CF positive: when BOTH arms of an If fully cover the
        // aggregate, the merge by intersection still yields full
        // coverage, so the post-If read is safe and coalescing
        // proceeds.
        let source = r#"
@fragment
fn fs_main(@location(0) cond: f32) -> @location(0) vec4f {
    var first: vec2<f32>;
    first = vec2<f32>(10.0, 20.0);
    let s1 = first.x + first.y;

    var second: vec2<f32>;
    if cond > 0.5 {
        second.x = 1.0;
        second.y = 2.0;
    } else {
        second.x = 3.0;
        second.y = 4.0;
    }
    let p = second.x + second.y;
    return vec4f(s1 + p, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            1,
            "both-arms full-coverage via partial writes should merge \
             via intersection to full coverage, leaving `second` \
             eligible to coalesce with `first`"
        );
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

        let usage = collect_local_usage(&function, &module.types);
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

        let usage = collect_local_usage(&function, &module.types);
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

    #[test]
    fn cooperative_store_data_pointer_extends_destination_local_live_range() {
        // The companion gap to the `target`-tracking case above: when
        // `CooperativeStore.data.pointer` resolves to a function-local
        // `dest`, the cooperative store WRITES the matrix value into
        // `dest`'s slot.  Without tracking `data.pointer`, `dest`'s
        // live range stops at its source-order last direct touch -
        // potentially BEFORE the cooperative store.  A later local
        // (`other`) whose range starts after `dest`'s tracked-last but
        // BEFORE the cooperative store could then be coalesced into
        // `dest`'s slot; when the cooperative store fires it
        // overwrites that shared slot with the matrix, clobbering
        // `other`'s value.  Subsequent reads of `other` would see the
        // matrix - a miscompile.
        //
        // Construct exactly that hazard:
        //   Store(dest, ...)         dest.first
        //   Emit(load_dest_a)        dest direct last touch
        //   Store(other, ...)        other.first (> dest direct last)
        //   Emit(load_other_a)       other direct last
        //   CooperativeStore { target: ptr_src, data.pointer: ptr_dest }
        //                            writes dest (UNTRACKED in old code!)
        //   Emit(load_other_b)       other read AFTER cooperative store
        //
        // Without `data.pointer` tracking:
        //   dest.range = [Store(dest), load_dest_a]
        //   other.range = [Store(other), load_other_b]
        //   dest.last < other.first => coalesce other into dest's slot
        //   at runtime: cooperative store writes matrix to dest's slot
        //               (= other's slot); load_other_b reads matrix
        //               instead of other's value.  Miscompile.
        //
        // With `data.pointer` tracking:
        //   dest.last is extended through the cooperative store, so it
        //   now overlaps other's range; the coalesce is correctly
        //   refused.
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

        let local_src = function.local_variables.append(
            naga::LocalVariable {
                name: Some("src".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let local_dest = function.local_variables.append(
            naga::LocalVariable {
                name: Some("dest".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let local_other = function.local_variables.append(
            naga::LocalVariable {
                name: Some("other".into()),
                ty: coop_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );

        let ptr_src = function.expressions.append(
            naga::Expression::LocalVariable(local_src),
            naga::Span::UNDEFINED,
        );
        let ptr_dest = function.expressions.append(
            naga::Expression::LocalVariable(local_dest),
            naga::Span::UNDEFINED,
        );
        let ptr_other = function.expressions.append(
            naga::Expression::LocalVariable(local_other),
            naga::Span::UNDEFINED,
        );
        let load_src = function.expressions.append(
            naga::Expression::Load { pointer: ptr_src },
            naga::Span::UNDEFINED,
        );
        let load_dest = function.expressions.append(
            naga::Expression::Load { pointer: ptr_dest },
            naga::Span::UNDEFINED,
        );
        let load_other_a = function.expressions.append(
            naga::Expression::Load { pointer: ptr_other },
            naga::Span::UNDEFINED,
        );
        let load_other_b = function.expressions.append(
            naga::Expression::Load { pointer: ptr_other },
            naga::Span::UNDEFINED,
        );
        let lit_one = function.expressions.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            naga::Span::UNDEFINED,
        );
        let stride = function.expressions.append(
            naga::Expression::Literal(naga::Literal::U32(16)),
            naga::Span::UNDEFINED,
        );

        // Statements layered to give dest.last < other.first under the
        // old (no-data-pointer-tracking) regime so the coalesce would
        // fire, and arrange the cooperative store + final other-read
        // to make the result observable.
        let mut body = naga::Block::new();
        // pos 0: src first (needed so target = ptr_src is "live")
        body.push(
            naga::Statement::Store {
                pointer: ptr_src,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        // pos 1: src read
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_src, load_src)),
            naga::Span::UNDEFINED,
        );
        // pos 2: dest first (direct)
        body.push(
            naga::Statement::Store {
                pointer: ptr_dest,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        // pos 3: dest direct last
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_dest, load_dest)),
            naga::Span::UNDEFINED,
        );
        // pos 4: other first
        body.push(
            naga::Statement::Store {
                pointer: ptr_other,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        // pos 5: other direct
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_other_a, load_other_a)),
            naga::Span::UNDEFINED,
        );
        // pos 6: cooperative store writes dest (THE UNTRACKED ONE)
        body.push(
            naga::Statement::CooperativeStore {
                target: ptr_src,
                data: naga::CooperativeData {
                    pointer: ptr_dest,
                    stride,
                    row_major: false,
                },
            },
            naga::Span::UNDEFINED,
        );
        // pos 7: other read AFTER the cooperative store - this is the
        // observation point where the miscompile would surface.
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_other_b, load_other_b)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        let usage = collect_local_usage(&function, &module.types);
        let info_dest = usage[&local_dest];
        let info_other = usage[&local_other];

        // After the fix `dest.last` reaches the cooperative-store
        // position (>= 6), overlapping with `other`'s range
        // [other.first(>=4), other.last(>=7)].
        assert!(
            info_dest.last >= info_other.first,
            "data.pointer tracking should extend dest's live range to overlap with other \
             (dest.last={}, other.first={})",
            info_dest.last,
            info_other.first,
        );

        // And the coalescer must refuse to merge `other` into `dest`'s
        // slot - otherwise the cooperative store would clobber `other`'s
        // value at the shared slot.
        let alias = build_alias_map(&usage);
        assert!(
            !alias.contains_key(&local_other) && !alias.contains_key(&local_dest),
            "data.pointer-as-local must extend that local's live range so an overlapping \
             local (`other`) cannot be coalesced into the slot the cooperative store writes; \
             alias = {alias:?}"
        );
    }
}
