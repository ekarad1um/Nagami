//! Variable coalescing: same-typed locals with disjoint live ranges share
//! one backing local, shrinking the declaration list and the rename
//! alphabet.
//!
//! Live ranges are `(first, last)` positions from a DFS statement walk; any
//! overlap in traversal order counts as simultaneously live, so only
//! unambiguously disjoint locals merge.  Locals with initialisers are
//! excluded (the init would have to be re-materialised at every alias site).
//!
//! DFS order alone misses two hazards: a loop body whose first touch of a
//! local is a read (served by zero-init on iteration 1 and by the back edge
//! afterwards), and an `if` whose one arm writes while the other reads.  A
//! per-scope first-touch gate (`coalesce_safe`) therefore refuses any local
//! whose first action in some control-flow scope is not a `Store`: such a
//! local reads the slot's prior contents on some path, and aliasing would
//! substitute another local's last value.  The gate is over-conservative
//! (`x = 1.0; if c { let y = x; }` is refused although the outer store
//! dominates) but correct without per-path dataflow.
//!
//! # Partial writes to aggregates
//!
//! `v.x = ..` / `arr[0] = ..` write one element and leave the rest at WGSL's
//! promised zero-init, so coalescing onto a prior local would leak its
//! residue into the unwritten bytes even though the first touch IS a store.
//! Each local carries an `ElementInit` bitset of written elements, and every
//! read site refuses coalescing unless the elements it touches are written
//! on every reaching path: `v.x=..; v.y=..; v.z=..; let p = v;` is covered
//! and safe, `arr[0]=..; let v = arr[2];` is not.  `If` merges by
//! intersection; `Switch` and `Loop` never propagate writes, since case
//! fall-through and early-`break` paths would otherwise be mis-classified.

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::pipeline::{Pass, PassContext};

/// Coalesce disjoint same-typed locals onto shared backing slots.
#[derive(Debug, Default)]
pub struct CoalescingPass;

/// Per-local liveness summary.  `coalesce_safe` stays `true` iff in every
/// block scope touching the local the first touch is a write (a `Store` /
/// `CooperativeStore` destination, or a child `If` / `Block` that
/// unconditionally writes it) AND every read resolves to elements provably
/// written on every reaching path; either gate failing means some execution
/// reads the slot's pre-coalesce contents.
#[derive(Debug, Clone, Copy)]
struct LocalUse {
    ty: naga::Handle<naga::Type>,
    first: usize,
    last: usize,
    used: bool,
    init_is_none: bool,
    coalesce_safe: bool,
}

/// Elements tracked per aggregate (one `u64` bit each).  Larger aggregates
/// can never reach full coverage, so whole and dynamic-index reads of them
/// refuse coalescing.  Covers every vector, matrix, typical struct and
/// arrays up to length 64.
const MAX_TRACKED_ELEMENTS: u32 = 64;

/// Which part of an aggregate local a pointer chain touches.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ElementSpec {
    /// The whole local: a full overwrite for a store, every byte for a load.
    Full,
    /// One-level `AccessIndex(LocalVariable(L), i)`: element / member /
    /// column `i`.
    Index(u32),
    /// Runtime-indexed `Access` or a nested chain: reads demand full
    /// coverage, writes add none.
    Dynamic,
}

/// Per-local written-element state; catches a partial store followed by an
/// uncovered read, which the first-touch gate cannot see (the first touch
/// IS a store, just a partial one).
#[derive(Debug, Clone, Copy)]
struct ElementInit {
    /// `0` for untrackable types (runtime- / override-sized arrays, opaque
    /// types) and `MAX_TRACKED_ELEMENTS + 1` for oversize aggregates; both
    /// make full coverage reachable only through an explicit full store.
    element_count: u32,
    /// Bit `i`: element `i` written through a one-level `AccessIndex` store.
    elements_written: u64,
    /// A whole-local store has fired; subsumes every bit.
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

    fn covers_element(&self, idx: u32) -> bool {
        if self.fully_written {
            return true;
        }
        if idx >= MAX_TRACKED_ELEMENTS {
            return false;
        }
        (self.elements_written & (1u64 << idx)) != 0
    }

    /// If-merge: only what BOTH arms wrote survives.  The inputs describe
    /// one local, so `element_count` agrees by construction; `max` is a
    /// no-op defence and the debug-assert catches drift.
    fn intersect(self, other: Self) -> Self {
        debug_assert!(
            self.element_count == other.element_count
                || self.element_count == 0
                || other.element_count == 0,
            "intersect of two ElementInits for the same local must agree on element_count \
             (got {} vs {})",
            self.element_count,
            other.element_count,
        );
        Self {
            element_count: self.element_count.max(other.element_count),
            elements_written: self.elements_written & other.elements_written,
            fully_written: self.fully_written && other.fully_written,
        }
    }

    fn is_empty(self) -> bool {
        !self.fully_written && self.elements_written == 0
    }
}

/// Trackable element count of `ty`: `MAX_TRACKED_ELEMENTS + 1` for oversize
/// aggregates (full coverage then needs a whole store) and `0` for types
/// with no enumerable structure (pointers, images, samplers, runtime- /
/// override-sized arrays, atomics).
fn element_count_for_type(
    ty: naga::Handle<naga::Type>,
    types: &naga::UniqueArena<naga::Type>,
) -> u32 {
    match types[ty].inner {
        naga::TypeInner::Scalar(_) => 1,
        naga::TypeInner::Vector { size, .. } => size as u32,
        // Matrices are addressed by COLUMN (`AccessIndex(m, i)`, `i` in
        // `0..columns`), so counting columns lets a column-built matrix
        // reach full coverage; sub-column writes (`m[i].x`) are depth-2 and
        // resolve to `Dynamic`, which still demands full coverage.
        naga::TypeInner::Matrix { columns, .. } => columns as u32,
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

/// Root `LocalVariable` of a pointer chain plus which part of it the chain
/// targets: a one-level constant `AccessIndex` is `Index`, a runtime
/// `Access` or deeper chain is `Dynamic`.
pub(crate) fn resolve_local_and_element(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<(naga::Handle<naga::LocalVariable>, ElementSpec)> {
    match expressions[expr] {
        naga::Expression::LocalVariable(lh) => Some((lh, ElementSpec::Full)),
        naga::Expression::AccessIndex { base, index } => {
            let (local, parent) = resolve_local_and_element(base, expressions)?;
            // The flat bitset cannot describe an element of an element, so
            // only a one-level `AccessIndex` off the local is tracked;
            // nested aggregates (struct of struct, array of struct) coalesce
            // only when overwritten whole.
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

/// A chain of disjoint live windows sharing one representative; `last` is
/// the chain's current end.
#[derive(Debug, Clone, Copy)]
struct Lane {
    representative: naga::Handle<naga::LocalVariable>,
    last: usize,
}

/// Sorted by `(ty, first, last, handle)` so lane packing sees a
/// deterministic, live-range-ordered input.
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

/// Per-local [`LocalUse`] table from one DFS of the body.
fn collect_local_usage(
    function: &naga::Function,
    types: &naga::UniqueArena<naga::Type>,
) -> HandleMap<naga::LocalVariable, LocalUse> {
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
                    coalesce_safe: true,
                },
            )
        })
        .collect::<HandleMap<_, _>>();

    // Loads are pre-resolved to `(root_local, element_spec)` so the DFS never
    // re-walks pointer chains; arena handles are dense, so a `Vec` indexed
    // by `handle.index()` beats a hash map.
    let mut load_to_local_and_element: Vec<
        Option<(naga::Handle<naga::LocalVariable>, ElementSpec)>,
    > = vec![None; function.expressions.len()];
    for (eh, expr) in function.expressions.iter() {
        if let naga::Expression::Load { pointer } = *expr
            && let Some(pair) = resolve_local_and_element(pointer, &function.expressions)
        {
            load_to_local_and_element[eh.index()] = Some(pair);
        }
    }

    let mut local_element_count: Vec<Option<u32>> = vec![None; function.local_variables.len()];
    for (handle, local) in function.local_variables.iter() {
        local_element_count[handle.index()] = Some(element_count_for_type(local.ty, types));
    }

    // The body's own write set has no enclosing scope to propagate to.
    let mut pos = 0usize;
    let mut local_init: HandleMap<naga::LocalVariable, ElementInit> = Default::default();
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

/// Widen `local`'s `(first, last)` window.  `coalesce_safe` is tracked
/// separately, per block, so the gate is independent of DFS position order.
fn mark_used(
    usage: &mut HandleMap<naga::LocalVariable, LocalUse>,
    local: naga::Handle<naga::LocalVariable>,
    pos: usize,
) {
    if let Some(info) = usage.get_mut(local) {
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

/// First-touch gate: a scope's first touch of `local` that is not a store
/// clears `coalesce_safe`.  `block_seen` is fresh per scope (function body,
/// each If arm, case body, loop body / continuing, nested block).
fn mark_block_first(
    usage: &mut HandleMap<naga::LocalVariable, LocalUse>,
    block_seen: &mut HandleSet<naga::LocalVariable>,
    local: naga::Handle<naga::LocalVariable>,
    is_store: bool,
) {
    if block_seen.insert(local)
        && !is_store
        && let Some(info) = usage.get_mut(local)
    {
        info.coalesce_safe = false;
    }
}

/// DFS attributing every read, store, call argument and pointer operand to
/// its root local, widening live ranges and running the first-touch and
/// element-coverage gates with a fresh `block_seen` per scope.
///
/// Returns the locals every path through `block` stores whole before its
/// end.  A child `If` (both arms) or `Block` that unconditionally writes a
/// local counts as a store-first touch of the parent scope, so
/// `if c { x = a; } else { x = b; } let y = x;` keeps `x` coalescable even
/// though the parent's syntactic first touch is the post-If load.
fn scan_block_usage(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    load_to_local_and_element: &[Option<(naga::Handle<naga::LocalVariable>, ElementSpec)>],
    local_element_count: &[Option<u32>],
    pos: &mut usize,
    usage: &mut HandleMap<naga::LocalVariable, LocalUse>,
    local_init: &mut HandleMap<naga::LocalVariable, ElementInit>,
) -> HandleSet<naga::LocalVariable> {
    let mut block_seen: HandleSet<naga::LocalVariable> = Default::default();
    // Locals stored whole on every path so far.  `Switch` / `Loop` never
    // contribute: a switch needs default + fall-through + per-case analysis,
    // and a loop body's writes count only when it provably runs to the end.
    let mut block_writes: HandleSet<naga::LocalVariable> = Default::default();

    for stmt in block {
        let current = *pos;
        *pos += 1;
        match stmt {
            naga::Statement::Emit(range) => {
                // Loads: first-touch gate plus coverage - an element not
                // provably written on every reaching path would read the
                // predecessor local's residue.
                for h in range.clone() {
                    if let Some(&(local, spec)) = load_to_local_and_element
                        .get(h.index())
                        .and_then(|o| o.as_ref())
                    {
                        mark_used(usage, local, current);
                        mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                        if !load_covers(local_init.get(local), spec)
                            && let Some(info) = usage.get_mut(local)
                        {
                            info.coalesce_safe = false;
                        }
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                // Only a whole-local store claims the slot for
                // `block_writes`; an access-chained store leaves the other
                // bytes as they were.
                if let Some((local, spec)) = resolve_local_and_element(*pointer, expressions) {
                    mark_used(usage, local, current);
                    let element_count = local_element_count
                        .get(local.index())
                        .and_then(|o| *o)
                        .unwrap_or(0);
                    let init = local_init
                        .entry(local)
                        .or_insert_with(|| ElementInit::new(element_count));
                    let is_full_store = update_init_for_store(init, spec);
                    // A partial store still counts as a store-first touch:
                    // the gate exists for reads of zero-init / loop-carried
                    // bytes, and the coverage check at the next read decides
                    // whether the partial stores suffice (otherwise
                    // `v.x=..; v.y=..; v.z=..;` before a read would be
                    // refused).
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    if is_full_store {
                        block_writes.insert(local);
                    }
                }
            }
            naga::Statement::Call { arguments, .. } => {
                // The callee may read through the pointer before writing (or
                // never write): a coverage-gated read.  The gate matters
                // because a partial store followed by the escape is the
                // scope's first touch, so the first-touch gate never fires.
                for &arg in arguments {
                    if let Some((local, spec)) = resolve_local_and_element(arg, expressions) {
                        mark_used(usage, local, current);
                        mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                        if !load_covers(local_init.get(local), spec)
                            && let Some(info) = usage.get_mut(local)
                        {
                            info.coalesce_safe = false;
                        }
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                // Read-modify-write: a coverage-gated read.
                if let Some((local, spec)) = resolve_local_and_element(*pointer, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    if !load_covers(local_init.get(local), spec)
                        && let Some(info) = usage.get_mut(local)
                    {
                        info.coalesce_safe = false;
                    }
                }
            }
            naga::Statement::RayQuery { query, .. } => {
                // The query object's prior bytes matter: a coverage-gated read.
                if let Some((local, spec)) = resolve_local_and_element(*query, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    if !load_covers(local_init.get(local), spec)
                        && let Some(info) = usage.get_mut(local)
                    {
                        info.coalesce_safe = false;
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => {
                // The payload is read as input: a coverage-gated read.
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some((local, spec)) = resolve_local_and_element(*payload, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    if !load_covers(local_init.get(local), spec)
                        && let Some(info) = usage.get_mut(local)
                    {
                        info.coalesce_safe = false;
                    }
                }
            }
            naga::Statement::CooperativeStore { target, data } => {
                // `target` must be a `CooperativeMatrix` value, which the
                // pointer-chain resolver never matches on validator-clean
                // IR; kept as defence in depth (over-extending a range is
                // safe, under-tracking miscompiles).  `data.pointer` is the
                // write destination: a regular store touch.
                if let Some((local, spec)) = resolve_local_and_element(*target, expressions) {
                    mark_used(usage, local, current);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ false);
                    if !load_covers(local_init.get(local), spec)
                        && let Some(info) = usage.get_mut(local)
                    {
                        info.coalesce_safe = false;
                    }
                }
                if let Some((local, spec)) = resolve_local_and_element(data.pointer, expressions) {
                    mark_used(usage, local, current);
                    let element_count = local_element_count
                        .get(local.index())
                        .and_then(|o| *o)
                        .unwrap_or(0);
                    let init = local_init
                        .entry(local)
                        .or_insert_with(|| ElementInit::new(element_count));
                    let is_full_store = update_init_for_store(init, spec);
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    if is_full_store {
                        block_writes.insert(local);
                    }
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                // Each arm runs on a copy of the entry state; the post-states
                // merge by intersection so only guarantees holding on BOTH
                // paths survive (accept writes `v.x`, reject `v.y`: neither
                // bit survives).
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

                let newly_fully_covered = merge_inits_into(local_init, accept_init, reject_init);

                // Locals stored whole on both arms are unconditional writes
                // of the If itself: a store-first touch here, propagated up.
                for &local in accept_writes.intersection(&reject_writes) {
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    block_writes.insert(local);
                }
                // Likewise aggregates whose partial stores reached full
                // coverage on both arms.
                for local in newly_fully_covered {
                    mark_block_first(usage, &mut block_seen, local, /*is_store=*/ true);
                    block_writes.insert(local);
                }
            }
            naga::Statement::Switch { cases, .. } => {
                // Deliberately conservative: each case runs on the entry
                // state and nothing merges or propagates.  A precise meet
                // needs Default + fall-through analysis, and mis-classifying
                // an arm miscompiles.
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
                // Nothing propagates out (an early break / Return bypasses
                // later writes), and body / continuing start from EMPTY
                // coverage: pre-loop coverage does not survive the back edge,
                // and a loop-carried read of a partially initialised
                // aggregate judged covered would keep the local coalesce-safe
                // while an aliased local clobbers the carried value.  Empty
                // is the sound meet over entry edges.
                let entry_init = local_init.clone();
                local_init.clear();
                let _ = scan_block_usage(
                    body,
                    expressions,
                    load_to_local_and_element,
                    local_element_count,
                    pos,
                    usage,
                    local_init,
                );
                local_init.clear();
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
                // A nested block is a flat passthrough: coverage and
                // unconditional writes flow through unchanged.
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
            // No catch-all: a new pointer-bearing naga variant must fail to
            // compile here rather than bypass the analysis.  Break / Continue
            // / Return / Kill / barriers touch no local; the rest take their
            // operands through a preceding `Emit`, whose loads are already
            // attributed above (`WorkGroupUniformLoad` requires a
            // `ptr<workgroup>`, never a function local).
            naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Return { .. }
            | naga::Statement::Kill
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_)
            | naga::Statement::WorkGroupUniformLoad { .. }
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::ImageStore { .. }
            | naga::Statement::ImageAtomic { .. } => {}
        }
    }

    block_writes
}

/// Record a store; `true` iff it overwrites the whole local.
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
        // Untrackable index: adds no coverage, i.e. a partial store.
        ElementSpec::Index(_) | ElementSpec::Dynamic => false,
    }
}

/// `None`: never written in this scope, so any read is uncovered.
fn load_covers(init: Option<&ElementInit>, spec: ElementSpec) -> bool {
    let Some(init) = init else { return false };
    match spec {
        ElementSpec::Full => init.is_fully_covered(),
        ElementSpec::Index(idx) => init.covers_element(idx),
        ElementSpec::Dynamic => init.is_fully_covered(),
    }
}

/// If-merge into the parent's `local_init`: a local is initialised post-If
/// only where BOTH arms agree; locals touched on one arm only keep the
/// parent's entry state.  Returns the locals the merge made fully covered,
/// which the caller treats as store-first touches so a post-If load is not
/// mis-flagged.
fn merge_inits_into(
    local_init: &mut HandleMap<naga::LocalVariable, ElementInit>,
    accept: HandleMap<naga::LocalVariable, ElementInit>,
    reject: HandleMap<naga::LocalVariable, ElementInit>,
) -> HandleSet<naga::LocalVariable> {
    let mut newly_fully_covered: HandleSet<naga::LocalVariable> = Default::default();
    for (local, a) in accept {
        if let Some(b) = reject.get(local).copied() {
            let merged = a.intersect(b);
            if merged.is_empty() {
                continue;
            }
            let was_fully_covered = local_init
                .get(local)
                .map(|i| i.is_fully_covered())
                .unwrap_or(false);
            // Post-If knowledge = entry knowledge + what both arms added.
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
                    .get(local)
                    .map(|i| i.is_fully_covered())
                    .unwrap_or(false)
            {
                newly_fully_covered.insert(local);
            }
        }
    }
    newly_fully_covered
}

/// Pack disjoint live ranges into type-keyed lanes; the result maps each
/// coalesced local to its lane representative.
///
/// Only `used && init_is_none && coalesce_safe` locals participate, so a
/// local's first observed value always comes from its own write and neither
/// an initialiser nor a zero-init / loop-carried read can leak across slots.
/// Within a type, a local joins the lane whose `last` is latest yet still
/// before its `first`, greedily reusing hot lanes.
fn build_alias_map(
    usage: &HandleMap<naga::LocalVariable, LocalUse>,
) -> HandleMap<naga::LocalVariable, naga::Handle<naga::LocalVariable>> {
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

    let mut lanes_by_type: HandleMap<naga::Type, Vec<Lane>> = Default::default();
    let mut alias = HandleMap::default();

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

/// Follow `alias` to the representative; a self-loop stops the walk.
fn resolve_alias(
    mut handle: naga::Handle<naga::LocalVariable>,
    alias: &HandleMap<naga::LocalVariable, naga::Handle<naga::LocalVariable>>,
) -> naga::Handle<naga::LocalVariable> {
    while let Some(next) = alias.get(handle).copied() {
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
            name_log: None,
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
            .collect::<HandleSet<_>>()
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

    /// Distinct `LocalVariable` handles still referenced; each coalesced
    /// victim lowers it by one.
    fn distinct_local_handles_referenced(module: &naga::Module) -> usize {
        module.entry_points[0]
            .function
            .expressions
            .iter()
            .filter_map(|(_, e)| match e {
                naga::Expression::LocalVariable(h) => Some(*h),
                _ => None,
            })
            .collect::<HandleSet<_>>()
            .len()
    }

    #[test]
    fn no_coalesce_loop_carried_local_with_inner_only_local() {
        // `carried`'s first touch in the loop body is a read (zero-init on
        // iteration 1, the back edge afterwards).  DFS order alone puts
        // `inner.first` after `carried.last`, and a shared slot would make
        // iteration 2 read `inner`'s value.
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

        // `output` has an initialiser and is ineligible regardless.
        assert_eq!(
            distinct_local_handles_referenced(&module),
            3,
            "loop-carried local `carried` must not share a slot with `inner`, \
             which is written inside the same loop body (miscompile risk)"
        );
    }

    #[test]
    fn no_coalesce_when_branch_arm_reads_before_any_outer_store() {
        // DFS order sees `b`'s store (accept) before its read (reject), but
        // the cond=false path reads `b`'s zero-init; aliasing onto `a` would
        // substitute `1.0`.
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
        // Both arms store `b` whole, so the post-If read never sees the
        // pre-If slot: the parent's syntactic first touch being that read
        // must not refuse `b`.
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
        // `arr[2]` must read WGSL's zero-init; sharing `l_full`'s slot would
        // read `l_full[2]`.
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
    fn no_coalesce_partial_write_then_pointer_escape_via_call() {
        // The uncovered read happens inside the callee, and the partial
        // store is the scope's first touch, so only the coverage gate on the
        // pointer-escape arm catches it.
        let source = r#"
fn sink(p: ptr<function, array<f32, 4>>) -> f32 { return (*p)[2]; }
@fragment
fn fs_main(@location(0) idx: f32) -> @location(0) vec4f {
    var l_full: array<f32, 4>;
    l_full = array<f32, 4>(10.0, 20.0, 30.0, 40.0);
    let l_sum = l_full[0] + l_full[1] + l_full[2] + l_full[3];

    var arr: array<f32, 4>;
    arr[i32(idx)] = 1.0;
    let v = sink(&arr);
    return vec4f(l_sum + v, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            2,
            "partially-written `arr` escaping by pointer to `sink` must not \
             coalesce onto `l_full` (the callee would read l_full's residue \
             where WGSL guarantees zero-init)"
        );
    }

    #[test]
    fn coalesces_fully_written_aggregate_escaping_by_pointer() {
        // The coverage gate is precise: a fully written escapee exposes no
        // residue.
        let source = r#"
fn sink(p: ptr<function, array<f32, 4>>) -> f32 { return (*p)[2]; }
@fragment
fn fs_main() -> @location(0) vec4f {
    var l_full: array<f32, 4>;
    l_full = array<f32, 4>(10.0, 20.0, 30.0, 40.0);
    let l_sum = l_full[0] + l_full[1] + l_full[2] + l_full[3];

    var arr: array<f32, 4>;
    arr = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    let v = sink(&arr);
    return vec4f(l_sum + v, 0.0, 0.0, 1.0);
}
"#;
        let (changed, module) = run_pass(source);
        assert!(
            changed,
            "fully-written aggregate escaping by pointer should still coalesce"
        );
        assert_eq!(
            distinct_local_handles_referenced(&module),
            1,
            "fully-written `arr` exposes no residue, so it should share \
             `l_full`'s slot"
        );
    }

    #[test]
    fn coalesces_matrix_fully_written_by_columns() {
        // Coverage is per column, so column-wise writes reach full coverage.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var p: mat2x2<f32>;
    p[0] = vec2<f32>(10.0, 20.0);
    p[1] = vec2<f32>(30.0, 40.0);
    let s1 = p[0].x + p[1].y;

    var m: mat2x2<f32>;
    m[0] = vec2<f32>(1.0, 2.0);
    m[1] = vec2<f32>(3.0, 4.0);
    let s2 = m[0].x + m[1].y;

    return vec4f(s1 + s2, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            1,
            "matrix fully written via per-column Stores should coalesce"
        );
    }

    #[test]
    fn no_coalesce_matrix_partially_written_by_columns() {
        // `m[1]` reads zero-init.
        let source = r#"
@fragment
fn fs_main() -> @location(0) vec4f {
    var p: mat2x2<f32>;
    p[0] = vec2<f32>(10.0, 20.0);
    p[1] = vec2<f32>(30.0, 40.0);
    let s1 = p[0].x + p[1].y;

    var m: mat2x2<f32>;
    m[0] = vec2<f32>(1.0, 2.0);
    let s2 = m[0].x + m[1].y;

    return vec4f(s1 + s2, 0.0, 0.0, 1.0);
}
"#;
        let (_, module) = run_pass(source);
        assert_eq!(
            distinct_local_handles_referenced(&module),
            2,
            "partially-column-written matrix must not coalesce (m[1] reads zero-init)"
        );
    }

    #[test]
    fn coalesces_aggregate_fully_initialised_via_partial_writes() {
        // Four element stores cover all four bits before the first read.
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
        // Post-If coverage is the intersection {x} & {y} = {}, so the read
        // of `v` is uncovered.
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
        // Both arms fully cover `second`, so the intersection is still full.
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
        // `a` is the TraceRay payload between the two locals' direct uses;
        // untracked, `a.last` would end before `b.first`.

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

        // Store(a); Emit(load_a); Store(b); TraceRay(payload = a); Emit(load_b)
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

        assert!(
            info_a.last >= info_b.first,
            "TraceRay should extend a's live range to overlap with b (a.last={}, b.first={})",
            info_a.last,
            info_b.first,
        );

        let alias = build_alias_map(&usage);
        assert!(
            alias.is_empty(),
            "overlapping locals should not be coalesced when TraceRay extends the range"
        );
    }

    #[test]
    fn cooperative_store_extends_local_live_range() {
        // `a` is the CooperativeStore target between the two locals' direct
        // uses.

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

        // Store(a); Emit(load_a); Store(b); CooperativeStore(target = a);
        // Emit(load_b)
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

        assert!(
            info_a.last >= info_b.first,
            "CooperativeStore should extend a's live range to overlap with b (a.last={}, b.first={})",
            info_a.last,
            info_b.first,
        );

        let alias = build_alias_map(&usage);
        assert!(
            alias.is_empty(),
            "overlapping locals should not be coalesced when CooperativeStore extends the range"
        );
    }

    #[test]
    fn cooperative_store_data_pointer_extends_destination_local_live_range() {
        // `data.pointer` is a write destination: `other`'s range starts
        // after `dest`'s last direct touch but before the cooperative store
        // into `dest`, so a shared slot would let that store clobber `other`
        // before its final read.
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

        // Store(src); Emit(load_src); Store(dest); Emit(load_dest);
        // Store(other); Emit(load_other_a);
        // CooperativeStore(target = src, data.pointer = dest);
        // Emit(load_other_b)
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Store {
                pointer: ptr_src,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_src, load_src)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: ptr_dest,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_dest, load_dest)),
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Store {
                pointer: ptr_other,
                value: lit_one,
            },
            naga::Span::UNDEFINED,
        );
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_other_a, load_other_a)),
            naga::Span::UNDEFINED,
        );
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
        body.push(
            naga::Statement::Emit(naga::Range::new_from_bounds(load_other_b, load_other_b)),
            naga::Span::UNDEFINED,
        );
        function.body = body;

        let usage = collect_local_usage(&function, &module.types);
        let info_dest = usage[&local_dest];
        let info_other = usage[&local_other];

        assert!(
            info_dest.last >= info_other.first,
            "data.pointer tracking should extend dest's live range to overlap with other \
             (dest.last={}, other.first={})",
            info_dest.last,
            info_other.first,
        );

        let alias = build_alias_map(&usage);
        assert!(
            !alias.contains_key(local_other) && !alias.contains_key(local_dest),
            "data.pointer-as-local must extend that local's live range so an overlapping \
             local (`other`) cannot be coalesced into the slot the cooperative store writes; \
             alias = {alias:?}"
        );
    }
}
