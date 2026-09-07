//! Mutated-load binding analysis: which emitted loads must become
//! `let` bindings because a write to their place intervenes before a use.

use crate::handle_set::HandleSet;
use crate::passes::expr_util::{const_index_value, visit_expression_children};
use rustc_hash::FxHashMap;

/// A pointer's memory location: a root variable plus one level of refinement.
/// Two places sharing a root with distinct constant first-level indices are
/// provably disjoint; anything coarser (`Whole`, a dynamic `Opaque` index)
/// aliases everything in the root.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceRoot {
    Local(naga::Handle<naga::LocalVariable>),
    Global(naga::Handle<naga::GlobalVariable>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Refine {
    /// The whole variable (no access applied off the root).
    Whole,
    /// A single statically-known field / element index off the root.
    Field(u32),
    /// A dynamically-indexed (or otherwise unknown) location in the root.
    Opaque,
}

#[derive(Clone, Copy)]
struct Place {
    root: PlaceRoot,
    refine: Refine,
}

/// Conservative may-alias: `false` only for proven-disjoint pairs.
fn places_may_alias(a: Place, b: Place) -> bool {
    if a.root != b.root {
        return false;
    }
    match (a.refine, b.refine) {
        (Refine::Field(x), Refine::Field(y)) => x == y,
        _ => true,
    }
}

/// Lower a pointer expression to its [`Place`]: the root variable plus the
/// access applied directly to it; deeper accesses keep that first-level
/// refinement (sub-locations of one component alias).  `None` when the root is
/// not a concrete variable (a function-argument pointer, an exotic pointer),
/// which the caller treats as aliasing everything.
fn resolve_place(
    pointer: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<Place> {
    match &expressions[pointer] {
        naga::Expression::LocalVariable(l) => Some(Place {
            root: PlaceRoot::Local(*l),
            refine: Refine::Whole,
        }),
        naga::Expression::GlobalVariable(g) => Some(Place {
            root: PlaceRoot::Global(*g),
            refine: Refine::Whole,
        }),
        naga::Expression::AccessIndex { base, index } => {
            let base_place = resolve_place(*base, expressions)?;
            if matches!(base_place.refine, Refine::Whole) {
                Some(Place {
                    root: base_place.root,
                    refine: Refine::Field(*index),
                })
            } else {
                Some(base_place)
            }
        }
        naga::Expression::Access { base, index } => {
            let base_place = resolve_place(*base, expressions)?;
            if matches!(base_place.refine, Refine::Whole) {
                let refine = const_index_value(*index, expressions)
                    .and_then(|v| u32::try_from(v).ok())
                    .map(Refine::Field)
                    .unwrap_or(Refine::Opaque);
                Some(Place {
                    root: base_place.root,
                    refine,
                })
            } else {
                Some(base_place)
            }
        }
        _ => None,
    }
}

/// `true` when a callee or another invocation could write this global, so a
/// load of it can go stale; immutable address spaces never produce a hazard.
fn global_is_writable(module: &naga::Module, g: naga::Handle<naga::GlobalVariable>) -> bool {
    match module.global_variables[g].space {
        naga::AddressSpace::Uniform
        | naga::AddressSpace::Handle
        | naga::AddressSpace::Immediate => false,
        naga::AddressSpace::Storage { access } => access.contains(naga::StorageAccess::STORE),
        _ => true,
    }
}

/// `true` when texture global `g` is a STORE-access storage texture, so a
/// `textureStore` / `textureAtomic` (here or in a callee) can stale a prior
/// `textureLoad`.  Textures live in `AddressSpace::Handle`, which
/// [`global_is_writable`] reports immutable: texture writability is a property
/// of the storage access, so the `ImageLoad` hazard must route through this
/// helper.  A `binding_array<texture_storage_*>` global is a `BindingArray`,
/// peeled one level (nested binding arrays are forbidden); without the peel a
/// `textureLoad(texs[i], ..)` would inline past a `textureStore` to the same
/// element.
fn image_is_writable_storage(module: &naga::Module, g: naga::Handle<naga::GlobalVariable>) -> bool {
    let mut inner = &module.types[module.global_variables[g].ty].inner;
    if let naga::TypeInner::BindingArray { base, .. } = inner {
        inner = &module.types[*base].inner;
    }
    matches!(
        inner,
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { access, .. },
            ..
        } if access.contains(naga::StorageAccess::STORE)
    )
}

/// A write a statement performs.  [`statement_write_effects`] classifies every
/// statement variant with no wildcard arm, so a new writing statement kind is a
/// compile error rather than a silent miss.
enum WriteEffect {
    Place(Place),
    /// May write some writable global (a callee, barrier, or param-pointer
    /// store): invalidates loads rooted at a global and every Unknown-place
    /// load (a param pointer may itself target a global).
    Globals,
}

impl WriteEffect {
    /// Whether this write could invalidate a tracked load at `load` (`None` =
    /// an Unknown place that aliases everything).
    fn invalidates(&self, load: &Option<Place>) -> bool {
        match self {
            WriteEffect::Globals => match load {
                None => true,
                Some(p) => matches!(p.root, PlaceRoot::Global(_)),
            },
            WriteEffect::Place(w) => match load {
                // A `None` (function-argument pointer) load reads caller memory
                // or a global, never a named local of THIS function, so a store
                // to a resolved LOCAL place cannot alias it; a store to a GLOBAL
                // might.
                None => !matches!(w.root, PlaceRoot::Local(_)),
                Some(p) => places_may_alias(*w, *p),
            },
        }
    }
}

/// Record the pointer-to-LOCAL write place a call argument exposes: a callee
/// taking `ptr<function, T>` may write the pointee, which is exactly the
/// argument's own place (a naga pointer argument is a root variable with
/// optional `Access` / `AccessIndex` refinement).  Pointers are non-storable,
/// so no sub-expression carries a second pointee and an index sub-expression is
/// only read; hence no recursion.  Global and param-pointer roots are covered
/// by the blanket [`WriteEffect::Globals`] every `Call` records.
fn collect_ptr_local_writes(
    arg: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    if matches!(
        expressions[arg],
        naga::Expression::LocalVariable(_)
            | naga::Expression::Access { .. }
            | naga::Expression::AccessIndex { .. }
    ) && let Some(p) = resolve_place(arg, expressions)
        && matches!(p.root, PlaceRoot::Local(_))
    {
        out.push(WriteEffect::Place(p));
    }
}

/// Every [`WriteEffect`] of one statement; nested blocks are walked separately.
fn statement_write_effects(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    use naga::Statement as S;
    match stmt {
        S::Store { pointer, .. } | S::Atomic { pointer, .. } => {
            match resolve_place(*pointer, expressions) {
                Some(p) => out.push(WriteEffect::Place(p)),
                // Through a function-argument pointer: any global, never a local.
                None => out.push(WriteEffect::Globals),
            }
        }
        // The destination is `data.pointer`; `target` is the stored matrix
        // value, a read.
        S::CooperativeStore { data, .. } => match resolve_place(data.pointer, expressions) {
            Some(p) => out.push(WriteEffect::Place(p)),
            None => out.push(WriteEffect::Globals),
        },
        S::Call { arguments, .. } => {
            out.push(WriteEffect::Globals);
            for &arg in arguments {
                collect_ptr_local_writes(arg, expressions, out);
            }
        }
        // A barrier makes other invocations' stores to shared globals
        // observable, so a pre-barrier load can differ from a post-barrier
        // re-read; `Globals` also over-invalidates private-space loads, a
        // harmless over-binding.
        S::ControlBarrier(_) | S::MemoryBarrier(_) | S::WorkGroupUniformLoad { .. } => {
            out.push(WriteEffect::Globals)
        }
        S::RayQuery { query, .. } => {
            out.push(WriteEffect::Globals);
            if let Some(p) = resolve_place(*query, expressions) {
                out.push(WriteEffect::Place(p));
            }
        }
        S::RayPipelineFunction(fun) => {
            out.push(WriteEffect::Globals);
            let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
            if let Some(p) = resolve_place(*payload, expressions) {
                out.push(WriteEffect::Place(p));
            }
        }
        // A pending `ImageLoad` reaches a storage texture, so its write must
        // invalidate it; the stored value / atomic operand is a read.
        S::ImageStore { image, .. } | S::ImageAtomic { image, .. } => {
            match resolve_place(*image, expressions) {
                Some(p) => out.push(WriteEffect::Place(p)),
                // A function-argument texture could be any texture global.
                None => out.push(WriteEffect::Globals),
            }
        }
        // Subgroup ops exchange register values across lanes: no memory access,
        // no ordering, nothing to stale.
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => {}
        S::Emit(_)
        | S::Block(_)
        | S::If { .. }
        | S::Switch { .. }
        | S::Loop { .. }
        | S::Return { .. }
        | S::Break
        | S::Continue
        | S::Kill => {}
    }
}

/// Every [`WriteEffect`] anywhere inside `block`, nested control flow included.
fn collect_block_write_effects(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    crate::passes::expr_util::for_each_statement(block, &mut |stmt| {
        statement_write_effects(stmt, expressions, out)
    });
}

/// An emitted `Load` still in flight: its place, and whether a write to that
/// place has been observed since its `Emit`.
#[derive(Clone)]
struct PendingLoad {
    /// `None` = an Unknown place (function-argument pointer), aliases all.
    place: Option<Place>,
    written: bool,
}

type Pending = FxHashMap<naga::Handle<naga::Expression>, PendingLoad>;

/// "Already entered" marks over the expression arena, one stamp per handle;
/// a walk is a generation, so starting one is a counter bump, not a fresh set.
struct Visited {
    stamps: Vec<u32>,
    generation: u32,
}

impl Visited {
    fn new(expression_count: usize) -> Self {
        Self {
            stamps: vec![0; expression_count],
            generation: 0,
        }
    }

    fn walk(&mut self) -> Walk<'_> {
        self.generation += 1;
        Walk(self)
    }
}

/// One walk's marks; only [`Visited::walk`] hands one out, so no walk can
/// start on a previous walk's marks.
struct Walk<'a>(&'a mut Visited);

impl Walk<'_> {
    /// `true` the first time `h` is entered in this walk.
    fn enter(&mut self, h: naga::Handle<naga::Expression>) -> bool {
        let slot = &mut self.0.stamps[h.index()];
        if *slot == self.0.generation {
            false
        } else {
            *slot = self.0.generation;
            true
        }
    }
}

/// Join two successor states into `a`: written on either path is written after
/// the join, and keys from both sides are kept so a branch-local load that
/// outlives its branch stays tracked.  Commutative, so a caller may run one
/// successor over the pre-branch state in place and fold the other in.
fn merge_pending_into(a: &mut Pending, b: Pending) {
    for (h, pl) in b {
        a.entry(h)
            .and_modify(|e| e.written |= pl.written)
            .or_insert(pl);
    }
}

/// Flag every in-flight load in `root`'s operand cone that is already marked
/// written: it is read AFTER its place was overwritten, so it must be bound.
/// `walk` keeps the traversal linear over shared sub-DAGs.
fn flag_used_loads(
    root: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    pending: &Pending,
    must_bind: &mut HandleSet<naga::Expression>,
    walk: &mut Walk<'_>,
) {
    if !walk.enter(root) {
        return;
    }
    if let Some(pl) = pending.get(&root)
        && pl.written
    {
        must_bind.insert(root);
        // A written `root` emits as a `let` at its own Emit site, freezing its
        // whole operand cone lexically before the write, nested written loads
        // included, so re-pinning a child changes nothing; a child also used
        // outside this parent is pinned at that other use (only `root` is
        // marked, and each statement starts a fresh walk).
        return;
    }
    visit_expression_children(&expressions[root], |child| {
        flag_used_loads(child, expressions, pending, must_bind, walk);
    });
}

fn analyze_block(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    pending: &mut Pending,
    must_bind: &mut HandleSet<naga::Expression>,
    visited: &mut Visited,
) {
    for stmt in block.iter() {
        analyze_statement(stmt, expressions, module, pending, must_bind, visited);
    }
}

/// Mark every in-flight load `written` whose place a write effect of `stmt` may
/// alias.  Monotone (only flips unwritten -> written) and skips already-written
/// loads, so it is idempotent across the loop pre-mark and the linear pass.
fn apply_writes(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut Pending,
) {
    let mut effects = Vec::new();
    statement_write_effects(stmt, expressions, &mut effects);
    if effects.is_empty() {
        return;
    }
    for pl in pending.values_mut() {
        if !pl.written && effects.iter().any(|e| e.invalidates(&pl.place)) {
            pl.written = true;
        }
    }
}

fn analyze_statement(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    pending: &mut Pending,
    must_bind: &mut HandleSet<naga::Expression>,
    visited: &mut Visited,
) {
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => {
            // Uses first: a write never occurs within an Emit, and a load defined
            // in this range is not yet pending, so a sibling consuming it is
            // correctly not flagged.
            let mut walk = visited.walk();
            for h in range.clone() {
                flag_used_loads(h, expressions, pending, must_bind, &mut walk);
            }
            for h in range.clone() {
                match &expressions[h] {
                    naga::Expression::Load { pointer } => {
                        let place = resolve_place(*pointer, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => global_is_writable(module, g),
                                PlaceRoot::Local(_) => true,
                            },
                            // Function-argument pointer: any later write may alias it.
                            None => true,
                        };
                        if track {
                            pending.insert(
                                h,
                                PendingLoad {
                                    place,
                                    written: false,
                                },
                            );
                        }
                    }
                    // A `textureLoad` stales like a buffer `Load` under a later
                    // `textureStore` / `textureAtomic` (or callee); gate on
                    // `image_is_writable_storage`, since `global_is_writable`
                    // reports textures immutable.
                    naga::Expression::ImageLoad { image, .. } => {
                        let place = resolve_place(*image, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => image_is_writable_storage(module, g),
                                // A texture is never a local; stay conservative.
                                PlaceRoot::Local(_) => true,
                            },
                            // Value-parameter texture: a callee may store to it.
                            None => true,
                        };
                        if track {
                            pending.insert(
                                h,
                                PendingLoad {
                                    place,
                                    written: false,
                                },
                            );
                        }
                    }
                    // Both read the query's CURRENT traversal state, which a
                    // later `Statement::RayQuery` (modeled as
                    // `WriteEffect::Place(query)`) stales like a `Store`; the
                    // query is always function-space, so no writability gate.
                    naga::Expression::RayQueryGetIntersection { query, .. }
                    | naga::Expression::RayQueryVertexPositions { query, .. } => {
                        pending.insert(
                            h,
                            PendingLoad {
                                place: resolve_place(*query, expressions),
                                written: false,
                            },
                        );
                    }
                    _ => {}
                }
            }
        }
        S::Block(inner) => analyze_block(inner, expressions, module, pending, must_bind, visited),
        S::If {
            condition,
            accept,
            reject,
        } => {
            flag_used_loads(
                *condition,
                expressions,
                pending,
                must_bind,
                &mut visited.walk(),
            );
            // One clone, not two: the reject arm may run over the pre-branch
            // state in place because the join is commutative.
            let mut accept_state = pending.clone();
            analyze_block(
                accept,
                expressions,
                module,
                &mut accept_state,
                must_bind,
                visited,
            );
            analyze_block(reject, expressions, module, pending, must_bind, visited);
            merge_pending_into(pending, accept_state);
        }
        S::Switch { selector, cases } => {
            flag_used_loads(
                *selector,
                expressions,
                pending,
                must_bind,
                &mut visited.walk(),
            );
            if cases.iter().any(|c| c.fall_through) {
                // A fall-through case chains into the next, so thread one state
                // through the cases sequentially; assuming a directly-entered
                // case ran after its predecessors only over-binds.
                for case in cases {
                    analyze_block(&case.body, expressions, module, pending, must_bind, visited);
                }
            } else {
                // Mutually exclusive cases: each from the pre-switch state, then
                // the union of the post-states.
                // The last case runs over the pre-switch state in place, so
                // `cases` costs one clone fewer than it has arms.
                let mut merged: Option<Pending> = None;
                for case in &cases[..cases.len().saturating_sub(1)] {
                    let mut case_state = pending.clone();
                    analyze_block(
                        &case.body,
                        expressions,
                        module,
                        &mut case_state,
                        must_bind,
                        visited,
                    );
                    match &mut merged {
                        None => merged = Some(case_state),
                        Some(m) => merge_pending_into(m, case_state),
                    }
                }
                if let Some(last) = cases.last() {
                    analyze_block(&last.body, expressions, module, pending, must_bind, visited);
                }
                if let Some(m) = merged {
                    merge_pending_into(pending, m);
                }
            }
        }
        S::Loop {
            body,
            continuing,
            break_if,
        } => {
            // Back-edge: a write anywhere in the loop runs after a load emitted
            // before it and before a next-iteration use, so pre-mark every
            // outer load the loop may write.
            let mut loop_writes = Vec::new();
            collect_block_write_effects(body, expressions, &mut loop_writes);
            collect_block_write_effects(continuing, expressions, &mut loop_writes);
            if !loop_writes.is_empty() {
                for pl in pending.values_mut() {
                    if !pl.written && loop_writes.iter().any(|e| e.invalidates(&pl.place)) {
                        pl.written = true;
                    }
                }
            }
            // Loads emitted inside the loop are re-evaluated each iteration
            // (only memory crosses the back-edge), so one linear pass over
            // body + continuing is exact for them.
            analyze_block(body, expressions, module, pending, must_bind, visited);
            analyze_block(continuing, expressions, module, pending, must_bind, visited);
            if let Some(h) = break_if {
                flag_used_loads(*h, expressions, pending, must_bind, &mut visited.walk());
            }
        }
        // Leaf statement: operands are uses, then its writes apply.
        _ => {
            let mut walk = visited.walk();
            crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
                flag_used_loads(h, expressions, pending, must_bind, &mut walk);
            });
            apply_writes(stmt, expressions, pending);
        }
    }
}

/// Every `Load` in `func` that must be `let`-bound rather than inlined because
/// the place it reads is written between the `Load`'s `Emit` and a use of its
/// value; inlining would relocate the read past the write and yield the
/// post-write value.  One forward pass over the structured statement tree,
/// over-approximating the hazard (binding is always safe and costs only
/// bytes): unresolved places, branches, loops and call / barrier write effects
/// are handled conservatively, while read-only globals and locals are never
/// flagged.
pub(super) fn compute_must_bind_loads(
    func: &naga::Function,
    module: &naga::Module,
) -> HandleSet<naga::Expression> {
    let mut pending: Pending = Default::default();
    let mut must_bind = Default::default();
    let mut visited = Visited::new(func.expressions.len());
    analyze_block(
        &func.body,
        &func.expressions,
        module,
        &mut pending,
        &mut must_bind,
        &mut visited,
    );
    must_bind
}
