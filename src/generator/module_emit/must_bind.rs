//! Mutated-load binding analysis: which emitted loads must become
//! `let` bindings because a write to their place intervenes before a use.

use crate::passes::expr_util::visit_expression_children;

/// The memory location a pointer refers to, resolved to a root variable
/// plus one level of refinement off that root.  Two places that share a
/// root but carry distinct *constant* first-level indices are provably
/// disjoint; anything coarser (`Whole` or a dynamic `Opaque` index)
/// conservatively aliases everything in the root.
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

/// Conservative may-alias test.  Sound direction: returns `true` whenever
/// the two places *might* name overlapping memory.  Only proven-disjoint
/// pairs (same root, distinct constant first-level indices) return `false`.
fn places_may_alias(a: Place, b: Place) -> bool {
    if a.root != b.root {
        return false;
    }
    match (a.refine, b.refine) {
        (Refine::Field(x), Refine::Field(y)) => x == y,
        // `Whole` contains every field; `Opaque` could be any field.
        _ => true,
    }
}

/// Lower a pointer expression to its [`Place`], walking `Access` /
/// `AccessIndex` chains down to the root variable.  The refinement is the
/// access applied DIRECTLY to the root (the first level); deeper accesses
/// keep that first-level refinement (conservative - sub-locations of the
/// same first-level component are treated as aliasing).
///
/// Returns `None` when the root is not a concrete variable (a
/// function-argument pointer, or an exotic pointer expression).  Such a
/// place is treated as "Unknown" by the caller and aliases everything.
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

/// The constant value of an index expression, if it is a non-negative
/// integer `Literal`; otherwise `None` (a dynamic or non-integer index).
fn const_index_value(
    index: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<u32> {
    let naga::Expression::Literal(lit) = &expressions[index] else {
        return None;
    };
    match lit {
        naga::Literal::U32(v) => Some(*v),
        naga::Literal::I32(v) => u32::try_from(*v).ok(),
        naga::Literal::U64(v) => u32::try_from(*v).ok(),
        naga::Literal::I64(v) => u32::try_from(*v).ok(),
        naga::Literal::AbstractInt(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

/// `true` when a callee or another invocation could write this global
/// (so a load of it can become stale).  Immutable address spaces
/// (`uniform`, resource handles, immediate/push-constant, read-only
/// storage) can never change and never produce a hazard.
fn global_is_writable(module: &naga::Module, g: naga::Handle<naga::GlobalVariable>) -> bool {
    match module.global_variables[g].space {
        naga::AddressSpace::Uniform
        | naga::AddressSpace::Handle
        | naga::AddressSpace::Immediate => false,
        naga::AddressSpace::Storage { access } => access.contains(naga::StorageAccess::STORE),
        // Function (locals), Private, WorkGroup, ray/task payloads: writable.
        _ => true,
    }
}

/// `true` when texture global `g` is a STORE-access storage texture, i.e.
/// a `textureStore`/`textureAtomic` (here or in a callee) can mutate it, so
/// a prior `textureLoad` of it can go stale.  Sampled textures and
/// read-only storage textures are immutable resources and never produce a
/// hazard.  Note: textures live in `AddressSpace::Handle`, for which
/// [`global_is_writable`] returns `false` - writability for a texture is a
/// property of its storage *access*, not its address space, so the
/// `ImageLoad` hazard test MUST route through this helper, not that one.
///
/// A `binding_array<texture_storage_*<...>>` global has type
/// `TypeInner::BindingArray`, not `Image`, so peel one level to its element
/// type before classifying (WGSL/naga forbid nested binding arrays, so a
/// single peel suffices).  Without it a `textureLoad(texs[i], ..)` would be
/// dropped from the hazard set and inlined past a `textureStore` to the same
/// element - a silent miscompile.
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

/// A write a statement performs, as seen by the load-hazard analysis.
///
/// Every `naga::Statement` variant is classified exhaustively in
/// [`statement_write_effects`] (no wildcard arm), so a future statement
/// kind that can write memory forces a compile error there rather than a
/// silent miss - this enum needs no catch-all "writes everything" case.
enum WriteEffect {
    /// Writes a specific resolved place.
    Place(Place),
    /// May write some writable global (a callee, barrier, or
    /// param-pointer store): invalidates loads rooted at a global, plus
    /// any Unknown-place load (a param pointer may itself target a global).
    Globals,
}

impl WriteEffect {
    /// Whether this write could invalidate a tracked load whose place is
    /// `load` (`None` = an Unknown place that aliases everything).
    fn invalidates(&self, load: &Option<Place>) -> bool {
        match self {
            WriteEffect::Globals => match load {
                None => true,
                Some(p) => matches!(p.root, PlaceRoot::Global(_)),
            },
            WriteEffect::Place(w) => match load {
                // A `None` (function-argument pointer) load reads caller memory
                // or a global - NEVER a named local of THIS function (the caller
                // cannot hold a pointer to a local that does not exist in its
                // scope).  So a store to a resolved LOCAL place can never alias
                // it; a store to a GLOBAL still might (the param could point
                // there), so stay conservative for globals.
                None => !matches!(w.root, PlaceRoot::Local(_)),
                Some(p) => places_may_alias(*w, *p),
            },
        }
    }
}

/// Record the pointer-to-LOCAL write place a call argument exposes: a callee
/// taking `ptr<function, T>` may write the pointee.  A naga pointer argument is
/// a root variable (`LocalVariable` / `GlobalVariable` / `FunctionArgument`)
/// with optional `Access`/`AccessIndex` refinement, which `resolve_place` walks
/// to the root - so the pointee is exactly the argument's own place.  Pointers
/// are non-storable (no loadable pointer values, no pointer aggregates), so no
/// sub-expression can carry a second writable pointee, and an index
/// sub-expression is a value the callee reads, never writes; hence NO recursion
/// into children.  Only a LOCAL root is recorded - pointers to globals, and
/// param-pointer roots (which resolve to `None`), are already covered by the
/// blanket [`WriteEffect::Globals`] every `Call` records.
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

/// Append every [`WriteEffect`] a single statement performs to `out`.
/// Control-flow statements (`Block`/`If`/`Switch`/`Loop`) contribute
/// nothing here - their nested blocks are walked separately.
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
                // A store through an unresolved (function-argument) pointer
                // could land in any global; it cannot reach our own locals.
                None => out.push(WriteEffect::Globals),
            }
        }
        // The write is through `data.pointer` (the destination); `target` is the
        // matrix VALUE being stored, a read - not a written place.  (Latent today:
        // the generator cannot yet emit CooperativeStore; kept correct so the
        // exhaustive match holds no silent miss.)
        S::CooperativeStore { data, .. } => match resolve_place(data.pointer, expressions) {
            Some(p) => out.push(WriteEffect::Place(p)),
            None => out.push(WriteEffect::Globals),
        },
        S::Call { arguments, .. } => {
            // A callee may write any global, plus any local it receives by pointer.
            out.push(WriteEffect::Globals);
            for &arg in arguments {
                collect_ptr_local_writes(arg, expressions, out);
            }
        }
        // Memory-synchronisation points make other invocations' prior stores
        // to shared globals observable, so a pre-barrier load of such a global
        // can differ from a post-barrier re-read.  (Conservatively `Globals`;
        // this also over-invalidates private-space loads - harmless over-binding,
        // since a barrier cannot change a per-invocation private value.)
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
        // Image stores/atomics mutate a storage texture.  A buffer `Load`
        // never reaches a texture, but an `ImageLoad` (registered as a
        // pending load below) does, so the write must invalidate it.  The
        // destination is `image`; the stored value / atomic operand is a
        // read handled by the use-detection path.
        S::ImageStore { image, .. } | S::ImageAtomic { image, .. } => {
            match resolve_place(*image, expressions) {
                Some(p) => out.push(WriteEffect::Place(p)),
                // An image reached through an unresolved (function-argument)
                // value could be any texture global; stay conservative.
                None => out.push(WriteEffect::Globals),
            }
        }
        // Subgroup operations exchange already-computed values across lanes via
        // registers; they perform NO memory access and impose no memory
        // ordering, so they cannot stale any load.  (Their argument operands
        // are still seen as uses via the leaf use-detection path.)
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => {}
        // No memory write of their own (control flow handled by recursion).
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

/// Accumulate every [`WriteEffect`] performed anywhere inside `block`,
/// recursing through nested control flow.  Used to pre-mark loads that
/// outlive a loop's back-edge.
fn collect_block_write_effects(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    for stmt in block.iter() {
        statement_write_effects(stmt, expressions, out);
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            collect_block_write_effects(nested, expressions, out);
        }
    }
}

/// A `Load` that has been emitted and is still in flight: its place plus
/// whether a write to that place has been observed since its `Emit`.
#[derive(Clone)]
struct PendingLoad {
    /// `None` = an Unknown place (function-argument pointer) - aliases all.
    place: Option<Place>,
    written: bool,
}

type Pending = std::collections::HashMap<naga::Handle<naga::Expression>, PendingLoad>;

/// Merge two control-flow successor states: a load is "written" after the
/// join if it was written on EITHER path (conservative).  Keys from both
/// sides are kept so a branch-local load that (legally) outlives its
/// branch is still tracked downstream.
fn merge_pending(mut a: Pending, b: Pending) -> Pending {
    for (h, pl) in b {
        a.entry(h)
            .and_modify(|e| e.written |= pl.written)
            .or_insert(pl);
    }
    a
}

/// Walk the operand cone of `root`, flagging every in-flight load that is
/// (a) reachable from `root` and (b) already marked written.  Such a load
/// is read AFTER its place was overwritten, so it must be bound.  The
/// `visited` set keeps the walk linear over shared sub-DAGs.
fn flag_used_loads(
    root: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    pending: &Pending,
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    if !visited.insert(root) {
        return;
    }
    if let Some(pl) = pending.get(&root)
        && pl.written
    {
        must_bind.insert(root);
        // Stop here.  A written `root` is added to `must_bind`, so it emits as
        // a `let` at its own Emit site, freezing its WHOLE operand cone
        // (unbound children inlined, bound children naming their own earlier
        // `let`s) lexically before the write that marked it `written` -
        // including any nested written load reachable only via `root`.
        // Re-pinning a child would therefore change nothing.  A child also used
        // OUTSIDE this parent is still pinned at that other use: the early
        // return records only `root` in `visited` (children stay walkable), and
        // each statement walk starts a fresh `visited`.
        return;
    }
    visit_expression_children(&expressions[root], |child| {
        flag_used_loads(child, expressions, pending, must_bind, visited);
    });
}

/// Forward dataflow over one block, threading `pending` (in-flight loads)
/// and accumulating into `must_bind`.  See [`compute_must_bind_loads`].
fn analyze_block(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    pending: &mut Pending,
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    for stmt in block.iter() {
        analyze_statement(stmt, expressions, module, pending, must_bind);
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
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => {
            // Uses first (a write never occurs within an Emit): a load defined
            // in this same range is not yet pending, so a sibling consuming it
            // is correctly not flagged.
            let mut visited = std::collections::HashSet::new();
            for h in range.clone() {
                flag_used_loads(h, expressions, pending, must_bind, &mut visited);
            }
            // Then register the loads this Emit introduces.
            for h in range.clone() {
                match &expressions[h] {
                    naga::Expression::Load { pointer } => {
                        let place = resolve_place(*pointer, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => global_is_writable(module, g),
                                PlaceRoot::Local(_) => true,
                            },
                            // Unknown place (function-argument pointer): track it -
                            // any later write may alias the pointee.
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
                    // `textureLoad` reads a texel; a later `textureStore` /
                    // `textureAtomic` (or a callee) to the same storage texture
                    // can stale it, exactly like a buffer `Load`.  Track it so a
                    // single-use `textureLoad` is bound rather than inlined past
                    // the write.  Gate on storage-texture writability via
                    // `image_is_writable_storage` - NOT `global_is_writable`,
                    // which reports textures (Handle space) as non-writable and
                    // would silently drop the hazard.
                    naga::Expression::ImageLoad { image, .. } => {
                        let place = resolve_place(*image, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => image_is_writable_storage(module, g),
                                // A texture is always a Handle-space global,
                                // never a local; treat a malformed Local root
                                // conservatively.
                                PlaceRoot::Local(_) => true,
                            },
                            // Texture passed as a value parameter: a callee may
                            // hold and store to it - track conservatively.
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
                    // Both read the query object's CURRENT traversal state; a
                    // later `Statement::RayQuery` (Proceed / Confirm /
                    // Terminate / GenerateIntersection - all modeled as
                    // `WriteEffect::Place(query)`) stales them exactly like a
                    // buffer `Load` crossing a `Store`, so a crossing read
                    // must bind rather than re-evaluate at its use site.  The
                    // query is always function-address-space (a `ray_query`
                    // local, or a pointer argument resolving to `None` place),
                    // so there is no writability gate to consult.
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
        S::Block(inner) => analyze_block(inner, expressions, module, pending, must_bind),
        S::If {
            condition,
            accept,
            reject,
        } => {
            let mut visited = std::collections::HashSet::new();
            flag_used_loads(*condition, expressions, pending, must_bind, &mut visited);
            let mut accept_state = pending.clone();
            analyze_block(accept, expressions, module, &mut accept_state, must_bind);
            let mut reject_state = pending.clone();
            analyze_block(reject, expressions, module, &mut reject_state, must_bind);
            *pending = merge_pending(accept_state, reject_state);
        }
        S::Switch { selector, cases } => {
            let mut visited = std::collections::HashSet::new();
            flag_used_loads(*selector, expressions, pending, must_bind, &mut visited);
            if cases.iter().any(|c| c.fall_through) {
                // A fall-through case chains into the next, so a write in one
                // case can reach a use in a later one.  WGSL source never
                // produces fall-through, but handle it soundly by threading the
                // SAME state sequentially through the cases (a conservative
                // over-approximation: it also assumes a directly-entered case
                // ran after its predecessors, which only ever over-binds).
                for case in cases {
                    analyze_block(&case.body, expressions, module, pending, must_bind);
                }
            } else {
                // Cases are mutually exclusive: analyse each from the pre-switch
                // state and union (OR) their post-states.  Every case is
                // reachable (WGSL switches are exhaustive).
                let mut merged: Option<Pending> = None;
                for case in cases {
                    let mut case_state = pending.clone();
                    analyze_block(&case.body, expressions, module, &mut case_state, must_bind);
                    merged = Some(match merged {
                        None => case_state,
                        Some(m) => merge_pending(m, case_state),
                    });
                }
                if let Some(m) = merged {
                    *pending = m;
                }
            }
        }
        S::Loop {
            body,
            continuing,
            break_if,
        } => {
            // Back-edge: a write anywhere in the loop can execute before an
            // earlier-or-later use (next iteration) and after a load emitted
            // before the loop.  Pre-mark every outer load the loop may write.
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
            // Loads emitted INSIDE the loop are re-evaluated each iteration
            // (expression values never cross the back-edge - only memory does),
            // so a single linear pass over body+continuing is exact for them.
            analyze_block(body, expressions, module, pending, must_bind);
            analyze_block(continuing, expressions, module, pending, must_bind);
            if let Some(h) = break_if {
                let mut visited = std::collections::HashSet::new();
                flag_used_loads(*h, expressions, pending, must_bind, &mut visited);
            }
        }
        // Leaf statements: their operands are uses, then their writes apply.
        _ => {
            let mut visited = std::collections::HashSet::new();
            crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
                flag_used_loads(h, expressions, pending, must_bind, &mut visited);
            });
            apply_writes(stmt, expressions, pending);
        }
    }
}

/// Identify every `Load` expression in `func` that must be bound to a
/// `let` rather than inlined, because the place it reads is written
/// between the `Load`'s `Emit` and a use of its value.  Inlining such a
/// load relocates the memory read past the write and yields the
/// post-write value - a silent miscompile.
///
/// The analysis is a single forward pass over the structured statement
/// tree (`analyze_*`).  It deliberately OVER-approximates the hazard
/// (binding a load is always semantically safe; the only cost is a few
/// bytes), so unresolved places, branches, loops, and call/barrier write
/// effects are all handled conservatively.  Read-only globals and pure
/// read-only locals are never flagged, so the common case stays inlined.
pub(super) fn compute_must_bind_loads(
    func: &naga::Function,
    module: &naga::Module,
) -> std::collections::HashSet<naga::Handle<naga::Expression>> {
    let mut pending: Pending = std::collections::HashMap::new();
    let mut must_bind = std::collections::HashSet::new();
    analyze_block(
        &func.body,
        &func.expressions,
        module,
        &mut pending,
        &mut must_bind,
    );
    must_bind
}
