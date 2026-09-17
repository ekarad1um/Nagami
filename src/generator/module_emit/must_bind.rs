//! Forced-binding analysis: which emitted expressions must become `let`
//! bindings regardless of byte cost - loads whose place is written before a
//! use, and work a use would drag into a loop.

use crate::analysis::{Effect, FnEffects, statement_effects};
use crate::handle_set::{HandleMap, HandleSet};
use crate::ir::visit::visit_expression_children;
use crate::ir::visit::{Scope, Slot, Visitor, walk_block};
use crate::passes::expr_util::{const_index_value, is_expensive_expr};
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

/// A write a statement performs, as the pending-load tracking asks it: the
/// [`Place`] it lands in, or any writable global.
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

/// One function's in-flight-load analysis: its inputs and the marks it
/// builds.  The pending state is an argument, not a field, because branches
/// fork it.
struct LoadAnalysis<'a> {
    expressions: &'a naga::Arena<naga::Expression>,
    module: &'a naga::Module,
    fn_effects: &'a [FnEffects],
    must_bind: HandleSet<naga::Expression>,
    visited: Visited,
}

impl LoadAnalysis<'_> {
    /// Every [`WriteEffect`] of one statement ([`statement_effects`] resolved
    /// to places); nested blocks are walked separately.  A pointer that
    /// resolves to no place is a function-argument pointer: any global, never
    /// a local.  A pending `ImageLoad` reaches a storage texture, so a texture
    /// write invalidates it like a store; a texture parameter could be any
    /// texture global.  A barrier makes other invocations' stores to shared
    /// globals observable, so a pre-barrier load can differ from a
    /// post-barrier re-read; `Globals` also over-invalidates private-space
    /// loads, a harmless over-binding.  A `discard` or subgroup op touches no
    /// memory.
    fn write_effects(&self, stmt: &naga::Statement, out: &mut Vec<WriteEffect>) {
        statement_effects(
            stmt,
            &mut |callee| self.fn_effects[callee.index()],
            &mut |effect| match effect {
                Effect::Write(pointer) | Effect::ImageWrite(pointer) => {
                    out.push(match resolve_place(pointer, self.expressions) {
                        Some(p) => WriteEffect::Place(p),
                        None => WriteEffect::Globals,
                    })
                }
                Effect::Globals => out.push(WriteEffect::Globals),
                Effect::Observable => {}
            },
        );
    }

    /// Mark every in-flight load `written` whose place one of `effects` may
    /// alias.  Monotone (only flips unwritten -> written) and skips
    /// already-written loads, so it is idempotent across the loop pre-mark
    /// and the linear pass.
    fn apply_writes(effects: &[WriteEffect], pending: &mut Pending) {
        if effects.is_empty() {
            return;
        }
        for pl in pending.values_mut() {
            if !pl.written && effects.iter().any(|e| e.invalidates(&pl.place)) {
                pl.written = true;
            }
        }
    }

    fn flag_used(&mut self, root: naga::Handle<naga::Expression>, pending: &Pending) {
        let mut walk = self.visited.walk();
        flag_used_loads(
            root,
            self.expressions,
            pending,
            &mut self.must_bind,
            &mut walk,
        );
    }

    fn block(&mut self, block: &naga::Block, pending: &mut Pending) {
        for stmt in block.iter() {
            self.statement(stmt, pending);
        }
    }

    fn statement(&mut self, stmt: &naga::Statement, pending: &mut Pending) {
        use naga::Statement as S;
        match stmt {
            S::Emit(range) => {
                // Uses first: a write never occurs within an Emit, and a load
                // defined in this range is not yet pending, so a sibling
                // consuming it is correctly not flagged.
                let mut walk = self.visited.walk();
                for h in range.clone() {
                    flag_used_loads(h, self.expressions, pending, &mut self.must_bind, &mut walk);
                }
                for h in range.clone() {
                    // A read-only global's load is never staled; a local's,
                    // and a function-argument pointer's (any later write may
                    // alias it), always may be.
                    let global_root = |place: &Option<Place>| match place {
                        Some(Place {
                            root: PlaceRoot::Global(g),
                            ..
                        }) => Some(*g),
                        _ => None,
                    };
                    let place = match &self.expressions[h] {
                        naga::Expression::Load { pointer } => {
                            let place = resolve_place(*pointer, self.expressions);
                            if global_root(&place)
                                .is_some_and(|g| !global_is_writable(self.module, g))
                            {
                                continue;
                            }
                            place
                        }
                        // A `textureLoad` stales like a buffer `Load` under a
                        // later `textureStore` / `textureAtomic` (or callee);
                        // gate on `image_is_writable_storage`, since
                        // `global_is_writable` reports textures immutable.  A
                        // texture is never a local, and a value-parameter
                        // texture could be stored to by a callee: both stay
                        // tracked.
                        naga::Expression::ImageLoad { image, .. } => {
                            let place = resolve_place(*image, self.expressions);
                            if global_root(&place)
                                .is_some_and(|g| !image_is_writable_storage(self.module, g))
                            {
                                continue;
                            }
                            place
                        }
                        // Both read the query's CURRENT traversal state, which
                        // a later `Statement::RayQuery` (a write of the query)
                        // stales like a `Store`; the query is always
                        // function-space, so no writability gate.
                        naga::Expression::RayQueryGetIntersection { query, .. }
                        | naga::Expression::RayQueryVertexPositions { query, .. } => {
                            resolve_place(*query, self.expressions)
                        }
                        _ => continue,
                    };
                    pending.insert(
                        h,
                        PendingLoad {
                            place,
                            written: false,
                        },
                    );
                }
            }
            S::Block(inner) => self.block(inner, pending),
            S::If {
                condition,
                accept,
                reject,
            } => {
                self.flag_used(*condition, pending);
                // One clone, not two: the reject arm may run over the
                // pre-branch state in place because the join is commutative.
                let mut accept_state = pending.clone();
                self.block(accept, &mut accept_state);
                self.block(reject, pending);
                merge_pending_into(pending, accept_state);
            }
            S::Switch { selector, cases } => {
                self.flag_used(*selector, pending);
                if cases.iter().any(|c| c.fall_through) {
                    // A fall-through case chains into the next, so thread one
                    // state through the cases sequentially; assuming a
                    // directly-entered case ran after its predecessors only
                    // over-binds.
                    for case in cases {
                        self.block(&case.body, pending);
                    }
                } else {
                    // Mutually exclusive cases: each from the pre-switch state,
                    // then the union of the post-states.  The last case runs
                    // over the pre-switch state in place, so `cases` costs one
                    // clone fewer than it has arms.
                    let mut merged: Option<Pending> = None;
                    for case in &cases[..cases.len().saturating_sub(1)] {
                        let mut case_state = pending.clone();
                        self.block(&case.body, &mut case_state);
                        match &mut merged {
                            None => merged = Some(case_state),
                            Some(m) => merge_pending_into(m, case_state),
                        }
                    }
                    if let Some(last) = cases.last() {
                        self.block(&last.body, pending);
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
                // Back-edge: a write anywhere in the loop runs after a load
                // emitted before it and before a next-iteration use, so
                // pre-mark every outer load the loop may write.
                let mut loop_writes = Vec::new();
                for block in [body, continuing] {
                    crate::ir::visit::for_each_statement(block, &mut |stmt| {
                        self.write_effects(stmt, &mut loop_writes)
                    });
                }
                Self::apply_writes(&loop_writes, pending);
                // Loads emitted inside the loop are re-evaluated each iteration
                // (only memory crosses the back-edge), so one linear pass over
                // body + continuing is exact for them.
                self.block(body, pending);
                self.block(continuing, pending);
                if let Some(h) = break_if {
                    self.flag_used(*h, pending);
                }
            }
            // Leaf statement: operands are uses, then its writes apply.
            _ => {
                let mut walk = self.visited.walk();
                crate::ir::visit::visit_statement_expression_handles(stmt, false, &mut |h| {
                    flag_used_loads(h, self.expressions, pending, &mut self.must_bind, &mut walk);
                });
                let mut effects = Vec::new();
                self.write_effects(stmt, &mut effects);
                Self::apply_writes(&effects, pending);
            }
        }
    }
}

/// Every expression in `func` that must be `let`-bound rather than inlined:
/// a `Load` whose place is written between its `Emit` and a use of its value
/// (inlining would relocate the read past the write and yield the post-write
/// value), and work a use inside a loop would re-run every iteration
/// ([`loop_sunk_work`]).  The load analysis is one forward
/// pass over the structured statement tree, over-approximating the hazard
/// (binding is always safe and costs only bytes): unresolved places, branches,
/// loops and barriers are handled conservatively and a call writes what its
/// summary says, while read-only globals and locals are never flagged.
/// `ref_counts` are the undiscounted census counts.
pub(super) fn compute_must_bind(
    func: &naga::Function,
    module: &naga::Module,
    fn_effects: &[FnEffects],
    ref_counts: &[usize],
) -> HandleSet<naga::Expression> {
    let mut analysis = LoadAnalysis {
        expressions: &func.expressions,
        module,
        fn_effects,
        must_bind: Default::default(),
        visited: Visited::new(func.expressions.len()),
    };
    analysis.block(&func.body, &mut Pending::default());
    let mut must_bind = analysis.must_bind;
    let mut sunk = HandleSet::default();
    loop_sunk_work(func, module, &Bindings::Modelled { ref_counts }, &mut sunk);
    must_bind.extend(sunk.iter().copied());
    must_bind
}

// MARK: Loop-sunk work

/// Whether re-evaluating `h` on every iteration of a loop it was emitted
/// outside of is a cost: an image operation; arithmetic (`Binary`, `Math`,
/// `Select`, `Relational`); a load from a buffer, uniform, workgroup or
/// immediate variable; an array or struct constructor, which materialises
/// storage per evaluation.  A private variable's load is thread-local and
/// register-promoted downstream, so it moves freely, as do casts, swizzles,
/// splats, accesses and vector constructors; implicit-LOD samples and
/// derivatives are pinned by uniformity already.
fn is_loop_sinkable_work(
    h: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
) -> bool {
    match &expressions[h] {
        e if is_expensive_expr(e) => true,
        naga::Expression::Binary { .. }
        | naga::Expression::Math { .. }
        | naga::Expression::Select { .. }
        | naga::Expression::Relational { .. } => true,
        naga::Expression::Load { pointer } => match resolve_place(*pointer, expressions) {
            Some(Place {
                root: PlaceRoot::Global(g),
                ..
            }) => matches!(
                module.global_variables[g].space,
                naga::AddressSpace::Storage { .. }
                    | naga::AddressSpace::Uniform
                    | naga::AddressSpace::WorkGroup
                    | naga::AddressSpace::Immediate
            ),
            _ => false,
        },
        naga::Expression::Compose { ty, .. } => matches!(
            module.types[*ty].inner,
            naga::TypeInner::Array { .. } | naga::TypeInner::Struct { .. }
        ),
        _ => false,
    }
}

/// Whether anything in `func` can sink into a loop.
pub(super) fn has_loop(func: &naga::Function) -> bool {
    let mut has_loop = false;
    crate::ir::visit::for_each_statement(&func.body, &mut |s| {
        has_loop |= matches!(s, naga::Statement::Loop { .. });
    });
    has_loop
}

/// What [`loop_sunk_work`] knows of where each expression renders.
pub(super) enum Bindings<'a> {
    /// Before the function renders: a consumer with two or more references
    /// is taken to bind, so its cone is walked at its own `Emit`; a pointer
    /// chain rooted at a variable never binds and is walked through.  The
    /// bytes may still inline such a consumer, so work it holds is pinned
    /// when reached from a loop all the same.
    Modelled { ref_counts: &'a [usize] },
    /// After it rendered: the names the emitter gave, a stashed call's
    /// text (`stashed`) being no name.  Exact, so what renders as a name
    /// is never reported.
    Rendered {
        names: &'a HandleMap<naga::Expression, String>,
        stashed: &'a HandleSet<naga::Expression>,
        twins: &'a HandleMap<naga::Expression, naga::Handle<naga::Expression>>,
    },
}

impl Bindings<'_> {
    /// Whether `h` is known to render as a name.
    fn certain(&self, h: naga::Handle<naga::Expression>) -> bool {
        match self {
            Bindings::Modelled { .. } => false,
            Bindings::Rendered { names, stashed, .. } => {
                names.contains_key(h) && !stashed.contains(h)
            }
        }
    }

    /// Whether `h` is a twin rendering as the name of its first spelling:
    /// nothing of its own cone renders anywhere.
    fn reuses(&self, h: naga::Handle<naga::Expression>) -> bool {
        match self {
            Bindings::Modelled { .. } => false,
            Bindings::Rendered { names, twins, .. } => {
                twins.contains_key(h) && names.contains_key(h)
            }
        }
    }

    /// Whether the cone of `h` is walked at its `Emit`.
    fn bound(
        &self,
        h: naga::Handle<naga::Expression>,
        expressions: &naga::Arena<naga::Expression>,
    ) -> bool {
        match self {
            Bindings::Modelled { ref_counts } => {
                ref_counts[h.index()] >= 2 && resolve_place(h, expressions).is_none()
            }
            Bindings::Rendered { .. } => self.certain(h),
        }
    }

    /// Whether `h`'s own `Emit` starts a walk of its cone (a value with one
    /// consumer is reached from that consumer instead).
    fn walked_from_emit(&self, h: naga::Handle<naga::Expression>) -> bool {
        match self {
            Bindings::Modelled { ref_counts } => ref_counts[h.index()] >= 2,
            Bindings::Rendered { .. } => self.certain(h),
        }
    }
}

/// Work ([`is_loop_sinkable_work`]) emitted outside a loop and consumed
/// inside it, into `sunk`.  Rendered at its use it re-runs every
/// iteration, and the platform compiler does not undo that (Apple's
/// front-end left a shader's uniform loads and `normalize` where
/// single-use forwarding had sunk them, an array-of-struct constructor is
/// rebuilt per iteration, a fetch is never hoisted across a loop with other
/// memory traffic), so the input's loop depth is kept at the cost of one
/// `let`.
///
/// Render position follows `bindings`: a bound consumer's cone renders at
/// its `Emit`, an inlined one's at its use, a pinned node at its `Emit`
/// whatever its count - always valid, the operands being in scope there
/// and naga's `Load` reading at the `Emit`.  Modelled before the render,
/// the walk pins what the model guesses; rendered, it reports what the
/// bytes inlined into a loop after all (a short two-use value, a uniform
/// swizzle) and the function renders again with those pinned.
pub(super) fn loop_sunk_work(
    func: &naga::Function,
    module: &naga::Module,
    bindings: &Bindings<'_>,
    sunk: &mut HandleSet<naga::Expression>,
) {
    let expressions = &func.expressions;
    if !has_loop(func) {
        return;
    }
    // Loop depth of each `Emit`; `u16::MAX` for handles no `Emit` covers
    // (arguments, variables, statement results), which have no cone to sink.
    struct EmitDepths(Vec<u16>, u16);
    impl Visitor for EmitDepths {
        fn stmt(&mut self, _: &naga::Statement, scope: Scope) -> bool {
            self.1 = scope.loop_depth;
            true
        }
        fn handle(&mut self, h: naga::Handle<naga::Expression>, slot: Slot) {
            if slot == Slot::Emitted {
                self.0[h.index()] = self.1;
            }
        }
    }
    let mut depths = EmitDepths(vec![u16::MAX; expressions.len()], 0);
    walk_block(&func.body, Scope::default(), &mut depths);
    let emit_depth = depths.0;

    struct Walk<'a> {
        expressions: &'a naga::Arena<naga::Expression>,
        module: &'a naga::Module,
        bindings: &'a Bindings<'a>,
        emit_depth: &'a [u16],
        /// Bound consumers descend once, at their own `Emit` depth.
        entered: Vec<bool>,
        sunk: &'a mut HandleSet<naga::Expression>,
        /// The statement being walked: its loop depth, and its `break if`
        /// operand, which runs in the loop's `continuing` block, once per
        /// iteration - the loop's own depth, not the enclosing one.
        depth: u16,
        break_if: Option<naga::Handle<naga::Expression>>,
    }
    impl Walk<'_> {
        /// `h` is reached from loop depth `depth` and renders there unless
        /// bound; descend where it renders.
        fn render(&mut self, h: naga::Handle<naga::Expression>, depth: u16) {
            let i = h.index();
            let emitted = self.emit_depth[i] != u16::MAX;
            if !emitted || self.bindings.reuses(h) {
                return;
            }
            let sinks = depth > self.emit_depth[i]
                && !self.bindings.certain(h)
                && is_loop_sinkable_work(h, self.expressions, self.module);
            if sinks {
                self.sunk.insert(h);
            }
            let bound = sinks || self.bindings.bound(h, self.expressions);
            let depth = if bound {
                if std::mem::replace(&mut self.entered[i], true) {
                    return;
                }
                self.emit_depth[i]
            } else {
                depth
            };
            visit_expression_children(&self.expressions[h], |c| self.render(c, depth));
        }
    }
    impl Visitor for Walk<'_> {
        fn stmt(&mut self, stmt: &naga::Statement, scope: Scope) -> bool {
            self.depth = scope.loop_depth;
            self.break_if = match stmt {
                naga::Statement::Loop { break_if, .. } => *break_if,
                _ => None,
            };
            true
        }
        fn handle(&mut self, h: naga::Handle<naga::Expression>, slot: Slot) {
            match slot {
                Slot::Emitted if self.bindings.walked_from_emit(h) => self.render(h, self.depth),
                Slot::Operand => {
                    let depth = self.depth + u16::from(Some(h) == self.break_if);
                    self.render(h, depth);
                }
                Slot::Emitted | Slot::Result | Slot::WritePointer => {}
            }
        }
    }
    walk_block(
        &func.body,
        Scope::default(),
        &mut Walk {
            expressions,
            module,
            bindings,
            emit_depth: &emit_depth,
            entered: vec![false; expressions.len()],
            sunk,
            depth: 0,
            break_if: None,
        },
    );
}
