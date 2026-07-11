//! Struct field-wise build coalescing.
//!
//! A common hand-authored pattern declares an empty struct local and fills it
//! one member at a time:
//!
//! ```wgsl
//! var t: Transform2D;
//! t.pos = mix(a.pos, b.pos, k);
//! t.scale = mix(a.scale, b.scale, k);
//! // ...
//! return t;
//! ```
//!
//! Every member store costs `t.<member> = ...;`.  When the local is built up
//! completely and only THEN read, the whole thing is one struct constructor:
//!
//! ```wgsl
//! var t = Transform2D(mix(...), mix(...), ...);  // members in declaration order
//! ```
//!
//! ## Why this is value-safe
//!
//! In naga IR every store's right-hand side is a *pre-materialised* expression
//! handle - it was computed in an `Emit` range BEFORE the `Store`.  Rewriting
//! the member stores into one `Store(t, Compose{member handles in decl order})`
//! references those already-computed values at their original positions, so the
//! values AND their evaluation order are unchanged, regardless of whether a
//! right-hand side has side effects.  The only genuine hazards are excluded by
//! the gates below: a member store whose value reads `t` (the constructor would
//! read an unset member), a read of `t` before the build finishes, a pointer to
//! `t` escaping to a callee, or incomplete member coverage.
//!
//! Value-safety rests on the gates, not on validation: the pipeline's per-pass
//! re-validation rejects only structurally-invalid IR (rolling it back to a
//! no-op), so a gate that wrongly admits an unsafe build yields valid-but-wrong
//! IR that slips straight through.

use std::collections::HashMap;

use crate::error::Error;
use crate::passes::load_dedup::get_stored_local;
use crate::pipeline::{Pass, PassContext};

/// Pass entry point; the algorithm and its safety gates are the
/// module-level docs.
pub struct StructBuildPass;

/// `true` when the expression subtree rooted at `h` reads the local `g`
/// (a `Load` whose pointer roots at `g`).  Used to reject a member value that
/// depends on a sibling member (`t.b = t.a + 1`).
fn expr_loads_local(
    h: naga::Handle<naga::Expression>,
    g: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    expr_loads_local_memo(h, g, arena, &mut visited)
}

fn expr_loads_local_memo(
    h: naga::Handle<naga::Expression>,
    g: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    // A shared sub-expression forms a diamond in the DAG; without a visited set
    // each incoming edge re-walks it, making this exponential (2^depth) on wide
    // shared trees (a ~1KB shader can then hang the pass).  A handle is inserted
    // before it is explored and only ever memoises `false`: a handle whose
    // subtree DOES contain `g` returns `true` on its first visit and the
    // caller's `if !found` guard short-circuits every later edge, so it is never
    // revisited to yield a wrong `false`.
    if !visited.insert(h) {
        return false;
    }
    if let naga::Expression::Load { pointer } = arena[h]
        && get_stored_local(arena, pointer) == Some(g)
    {
        return true;
    }
    let mut found = false;
    crate::passes::expr_util::visit_expression_children(&arena[h], |child| {
        if !found {
            found = expr_loads_local_memo(child, g, arena, visited);
        }
    });
    found
}

/// Plan for one collapsible struct local.
struct BuildPlan {
    local: naga::Handle<naga::LocalVariable>,
    /// member index -> stored value handle, in declaration order.
    components: Vec<naga::Handle<naga::Expression>>,
    /// A `LocalVariable(local)` pointer handle to reuse for the whole-store.
    ptr: naga::Handle<naga::Expression>,
    /// Statement indices (in the top-level body) of the member stores to drop;
    /// the highest is where the constructor store is inserted.
    store_indices: Vec<usize>,
}

impl Pass for StructBuildPass {
    fn name(&self) -> &'static str {
        "struct-build"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = false;
        for (_, f) in module.functions.iter_mut() {
            changed |= collapse_in_function(f, &module.types);
        }
        for ep in module.entry_points.iter_mut() {
            changed |= collapse_in_function(&mut ep.function, &module.types);
        }
        Ok(changed)
    }
}

fn collapse_in_function(func: &mut naga::Function, types: &naga::UniqueArena<naga::Type>) -> bool {
    // Candidate locals: struct-typed, no initializer.
    let candidates: Vec<(naga::Handle<naga::LocalVariable>, usize)> = func
        .local_variables
        .iter()
        .filter_map(|(h, lv)| {
            if lv.init.is_some() {
                return None;
            }
            match &types[lv.ty].inner {
                naga::TypeInner::Struct { members, .. } => Some((h, members.len())),
                _ => None,
            }
        })
        .collect();
    if candidates.is_empty() {
        return false;
    }

    let mut plans: Vec<BuildPlan> = Vec::new();
    for (g, member_count) in candidates {
        if let Some(plan) = plan_local(func, g, member_count) {
            plans.push(plan);
        }
    }
    if plans.is_empty() {
        return false;
    }

    apply_plans(func, types, plans);
    true
}

/// Analyse local `g`; return a [`BuildPlan`] when it is built member-by-member
/// in the top-level body and only read afterwards.  Returns `None` (leave it
/// alone) on ANYTHING that does not match exactly - this is the whole safety
/// surface, so it is deliberately strict.
fn plan_local(
    func: &naga::Function,
    g: naga::Handle<naga::LocalVariable>,
    member_count: usize,
) -> Option<BuildPlan> {
    use naga::Statement as S;
    let arena = &func.expressions;

    // 1. Scan the top-level body once for member stores, escapes, and reads.
    let mut members: HashMap<u32, (usize, naga::Handle<naga::Expression>)> = HashMap::new();
    let mut ptr: Option<naga::Handle<naga::Expression>> = None;
    let mut last_store_idx: Option<usize> = None;
    let mut first_read_idx: Option<usize> = None;

    for (idx, stmt) in func.body.iter().enumerate() {
        // Member store?
        if let S::Store { pointer, value } = stmt
            && let Some((local, spec)) =
                crate::passes::coalescing::resolve_local_and_element(*pointer, arena)
            && local == g
        {
            match spec {
                crate::passes::coalescing::ElementSpec::Index(i) => {
                    // A member written twice, or a value that reads `g`, is out.
                    if members.contains_key(&i) || expr_loads_local(*value, g, arena) {
                        return None;
                    }
                    members.insert(i, (idx, *value));
                    last_store_idx = Some(idx);
                    if let naga::Expression::AccessIndex { base, .. } = arena[*pointer] {
                        ptr = Some(base);
                    }
                    continue;
                }
                // Whole-store or dynamic-index store to `g` -> too complex.
                _ => return None,
            }
        }
        // Escape (pointer passed to a callee) -> the callee may read/write it.
        if statement_escapes_local(stmt, g, arena) {
            return None;
        }
        // A VALUE read of `g` (a `Load` rooting at it).  Bare `AccessIndex` /
        // `LocalVariable` pointer materialisations - including the member-store
        // pointers emitted up front - are addresses, NOT reads, so they do not
        // count (otherwise the build would never look "completed before read").
        if statement_value_reads_local(stmt, g, arena) {
            first_read_idx = Some(first_read_idx.map_or(idx, |p| p.min(idx)));
        }
    }

    // 2. Gates: every member covered exactly once; first read strictly after
    //    the last member store.
    if members.len() != member_count {
        return None;
    }
    let last = last_store_idx?;
    if let Some(read) = first_read_idx
        && read <= last
    {
        return None;
    }

    // 3. Reject any reference to `g` inside a NESTED block - the positional
    //    "read after build" reasoning above only covers the top-level body, so
    //    any Load/Access/Store rooting at `g` (or `g` escaping) in a nested
    //    block is unsafe.  This is the one whole-nested-tree walk; done last so
    //    it runs only for a candidate that already passed the cheap top-level
    //    gates, not for every struct local.
    let mut nested_ref = false;
    walk_nested(&func.body, &mut |stmt| {
        if statement_references_local(stmt, g, arena) {
            nested_ref = true;
        }
    });
    if nested_ref {
        return None;
    }

    // 4. Build the component list in declaration (member-index) order.
    let mut components = Vec::with_capacity(member_count);
    let mut store_indices = Vec::with_capacity(member_count);
    for i in 0..member_count as u32 {
        let (sidx, val) = members.get(&i)?;
        components.push(*val);
        store_indices.push(*sidx);
    }
    Some(BuildPlan {
        local: g,
        components,
        ptr: ptr?,
        store_indices,
    })
}

/// Rewrite the member stores of every plan into one struct-constructor store
/// apiece, in a SINGLE rebuild of the body.
///
/// All plans' `store_indices` / `insert_at` are positions in the *current*
/// (pre-mutation) body.  Applying them one at a time would be unsound: each
/// rewrite changes the statement count, so a later plan's indices would point
/// at the wrong (shifted) statements and could drop a live, unrelated statement
/// or splice a constructor in the wrong place.  Instead we consult every plan's
/// original indices in one pass, so no index is ever invalidated.
///
/// The plans are mutually independent: each targets a distinct local, so their
/// member-store index sets are disjoint and their `insert_at` positions are
/// distinct; and the gates (a value read of a local before its build completes
/// blocks that local's plan) guarantee no plan's rewrite changes another's
/// observed values.
fn apply_plans(
    func: &mut naga::Function,
    types: &naga::UniqueArena<naga::Type>,
    plans: Vec<BuildPlan>,
) {
    // member-store indices to drop, and `insert_at` -> (compose, struct ptr).
    let mut drop: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut splice: HashMap<
        usize,
        (
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        ),
    > = HashMap::new();
    for plan in plans {
        let struct_ty = func.local_variables[plan.local].ty;
        // Sanity: the local's type must still be the struct we planned for.
        debug_assert!(matches!(
            types[struct_ty].inner,
            naga::TypeInner::Struct { .. }
        ));
        // Appending the Compose at the end of the arena is topologically safe:
        // it references only the (lower-handle) member values materialised
        // earlier.  naga permits a high-handle `Emit` range before later
        // lower-handle ones, so the splice position does not constrain ordering.
        let compose = func.expressions.append(
            naga::Expression::Compose {
                ty: struct_ty,
                components: plan.components,
            },
            naga::Span::UNDEFINED,
        );
        let insert_at = *plan.store_indices.iter().max().unwrap();
        drop.extend(plan.store_indices);
        splice.insert(insert_at, (compose, plan.ptr));
    }

    let original = std::mem::replace(&mut func.body, naga::Block::new());
    for (idx, (stmt, span)) in original.span_into_iter().enumerate() {
        if drop.contains(&idx) {
            // A member store: at the position of the LAST one for its local,
            // splice `Emit(Compose); Store(g, Compose)` (every member value is
            // already materialised before here).
            if let Some(&(compose, ptr)) = splice.get(&idx) {
                func.body.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(compose, compose)),
                    naga::Span::UNDEFINED,
                );
                func.body.push(
                    naga::Statement::Store {
                        pointer: ptr,
                        value: compose,
                    },
                    naga::Span::UNDEFINED,
                );
            }
            continue;
        }
        func.body.push(stmt, span);
    }
}

/// `true` when `stmt` references `local` in any way (Store target, Load,
/// value operand, escape).  Conservative: used both to reject nested refs and
/// to find reads in the body.
fn statement_references_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    let mut found = false;
    let mut check = |h: naga::Handle<naga::Expression>| {
        if !found && expr_mentions_local(h, local, arena) {
            found = true;
        }
    };
    visit_statement_expressions(stmt, arena, &mut check);
    found
}

/// `true` when `stmt` performs a VALUE read of `local` (a `Load` rooting at
/// it) - the genuine "use" that must come after the build, as opposed to the
/// bare pointer materialisations.
fn statement_value_reads_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    let mut found = false;
    let mut check = |h: naga::Handle<naga::Expression>| {
        if !found && expr_loads_local(h, local, arena) {
            found = true;
        }
    };
    visit_statement_expressions(stmt, arena, &mut check);
    found
}

/// `true` when `stmt` passes a pointer rooting at `local` to a callee / atomic
/// / other escape channel.
fn statement_escapes_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    match stmt {
        naga::Statement::Call { arguments, .. } => arguments
            .iter()
            .any(|&a| get_stored_local(arena, a) == Some(local)),
        naga::Statement::Atomic { pointer, .. }
        | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            get_stored_local(arena, *pointer) == Some(local)
        }
        _ => false,
    }
}

/// `true` when the expression subtree rooted at `h` mentions `local` (as the
/// base of an `AccessIndex`/`Access`, the pointer of a `Load`, or bare).
fn expr_mentions_local(
    h: naga::Handle<naga::Expression>,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    expr_mentions_local_memo(h, local, arena, &mut visited)
}

fn expr_mentions_local_memo(
    h: naga::Handle<naga::Expression>,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    // Memoised for the same reason as `expr_loads_local_memo`: without a visited
    // set, shared sub-expressions make this exponential on the DAG.
    if !visited.insert(h) {
        return false;
    }
    if matches!(arena[h], naga::Expression::LocalVariable(l) if l == local) {
        return true;
    }
    let mut found = false;
    crate::passes::expr_util::visit_expression_children(&arena[h], |child| {
        if !found {
            found = expr_mentions_local_memo(child, local, arena, visited);
        }
    });
    found
}

/// Visit every expression handle a statement reads (NOT recursing into nested
/// blocks; the caller drives block recursion).  Store pointers are included so
/// member-store targets count as references for the nested-ref guard.
fn visit_statement_expressions(
    stmt: &naga::Statement,
    arena: &naga::Arena<naga::Expression>,
    f: &mut impl FnMut(naga::Handle<naga::Expression>),
) {
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => {
            for h in range.clone() {
                f(h);
            }
        }
        S::Store { pointer, value } => {
            f(*pointer);
            f(*value);
        }
        S::Return { value: Some(v) } => f(*v),
        S::If { condition, .. } => f(*condition),
        S::Switch { selector, .. } => f(*selector),
        S::Loop {
            break_if: Some(b), ..
        } => f(*b),
        S::Call {
            arguments, result, ..
        } => {
            for &a in arguments {
                f(a);
            }
            if let Some(r) = result {
                f(*r);
            }
        }
        S::Atomic { pointer, value, .. } => {
            f(*pointer);
            f(*value);
        }
        S::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            f(*image);
            f(*coordinate);
            if let Some(a) = array_index {
                f(*a);
            }
            f(*value);
        }
        S::WorkGroupUniformLoad { pointer, result } => {
            f(*pointer);
            f(*result);
        }
        // The catch-all is sound for the local-mention consumers because
        // every COMPUTED expression is rooted through the `Emit` arm; a
        // variant may be skipped here iff its own handle fields are all
        // Emit'd computations, childless result expressions (which cannot
        // mention a local), or pointers that can never root at a
        // struct-typed function local (`RayQuery::query`,
        // `WorkGroupUniformLoad::pointer`).  Bare-`LocalVariable` fields
        // that CAN root at one - `Store::pointer`, `Call::arguments` -
        // must have explicit arms above.
        _ => {}
    }
    let _ = arena;
}

/// Run `f` on every statement that lives inside a NESTED block of `body` (i.e.
/// every statement reachable through a block-bearing statement, NOT the
/// top-level statements themselves).
fn walk_nested(body: &naga::Block, f: &mut impl FnMut(&naga::Statement)) {
    for stmt in body.iter() {
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            walk_block_all(nested, f);
        }
    }
}

fn walk_block_all(body: &naga::Block, f: &mut impl FnMut(&naga::Statement)) {
    for stmt in body.iter() {
        f(stmt);
        for nested in crate::passes::expr_util::nested_blocks(stmt) {
            walk_block_all(nested, f);
        }
    }
}
