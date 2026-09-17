//! Struct field-wise build coalescing: a struct local declared empty and
//! filled one member at a time (`var t: T; t.pos = ...; t.scale = ...;`),
//! completely built and only then read, becomes one constructor
//! (`var t = T(..., ...);`, members in declaration order).
//!
//! Value-safe because every store's right-hand side is a pre-materialised
//! expression handle computed in an `Emit` range before the `Store`:
//! `Store(t, Compose{member handles in decl order})` references those
//! already-computed values at their original positions, so values and
//! evaluation order are unchanged whatever side effects they carry.  The
//! genuine hazards are excluded by the gates: a member value that reads
//! `t` (the constructor would read an unset member), a read of `t` before
//! the build finishes, a pointer to `t` escaping to a callee, incomplete
//! member coverage.  Safety rests on the gates, not on validation: per-pass
//! re-validation rejects only structurally invalid IR, so a wrongly
//! admitted build yields valid-but-wrong IR that slips straight through.

use crate::error::Error;
use crate::handle_set::HandleMap;
use crate::ir::visit::for_each_function_mut;
use crate::passes::expr_util::root_local_var;
use crate::pipeline::{Pass, PassContext};

/// Collapses member-wise struct builds into one constructor store.
pub struct StructBuildPass;

/// A `Load` whose pointer roots at `g` anywhere under `h`; rejects a member
/// value that depends on a sibling member (`t.b = t.a + 1`).  Shared
/// sub-expressions make an unmemoised walk exponential (even a small shader
/// hangs the pass), and `memo` records both answers, so one memo serves every
/// query about the same `g`.
fn expr_loads_local(
    h: naga::Handle<naga::Expression>,
    g: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    memo: &mut HandleMap<naga::Expression, bool>,
) -> bool {
    if let Some(&known) = memo.get(h) {
        return known;
    }
    let mut found = match arena[h] {
        naga::Expression::Load { pointer } => root_local_var(pointer, arena) == Some(g),
        // A cooperative read goes through its pointer like a `Load`.
        naga::Expression::CooperativeLoad { ref data, .. } => {
            root_local_var(data.pointer, arena) == Some(g)
        }
        _ => false,
    };
    if !found {
        crate::ir::visit::visit_expression_children(&arena[h], |child| {
            if !found {
                found = expr_loads_local(child, g, arena, memo);
            }
        });
    }
    memo.insert(h, found);
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
        let types = &module.types;
        for_each_function_mut(&mut module.functions, &mut module.entry_points, &mut |f| {
            changed |= collapse_in_function(f, types);
        });
        Ok(changed)
    }
}

fn collapse_in_function(func: &mut naga::Function, types: &naga::UniqueArena<naga::Type>) -> bool {
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
    // Both memos answer "about `g`", so they reset per candidate and keep
    // their slots across candidates.
    let mut loads: HandleMap<naga::Expression, bool> = Default::default();
    let mut mentions: HandleMap<naga::Expression, bool> = Default::default();
    for (g, member_count) in candidates {
        loads.clear();
        mentions.clear();
        if let Some(plan) = plan_local(func, g, member_count, &mut loads, &mut mentions) {
            plans.push(plan);
        }
    }
    if plans.is_empty() {
        return false;
    }

    apply_plans(func, types, plans);
    true
}

/// A plan when `g` is built member-by-member in the top-level body and only
/// read afterwards; `None` on anything else, since this is the whole safety
/// surface.
fn plan_local(
    func: &naga::Function,
    g: naga::Handle<naga::LocalVariable>,
    member_count: usize,
    loads: &mut HandleMap<naga::Expression, bool>,
    mentions: &mut HandleMap<naga::Expression, bool>,
) -> Option<BuildPlan> {
    use naga::Statement as S;
    let arena = &func.expressions;

    // Per member: the index of its store and the value stored.
    let mut members: Vec<Option<(usize, naga::Handle<naga::Expression>)>> =
        vec![None; member_count];
    let mut ptr: Option<naga::Handle<naga::Expression>> = None;
    let mut last_store_idx: Option<usize> = None;
    let mut first_read_idx: Option<usize> = None;

    for (idx, stmt) in func.body.iter().enumerate() {
        if let S::Store { pointer, value } = stmt
            && let Some((local, spec)) =
                crate::passes::coalescing::resolve_local_and_element(*pointer, arena)
            && local == g
        {
            match spec {
                crate::passes::coalescing::ElementSpec::Index(i) => {
                    // A member written twice, or a value that reads `g`, is out.
                    let slot = members.get_mut(i as usize)?;
                    if slot.is_some() || expr_loads_local(*value, g, arena, loads) {
                        return None;
                    }
                    *slot = Some((idx, *value));
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
        // Pointer materialisations (including the member-store pointers
        // emitted up front) are addresses, not reads, or the build would never
        // look completed before its first read.
        if statement_value_reads_local(stmt, g, arena, loads) {
            first_read_idx = Some(first_read_idx.map_or(idx, |p| p.min(idx)));
        }
    }

    // Every member covered exactly once; first read strictly after the last
    // member store.
    if members.iter().any(Option::is_none) {
        return None;
    }
    let last = last_store_idx?;
    if let Some(read) = first_read_idx
        && read <= last
    {
        return None;
    }

    // The positional read-after-build reasoning covers only the top-level
    // body, so any reference to `g` in a nested block is unsafe.  The one
    // whole-tree walk, done last so only candidates past the cheap gates pay
    // for it.
    let mut nested_ref = false;
    walk_nested(&func.body, &mut |stmt| {
        if statement_references_local(stmt, g, arena, mentions) {
            nested_ref = true;
        }
    });
    if nested_ref {
        return None;
    }

    let mut components = Vec::with_capacity(member_count);
    let mut store_indices = Vec::with_capacity(member_count);
    for (sidx, val) in members.into_iter().flatten() {
        components.push(val);
        store_indices.push(sidx);
    }
    Some(BuildPlan {
        local: g,
        components,
        ptr: ptr?,
        store_indices,
    })
}

/// One rebuild of the body for every plan: all `store_indices` are
/// positions in the pre-mutation body, and applying plans one at a time
/// would shift a later plan's indices onto the wrong statements.  Plans are
/// independent: each targets a distinct local, so their index sets are
/// disjoint and their insertion points distinct, and the gates guarantee no
/// rewrite changes another plan's observed values.
fn apply_plans(
    func: &mut naga::Function,
    types: &naga::UniqueArena<naga::Type>,
    plans: Vec<BuildPlan>,
) {
    // Per top-level statement: whether it is a member store to drop, and
    // the (compose, struct ptr) to splice in at its position.
    let mut drop = vec![false; func.body.len()];
    let mut splice: Vec<
        Option<(
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        )>,
    > = vec![None; func.body.len()];
    for plan in plans {
        let struct_ty = func.local_variables[plan.local].ty;
        debug_assert!(matches!(
            types[struct_ty].inner,
            naga::TypeInner::Struct { .. }
        ));
        // Appending the Compose is topologically safe: it references only
        // lower-handle member values, and naga permits a high-handle `Emit`
        // range before later lower-handle ones.
        let compose = func.expressions.append(
            naga::Expression::Compose {
                ty: struct_ty,
                components: plan.components,
            },
            naga::Span::UNDEFINED,
        );
        let insert_at = *plan.store_indices.iter().max().unwrap();
        for i in plan.store_indices {
            drop[i] = true;
        }
        splice[insert_at] = Some((compose, plan.ptr));
    }

    let mut idx = 0usize;
    crate::ir::rewrite::rewrite_statements(&mut func.body, &mut |stmt, span, out| {
        let here = idx;
        idx += 1;
        if drop[here] {
            // At the last member store of a local every member value is
            // already materialised.
            if let Some((compose, ptr)) = splice[here] {
                out.push(
                    naga::Statement::Emit(naga::Range::new_from_bounds(compose, compose)),
                    naga::Span::UNDEFINED,
                );
                out.push(
                    naga::Statement::Store {
                        pointer: ptr,
                        value: compose,
                    },
                    naga::Span::UNDEFINED,
                );
            }
            return;
        }
        out.push(stmt, span);
    });
}

/// Any reference at all (store target, load, operand, escape); conservative
/// by design.
fn statement_references_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    memo: &mut HandleMap<naga::Expression, bool>,
) -> bool {
    let mut found = false;
    let mut check = |h: naga::Handle<naga::Expression>| {
        if !found && expr_mentions_local(h, local, arena, memo) {
            found = true;
        }
    };
    crate::ir::visit::visit_statement_operands(stmt, true, &mut check);
    found
}

/// A `Load` rooting at `local`: the genuine use that must follow the build,
/// unlike bare pointer materialisations.
fn statement_value_reads_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    memo: &mut HandleMap<naga::Expression, bool>,
) -> bool {
    let mut found = false;
    let mut check = |h: naga::Handle<naga::Expression>| {
        if !found && expr_loads_local(h, local, arena, memo) {
            found = true;
        }
    };
    crate::ir::visit::visit_statement_operands(stmt, true, &mut check);
    found
}

/// A pointer rooting at `local` handed to a callee, an atomic, a uniform
/// load or a cooperative store.
fn statement_escapes_local(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    match stmt {
        naga::Statement::Call { arguments, .. } => arguments
            .iter()
            .any(|&a| root_local_var(a, arena) == Some(local)),
        naga::Statement::Atomic { pointer, .. }
        | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            root_local_var(*pointer, arena) == Some(local)
        }
        naga::Statement::CooperativeStore { data, .. } => {
            root_local_var(data.pointer, arena) == Some(local)
        }
        _ => false,
    }
}

/// `LocalVariable(local)` anywhere under `h`; memoised like
/// [`expr_loads_local`].
fn expr_mentions_local(
    h: naga::Handle<naga::Expression>,
    local: naga::Handle<naga::LocalVariable>,
    arena: &naga::Arena<naga::Expression>,
    memo: &mut HandleMap<naga::Expression, bool>,
) -> bool {
    if let Some(&known) = memo.get(h) {
        return known;
    }
    let mut found = matches!(arena[h], naga::Expression::LocalVariable(l) if l == local);
    if !found {
        crate::ir::visit::visit_expression_children(&arena[h], |child| {
            if !found {
                found = expr_mentions_local(child, local, arena, memo);
            }
        });
    }
    memo.insert(h, found);
    found
}

/// Every statement inside a nested block of `body`, excluding the top-level
/// statements themselves.
fn walk_nested(body: &naga::Block, f: &mut impl FnMut(&naga::Statement)) {
    for stmt in body.iter() {
        for nested in crate::ir::visit::nested_blocks(stmt) {
            crate::ir::visit::for_each_statement(nested, f);
        }
    }
}
