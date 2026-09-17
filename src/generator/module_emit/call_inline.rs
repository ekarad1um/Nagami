//! Call purity and single-use call-inlining analysis.

use crate::analysis::{ExprClass, FnEffects, PointerRoot, resolve_pointer_root};
use crate::handle_set::{HandleMap, HandleSet};
use crate::ir::visit::visit_expression_children;
use crate::passes::expr_util::{root_local_var, short_circuit_rhs};

/// A single-use `Call` result that may still be inlined into a later use site,
/// with the function-locals its arguments load so a Store to a local drops
/// only the pending calls that read it.
struct PendingCall {
    /// Marked inlineable once a non-`Emit` statement consumes the entry.
    result: naga::Handle<naga::Expression>,
    /// The outermost expression currently carrying the call: `result` until an
    /// `Emit` wraps it (`f() + 1`), then the wrapper.  Staying pending through
    /// wrappers keeps the Store / control-flow clearing rules guarding the
    /// call, so a carrier cannot float it past a store to memory it reads.
    carrier: naga::Handle<naga::Expression>,
    reads_locals: HandleSet<naga::LocalVariable>,
}

/// Every function-local whose VALUE a call argument's evaluation depends on, so
/// a Store to it cannot be reordered before the pending call's re-evaluation at
/// a later use site: a `Load` rooted at a local, and a POINTER argument rooted
/// at one (`&d`, `&d.f`, `&arr[i]`), which lets the callee read the pointee at
/// call time - missing it inlines `let c = g(&d); d = ...;` past the store, so
/// the callee derefs the post-store value.  `call_reads` supplies the recorded
/// argument-locals of every pending pure call: a `CallResult` operand is a leaf
/// here, but relocating THIS call bakes the inner call's text inside it and
/// re-evaluates the inner arguments too.
fn collect_loaded_locals(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    call_reads: &CallReads,
    out: &mut HandleSet<naga::LocalVariable>,
    visited: &mut HandleSet<naga::Expression>,
) {
    // A subexpression shared across argument positions is a DAG diamond; the
    // visited set keeps the walk linear.
    if !visited.insert(expr) {
        return;
    }
    if let Some(inner) = call_reads.get(expr) {
        out.extend(inner.iter().copied());
    }
    match &expressions[expr] {
        naga::Expression::Load { pointer } => {
            if let Some(local) = root_local_var(*pointer, expressions) {
                out.insert(local);
            }
        }
        // A pointer to a local (or a component): passed to a callee or loaded
        // later, the local's value is observed at evaluation time.
        naga::Expression::LocalVariable(_)
        | naga::Expression::Access { .. }
        | naga::Expression::AccessIndex { .. } => {
            if let Some(local) = root_local_var(expr, expressions) {
                out.insert(local);
            }
        }
        _ => {}
    }
    visit_expression_children(&expressions[expr], |child| {
        collect_loaded_locals(child, expressions, call_reads, out, visited)
    });
}

/// Argument-locals of every pending pure call so far, keyed by `CallResult`.
/// Entries are never removed: a call that ends up `let`-bound is never
/// consulted through a stashed cone, and an over-approximated read set only
/// retains a pending call longer, the safe direction.
type CallReads = HandleMap<naga::Expression, HandleSet<naga::LocalVariable>>;

/// `Call` results that can be inlined at their single use site instead of
/// being `let`-bound: `ref_count == 1`; the callee is pure, or the call is
/// impure and its consuming statement evaluates no other memory access (an
/// impure call writes memory, and relocating it would reorder that write
/// against an intervening or sibling read - operand order is invisible here,
/// so purity gates the general case); and the consumer is reached without
/// crossing an interfering statement (another `Call`, `Atomic` / `ImageStore` /
/// `RayQuery` and the like, a non-local `Store`, or any control-flow boundary).
/// A pure call still depends on its argument VALUES, so a `Store` to a local
/// drops only the pending calls whose arguments read it.  Consumption is
/// recorded the moment a statement references a pending result, so a later
/// clearing event cannot undo a use already made and program order is kept.
pub(super) fn find_inlineable_calls(
    block: &naga::Block,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
    fn_effects: &[FnEffects],
) -> HandleSet<naga::Expression> {
    // Function-wide: an inner call's stashed text is re-evaluated wherever the
    // OUTER pending call that consumed it ends up.
    let mut call_reads = CallReads::default();
    find_inlineable_calls_in_block(block, ref_counts, expressions, fn_effects, &mut call_reads)
}

fn find_inlineable_calls_in_block(
    block: &naga::Block,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
    fn_effects: &[FnEffects],
    call_reads: &mut CallReads,
) -> HandleSet<naga::Expression> {
    let mut result = HandleSet::default();
    let mut pending: Vec<PendingCall> = Vec::new();
    // An IMPURE call result from an earlier statement, still eligible to inline
    // into its consumer where its write cannot reorder against any other memory
    // access (`impure_call_inlines_safely`).  `Emit`s between call and consumer
    // only build the consuming expression; the first non-`Emit` decides.
    let mut last_impure: Option<naga::Handle<naga::Expression>> = None;

    for stmt in block.iter() {
        if let Some(h) = last_impure {
            let survives_emit = if let naga::Statement::Emit(range) = stmt {
                // An Emit may be skipped only when it assembles the call's own
                // consumer and reads no memory: a materialised `Load` (e.g.
                // `let x = W;` binding a global the call writes) would be
                // hoisted above the call's write.  The call result itself counts
                // as memory-free.  A wrapper the emitter renders at every use
                // ([`shared_pointer_chain`]) would run the call once per use.
                let mut memo = Default::default();
                range.clone().all(|root| {
                    let mut found = false;
                    expr_is_memory_free(root, h, expressions, &mut found, &mut memo)
                        && !(found && shared_pointer_chain(root, ref_counts, expressions))
                })
            } else {
                false
            };
            if survives_emit {
                // Still assembling the consumer; keep `h`.
            } else if matches!(stmt, naga::Statement::Emit(_)) {
                // A memory-reading Emit: the call stays `let`-bound at its own
                // statement.
                last_impure = None;
            } else {
                last_impure = None;
                if impure_call_inlines_safely(stmt, h, expressions) {
                    result.insert(h);
                }
            }
        }

        // Consumption BEFORE the clearing rules, so a control-flow statement
        // cannot retroactively drop a result whose use already happened.
        if !pending.is_empty() {
            consume_pending_for_statement(stmt, expressions, ref_counts, &mut pending, &mut result);
        }

        match stmt {
            naga::Statement::Call {
                result: Some(h),
                function,
                arguments,
                ..
            } if ref_counts[h.index()] == 1 => {
                if fn_effects[function.index()].inline_pure() {
                    // A pure callee is not a reordering barrier: prior pending
                    // pure calls survive it (those its arguments consumed were
                    // already moved to `result`), so adjacent pure-call `let`s
                    // collapse in one pass.  Its own argument locals feed the
                    // `Store` check.
                    let mut reads_locals = Default::default();
                    let mut visited = Default::default();
                    for &arg in arguments {
                        collect_loaded_locals(
                            arg,
                            expressions,
                            call_reads,
                            &mut reads_locals,
                            &mut visited,
                        );
                    }
                    call_reads.insert(*h, reads_locals.clone());
                    pending.push(PendingCall {
                        result: *h,
                        carrier: *h,
                        reads_locals,
                    });
                } else {
                    // Any prior pending call inlined past this write would
                    // reorder against it: drop them all (those consumed by this
                    // call's own arguments are already in `result`).
                    pending.clear();
                    last_impure = Some(*h);
                }
            }
            naga::Statement::Emit(_) | naga::Statement::Return { .. } => {
                // Non-side-effecting: keep pending calls
            }
            naga::Statement::Store { pointer, .. } => {
                // Only pending calls whose arguments read local `L` are
                // re-evaluated after a store to `L`; a non-local store is
                // observable by any callee, so clear everything.
                if let Some(stored) = root_local_var(*pointer, expressions) {
                    pending.retain(|p| !p.reads_locals.contains(stored));
                } else {
                    pending.clear();
                }
            }
            // Every other statement may be side-effecting: drop the pending set
            // and analyse nested blocks with their own empty one.
            _ => {
                pending.clear();
                for nested in crate::ir::visit::nested_blocks(stmt) {
                    result.extend(find_inlineable_calls_in_block(
                        nested,
                        ref_counts,
                        expressions,
                        fn_effects,
                        call_reads,
                    ));
                }
            }
        }
    }

    // A call still pending here has its consumer OUTSIDE the block (a loop's
    // continuing / `break if`, a `Block`'s parent); inlining it would relocate
    // the call past statements this walk never saw, so leftovers stay
    // `let`-bound.
    result
}

/// Whether an impure single-use call producing `call_result` can be inlined
/// into `stmt`, the first non-`Emit` statement after it: safe exactly when every
/// other expression `stmt` evaluates is memory-free (literals, constants,
/// by-value parameters, arithmetic over them), so the call is the statement's
/// only memory access besides its own terminal store and no reorder is
/// observable, whatever the operand evaluation order or `let`-binding choices.
/// Covers `out = (call() - .5) * k`, `if call() == k`, `switch call()` and
/// `arr[const] = call()`; any other consuming statement keeps the call bound.
fn impure_call_inlines_safely(
    stmt: &naga::Statement,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Statement as S;
    let mut found = false;
    let mut memo = Default::default();
    let mut memfree =
        |root| expr_is_memory_free(root, call_result, expressions, &mut found, &mut memo);
    let all_memfree = match stmt {
        // Both operands, no short-circuit, so `found` is set whichever side
        // carries the call.
        S::Store { pointer, value } => {
            let p = memfree(*pointer);
            let v = memfree(*value);
            p && v
        }
        S::Return { value: Some(v) } => memfree(*v),
        // Branch bodies are later statements that legitimately access memory.
        S::If { condition, .. } => memfree(*condition),
        S::Switch { selector, .. } => memfree(*selector),
        _ => return false,
    };
    found && all_memfree
}

/// `true` when evaluating the tree at `root` reads no memory and observes no
/// side effect, with `call_result` (the one call being inlined) a transparent
/// hole that sets `*found`.  A node outside [`ExprClass::MEMORY_FREE_NODE`]
/// (a `Load`, every effect result) is not memory-free, so the predicate is
/// conservative by construction.
///
/// Memory-freedom answers "is a reorder observable"; the call ALSO needs its
/// new position evaluated unconditionally, or its write is skipped on the
/// lanes where a short-circuit `&&` / `||` never reaches the right operand,
/// so a call under that operand makes the tree not memory-free.  Memoised
/// per node as (memory-free, holds the call): both compose from the
/// children, so a shared sub-DAG is walked once.
fn expr_is_memory_free(
    root: naga::Handle<naga::Expression>,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    found: &mut bool,
    memo: &mut HandleMap<naga::Expression, (bool, bool)>,
) -> bool {
    if root == call_result {
        // Reached on a unique path (`ref_count == 1`), so never memoised.
        *found = true;
        return true;
    }
    if let Some(&(memory_free, holds_call)) = memo.get(root) {
        *found |= holds_call;
        return memory_free;
    }
    let mut holds_call = false;
    // A leaf has no children, so the walk below is its `true`.
    let memory_free = ExprClass::node(&expressions[root]).any(ExprClass::MEMORY_FREE_NODE) && {
        let conditional = short_circuit_rhs(&expressions[root]);
        let mut ok = true;
        visit_expression_children(&expressions[root], |child| {
            let mut under_child = false;
            if !expr_is_memory_free(child, call_result, expressions, &mut under_child, memo) {
                ok = false;
            }
            if under_child {
                holds_call = true;
                if Some(child) == conditional {
                    ok = false;
                }
            }
        });
        ok
    };
    *found |= holds_call;
    memo.insert(root, (memory_free, holds_call));
    memory_free
}

/// An `Access` / `AccessIndex` chain rooted at a variable or parameter with
/// more than one consumer: the emitter never `let`-binds a pointer-typed
/// expression, so such a chain is spelled at every use - `let p = &a[f()];
/// *p = 1; x = *p;` renders `a[f()]` twice - and a call carried inside it
/// must stay bound at its own statement.  A by-value parameter's chain is
/// judged the same way (its type is out of reach here): bound in a `let` by
/// the emitter, the stash would have cost nothing, so the decline only
/// loses bytes.
fn shared_pointer_chain(
    h: naga::Handle<naga::Expression>,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    ref_counts[h.index()] > 1
        && matches!(
            expressions[h],
            naga::Expression::Access { .. } | naga::Expression::AccessIndex { .. }
        )
        && !matches!(resolve_pointer_root(h, expressions), PointerRoot::Other)
}

/// Consume the pending calls against `stmt`: a non-`Emit` statement is a real
/// use site and moves every pending call whose carrier it references into
/// `result`; an `Emit` only wraps the call in a larger expression, so it
/// advances the carrier to that wrapper and the guards stay active.
fn consume_pending_for_statement(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    ref_counts: &[usize],
    pending: &mut Vec<PendingCall>,
    result: &mut HandleSet<naga::Expression>,
) {
    // Drain EVERY pending call sharing the carrier: sibling single-use pure
    // calls (`if a()==b()`) merge onto one wrapper carrier and all survived the
    // same clears.
    let check = |h: naga::Handle<naga::Expression>,
                 pending: &mut Vec<PendingCall>,
                 result: &mut HandleSet<naga::Expression>| {
        while let Some(pos) = pending.iter().position(|p| p.carrier == h) {
            result.insert(pending.swap_remove(pos).result);
        }
    };

    match stmt {
        naga::Statement::Emit(range) => {
            // Range order is topological (children precede parents), so the
            // carrier bubbles up to the outermost wrapper.
            for h in range.clone() {
                // A wrapper evaluating the carrier conditionally cannot take
                // it: the call would run on fewer lanes than the input's
                // unconditional statement, and even a pure callee's
                // derivative / texture sample moved under non-uniform control
                // flow is a Dawn error.  The call stays `let`-bound.
                if let Some(rhs) = short_circuit_rhs(&expressions[h]) {
                    pending.retain(|p| p.carrier != rhs);
                }
                // Spelled at every use: the call it carries stays bound.
                let shared = shared_pointer_chain(h, ref_counts, expressions);
                visit_expression_children(&expressions[h], |child| {
                    if shared {
                        pending.retain(|p| p.carrier != child);
                    } else {
                        for p in pending.iter_mut() {
                            if p.carrier == child {
                                p.carrier = h;
                            }
                        }
                    }
                });
            }
        }
        // A `break_if` re-evaluates each iteration while a pending call from
        // the enclosing block was emitted once before the loop; consuming it
        // here would recompute it per iteration (the Store guard cannot see
        // the loop's nested writes).  Left unconsumed it clears at the Loop arm
        // and emits as a pre-loop `let`; calls inside the loop still inline via
        // the recursion's fresh pending set.
        naga::Statement::Loop { .. } => {}
        other => crate::ir::visit::visit_statement_operands(other, false, &mut |h| {
            check(h, pending, result)
        }),
    }
}
