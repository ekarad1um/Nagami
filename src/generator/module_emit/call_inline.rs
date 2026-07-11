//! Call purity and single-use call-inlining analysis.

use super::local_resolve::resolve_local_var;
use super::ref_counts::visit_expr_children;

/// A single-use `Call` result that may still be inlined into a later use
/// site, paired with the set of function-locals its arguments load.  The
/// locals set is what lets a Store to a local invalidate only the pending
/// calls that actually read that local (see the `Store` arm).
struct PendingCall {
    /// The `Call` result handle to mark inlineable once a non-`Emit` statement
    /// finally consumes this pending entry.
    result: naga::Handle<naga::Expression>,
    /// The outermost expression currently carrying the call.  Equal to
    /// `result` until an `Emit` wraps the call in a larger expression (e.g.
    /// `f() + 1`), after which consumption is matched against the wrapper.
    /// Keeping the call pending through such wrappers - rather than moving it
    /// straight to the inlineable set at the `Emit` - is what keeps the Store /
    /// control-flow clearing rules guarding it, so a single-use carrier cannot
    /// float the (pure) call past an intervening store to memory it reads.
    carrier: naga::Handle<naga::Expression>,
    reads_locals: std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
}

/// Collect every function-local whose VALUE a call argument's evaluation
/// depends on, so a Store to that local cannot be reordered before the
/// pending call's (re-)evaluation at a later use site.  Two dependency kinds:
///
/// * a `Load` rooted at a local reads the local's value directly; and
/// * a POINTER argument rooted at a local (`&d`, `&d.f`, `&arr[i]` -
///   i.e. a bare `LocalVariable` / `Access` / `AccessIndex` reference) lets
///   the callee read the pointee at call time, so the result still depends on
///   the local's value when the call is evaluated.
///
/// Missing the pointer case lets a single-use `let c = g(&d); d = ...;` call
/// be inlined past the store, so the callee derefs the post-store value -
/// a silent reorder miscompilation.
fn collect_loaded_locals(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    // A common subexpression shared across argument positions forms a diamond
    // in the expression DAG; without a visited set it would be re-walked once
    // per incoming edge (super-linear).  `out` is a set so re-inserting is
    // harmless - only the traversal cost is saved.
    if !visited.insert(expr) {
        return;
    }
    match &expressions[expr] {
        naga::Expression::Load { pointer } => {
            if let Some(local) = resolve_local_var(*pointer, expressions) {
                out.insert(local);
            }
        }
        // A reference/pointer to a local (or a component of one).  Whether it
        // is passed by pointer to a callee or loaded later, the local's value
        // is observed at evaluation time.
        naga::Expression::LocalVariable(_)
        | naga::Expression::Access { .. }
        | naga::Expression::AccessIndex { .. } => {
            if let Some(local) = resolve_local_var(expr, expressions) {
                out.insert(local);
            }
        }
        _ => {}
    }
    visit_expr_children(&expressions[expr], |child| {
        collect_loaded_locals(child, expressions, out, visited)
    });
}

/// The root a pointer expression resolves to, for the write-effect analysis.
enum PointerRoot {
    /// A function-local variable (a write here is contained in the function).
    Local,
    /// A module-scope global (a write here escapes to every caller).
    Global,
    /// The function's own pointer PARAMETER `idx` (a write through it lands in
    /// whatever the caller passed - escapes to the caller).
    Param(u32),
    /// An exotic pointer expression - conservatively treated as escaping.
    Other,
}

fn resolve_pointer_root(
    ptr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> PointerRoot {
    match &expressions[ptr] {
        naga::Expression::LocalVariable(_) => PointerRoot::Local,
        naga::Expression::GlobalVariable(_) => PointerRoot::Global,
        naga::Expression::FunctionArgument(i) => PointerRoot::Param(*i),
        naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
            resolve_pointer_root(*base, expressions)
        }
        _ => PointerRoot::Other,
    }
}

/// The memory effects of a function that are observable OUTSIDE a call to it.
///
/// Writes to the function's own locals never escape; the two ways an effect
/// reaches the caller are tracked separately so a caller that supplies its OWN
/// local to a param-writing helper stays pure (the helper's write lands in the
/// caller's local, not the caller's caller):
/// * `escapes` - a write to a global, an atomic / image store / barrier / ray /
///   subgroup / cooperative op / `discard`, or such an effect transitively via
///   a callee.  Always observable, regardless of how the function is called.
/// * `written_params` - the function writes through these of its OWN pointer
///   parameters (directly, or by forwarding them to a param-writing callee).
///   Whether THAT escapes depends on what each caller passes.
#[derive(Clone)]
struct FnEffects {
    escapes: bool,
    written_params: std::collections::HashSet<u32>,
}

/// Fold the effect of one statement into `eff`.  Nested control-flow blocks are
/// walked by [`accumulate_block_effects`].
fn accumulate_statement_effects(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
    eff: &mut FnEffects,
) {
    use naga::Statement as S;
    match stmt {
        S::Store { pointer, .. } | S::Atomic { pointer, .. } => {
            match resolve_pointer_root(*pointer, expressions) {
                PointerRoot::Local => {}
                PointerRoot::Param(i) => {
                    eff.written_params.insert(i);
                }
                PointerRoot::Global | PointerRoot::Other => eff.escapes = true,
            }
        }
        S::ImageStore { .. } | S::ImageAtomic { .. } | S::CooperativeStore { .. } => {
            eff.escapes = true
        }
        S::ControlBarrier(_) | S::MemoryBarrier(_) | S::WorkGroupUniformLoad { .. } => {
            eff.escapes = true
        }
        S::RayQuery { .. } | S::RayPipelineFunction(_) => eff.escapes = true,
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => eff.escapes = true,
        S::Kill => eff.escapes = true,
        S::Call {
            function,
            arguments,
            ..
        } => {
            let callee = function_effects(*function, module, memo);
            if callee.escapes {
                eff.escapes = true;
            }
            // Each global/local write the callee performs THROUGH a pointer
            // parameter lands in whatever WE passed for that parameter: our own
            // local stays contained, our own param forwards the escape outward,
            // a global (or an exotic pointer) escapes here and now.
            for &p in &callee.written_params {
                match arguments.get(p as usize) {
                    Some(&arg) => match resolve_pointer_root(arg, expressions) {
                        PointerRoot::Local => {}
                        PointerRoot::Param(i) => {
                            eff.written_params.insert(i);
                        }
                        PointerRoot::Global | PointerRoot::Other => eff.escapes = true,
                    },
                    None => eff.escapes = true, // arity mismatch - stay conservative
                }
            }
        }
        S::Block(inner) => accumulate_block_effects(inner, expressions, module, memo, eff),
        S::If { accept, reject, .. } => {
            accumulate_block_effects(accept, expressions, module, memo, eff);
            accumulate_block_effects(reject, expressions, module, memo, eff);
        }
        S::Switch { cases, .. } => {
            for case in cases {
                accumulate_block_effects(&case.body, expressions, module, memo, eff);
            }
        }
        S::Loop {
            body, continuing, ..
        } => {
            accumulate_block_effects(body, expressions, module, memo, eff);
            accumulate_block_effects(continuing, expressions, module, memo, eff);
        }
        S::Emit(_) | S::Return { .. } | S::Break | S::Continue => {}
    }
}

fn accumulate_block_effects(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
    eff: &mut FnEffects,
) {
    for stmt in block.iter() {
        accumulate_statement_effects(stmt, expressions, module, memo, eff);
    }
}

/// Memoised [`FnEffects`] of `module.functions[h]`.  The call graph is acyclic
/// (naga forbids recursion); the in-progress marker (`escapes = true`) both
/// memoises and makes any unexpected cycle resolve to the conservative
/// "escapes everything", so the recursion always terminates.
fn function_effects(
    h: naga::Handle<naga::Function>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
) -> FnEffects {
    if let Some(known) = &memo[h.index()] {
        return known.clone();
    }
    memo[h.index()] = Some(FnEffects {
        escapes: true,
        written_params: std::collections::HashSet::new(),
    });
    let func = &module.functions[h];
    let mut eff = FnEffects {
        escapes: false,
        written_params: std::collections::HashSet::new(),
    };
    accumulate_block_effects(&func.body, &func.expressions, module, memo, &mut eff);
    memo[h.index()] = Some(eff.clone());
    eff
}

/// Per-`module.functions` inline-purity bitmap, computed once per module and
/// shared by every `find_inlineable_calls` invocation: a single-use `Call` is
/// only ever relocated to an arbitrary use site when its callee is inline-pure,
/// i.e. its only effect observable by the caller is its return value.  That is
/// exactly `!escapes && written_params.is_empty()` - a function writing through
/// one of its OWN params is NOT inline-pure (the write reaches the caller), but
/// a function that merely calls such a helper with its OWN local is.
pub(super) fn compute_pure_functions(module: &naga::Module) -> Vec<bool> {
    let mut memo: Vec<Option<FnEffects>> = vec![None; module.functions.len()];
    for (h, _) in module.functions.iter() {
        function_effects(h, module, &mut memo);
    }
    memo.into_iter()
        .map(|e| match e {
            Some(eff) => !eff.escapes && eff.written_params.is_empty(),
            None => false,
        })
        .collect()
}

/// Identify `Call` results that can be safely inlined at their single use
/// site instead of being bound to a `let`.  A result is inlineable when:
///
/// 1. its `ref_count` is exactly 1 (used once);
/// 2. its callee is PURE (`pure_functions[callee]`), OR it is an impure call
///    whose consuming statement evaluates NO other memory access (see
///    `last_impure` / `impure_call_inlines_safely`): an impure call writes
///    memory, and relocating it to an arbitrary use site would re-order that
///    write against an intervening read OR an operand-evaluation-order sibling
///    read of the same memory - a silent miscompile.  Operand order is
///    invisible here, so purity is the gate for the general case; the exception
///    is a consuming statement whose every other operand is memory-free, where
///    the call is the sole memory access and nothing can be reordered; and
/// 3. the consuming statement is reached from the `Call` without crossing a
///    potentially-interfering side-effecting statement (another `Call`,
///    `Atomic` / `ImageStore` / `RayQuery` and the like, a non-local `Store`,
///    or any control-flow boundary: `If` / `Loop` / `Switch` / `Block`).
///
/// A pure call still depends on its arguments' VALUES at evaluation time, so a
/// `Store` to a function-local does NOT unconditionally clear the pending set;
/// it invalidates only the pending calls whose arguments read that local
/// (tracked per-local via [`PendingCall`]'s `reads_locals`; see the `Store`
/// arm).  Consumption is recorded the moment a statement references a pending
/// result - the handle moves to the inlineable set immediately, so a later
/// clearing event cannot undo a use already made.  Evaluation order is thus
/// preserved: the use site stays in program order and the (pure) call's
/// re-evaluation reads the same argument values it would have at the call site.
pub(super) fn find_inlineable_calls(
    block: &naga::Block,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
    pure_functions: &[bool],
) -> std::collections::HashSet<naga::Handle<naga::Expression>> {
    let mut result = std::collections::HashSet::new();
    let mut pending: Vec<PendingCall> = Vec::new();
    // The result of an IMPURE call from an earlier statement, still eligible to
    // be inlined into the statement that consumes it.  An impure call writes
    // memory, so it may be inlined only where its side effect cannot be
    // reordered against any other memory access - decided by
    // `impure_call_inlines_safely` (the consuming statement's every OTHER
    // operand must be memory-free).  `Emit` statements between the call and its
    // consumer only build the consuming expression and write nothing, so the
    // candidate survives them; the first non-`Emit` statement decides.
    let mut last_impure: Option<naga::Handle<naga::Expression>> = None;

    for stmt in block.iter() {
        if let Some(h) = last_impure {
            let survives_emit = if let naga::Statement::Emit(range) = stmt {
                // The impure call may skip an Emit ONLY when that Emit merely
                // assembles the call's own consuming expression and reads no
                // memory itself.  If the Emit materialises a `Load` (or any
                // other memory access - e.g. `let x = W;` binding a global the
                // call writes), inlining the call into a later consumer would
                // hoist that read above the call's write - a silent miscompile.
                // The call result itself counts as memory-free (it IS the
                // write being relocated), so a range that is otherwise
                // memory-free is safe to skip.
                let mut found = false;
                let mut memo = std::collections::HashMap::new();
                range
                    .clone()
                    .all(|root| expr_is_memory_free(root, h, expressions, &mut found, &mut memo))
            } else {
                false
            };
            if survives_emit {
                // Still assembling a memory-free consuming expression; keep `h`.
            } else if matches!(stmt, naga::Statement::Emit(_)) {
                // A memory-reading Emit: keep the impure call `let`-bound at its
                // own statement rather than relocating past the read.
                last_impure = None;
            } else {
                last_impure = None;
                if impure_call_inlines_safely(stmt, h, expressions) {
                    result.insert(h);
                }
            }
        }

        // Phase 1: record any pending handles consumed by this statement.
        // Order matters: we must detect consumption BEFORE applying the
        // statement's clearing rules, so a later control-flow statement
        // cannot retroactively drop a result whose use already happened.
        if !pending.is_empty() {
            consume_pending_for_statement(stmt, expressions, &mut pending, &mut result);
        }

        // Phase 2: apply the statement's effect on the pending set and
        // recurse into nested blocks.
        match stmt {
            naga::Statement::Call {
                result: Some(h),
                function,
                arguments,
                ..
            } if ref_counts[h.index()] == 1 => {
                // Only a PURE callee may be relocated to an arbitrary later use
                // site: an impure call's write would be re-ordered against an
                // intervening read - or, since operand evaluation order is not
                // visible here, against a sibling read in the use expression
                // itself.  A pure call still depends on its argument VALUES at
                // evaluation time, so track the locals its args read for the
                // `Store`-interference check below.
                if pure_functions[function.index()] {
                    // A pure callee writes no caller-visible memory, so it is
                    // NOT a reordering barrier: any *prior* pending pure call
                    // survives it (just as it survives an `Emit`/`Return`) and
                    // stays inlineable at its own later use site, so several
                    // adjacent pure-call `let`s collapse in ONE pass rather than
                    // one-per-re-minify.  We therefore do NOT clear `pending`.
                    // Phase 1 already moved any pending consumed by THIS call's
                    // arguments into `result`, so the survivors are exactly the
                    // ones whose evaluation is unaffected by this pure call.
                    let mut reads_locals = std::collections::HashSet::new();
                    let mut visited = std::collections::HashSet::new();
                    for &arg in arguments {
                        collect_loaded_locals(arg, expressions, &mut reads_locals, &mut visited);
                    }
                    pending.push(PendingCall {
                        result: *h,
                        carrier: *h,
                        reads_locals,
                    });
                } else {
                    // An impure call IS a side-effecting statement: any prior
                    // pending pure call inlined past it would reorder its
                    // evaluation against this call's write.  Drop them all
                    // (Phase 1 already captured any consumed by this call's own
                    // arguments).  The impure call itself is eligible to be
                    // inlined into its consuming statement only where that
                    // statement evaluates no other memory access (see
                    // `last_impure` / `impure_call_inlines_safely`).
                    pending.clear();
                    last_impure = Some(*h);
                }
            }
            naga::Statement::Emit(_) | naga::Statement::Return { .. } => {
                // Non-side-effecting: keep pending calls
            }
            naga::Statement::Store { pointer, .. } => {
                // A Store to local `L` only interferes with a pending call
                // whose ARGUMENTS load `L`: inlining the call to a later use
                // site re-evaluates its argument text after the store, so a
                // `Load(L)` argument would read the post-store value.  Drop
                // exactly those pending calls; calls whose args don't read
                // `L` stay inlineable.  A non-local Store (storage / workgroup
                // global) can be observed by any callee, so clear everything.
                if let Some(stored) = resolve_local_var(*pointer, expressions) {
                    pending.retain(|p| !p.reads_locals.contains(&stored));
                } else {
                    pending.clear();
                }
            }
            // Every other statement is (or may be) side-effecting: drop the
            // pending set, then analyse any nested blocks independently
            // (each starts with its own empty pending set).
            _ => {
                pending.clear();
                for nested in crate::passes::expr_util::nested_blocks(stmt) {
                    result.extend(find_inlineable_calls(
                        nested,
                        ref_counts,
                        expressions,
                        pure_functions,
                    ));
                }
            }
        }
    }

    result.extend(pending.into_iter().map(|p| p.result));
    result
}

/// Whether an impure single-use call producing `call_result` can be inlined
/// into `stmt`, the first non-`Emit` statement after the call.
///
/// Safe exactly when every expression `stmt` evaluates APART FROM the call is
/// memory-free (a literal / constant / by-value parameter, or arithmetic over
/// such).  Then the inlined call is the statement's ONLY memory access besides
/// its own terminal store, so its side effect cannot be reordered against a
/// sibling read, an intervening read, or a hoisted `let`-bound load - whatever
/// the call writes, nothing else in the statement observes it.  This makes the
/// operand evaluation order (and which sub-expressions the generator chooses to
/// `let`-bind) irrelevant, which is what keeps the analysis sound without
/// modelling either.
///
/// Subsumes the bare `out = call()` / `return call()` direct-value forms and
/// additionally recovers `out = (call() - .5) * k`, `if call() == k`,
/// `switch call()`, and `arr[const] = call()`.  An expression that reads memory
/// anywhere outside the call (e.g. `out = g + call()` with `g` a load) makes
/// the statement ineligible, so the call stays `let`-bound.  Only value-bearing
/// statement kinds are handled; any other consuming statement keeps the call
/// bound.
fn impure_call_inlines_safely(
    stmt: &naga::Statement,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Statement as S;
    let mut found = false;
    let mut memo = std::collections::HashMap::new();
    let mut memfree =
        |root| expr_is_memory_free(root, call_result, expressions, &mut found, &mut memo);
    let all_memfree = match stmt {
        // Evaluate BOTH operands (no short-circuit) so `found` is set whichever
        // side carries the call: an `arr[const] = call()` places it in `value`,
        // while a `bare = (call()..)` also keeps `pointer` memory-free.
        S::Store { pointer, value } => {
            let p = memfree(*pointer);
            let v = memfree(*value);
            p && v
        }
        S::Return { value: Some(v) } => memfree(*v),
        // Only the condition / selector is the consuming expression; the branch
        // bodies are later statements that legitimately access memory.
        S::If { condition, .. } => memfree(*condition),
        S::Switch { selector, .. } => memfree(*selector),
        _ => return false,
    };
    found && all_memfree
}

/// `true` when evaluating the expression tree rooted at `root` reads no memory
/// and observes no side effect, treating `call_result` as a transparent hole
/// (it is the one call we intend to inline) and setting `*found` when that hole
/// is reached.  A `Load`, any effect-result expression (call / atomic / image /
/// ray / subgroup / `arrayLength`), and any future variant default to NOT
/// memory-free, so the predicate is conservative by construction.
fn expr_is_memory_free(
    root: naga::Handle<naga::Expression>,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    found: &mut bool,
    memo: &mut std::collections::HashMap<naga::Handle<naga::Expression>, bool>,
) -> bool {
    if root == call_result {
        // The inlined call is reached on a unique path (`ref_count == 1`), so
        // this never collides with a memoised entry.
        *found = true;
        return true;
    }
    if let Some(&m) = memo.get(&root) {
        return m;
    }
    use naga::Expression as E;
    let memory_free = match &expressions[root] {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_) => true,
        E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. }
        | E::Compose { .. }
        | E::Derivative { .. } => {
            let mut ok = true;
            visit_expr_children(&expressions[root], |child| {
                if !expr_is_memory_free(child, call_result, expressions, found, memo) {
                    ok = false;
                }
            });
            ok
        }
        _ => false,
    };
    memo.insert(root, memory_free);
    memory_free
}

/// Consume the pending call list against `stmt`.  A non-`Emit` statement is a
/// real use site: every pending call whose carrier it references moves into
/// `result` (the inlineable set).  An `Emit` is not a use site - it only wraps
/// the call in a larger expression, so it advances each matching call's carrier
/// to that wrapper, keeping the Store / control-flow guards active until a real
/// consumer is reached.
fn consume_pending_for_statement(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut Vec<PendingCall>,
    result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    // A NON-`Emit` statement is a real consumption site: relocate the pending
    // call's result into the inlineable set (matched against its current
    // carrier, which an earlier `Emit` may have advanced past the raw call).
    // Drain EVERY pending call sharing the carrier, not just the first: sibling
    // single-use pure calls (`if a()==b()`) are merged onto one wrapper carrier
    // by the `Emit` arm, and all are safe to inline at this shared use site
    // (each survived the same Store / control-flow clears).
    let check =
        |h: naga::Handle<naga::Expression>,
         pending: &mut Vec<PendingCall>,
         result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>| {
            while let Some(pos) = pending.iter().position(|p| p.carrier == h) {
                result.insert(pending.swap_remove(pos).result);
            }
        };

    match stmt {
        naga::Statement::Emit(range) => {
            // An `Emit` does not FIX the call's emission point - it only builds a
            // larger expression around it that is itself emitted later.  Advance
            // each pending call's carrier to the wrapping expression instead of
            // moving it to the inlineable set, so the Store / control-flow
            // clearing rules keep guarding it until a non-`Emit` statement
            // consumes it.  Range order is topological (children precede
            // parents), so the carrier bubbles up to the outermost expression.
            for h in range.clone() {
                visit_expr_children(&expressions[h], |child| {
                    for p in pending.iter_mut() {
                        if p.carrier == child {
                            p.carrier = h;
                        }
                    }
                });
            }
        }
        naga::Statement::If { condition, .. } => check(*condition, pending, result),
        naga::Statement::Switch { selector, .. } => check(*selector, pending, result),
        // A loop's `break_if` is re-evaluated each iteration, but a pending
        // call from the ENCLOSING block was emitted once before the loop;
        // consuming it into `break_if` would recompute it per iteration - a
        // silent miscompile when a call argument reads a local the loop
        // mutates (the intra-block Store-interference guard cannot see those
        // nested writes).  Left unconsumed it falls to `pending.clear()` in
        // `find_inlineable_calls`'s Loop arm and emits as a pre-loop `let`;
        // calls emitted INSIDE the loop still inline via that function's
        // recursion into body/continuing with a fresh pending set.
        naga::Statement::Loop { .. } => {}
        naga::Statement::Return { value: Some(h) } => check(*h, pending, result),
        naga::Statement::Store { pointer, value } => {
            check(*pointer, pending, result);
            check(*value, pending, result);
        }
        naga::Statement::Call { arguments, .. } => {
            for a in arguments {
                check(*a, pending, result);
            }
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            check(*image, pending, result);
            check(*coordinate, pending, result);
            if let Some(i) = array_index {
                check(*i, pending, result);
            }
            check(*value, pending, result);
        }
        naga::Statement::Atomic {
            pointer,
            value,
            fun,
            ..
        } => {
            check(*pointer, pending, result);
            check(*value, pending, result);
            crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                check(h, pending, result)
            });
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            fun,
        } => {
            check(*image, pending, result);
            check(*coordinate, pending, result);
            if let Some(i) = array_index {
                check(*i, pending, result);
            }
            check(*value, pending, result);
            crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                check(h, pending, result)
            });
        }
        naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            check(*pointer, pending, result);
        }
        naga::Statement::SubgroupBallot {
            predicate: Some(p), ..
        } => check(*p, pending, result),
        naga::Statement::SubgroupGather { argument, mode, .. } => {
            check(*argument, pending, result);
            match mode {
                naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                naga::GatherMode::Broadcast(h)
                | naga::GatherMode::Shuffle(h)
                | naga::GatherMode::ShuffleDown(h)
                | naga::GatherMode::ShuffleUp(h)
                | naga::GatherMode::ShuffleXor(h)
                | naga::GatherMode::QuadBroadcast(h) => check(*h, pending, result),
            }
        }
        naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
            check(*argument, pending, result);
        }
        naga::Statement::RayQuery { query, fun } => {
            check(*query, pending, result);
            match fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => {
                    check(*acceleration_structure, pending, result);
                    check(*descriptor, pending, result);
                }
                naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                    check(*hit_t, pending, result);
                }
                _ => {}
            }
        }
        naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            acceleration_structure,
            descriptor,
            payload,
        }) => {
            check(*acceleration_structure, pending, result);
            check(*descriptor, pending, result);
            check(*payload, pending, result);
        }
        naga::Statement::CooperativeStore { target, data } => {
            check(*target, pending, result);
            check(data.pointer, pending, result);
            check(data.stride, pending, result);
        }
        // No direct expression references (or only nested blocks handled by recursion):
        naga::Statement::Block(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::Return { value: None }
        | naga::Statement::SubgroupBallot {
            predicate: None, ..
        } => {}
    }
}
