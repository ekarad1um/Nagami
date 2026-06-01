//! Dead-branch elimination.  Three primary phases plus several
//! structural simplifications folded into the constant-condition pass:
//!
//! 1. Short-circuit re-sugaring folds the `if/else store false`
//!    patterns naga's WGSL frontend emits for `&&` and `||` back into
//!    compact [`naga::Expression::Binary`] `LogicalAnd` / `LogicalOr`
//!    expressions before the next phase destroys their shape.
//! 2. Redundant `else`-store elimination removes writes that assign
//!    the same known literal already present in the variable on the
//!    opposite branch, shrinking unbalanced two-arm ifs into one arm.
//! 3. Constant-condition / structural cleanup (`eliminate_dead_branches`)
//!    strips `if (true)` / `if (false)` arms AND performs several
//!    cooperative simplifications in the same walk:
//!    * empty-If / empty-Switch / empty-Block elision,
//!    * else-block elision when the accept branch unconditionally
//!      terminates (return / break / continue / kill),
//!    * dead code after a terminating statement,
//!    * `loop { ... break if true; }` unwrap when no bare break/continue
//!      would mis-target after unwrapping,
//!    * `break if false` dropped (loop never terminates via that branch),
//!    * single-default-case Switch splicing.
//!
//! Phase order is load-bearing: short-circuit patterns rely on the
//! untransformed frontend output, so re-sugaring must run before the
//! later phases mutate the statement shape.
//!
//! The "branch flipping" optimisation listed in the project README
//! (`if c {} else { x; }` -> `if !c { x; }`) is NOT implemented here;
//! it lives at emit time in `generator::stmt_emit` (search for
//! "flip the condition").

use std::collections::{HashMap, HashSet};

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// `true` when `expr` is a "value-only declarative" expression: one
/// that names a value in the function's permanent scope (Literal,
/// Constant, Override, ZeroValue, FunctionArgument, GlobalVariable,
/// LocalVariable) and carries no implicit `let` binding scoped to a
/// statement.  Tightens `expression_needs_emit` for the short-circuit
/// re-sugar invariant: `!expression_needs_emit` would also pass for
/// `CallResult`/`AtomicResult`/etc., which `single_store_info` already
/// excludes for orthogonal reasons (the originating Call would be a
/// second statement in the branch), but mirroring the assertion to
/// the documented declarative-only invariant survives future
/// relaxations of `single_store_info` that might no longer guarantee
/// the exclusion.
fn is_value_only_declarative(expr: &naga::Expression) -> bool {
    let result = matches!(
        expr,
        naga::Expression::Literal(_)
            | naga::Expression::Constant(_)
            | naga::Expression::Override(_)
            | naga::Expression::ZeroValue(_)
            | naga::Expression::FunctionArgument(_)
            | naga::Expression::GlobalVariable(_)
            | naga::Expression::LocalVariable(_)
    );
    // Invariant: this set is a subset of `!expression_needs_emit`.
    // A variant added here that needs an Emit slot would silently
    // break the short-circuit re-sugar.
    debug_assert!(
        !result || !super::expr_util::expression_needs_emit(expr),
        "is_value_only_declarative classified {expr:?} as declarative but \
         expression_needs_emit reports it requires an Emit slot",
    );
    result
}

use super::load_dedup::{collect_modified_locals, get_stored_local, is_zero_literal};
use super::scoped_map::ScopedMap;

/// Dead-branch pass.  See the module-level doc for the three phases
/// this pass runs per function on every sweep.
#[derive(Debug, Default)]
pub struct DeadBranchPass;

impl Pass for DeadBranchPass {
    fn name(&self) -> &'static str {
        "dead_branch_elimination"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // Compute the constant-handle -> `Literal` cache once so the
        // mutable function iteration below does not have to re-borrow
        // `module.constants` on every lookup.
        let const_lits = build_const_literal_cache(module);

        for (_, function) in module.functions.iter_mut() {
            // Phase 0: short-circuit re-sugaring runs before the else
            // store phase destroys the lowered patterns.
            changed += desugar_short_circuit(&mut function.body, &mut function.expressions);
            changed += eliminate_redundant_else_stores_in_function(function, &const_lits);
            changed +=
                eliminate_dead_branches(&mut function.body, &function.expressions, &const_lits);
        }
        for entry in module.entry_points.iter_mut() {
            changed +=
                desugar_short_circuit(&mut entry.function.body, &mut entry.function.expressions);
            changed +=
                eliminate_redundant_else_stores_in_function(&mut entry.function, &const_lits);
            changed += eliminate_dead_branches(
                &mut entry.function.body,
                &entry.function.expressions,
                &const_lits,
            );
        }

        Ok(changed > 0)
    }
}

// MARK: Short-circuit re-sugaring

// naga's WGSL front-end lowers short-circuit operators into explicit
// if/else statements that write the intermediate result to a local:
//
//   // `a && b`
//   var d: bool;
//   if (a) { d = b; } else { d = false; }
//
//   // `a || b`
//   var d: bool;
//   if (!a) { d = b; } else { d = true; }
//
// This phase detects both shapes and folds them back into
// `Binary(LogicalAnd)` / `Binary(LogicalOr)` expressions so downstream
// passes (and the generator) see the compact form.

/// Recursively fold short-circuit if/else store patterns into
/// `Binary` logical-and / logical-or expressions.  Returns the number
/// of replacements performed so the caller can aggregate a change
/// count across phases.
fn desugar_short_circuit(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;

    for (mut statement, span) in original.span_into_iter() {
        // Step 1: recurse into nested blocks.  Leaf statements are
        // enumerated explicitly so a future naga release adding a
        // new block-bearing variant breaks the build here instead
        // of silently bypassing recursion.
        match &mut statement {
            naga::Statement::Block(inner) => {
                changed += desugar_short_circuit(inner, expressions);
            }
            naga::Statement::If { accept, reject, .. } => {
                changed += desugar_short_circuit(accept, expressions);
                changed += desugar_short_circuit(reject, expressions);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    changed += desugar_short_circuit(&mut case.body, expressions);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed += desugar_short_circuit(body, expressions);
                changed += desugar_short_circuit(continuing, expressions);
            }
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

        // Step 2: check for short-circuit patterns.
        match statement {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // Pattern 1 (&&): `if (cond) { d = val; } else { d = false; }`
                // -> `d = cond && val;`
                //
                // Scope invariant: the rewritten Binary lives in the
                // parent block, so both `val_a` and `condition` must
                // already be in scope there.
                //   * `condition`: a WGSL `if` evaluates its condition
                //     before entering either branch, so its `Emit`
                //     dominates the If and lives in the parent block.
                //   * `val_a`: `single_store_info` rejects any branch
                //     with an `Emit`, leaving `val_a` declarative
                //     (Literal/Constant/Override/ZeroValue/
                //     FunctionArgument/{Global,Local}Variable - all
                //     "in scope everywhere").  The explicit
                //     `is_value_only_declarative` guard mirrors that
                //     invariant: a malformed IR where the producing
                //     statement for a `CallResult`/`AtomicResult` was
                //     removed but the result expression handle survived
                //     would otherwise pass `single_store_info` (bare
                //     Store) and the resugared Binary would reference
                //     an out-of-scope handle.
                if let (Some((ptr_a, val_a)), Some((ptr_r, val_r))) =
                    (single_store_info(&accept), single_store_info(&reject))
                    && same_local_pointer(ptr_a, ptr_r, expressions)
                    && is_bool_false(expressions, val_r)
                    && is_value_only_declarative(&expressions[val_a])
                {
                    let binary = expressions.append(
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::LogicalAnd,
                            left: condition,
                            right: val_a,
                        },
                        naga::Span::default(),
                    );
                    // `single_store_info` requires both branches to
                    // contain exactly one `Store` statement (no `Emit`s,
                    // no nested control flow), so `accept`/`reject`
                    // carry no expressions worth preserving here -
                    // drop both blocks.  The new `Emit` + `Store` above
                    // already represents the entire post-fold body.
                    drop(accept);
                    drop(reject);
                    rebuilt.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(binary, binary)),
                        span,
                    );
                    rebuilt.push(
                        naga::Statement::Store {
                            pointer: ptr_a,
                            value: binary,
                        },
                        span,
                    );
                    changed += 1;
                    continue;
                }

                // Pattern 2 (||): `if (!cond) { d = val; } else { d = true; }`
                // -> `d = !cond || val;` (rewritten as `inner_cond || val`).
                //
                // Same scope and `is_value_only_declarative` invariants
                // as pattern 1.  `inner_cond` (the operand of the
                // LogicalNot) is in scope transitively: the LogicalNot's
                // Emit dominates the If, and the LogicalNot can only
                // reference operands whose Emit dominates IT.
                if let Some(inner_cond) = unwrap_logical_not(condition, expressions)
                    && let (Some((ptr_a, val_a)), Some((ptr_r, val_r))) =
                        (single_store_info(&accept), single_store_info(&reject))
                    && same_local_pointer(ptr_a, ptr_r, expressions)
                    && is_bool_true(expressions, val_r)
                    && is_value_only_declarative(&expressions[val_a])
                {
                    let binary = expressions.append(
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::LogicalOr,
                            left: inner_cond,
                            right: val_a,
                        },
                        naga::Span::default(),
                    );
                    // `single_store_info` requires both branches to
                    // contain exactly one `Store` statement (no `Emit`s,
                    // no nested control flow), so `accept`/`reject`
                    // carry no expressions worth preserving here -
                    // drop both blocks.  The new `Emit` + `Store` above
                    // already represents the entire post-fold body.
                    drop(accept);
                    drop(reject);
                    rebuilt.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(binary, binary)),
                        span,
                    );
                    rebuilt.push(
                        naga::Statement::Store {
                            pointer: ptr_a,
                            value: binary,
                        },
                        span,
                    );
                    changed += 1;
                    continue;
                }

                // No pattern matched - keep the If.
                rebuilt.push(
                    naga::Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            other => {
                rebuilt.push(other, span);
            }
        }
    }

    *block = rebuilt;
    changed
}

/// Return the pointer and value of a block whose sole statement is a
/// single `Store`.  Yields `None` when the block contains any `Emit`,
/// has zero stores, more than one store, or any other statement.
/// The desugaring caller uses this to recognise the single-assign
/// arm of the lowered short-circuit patterns AND to guarantee that
/// no expensive intermediate computation is moved out of the branch.
fn single_store_info(
    block: &naga::Block,
) -> Option<(
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
)> {
    // Reject branches that carry any `Emit` range.  An `Emit`
    // represents intermediate-expression evaluation (e.g., a load, a
    // binary op, an array index), and the short-circuit re-sugar
    // would *hoist* every such `Emit` out of the branch in
    // `hoist_emits`.  Once hoisted, the computation runs
    // unconditionally - even when WGSL's `&&` / `||` would
    // short-circuit at runtime.  For the conditional-load case
    // (`if c { d = arr[i]; } else { d = false; }`) hoisting the
    // `arr[i]` load makes it observable on every iteration, which is
    // both wasteful and (for `i` out of bounds) potentially observable
    // through implementation-defined out-of-range behaviour.  We
    // therefore restrict the pattern to branches that store a
    // declarative-only value (literal, FunctionArgument, LocalVariable
    // ref, GlobalVariable ref, Constant, ZeroValue, ...) where no
    // `Emit` is required inside the branch in the first place.
    let mut result = None;
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(_) => return None,
            naga::Statement::Store { pointer, value } => {
                if result.is_some() {
                    return None;
                }
                result = Some((*pointer, *value));
            }
            _ => return None,
        }
    }
    result
}

/// `true` when `a` and `b` both resolve to the same `LocalVariable`
/// handle.  Only direct local references are compared; swizzle,
/// access, and pointer arithmetic all bail out as "not same".
fn same_local_pointer(
    a: naga::Handle<naga::Expression>,
    b: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    if let (naga::Expression::LocalVariable(la), naga::Expression::LocalVariable(lb)) =
        (&expressions[a], &expressions[b])
    {
        la == lb
    } else {
        false
    }
}

/// Check if an expression is the boolean literal `false`.
fn is_bool_false(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        &expressions[handle],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Check if an expression is the boolean literal `true`.
fn is_bool_true(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
) -> bool {
    matches!(
        &expressions[handle],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

/// Unwrap `!expr` and return the inner handle, or `None` when the
/// condition is not a `LogicalNot`.
fn unwrap_logical_not(
    condition: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[condition]
    {
        Some(*inner)
    } else {
        None
    }
}

// MARK: Constant-condition elimination

/// `true` when `block` (recursively) contains a statement that produces a
/// result EXPRESSION (`Call`/`Atomic`/`WorkGroupUniformLoad`/`RayQuery`/
/// `Subgroup*`).  A constant-condition collapse KEEPS such an `if`/`switch`
/// branch intact rather than dropping it: dropping it would orphan the result
/// expression (its producer statement is gone), which fails validation and
/// rolls the whole pass back every sweep.  Variant-only and conservative - a
/// result-less `Call` also trips it, at the cost of one kept-but-dead branch.
fn block_has_result_producer(block: &naga::Block) -> bool {
    use naga::Statement as S;
    block.iter().any(|stmt| match stmt {
        S::Call { .. }
        | S::Atomic { .. }
        | S::WorkGroupUniformLoad { .. }
        | S::RayQuery { .. }
        | S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => true,
        S::Block(inner) => block_has_result_producer(inner),
        S::If { accept, reject, .. } => {
            block_has_result_producer(accept) || block_has_result_producer(reject)
        }
        S::Switch { cases, .. } => cases.iter().any(|c| block_has_result_producer(&c.body)),
        S::Loop {
            body, continuing, ..
        } => block_has_result_producer(body) || block_has_result_producer(continuing),
        _ => false,
    })
}

/// Recursively walk `block`, folding branches whose condition is a
/// compile-time literal `true` / `false` and pruning unreachable switch
/// cases.  Returns the number of transformations applied so the caller can
/// aggregate change counts across phases.
fn eliminate_dead_branches(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> usize {
    let original = std::mem::take(block);
    let mut rebuilt = naga::Block::with_capacity(original.len());
    let mut changed = 0usize;
    let total = original.len();
    let mut processed = 0usize;

    for (mut statement, span) in original.span_into_iter() {
        processed += 1;

        // Step 1: recurse into nested blocks first.  Leaf statements
        // are enumerated explicitly so a future naga release adding
        // a new block-bearing variant breaks the build instead of
        // silently bypassing recursion.
        match &mut statement {
            naga::Statement::Block(inner) => {
                changed += eliminate_dead_branches(inner, expressions, const_lits);
            }
            naga::Statement::If { accept, reject, .. } => {
                changed += eliminate_dead_branches(accept, expressions, const_lits);
                changed += eliminate_dead_branches(reject, expressions, const_lits);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases.iter_mut() {
                    changed += eliminate_dead_branches(&mut case.body, expressions, const_lits);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                changed += eliminate_dead_branches(body, expressions, const_lits);
                changed += eliminate_dead_branches(continuing, expressions, const_lits);
            }
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

        // Step 2: structural simplification - fold constant
        // If/Switch conditions, drain empty branches, unwrap
        // degenerate Switch, drop `break if true/false`, elide
        // empty nested Blocks.  See the per-arm comments below.
        match statement {
            // If with constant boolean condition
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => match resolve_to_literal(expressions, condition, const_lits) {
                // `resolve_to_literal` also folds an
                // `Expression::Constant(c)` whose init is a bool
                // literal, so a branch on a named `const X: bool = ...;`
                // is eliminated even before const_fold inlines it.
                Some(naga::Literal::Bool(true)) => {
                    // `accept` is taken; `reject` is dropped.  If `reject`
                    // produced a statement result (`Call`/`Atomic`/...), dropping
                    // it would orphan that result EXPRESSION (its producer is
                    // gone) - invalid IR the validator rejects, rolling the whole
                    // pass back every sweep.  Keep the `if` intact in that case:
                    // the branch is dead-but-valid, and leaving it avoids both
                    // the invalid IR and the wasted per-sweep revalidation.
                    if block_has_result_producer(&reject) {
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                    } else {
                        splice_block(&mut rebuilt, accept);
                        changed += 1;
                    }
                }
                Some(naga::Literal::Bool(false)) => {
                    // `accept` is dropped: same orphaned-result hazard.
                    if block_has_result_producer(&accept) {
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                    } else {
                        splice_block(&mut rebuilt, reject);
                        changed += 1;
                    }
                }
                _ => {
                    // After recursion, if both branches are empty the
                    // entire If is a no-op and can be discarded.
                    if accept.is_empty() && reject.is_empty() {
                        changed += 1;
                    } else if !reject.is_empty() && block_definitely_terminates(&accept) {
                        // Else-elision: `if c { return; } else { x; }` ->
                        // `if c { return; } x;` drops the `else{}`
                        // scaffolding.  Still fires when reject ALSO
                        // terminates with a different value-bearing
                        // return - the saved scaffolding outweighs the
                        // dead-but-conditional `if c { return v1; }`.
                        // Only symmetric value-less returns make the
                        // trailing return redundant; we don't collapse
                        // that case.
                        let hoisted = reject;
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject: naga::Block::new(),
                            },
                            span,
                        );
                        splice_block(&mut rebuilt, hoisted);
                        changed += 1;
                    } else {
                        rebuilt.push(
                            naga::Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        );
                    }
                }
            },

            // Switch with constant integer selector
            naga::Statement::Switch { selector, cases } => {
                if let Some(value) = resolve_switch_value(selector, expressions, const_lits) {
                    // A constant-selector collapse splices the matched case and
                    // DROPS the others (or all, on no-match).  Dropping a case
                    // that produced a statement result orphans that result
                    // expression - invalid IR that rolls the WHOLE pass back
                    // every sweep (also discarding unrelated dead_branch work).
                    // Keep the switch intact when any case body carries a result
                    // producer, mirroring the `if`-arm guard.  (The degenerate
                    // sole-`default` splice below drops nothing, so needs no
                    // guard.)
                    if cases.iter().any(|c| block_has_result_producer(&c.body)) {
                        rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                    } else {
                        match find_matching_case_index(&cases, value) {
                            Some(start_idx) if !case_body_has_bare_break(&cases, start_idx) => {
                                let body = collect_case_body(cases, start_idx);
                                splice_block(&mut rebuilt, body);
                                changed += 1;
                            }
                            None => {
                                // No match and no default -> the switch is a no-op.
                                changed += 1;
                            }
                            // Matched case body has a bare Break that targets the
                            // switch.  Splicing would mis-target it, so keep the
                            // switch as-is.
                            Some(_) => {
                                rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                            }
                        }
                    }
                } else {
                    // Degenerate switch: exactly one case that is Default.
                    // The body always executes, so splice it directly
                    // (unless it contains a bare Break targeting the switch).
                    if cases.len() == 1
                        && cases[0].value == naga::SwitchValue::Default
                        && !contains_bare_break(&cases[0].body)
                    {
                        let body = cases.into_iter().next().unwrap().body;
                        splice_block(&mut rebuilt, body);
                        changed += 1;
                    }
                    // After recursion, if every case body is empty the
                    // entire Switch is a no-op and can be discarded.
                    else if cases.iter().all(|c| c.body.is_empty()) {
                        changed += 1;
                    } else {
                        rebuilt.push(naga::Statement::Switch { selector, cases }, span);
                    }
                }
            }

            // Loop with constant break_if.  Routed through
            // `resolve_to_literal` so `break if NAMED_CONST` folds
            // without waiting for `const_fold` to inline the
            // constant - matches the If / Switch arms above.
            naga::Statement::Loop {
                body,
                continuing,
                break_if: Some(bi),
            } => match resolve_to_literal(expressions, bi, const_lits) {
                Some(naga::Literal::Bool(true)) => {
                    // `break if true` -> loop executes body + continuing once.
                    // Only safe when body/continuing have no bare Break/Continue
                    // that would target this loop (those would mis-target after
                    // unwrapping).
                    if !contains_bare_loop_control(&body)
                        && !contains_bare_loop_control(&continuing)
                    {
                        splice_block(&mut rebuilt, body);
                        splice_block(&mut rebuilt, continuing);
                        changed += 1;
                    } else {
                        // Unsafe to unwrap - keep the loop.
                        rebuilt.push(
                            naga::Statement::Loop {
                                body,
                                continuing,
                                break_if: Some(bi),
                            },
                            span,
                        );
                    }
                }
                Some(naga::Literal::Bool(false)) => {
                    // `break if false` -> never breaks via break_if; drop it.
                    rebuilt.push(
                        naga::Statement::Loop {
                            body,
                            continuing,
                            break_if: None,
                        },
                        span,
                    );
                    changed += 1;
                }
                _ => {
                    rebuilt.push(
                        naga::Statement::Loop {
                            body,
                            continuing,
                            break_if: Some(bi),
                        },
                        span,
                    );
                }
            },

            // Drop nested `Block(inner)` if recursion emptied it;
            // mirrors the empty-If / empty-Switch eliminations
            // above.  Without this an upstream fold that drains
            // the body leaves a vacuous `{}` in the output.
            naga::Statement::Block(inner) if inner.is_empty() => {
                changed += 1;
            }

            // Non-empty Block: kept verbatim.  Explicit arm (vs
            // catch-all) so a reader can see the empty-Block
            // elision above doesn't accidentally swallow live blocks.
            naga::Statement::Block(inner) => {
                rebuilt.push(naga::Statement::Block(inner), span);
            }

            // Catch-all on a transform rewriter (not a walker): the
            // safe default for an unknown statement variant is
            // "preserve unchanged".  A future naga variant that
            // warrants a dedicated transform must be added above; if
            // none does, this keeps the pass sound.
            other => {
                rebuilt.push(other, span);
            }
        }

        // Step 3: if the last statement in `rebuilt` definitely terminates,
        // all remaining statements in the original block are dead.
        if rebuilt.last().is_some_and(definitely_terminates) {
            let dead = total - processed;
            changed += dead;
            break;
        }
    }

    *block = rebuilt;
    changed
}

/// Move all statements from `source` into `target`, preserving spans.
fn splice_block(target: &mut naga::Block, source: naga::Block) {
    for (stmt, sp) in source.span_into_iter() {
        target.push(stmt, sp);
    }
}

/// Returns `true` when `stmt` unconditionally terminates control flow in
/// its enclosing block (i.e. no path through `stmt` falls through to the
/// next statement).
fn definitely_terminates(stmt: &naga::Statement) -> bool {
    match stmt {
        naga::Statement::Return { .. }
        | naga::Statement::Kill
        | naga::Statement::Break
        | naga::Statement::Continue => true,
        naga::Statement::Block(inner) => block_definitely_terminates(inner),
        naga::Statement::If { accept, reject, .. } => {
            block_definitely_terminates(accept) && block_definitely_terminates(reject)
        }
        // A loop whose body always terminates *without* Break/Continue
        // (which would exit/restart the loop rather than the enclosing
        // scope) never falls through to the next statement.
        naga::Statement::Loop { body, .. } => {
            block_definitely_terminates(body) && !contains_bare_loop_control(body)
        }
        // A switch terminates the outer block iff every non-
        // fall-through case exits *beyond* the switch
        // (Return/Kill/Continue), a Default case exists, AND the
        // last case does NOT have `fall_through: true`.  A bare
        // Break exits the switch only (resumes after it), so it
        // does not qualify; see [`case_body_terminates_beyond_switch`].
        // The last-case fall-through guard catches IR rebuilt by
        // inlining / CSE - naga's frontend never emits this shape,
        // but the rebuilders could; without the guard those switches
        // mis-classify as terminating.
        naga::Statement::Switch { cases, .. } => {
            let last_falls_through = cases.last().is_some_and(|c| c.fall_through);
            cases
                .iter()
                .all(|c| c.fall_through || case_body_terminates_beyond_switch(&c.body))
                && cases.iter().any(|c| c.value == naga::SwitchValue::Default)
                && !last_falls_through
        }
        _ => false,
    }
}

/// Returns `true` when the last statement of `block` definitely terminates.
fn block_definitely_terminates(block: &naga::Block) -> bool {
    block.last().is_some_and(definitely_terminates)
}
/// Returns `true` when a statement inside a switch case terminates control
/// flow *beyond* the switch itself (i.e., exits the function or enclosing
/// loop, not just the switch).
///
/// `Break` inside a switch case exits the switch and lets execution resume
/// after the switch statement - it does NOT prevent subsequent statements in
/// the outer block from running.  Therefore `Break` must NOT count here.
/// `Continue` inside a switch-case-body-inside-a-loop does jump past the
/// switch to the loop's continuing block, so it IS a beyond-switch terminator.
fn case_body_terminates_beyond_switch(block: &naga::Block) -> bool {
    block.last().is_some_and(|stmt| match stmt {
        naga::Statement::Return { .. } | naga::Statement::Kill | naga::Statement::Continue => true,
        // Break in a switch case exits the switch only.
        naga::Statement::Break => false,
        naga::Statement::Block(inner) => case_body_terminates_beyond_switch(inner),
        naga::Statement::If { accept, reject, .. } => {
            case_body_terminates_beyond_switch(accept) && case_body_terminates_beyond_switch(reject)
        }
        // A `loop` inside a switch case only falls through if it contains a
        // bare `break` (which exits the loop, not the switch).  When the loop
        // body definitely terminates *and* has no bare loop-control
        // statements, the only way out is Return/Kill, both of which exit
        // beyond the switch.  Matches the reasoning in `definitely_terminates`.
        naga::Statement::Loop { body, .. } => {
            block_definitely_terminates(body) && !contains_bare_loop_control(body)
        }
        // A nested switch only terminates beyond if all its cases do
        // so AND the last case does not fall through.  See the
        // identical reasoning on `definitely_terminates`'s Switch
        // arm: a `fall_through: true` last case means execution
        // falls past the (nonexistent) next case and out of the
        // switch, which is Break-equivalent and therefore does NOT
        // terminate beyond.  Without this gate, a nested switch
        // wrapped in another switch case (or a function body using
        // this predicate transitively) would mis-classify and let
        // upstream dead-code elimination drop statements that
        // execution can actually reach.
        naga::Statement::Switch { cases, .. } => {
            let last_falls_through = cases.last().is_some_and(|c| c.fall_through);
            cases
                .iter()
                .all(|c| c.fall_through || case_body_terminates_beyond_switch(&c.body))
                && cases.iter().any(|c| c.value == naga::SwitchValue::Default)
                && !last_falls_through
        }
        _ => false,
    })
}

/// Try to resolve a switch selector expression to a concrete `SwitchValue`.
///
/// Consults `const_lits` so a selector that is `Expression::Constant(c)`
/// (a named constant whose init is an integer literal) is also resolved.
fn resolve_switch_value(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> Option<naga::SwitchValue> {
    match resolve_to_literal(expressions, handle, const_lits)? {
        naga::Literal::I32(v) => Some(naga::SwitchValue::I32(v)),
        naga::Literal::U32(v) => Some(naga::SwitchValue::U32(v)),
        _ => None,
    }
}

/// Find the body to execute for a given constant switch value.
///
/// Handles fall-through: in naga IR a case with `fall_through: true` and an
/// empty body represents a multi-value match (e.g. `case 1, 2:`).  We walk
/// forward from the matched case collecting bodies until we reach a case
/// with `fall_through: false`.
///
/// If no case matches and a `Default` case exists, its index is returned.
fn find_matching_case_index(cases: &[naga::SwitchCase], value: naga::SwitchValue) -> Option<usize> {
    cases.iter().position(|c| c.value == value).or_else(|| {
        cases
            .iter()
            .position(|c| c.value == naga::SwitchValue::Default)
    })
}

/// Collect the combined case body starting at `start_idx`, following
/// fall-through chains.  Consumes the `cases` vector.
fn collect_case_body(cases: Vec<naga::SwitchCase>, start_idx: usize) -> naga::Block {
    let mut combined = naga::Block::new();
    for case in cases.into_iter().skip(start_idx) {
        let done = !case.fall_through;
        splice_block(&mut combined, case.body);
        if done {
            break;
        }
    }
    combined
}

/// Returns `true` if the combined case body starting at `start_idx`
/// (following fall-through) contains a bare `Break`.
fn case_body_has_bare_break(cases: &[naga::SwitchCase], start_idx: usize) -> bool {
    for case in &cases[start_idx..] {
        if contains_bare_break(&case.body) {
            return true;
        }
        if !case.fall_through {
            break;
        }
    }
    false
}

/// Returns `true` if `block` contains a bare `Break` or `Continue` that
/// targets the immediately enclosing loop.
///
/// In naga IR `Break` exits the innermost `Loop` **or** `Switch`, while
/// `Continue` targets only the innermost `Loop`.  Therefore:
/// - do NOT recurse into `Loop` (captures both Break and Continue).
/// - do NOT recurse into `Switch` for `Break` (captured by Switch).
/// - DO recurse into `Switch` for `Continue` (Switch does not capture it).
fn contains_bare_loop_control(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Break | naga::Statement::Continue => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_loop_control(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_loop_control(accept) || contains_bare_loop_control(reject) {
                    return true;
                }
            }
            naga::Statement::Switch { cases, .. } => {
                // Switch captures Break but NOT Continue.  A `continue`
                // inside a switch case still targets the enclosing loop.
                for case in cases {
                    if contains_bare_continue(&case.body) {
                        return true;
                    }
                }
            }
            // Do NOT recurse into Loop - it captures both Break and Continue.
            naga::Statement::Loop { .. } => {}
            // Leaf statements that cannot carry a bare Break / Continue
            // and have no nested blocks.  Enumerated explicitly so a
            // future naga release adding a new block-bearing variant
            // breaks the build here instead of silently bypassing
            // the loop-control detection.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
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
    false
}

/// Returns `true` if `block` contains a bare `Continue` targeting the
/// enclosing loop (not captured by any nested `Loop`).
///
/// Unlike [`contains_bare_loop_control`] this only looks for `Continue` and
/// therefore recurses into `Switch` (which does not capture `Continue`).
fn contains_bare_continue(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Continue => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_continue(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_continue(accept) || contains_bare_continue(reject) {
                    return true;
                }
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    if contains_bare_continue(&case.body) {
                        return true;
                    }
                }
            }
            // Loop captures Continue - do not recurse.
            naga::Statement::Loop { .. } => {}
            // Leaf statements that cannot carry a bare Continue and
            // have no nested blocks.  Enumerated explicitly so a
            // future naga release adding a new block-bearing variant
            // breaks the build here instead of silently missing a
            // Continue that targets the enclosing loop.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
            | naga::Statement::Break
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
    false
}

/// Returns `true` if `block` contains a bare `Break` targeting the
/// immediately enclosing `Switch` or `Loop`.
///
/// Used to guard switch-case splicing: after removing the switch wrapper a
/// bare `Break` would mis-target the next enclosing construct.
fn contains_bare_break(block: &naga::Block) -> bool {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Break => return true,
            naga::Statement::Block(inner) => {
                if contains_bare_break(inner) {
                    return true;
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                if contains_bare_break(accept) || contains_bare_break(reject) {
                    return true;
                }
            }
            // Both Loop and Switch capture Break - do not recurse.
            naga::Statement::Loop { .. } | naga::Statement::Switch { .. } => {}
            // Leaf statements that cannot carry a bare Break and have
            // no nested blocks.  Enumerated explicitly so a future
            // naga release adding a new block-bearing variant breaks
            // the build here instead of silently missing a Break
            // that targets the enclosing Switch / Loop.
            naga::Statement::Emit(_)
            | naga::Statement::Store { .. }
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
    false
}

// Redundant else-store elimination.  naga's WGSL frontend lowers short-circuit
// `&&` into if-else chains:
//
//     var d: bool;                          // zero-initialized to false
//     if (a) { d = b; } else { d = false; }
//     if (d) { d = c; } else { d = false; }
//
// The else branches store the same value the variable already holds:
//   - Pattern A: condition is `load(d)` -> d must be false in the else branch,
//     so `d = false` is always a no-op.
//   - Pattern B: variable was zero-initialized (init: None) and has not been
//     modified, so `d = false` is a no-op.
//
// Similarly, `||` is lowered as (note the negated condition):
//
//     var d: bool;                          // zero-initialized to false
//     if (!a) { d = b; } else { d = true; }
//     if (!d) { d = c; } else { d = true; }
//
// The else-branches store `true` to a variable already known to be `true`:
//   - Pattern A': condition is `Load(d)` -> d must be true in the accept
//     branch, so `if (d) { d = true; }` is always a no-op.
//   - Pattern A'': condition is `!Load(d)` -> d must be true in the reject
//     branch, so `else { d = true; }` is always a no-op.  (This is the
//     form naga's WGSL frontend actually emits for `||`.)
//
// This pass detects and removes such redundant branches.

/// A value that a local variable is known to hold at a given program point.
#[derive(Clone, Debug, PartialEq)]
enum KnownValue {
    /// The type's zero/default value (matches any zero literal or `ZeroValue`).
    Zero,
    /// A specific literal value.
    Literal(naga::Literal),
}

// MARK: Redundant else-store elimination

/// Build a map from module-level `Constant` handles to their
/// `Literal` values.  Only constants whose init expression is
/// already a `Literal` are included; compound-expression inits fall
/// back to a runtime lookup.  The cache is populated once per run
/// so the else-store phase can read constant values without
/// re-borrowing `module.constants` in the inner loop.
fn build_const_literal_cache(
    module: &naga::Module,
) -> HashMap<naga::Handle<naga::Constant>, naga::Literal> {
    module
        .constants
        .iter()
        .filter_map(|(ch, c)| {
            if let naga::Expression::Literal(lit) = module.global_expressions[c.init] {
                Some((ch, lit))
            } else {
                None
            }
        })
        .collect()
}

/// Initialise known values for local variables.  Variables declared without
/// an explicit initializer are zero-initialised by WGSL, so they start as
/// `KnownValue::Zero`.  Variables with a literal init get
/// `KnownValue::Literal`.
fn init_known_values(
    locals: &naga::Arena<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> HashMap<naga::Handle<naga::LocalVariable>, KnownValue> {
    locals
        .iter()
        .filter_map(|(lh, lv)| match lv.init {
            None => Some((lh, KnownValue::Zero)),
            Some(init_h) => {
                let lit = resolve_to_literal(expressions, init_h, const_lits)?;
                if is_zero_literal(&lit) {
                    Some((lh, KnownValue::Zero))
                } else {
                    Some((lh, KnownValue::Literal(lit)))
                }
            }
        })
        .collect()
}

/// Entry point: run redundant-store elimination on a single function.
/// Seed the known-values scoped map with every uninitialised local
/// that has an obvious default, then recurse into the function body.
/// Returns the change count so the caller can aggregate.
fn eliminate_redundant_else_stores_in_function(
    function: &mut naga::Function,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> usize {
    let mut known_values = ScopedMap::new();
    for (lh, kv) in init_known_values(&function.local_variables, &function.expressions, const_lits)
    {
        known_values.insert(lh, kv);
    }
    // Tracks, per local, the handle of its most recently materialised
    // `Load(LocalVariable)` that is still *fresh* - i.e. no store to that
    // local has happened since.  Condition narrowing is only sound when the
    // condition's `Load` is fresh; a stale forwarded load (e.g. `let t = d;
    // d = false; if t {...}`) reflects the value BEFORE the store, so
    // narrowing on it would clobber the correct post-store known value and
    // drop a live branch.  See `condition_load_is_fresh`.
    let mut fresh_loads = HashMap::new();
    eliminate_redundant_else_stores(
        &mut function.body,
        &function.expressions,
        const_lits,
        &mut known_values,
        &mut fresh_loads,
    )
}

/// `true` when `condition`'s underlying `Load(LocalVariable)` is still the
/// freshest load of that local recorded in `fresh_loads` - meaning no store
/// to the local has intervened since the load was materialised, so the
/// loaded value equals the local's current value.  Covers the bare
/// `Load(d)` (Pattern A) and `!Load(d)` (Pattern A'') condition shapes that
/// [`narrow_for_accept`] / [`narrow_for_reject`] act on; returns `false` for
/// any other shape (which those helpers ignore anyway).
fn condition_load_is_fresh(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    fresh_loads: &HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::Expression>>,
) -> bool {
    match &expressions[*condition] {
        naga::Expression::Load { pointer } => {
            if let naga::Expression::LocalVariable(d) = expressions[*pointer] {
                fresh_loads.get(&d) == Some(condition)
            } else {
                false
            }
        }
        naga::Expression::Unary {
            op: naga::UnaryOperator::LogicalNot,
            expr: inner,
        } => {
            if let naga::Expression::Load { pointer } = &expressions[*inner]
                && let naga::Expression::LocalVariable(d) = expressions[*pointer]
            {
                fresh_loads.get(&d) == Some(inner)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Recursively walk a block, tracking known values of locals, and clear
/// branches that only store a value the variable already holds.
fn eliminate_redundant_else_stores(
    block: &mut naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    fresh_loads: &mut HashMap<naga::Handle<naga::LocalVariable>, naga::Handle<naga::Expression>>,
) -> usize {
    let mut changed = 0usize;

    for stmt in block.iter_mut() {
        match stmt {
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                // Per-branch scoped undo-log walk:
                //   1. Snapshot pre-if state.
                //   2. Apply accept-narrowing, recurse, roll back the
                //      recursion's mutations to the post-narrowing
                //      state, then run the redundancy check.
                //   3. Roll back to pre-if state, repeat for reject.
                //   4. Roll back fully, then permanently drop any
                //      locals modified in either branch (logged for an
                //      outer scope's rollback).
                // Condition narrowing is sound only when the condition's
                // `Load` reflects the local's CURRENT value (no store since
                // the load was materialised).  Decide this once, before
                // recursing, so an accept-branch store cannot perturb the
                // reject-phase decision.
                let cond_fresh = condition_load_is_fresh(condition, expressions, fresh_loads);
                let cp_pre_if = known_values.checkpoint();

                // Accept phase
                if cond_fresh {
                    narrow_for_accept(condition, expressions, known_values);
                }
                let cp_accept_entry = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    accept,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_accept_entry);
                let accept_redundant = !accept.is_empty()
                    && block_only_has_redundant_known_stores(
                        accept,
                        expressions,
                        known_values.as_map(),
                        const_lits,
                    );
                known_values.rollback_to(cp_pre_if);

                // Reject phase
                if cond_fresh {
                    narrow_for_reject(condition, expressions, known_values);
                }
                let cp_reject_entry = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    reject,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_reject_entry);
                let reject_redundant = !reject.is_empty()
                    && block_only_has_redundant_known_stores(
                        reject,
                        expressions,
                        known_values.as_map(),
                        const_lits,
                    );
                known_values.rollback_to(cp_pre_if);

                if reject_redundant {
                    *reject = naga::Block::new();
                    changed += 1;
                }
                if accept_redundant {
                    *accept = naga::Block::new();
                    changed += 1;
                }

                // Permanent update: conservatively remove any locals
                // modified in either branch.  Logged so outer scopes can
                // roll back if needed.  A branch may conditionally store a
                // local, so any prior load of it is no longer fresh.
                let mut modified = HashSet::new();
                collect_modified_locals(accept, expressions, &mut modified);
                collect_modified_locals(reject, expressions, &mut modified);
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::Emit(range) => {
                // Materialising a `Load(LocalVariable)` records it as the
                // freshest load of that local; the entry is invalidated by
                // any subsequent store to the local (below and in the branch
                // / call / atomic arms).  This is what lets condition
                // narrowing tell a fresh `if d {...}` from a stale forwarded
                // `let t = d; d = ...; if t {...}`.
                for h in range.clone() {
                    if let naga::Expression::Load { pointer } = expressions[h]
                        && let naga::Expression::LocalVariable(d) = expressions[pointer]
                    {
                        fresh_loads.insert(d, h);
                    }
                }
            }

            naga::Statement::Store { pointer, value } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    // The store changes the local, so any earlier load of it
                    // is no longer fresh for narrowing.
                    fresh_loads.remove(&lh);
                    if let Some(lit) = resolve_to_literal(expressions, *value, const_lits) {
                        if is_zero_literal(&lit) {
                            known_values.insert(lh, KnownValue::Zero);
                        } else {
                            known_values.insert(lh, KnownValue::Literal(lit));
                        }
                    } else if is_zero_value(expressions, *value, const_lits) {
                        known_values.insert(lh, KnownValue::Zero);
                    } else {
                        known_values.remove(&lh);
                    }
                } else if let Some(lh) = get_stored_local(expressions, *pointer) {
                    // Partial store - conservatively remove.
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::Switch { cases, .. } => {
                let cp = known_values.checkpoint();
                for case in cases.iter_mut() {
                    changed += eliminate_redundant_else_stores(
                        &mut case.body,
                        expressions,
                        const_lits,
                        known_values,
                        fresh_loads,
                    );
                    known_values.rollback_to(cp);
                }
                let mut modified = HashSet::new();
                for case in cases.iter() {
                    collect_modified_locals(&case.body, expressions, &mut modified);
                }
                for lh in modified {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Variables modified inside the loop may not hold the same
                // value on subsequent iterations - strip them from the
                // known-value set before recursing.  The removals are
                // permanent (persisted past the loop); the body's interior
                // mutations are rolled back after the loop body is done.
                let mut modified = HashSet::new();
                collect_modified_locals(body, expressions, &mut modified);
                collect_modified_locals(continuing, expressions, &mut modified);

                for lh in &modified {
                    known_values.remove(lh);
                    fresh_loads.remove(lh);
                }
                let cp_loop = known_values.checkpoint();
                changed += eliminate_redundant_else_stores(
                    body,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                changed += eliminate_redundant_else_stores(
                    continuing,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
                known_values.rollback_to(cp_loop);
            }

            naga::Statement::Block(inner) => {
                changed += eliminate_redundant_else_stores(
                    inner,
                    expressions,
                    const_lits,
                    known_values,
                    fresh_loads,
                );
            }

            naga::Statement::Call { arguments, .. } => {
                // Pointer arguments may be written through by the callee.
                for &arg in arguments.iter() {
                    if let Some(lh) = get_stored_local(expressions, arg) {
                        known_values.remove(&lh);
                        fresh_loads.remove(&lh);
                    }
                }
            }

            naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = get_stored_local(expressions, *pointer) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::RayQuery { query, .. } => {
                if let Some(lh) = get_stored_local(expressions, *query) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::RayPipelineFunction(fun) => {
                let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
                if let Some(lh) = get_stored_local(expressions, *payload) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            naga::Statement::CooperativeStore { data, .. } => {
                // `CooperativeStore { target, data }`: `data.pointer` is
                // the validator-required STORE-space write destination;
                // `target` is the matrix value being read.  Invalidate
                // the local rooted at the write side.
                if let Some(lh) = get_stored_local(expressions, data.pointer) {
                    known_values.remove(&lh);
                    fresh_loads.remove(&lh);
                }
            }

            // Statements that neither modify known-value tracking nor
            // contain nested blocks - enumerated explicitly so a
            // future naga release adding a new pointer-bearing variant
            // breaks the build here instead of silently leaving a
            // known_values entry stale across the new statement type.
            // (`Emit` has its own arm above: it materialises loads that
            // feed condition narrowing.)
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

    changed
}

/// Apply "accept branch" condition-derived narrowing to `known_values`.
///
/// Pattern A: condition is `Load(cond_local)` -> in the accept branch,
/// `cond_local` must be `true`.
/// Pattern A'': condition is `!Load(cond_local)` -> in the accept branch,
/// `cond_local` must be zero/false.
///
/// All mutations go through `ScopedMap::insert` so the narrowing can be
/// rolled back via a caller-held checkpoint.
fn narrow_for_accept(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
) {
    if let naga::Expression::Load { pointer } = &expressions[*condition]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Literal(naga::Literal::Bool(true)));
    }
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[*condition]
        && let naga::Expression::Load { pointer } = &expressions[*inner]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Zero);
    }
}

/// Apply "reject branch" condition-derived narrowing.  Mirror image of
/// [`narrow_for_accept`].
fn narrow_for_reject(
    condition: &naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &mut ScopedMap<naga::Handle<naga::LocalVariable>, KnownValue>,
) {
    if let naga::Expression::Load { pointer } = &expressions[*condition]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Zero);
    }
    if let naga::Expression::Unary {
        op: naga::UnaryOperator::LogicalNot,
        expr: inner,
    } = &expressions[*condition]
        && let naga::Expression::Load { pointer } = &expressions[*inner]
        && let naga::Expression::LocalVariable(cond_local) = expressions[*pointer]
    {
        known_values.insert(cond_local, KnownValue::Literal(naga::Literal::Bool(true)));
    }
}

/// Return `true` when every statement in `block` is either an `Emit` (no
/// side-effect) or a `Store` whose value matches the known value of the
/// target local.  At least one such `Store` must be present.
fn block_only_has_redundant_known_stores(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    known_values: &HashMap<naga::Handle<naga::LocalVariable>, KnownValue>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> bool {
    let mut has_store = false;
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(_) => continue,
            naga::Statement::Store { pointer, value } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer]
                    && let Some(known) = known_values.get(&lh)
                    && expr_matches_known(expressions, *value, known, const_lits)
                {
                    has_store = true;
                    continue;
                }
                return false;
            }
            _ => return false,
        }
    }
    has_store
}

/// Check whether an expression's value matches a `KnownValue`.
fn expr_matches_known(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    known: &KnownValue,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> bool {
    match known {
        KnownValue::Zero => is_zero_value(expressions, handle, const_lits),
        KnownValue::Literal(lit) => resolve_to_literal(expressions, handle, const_lits)
            .is_some_and(|resolved| literal_bit_eq(&resolved, lit)),
    }
}

/// Bit-exact literal equality.  Float variants (`F16`/`F32`/`F64`/
/// `AbstractFloat`) compare bit patterns, so `+0.0` and `-0.0` - IEEE-equal
/// but distinct bits - are NOT conflated, and two identical NaN payloads
/// ARE.  That is exactly the right test for "does storing `a` actually
/// change a value already known to be `b`": a store is redundant only when
/// it writes the identical bit pattern.  naga's derived `PartialEq` uses
/// IEEE `==` (where `+0.0 == -0.0`), which would let the redundant-store
/// elimination drop a sign-flipping store - a silent miscompile for
/// sign-of-zero-sensitive ops (`1.0/x`, `sign`, `copysign`, bit reinterpret).
/// Int/bool variants are already bit-exact under `==`, and mismatched
/// variants are correctly unequal.
fn literal_bit_eq(a: &naga::Literal, b: &naga::Literal) -> bool {
    use naga::Literal as L;
    match (a, b) {
        (L::F16(x), L::F16(y)) => x.to_bits() == y.to_bits(),
        (L::F32(x), L::F32(y)) => x.to_bits() == y.to_bits(),
        (L::F64(x), L::F64(y)) => x.to_bits() == y.to_bits(),
        (L::AbstractFloat(x), L::AbstractFloat(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Resolve an expression to a concrete `naga::Literal`, or `None`
/// when the value isn't known at compile time.
///
/// `Override` is omitted on purpose: its init is only a *default*
/// that the pipeline can replace at draw time, so folding through it
/// would let dead_branch erase code based on a value that changes
/// post-compile.  `Constant` resolves through `const_lits`, which
/// already filters out abstract literals upstream.
fn resolve_to_literal(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> Option<naga::Literal> {
    match &expressions[handle] {
        naga::Expression::Literal(lit) => Some(*lit),
        naga::Expression::Constant(c) => const_lits.get(c).copied(),
        _ => None,
    }
}

/// Check whether an expression evaluates to zero / false.
fn is_zero_value(
    expressions: &naga::Arena<naga::Expression>,
    handle: naga::Handle<naga::Expression>,
    const_lits: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
) -> bool {
    match &expressions[handle] {
        naga::Expression::Literal(lit) => is_zero_literal(lit),
        naga::Expression::ZeroValue(_) => true,
        naga::Expression::Constant(c) => const_lits.get(c).is_some_and(is_zero_literal),
        _ => false,
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(source: &str) -> (bool, naga::Module) {
        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let mut pass = DeadBranchPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("pass should run");

        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("module should remain valid after pass");

        (changed, module)
    }

    /// Count If statements recursively in a block.
    fn count_ifs(block: &naga::Block) -> usize {
        let mut n = 0;
        for stmt in block.iter() {
            match stmt {
                naga::Statement::If { accept, reject, .. } => {
                    n += 1;
                    n += count_ifs(accept);
                    n += count_ifs(reject);
                }
                naga::Statement::Block(inner) => n += count_ifs(inner),
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        n += count_ifs(&case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    n += count_ifs(body);
                    n += count_ifs(continuing);
                }
                _ => {}
            }
        }
        n
    }

    /// Count Switch statements recursively in a block.
    fn count_switches(block: &naga::Block) -> usize {
        let mut n = 0;
        for stmt in block.iter() {
            match stmt {
                naga::Statement::Switch { cases, .. } => {
                    n += 1;
                    for case in cases {
                        n += count_switches(&case.body);
                    }
                }
                naga::Statement::If { accept, reject, .. } => {
                    n += count_switches(accept);
                    n += count_switches(reject);
                }
                naga::Statement::Block(inner) => n += count_switches(inner),
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    n += count_switches(body);
                    n += count_switches(continuing);
                }
                _ => {}
            }
        }
        n
    }

    // If: condition folded to true

    /// Collapsing a constant-selector `switch` whose DROPPED case produced a
    /// statement result (a non-inlined call) would orphan that result
    /// expression - invalid IR.  The guard keeps the switch intact instead.
    /// `run_pass` validates the post-pass module and panics on invalid IR, so
    /// this test fails if the guard is removed.
    #[test]
    fn constant_switch_with_result_producing_dropped_case_stays_valid() {
        let src = r#"
fn helper() -> i32 { return 7; }
@compute @workgroup_size(1)
fn main() {
    var x: i32 = 0;
    switch 1 {
        case 0: { x = helper(); }
        default: { x = 5; }
    }
    _ = x;
}
"#;
        // Must not panic: the pass leaves valid IR (switch kept, result
        // expression keeps its producer).
        let (_changed, module) = run_pass(src);
        // The switch survives (was NOT collapsed) because case 0 carries a
        // call result the collapse cannot safely drop.
        let main = module
            .entry_points
            .iter()
            .find(|e| e.name == "main")
            .expect("main entry point");
        fn has_switch(block: &naga::Block) -> bool {
            block.iter().any(|s| match s {
                naga::Statement::Switch { .. } => true,
                naga::Statement::Block(b) => has_switch(b),
                naga::Statement::If { accept, reject, .. } => {
                    has_switch(accept) || has_switch(reject)
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => has_switch(body) || has_switch(continuing),
                _ => false,
            })
        }
        assert!(
            has_switch(&main.function.body),
            "switch with a result-producing dropped case must be kept intact"
        );
    }

    #[test]
    fn eliminates_if_true_accept_branch() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should report a change");
        let body = &module.entry_points[0].function.body;
        assert_eq!(count_ifs(body), 0, "if should be eliminated");
    }

    // If: condition folded to false

    #[test]
    fn eliminates_if_false_reject_branch() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if false { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should report a change");
        let body = &module.entry_points[0].function.body;
        assert_eq!(count_ifs(body), 0, "if should be eliminated");
    }

    // If: true with no else (empty reject)

    #[test]
    fn eliminates_if_true_no_else() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true { x = 1.0; }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(count_ifs(&module.entry_points[0].function.body), 0);
    }

    // If: false with no else -> entire if dropped

    #[test]
    fn eliminates_if_false_no_else() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if false { x = 1.0; }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(count_ifs(&module.entry_points[0].function.body), 0);
    }

    // Nested dead branches

    #[test]
    fn eliminates_nested_dead_branches() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    if true {
        if false { x = 1.0; } else { x = 2.0; }
    } else {
        x = 3.0;
    }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(
            count_ifs(&module.entry_points[0].function.body),
            0,
            "both nested ifs should be eliminated"
        );
    }

    // Non-constant condition is preserved

    #[test]
    fn preserves_non_constant_if() {
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "non-constant condition should not be eliminated");
    }

    // Switch with constant selector

    #[test]
    fn eliminates_switch_with_constant_selector() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 1i {
        case 1i { x = 1.0; }
        case 2i { x = 2.0; }
        default { x = 3.0; }
    }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(count_switches(&module.entry_points[0].function.body), 0);
    }

    // Switch: no match falls to default

    #[test]
    fn switch_constant_falls_to_default() {
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 99i {
        case 1i { x = 1.0; }
        default { x = 3.0; }
    }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(count_switches(&module.entry_points[0].function.body), 0);
    }

    // Pass reports no change when nothing to do

    #[test]
    fn reports_no_change_when_nothing_to_eliminate() {
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; }
    return vec4f(x);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed);
    }

    // The constant-condition branch arm must also resolve
    // `Expression::Constant(c)` whose init is a bool literal, not just
    // raw `Literal::Bool` - otherwise a branch on a named `const X:
    // bool = false;` slips past until const_fold has inlined it.
    #[test]
    fn eliminates_branch_with_const_bool_condition() {
        let src = r#"
const ENABLE_FEATURE: bool = false;

@fragment
fn fs() -> @location(0) vec4f {
    if ENABLE_FEATURE {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
    return vec4f(0.0);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "branch on const false-bool should be statically eliminated"
        );
        // The entry point's body should have no If statements; the
        // dead `if ENABLE_FEATURE { ... }` branch should be removed.
        let ep_body = &module.entry_points[0].function.body;
        assert_eq!(
            count_ifs(ep_body),
            0,
            "if/else on a const-bool selector must be eliminated"
        );
    }

    #[test]
    fn eliminates_switch_with_const_int_selector() {
        let src = r#"
const SELECTOR: i32 = 1;

@fragment
fn fs() -> @location(0) vec4f {
    var v = 0.0;
    switch SELECTOR {
        case 0: { v = 1.0; }
        case 1: { v = 2.0; }
        default: { v = 3.0; }
    }
    return vec4f(v);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "switch on const integer should be statically resolved"
        );
        // Walk the entry point looking for any Switch statement; there
        // shouldn't be one (the matching case has been spliced in).
        fn has_switch(block: &naga::Block) -> bool {
            block.iter().any(|s| match s {
                naga::Statement::Switch { .. } => true,
                naga::Statement::Block(b) => has_switch(b),
                naga::Statement::If { accept, reject, .. } => {
                    has_switch(accept) || has_switch(reject)
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => has_switch(body) || has_switch(continuing),
                _ => false,
            })
        }
        assert!(
            !has_switch(&module.entry_points[0].function.body),
            "switch on const-int selector must be resolved away"
        );
    }

    // Function (not just entry point)

    #[test]
    fn eliminates_dead_branch_in_regular_function() {
        let src = r#"
fn helper() -> f32 {
    if true { return 1.0; } else { return 2.0; }
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(helper());
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        // The helper function's body should have no If statements.
        for (_, func) in module.functions.iter() {
            assert_eq!(count_ifs(&func.body), 0);
        }
    }

    // contains_bare_break / contains_bare_continue helpers

    #[test]
    fn bare_break_detected_at_top_level() {
        let mut block = naga::Block::new();
        block.push(naga::Statement::Break, Default::default());
        assert!(contains_bare_break(&block));
    }

    #[test]
    fn bare_break_detected_inside_if_block() {
        // Parse a shader that has a break inside an if inside a loop,
        // then inspect the loop body.
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    loop {
        if x > 1.0 { break; }
        x = x + 1.0;
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let loop_body = module.entry_points[0]
            .function
            .body
            .iter()
            .find_map(|s| {
                if let naga::Statement::Loop { body, .. } = s {
                    Some(body)
                } else {
                    None
                }
            })
            .unwrap();
        assert!(contains_bare_break(loop_body));
    }

    #[test]
    fn bare_break_not_detected_inside_nested_switch() {
        // A break inside a nested switch targets that switch, not the parent.
        let src = r#"
@fragment
fn fs(@location(0) v: i32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        switch v {
            case 1i {
                x = 1.0;
                break;
            }
            default {
                x = 2.0;
            }
        }
        continuing { break if x > 0.0; }
    }
    return vec4f(x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).unwrap();
        // The loop body contains a switch with a break - but that break
        // targets the switch, NOT the loop.
        let loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
            if let naga::Statement::Loop { body, .. } = s {
                Some(body)
            } else {
                None
            }
        });
        assert!(loop_body.is_some());
        // contains_bare_break should NOT see through the nested switch
        assert!(!contains_bare_break(loop_body.unwrap()));
    }

    #[test]
    fn bare_continue_detected_inside_switch() {
        // In naga IR, Switch does NOT capture Continue - it still targets
        // the enclosing loop.  contains_bare_continue must find it.
        let src = r#"
@fragment
fn fs(@location(0) v: i32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        switch v {
            case 1i { continue; }
            default { x = 2.0; }
        }
        x = x + 1.0;
        continuing { break if x > 5.0; }
    }
    return vec4f(x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
            if let naga::Statement::Loop { body, .. } = s {
                Some(body)
            } else {
                None
            }
        });
        assert!(loop_body.is_some());
        assert!(
            contains_bare_continue(loop_body.unwrap()),
            "continue inside switch should be detected by contains_bare_continue"
        );
        // contains_bare_loop_control should also detect it
        assert!(
            contains_bare_loop_control(loop_body.unwrap()),
            "continue inside switch should be detected by contains_bare_loop_control"
        );
    }

    #[test]
    fn bare_continue_not_detected_inside_nested_loop() {
        // Continue inside a nested loop targets that inner loop.
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    loop {
        loop {
            if x > 1.0 { continue; }
            x = x + 1.0;
            continuing { break if x > 3.0; }
        }
        x = x + 1.0;
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let outer_loop_body = module.entry_points[0].function.body.iter().find_map(|s| {
            if let naga::Statement::Loop { body, .. } = s {
                Some(body)
            } else {
                None
            }
        });
        assert!(outer_loop_body.is_some());
        // The outer loop body contains a nested loop with continue.
        // contains_bare_continue should NOT see through the nested loop.
        assert!(
            !contains_bare_continue(outer_loop_body.unwrap()),
            "continue inside nested loop should NOT be detected"
        );
    }

    /// Count non-empty reject blocks in If statements (recursive).
    fn count_non_empty_rejects(block: &naga::Block) -> usize {
        let mut n = 0;
        for stmt in block.iter() {
            match stmt {
                naga::Statement::If { accept, reject, .. } => {
                    if !reject.is_empty() {
                        n += 1;
                    }
                    n += count_non_empty_rejects(accept);
                    n += count_non_empty_rejects(reject);
                }
                naga::Statement::Block(inner) => n += count_non_empty_rejects(inner),
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        n += count_non_empty_rejects(&case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    n += count_non_empty_rejects(body);
                    n += count_non_empty_rejects(continuing);
                }
                _ => {}
            }
        }
        n
    }

    /// Count If statements with non-empty accept blocks recursively.
    fn count_non_empty_accepts(block: &naga::Block) -> usize {
        let mut n = 0;
        for stmt in block.iter() {
            match stmt {
                naga::Statement::If { accept, reject, .. } => {
                    if !accept.is_empty() {
                        n += 1;
                    }
                    n += count_non_empty_accepts(accept);
                    n += count_non_empty_accepts(reject);
                }
                naga::Statement::Block(inner) => n += count_non_empty_accepts(inner),
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        n += count_non_empty_accepts(&case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    n += count_non_empty_accepts(body);
                    n += count_non_empty_accepts(continuing);
                }
                _ => {}
            }
        }
        n
    }

    // Pattern A: condition is Load(d), so d is false in the reject branch.
    // Storing `false` to d in the else is a no-op.

    #[test]
    fn redundant_else_pattern_a_load_condition() {
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should report a change");
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_non_empty_rejects(&func.body),
                0,
                "all else {{ d = false; }} branches should be cleared"
            );
        }
    }

    // Pattern B: var d: bool (zero-init), first if's else stores false
    // before any modification.

    #[test]
    fn redundant_else_pattern_b_zero_init() {
        let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(count_non_empty_rejects(&func.body), 0);
        }
    }

    // Negative: else stores a non-zero value - must be preserved.

    #[test]
    fn preserves_else_storing_non_zero() {
        let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = false; } else { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, module) = run_pass(src);
        // The accept stores `false` to d - redundant (d is zero-init) ->
        // accept cleared.  The else stores `true` -> NOT redundant -> kept.
        assert!(
            changed,
            "accept zero-store to zero-init var should be cleared"
        );
        for (_, func) in module.functions.iter() {
            // The if should still exist (reject is non-empty).
            assert!(
                count_ifs(&func.body) >= 1,
                "if should be preserved because reject stores non-zero"
            );
        }
    }

    // Negative: variable is not known-zero (has non-zero init), so
    // else { d = 0 } is meaningful.

    #[test]
    fn preserves_else_when_var_not_known_zero() {
        let src = r#"
fn f(a: bool) -> f32 {
    var x: f32 = 1.0;
    if a { x = 2.0; } else { x = 0.0; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
        let (changed, _module) = run_pass(src);
        assert!(
            !changed,
            "else {{ x = 0.0 }} is not redundant when x was 1.0"
        );
    }

    // Short-circuit desugaring subsumes the "prior store" edge-case:
    // both ifs match the && pattern and are replaced with Binary stores.

    #[test]
    fn short_circuit_desugars_despite_prior_store() {
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    d = true;
    if a { d = b; } else { d = false; }
    if d { d = true; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        // Phase 0 desugars both ifs into d = a&&b and d = d&&true.
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_ifs(&func.body),
                0,
                "both ifs should be desugared by short-circuit pass"
            );
        }
    }

    // Chained `&&` pattern: multiple if-else in sequence.

    #[test]
    fn redundant_else_chained_and() {
        let src = r#"
fn f(a: bool, b: bool, c: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = c; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_non_empty_rejects(&func.body),
                0,
                "all 2 else branches should be cleared"
            );
        }
    }

    // Loop: short-circuit desugaring still applies inside loops
    // (if cond { d = val; } else { d = false; } -> d = cond && val
    // is valid regardless of d's prior value).

    #[test]
    fn short_circuit_inside_loop() {
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    loop {
        if a { d = true; } else { d = false; }
        continuing { break if b; }
    }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        // Phase 0 desugars the if inside the loop to d = a && true.
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_non_empty_rejects(&func.body),
                0,
                "short-circuit desugaring should remove the if"
            );
        }
    }

    // Short-circuit ||: if(!cond) { d = val; } else { d = true; }

    #[test]
    fn short_circuit_basic_or_replacement() {
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if !a { d = b; } else { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_ifs(&func.body),
                0,
                "if should be desugared to d = a || b"
            );
        }
    }

    // Negative: non-boolean types must NOT be matched.

    #[test]
    fn short_circuit_preserves_non_bool_if_else() {
        // f32 stores: reject stores 0.0, but LogicalAnd only works on bool.
        let src = r#"
fn f(a: bool) -> f32 {
    var d: f32;
    if a { d = 1.0; } else { d = 0.0; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
        let (_, module) = run_pass(src);
        for (_, func) in module.functions.iter() {
            // The if must be preserved (not a bool short-circuit pattern).
            assert!(
                count_ifs(&func.body) >= 1,
                "non-bool if-else must not be desugared"
            );
        }
    }

    // Negative: reject stores non-false value -> not a short-circuit.

    #[test]
    fn short_circuit_preserves_reject_storing_non_false() {
        let src = r#"
fn f(a: bool, b: bool, c: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = c; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true, false))); }
"#;
        let (_, module) = run_pass(src);
        for (_, func) in module.functions.iter() {
            assert!(
                count_ifs(&func.body) >= 1,
                "reject stores c (not false), must not desugar"
            );
        }
    }

    // No change when there is nothing to eliminate.

    #[test]
    fn redundant_else_no_change_when_nothing_to_do() {
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "no redundant else stores present");
    }

    // Entry point (not just regular function).

    #[test]
    fn redundant_else_in_entry_point() {
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var d: bool;
    if v > 0.5 { d = true; } else { d = false; }
    return vec4f(f32(d));
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        assert_eq!(
            count_non_empty_rejects(&module.entry_points[0].function.body),
            0,
            "else in entry point should be cleared"
        );
    }

    // OR pattern (accept side): condition Load(d) -> d is true in accept.
    // `if d { d = true; } else { d = b; }` - the accept `d = true` is
    // redundant because d is already true.

    #[test]
    fn redundant_accept_true_store_when_condition_is_load() {
        // Manually write the pattern: `if d { d = true; } else { d = b; }`
        // The accept branch stores `true` to a variable that the condition
        // already proves is `true`.
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = true; } else { d = b; }
    if d { d = true; } else { d = b; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should clear redundant accept `d = true`");
        for (_, func) in module.functions.iter() {
            // The first if: no pattern fires (a is a function arg, not a local).
            // The second if: condition = Load(d), accept = {d = true} -> redundant.
            // After clearing, the second if should have an empty accept.
            let non_empty_accepts = count_non_empty_accepts(&func.body);
            assert!(
                non_empty_accepts <= 1,
                "second if's accept `d = true` should be cleared, got {non_empty_accepts} non-empty accepts"
            );
        }
    }

    // Regression: condition narrowing must NOT fire on a STALE forwarded
    // load.  `let t = d; d = false; if t { d = true; }` - the condition `t`
    // captured d's value BEFORE the `d = false` store, so the loaded value
    // (true) does not reflect d's current value (false).  Narrowing would
    // clobber the correct post-store known value and wrongly classify the
    // live `d = true` store as redundant, dropping it and returning false
    // instead of true.  The fresh-load tracking must keep the branch.
    #[test]
    fn narrowing_skips_stale_forwarded_load() {
        let src = r#"
fn f() -> bool {
    var d: bool = true;
    let t = d;
    d = false;
    if t { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f())); }
"#;
        let (_, module) = run_pass(src);
        // The `if t { d = true; }` accept must survive: the store is live.
        let total_non_empty_accepts: usize = module
            .functions
            .iter()
            .map(|(_, func)| count_non_empty_accepts(&func.body))
            .sum();
        assert!(
            total_non_empty_accepts >= 1,
            "live `d = true` store behind a stale-load condition must be preserved"
        );
    }

    // Positive companion: when the SAME local is re-loaded fresh for the
    // condition (no intervening store), narrowing still fires.  Distinguishes
    // the fix from a blanket disable of load-condition narrowing.
    #[test]
    fn narrowing_fires_on_fresh_load_after_store() {
        let src = r#"
fn f(a: bool) -> bool {
    var d: bool = true;
    if a { d = false; } else { d = true; }
    if d { d = true; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, _) = run_pass(src);
        assert!(
            changed,
            "fresh `if d {{ d = true; }}` after a branch write should still be eliminated"
        );
    }

    #[test]
    fn short_circuit_chained_and_with_non_literal_value() {
        // Both ifs match the && pattern (reject stores false to same local).
        // Phase 0 desugars them into Binary(LogicalAnd) stores.
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    if d { d = b; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_ifs(&func.body),
                0,
                "both ifs should be desugared by short-circuit pass"
            );
        }
    }

    // Empty construct elimination: if both branches are empty after
    // recursion, the whole If is dropped.

    #[test]
    fn eliminates_empty_if_after_both_branches_cleared() {
        // Both branches store zero to a zero-initialized local -> both get
        // cleared by the redundant-else-store pass.  The resulting empty If
        // should then be discarded.
        let src = r#"
fn f(a: bool) -> bool {
    var d: bool;
    if a { d = false; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "pass should report a change");
        // After clearing both branches, the if should be gone.
        for (_, func) in module.functions.iter() {
            assert_eq!(count_ifs(&func.body), 0, "empty if should be eliminated");
        }
    }

    #[test]
    fn short_circuit_basic_and_replacement() {
        // The && lowered pattern: if a { d = b; } else { d = false; }
        // should be entirely desugared into d = a && b (no If left).
        let src = r#"
fn f(a: bool, b: bool) -> bool {
    var d: bool;
    if a { d = b; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_ifs(&func.body),
                0,
                "if should be desugared by short-circuit pass"
            );
        }
    }

    #[test]
    fn short_circuit_preserves_branch_with_emit_in_value() {
        // The accept branch stores a computed expression
        // (`array[idx]` requires an `Emit` inside the branch to load
        // the array entry).  Re-sugaring into `d = a && array[idx]`
        // would hoist that load into the outer scope, defeating the
        // short-circuit: when `a` is false the load now runs anyway,
        // observable through implementation-defined out-of-bounds
        // behaviour at runtime.  The guard in `single_store_info`
        // forces the `If` to remain.
        let src = r#"
fn f(a: bool, idx: u32) -> bool {
    let arr = array<bool, 4>(true, false, true, false);
    var d: bool;
    if a { d = arr[idx]; } else { d = false; }
    return d;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true, 0u))); }
"#;
        let (_changed, module) = run_pass(src);
        // The `f` function must still contain at least one `If`
        // statement: the conditional `arr[idx]` evaluation cannot be
        // hoisted safely, so the desugar pattern must not fire.
        let f_function = module
            .functions
            .iter()
            .find(|(_, func)| func.name.as_deref() == Some("f"))
            .map(|(_, func)| func)
            .expect("function `f` should survive the pass");
        assert!(
            count_ifs(&f_function.body) >= 1,
            "If with a non-trivial (Emit-required) store value must NOT be desugared \
             into `&&` because hoisting the Emit would defeat short-circuit semantics"
        );
    }

    // Empty construct elimination for Switch: all cases empty.

    #[test]
    fn eliminates_empty_switch_all_cases_cleared() {
        // Switch with a constant selector resolves to a case with an empty
        // body -> entire switch is removed.  For a non-constant selector,
        // if all cases are empty after recursion, the switch is a no-op.
        let src = r#"
@fragment
fn fs() -> @location(0) vec4f {
    var x = 0.0;
    switch 1i { case 1i: { x = 1.0; } default: {} }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "constant-selector switch should be eliminated");
        assert_eq!(
            count_switches(&module.entry_points[0].function.body),
            0,
            "switch should be eliminated"
        );
    }

    // Degenerate switch: one Default case with runtime selector -> unwrap.

    #[test]
    fn unwraps_degenerate_switch_default_only() {
        // Switch with only a Default case and a runtime selector.
        // The body always executes, so it should be spliced directly.
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { default: { x = 1.0; } }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "degenerate switch should be unwrapped");
        assert_eq!(
            count_switches(&module.entry_points[0].function.body),
            0,
            "switch wrapper should be removed"
        );
    }

    #[test]
    fn preserves_degenerate_switch_with_bare_break() {
        // Default-only switch whose body has a bare Break - unsafe to
        // splice because the Break targets the switch.
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { default: { x = 1.0; break; } }
    return vec4f(x);
}
"#;
        let (_changed, module) = run_pass(src);
        // Should NOT unwrap because of the bare Break.
        assert_eq!(
            count_switches(&module.entry_points[0].function.body),
            1,
            "switch with bare break must be preserved"
        );
    }

    #[test]
    fn preserves_switch_with_multiple_cases() {
        // Switch with multiple cases - not degenerate.
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    switch i32(v) { case 0i: { x = 1.0; } default: { x = 2.0; } }
    return vec4f(x);
}
"#;
        let (_changed, module) = run_pass(src);
        assert_eq!(
            count_switches(&module.entry_points[0].function.body),
            1,
            "multi-case switch must be preserved"
        );
    }

    // MARK: Else block elision (CFG flattening) tests

    #[test]
    fn else_elision_when_accept_returns() {
        // if (cond) { return ...; } else { x = 2.0; }
        // After elision: if (cond) { return ...; }  x = 2.0;
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { return vec4f(1.0); } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "else elision should fire");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_non_empty_rejects(body),
            0,
            "reject block should be empty after elision"
        );
    }

    #[test]
    fn else_elision_when_accept_breaks() {
        // Inside a loop: if (cond) { break; } else { x = 2.0; }
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    loop {
        if v > 0.5 { break; } else { x = 2.0; }
        continuing { break if x > 10.0; }
    }
    return vec4f(x);
}
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "else elision should fire when accept has break");
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_non_empty_rejects(body),
            0,
            "reject block should be empty after elision"
        );
    }

    #[test]
    fn no_else_elision_when_accept_does_not_terminate() {
        // Accept block does NOT definitely terminate.
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { x = 1.0; } else { x = 2.0; }
    return vec4f(x);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "no elision when accept doesn't terminate");
    }

    #[test]
    fn no_else_elision_when_reject_is_empty() {
        // Accept terminates but reject is already empty - nothing to hoist.
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    var x = 0.0;
    if v > 0.5 { return vec4f(1.0); }
    return vec4f(x);
}
"#;
        let (changed, _) = run_pass(src);
        assert!(!changed, "no change when reject is already empty");
    }

    /// Else-elision fires even when both arms terminate with
    /// value-bearing returns - the saved `else{...}` scaffolding
    /// outweighs the residual `if c { return v1; }`.  Gating on
    /// `!reject.terminates` made the corpus strictly larger on real
    /// shaders; the only collapse the elision misses is symmetric
    /// `return;` arms, which we don't fold anyway.
    #[test]
    fn else_elision_fires_when_both_arms_return_values() {
        let src = r#"
@fragment
fn fs(@location(0) v: f32) -> @location(0) vec4f {
    if v > 0.5 { return vec4f(1.0); } else { return vec4f(0.0); }
}
"#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "else-elision should fire even when both arms terminate \
             (value-bearing returns still save the `else{{}}` overhead)"
        );
        let body = &module.entry_points[0].function.body;
        assert_eq!(
            count_non_empty_rejects(body),
            0,
            "reject block must be hoisted out even when it terminates - \
             the resulting `if c {{ return v1; }} return v2;` is shorter \
             than `if c {{ return v1; }} else {{ return v2; }}`"
        );
    }

    /// Nested `Block(empty)` (either authored `{ }` or one drained
    /// by an upstream fold) must be dropped after recursion; otherwise
    /// a vacuous `{}` leaks into the output.
    #[test]
    fn empty_nested_block_is_dropped() {
        // WGSL source has no syntax for a bare nested `{ }` inside
        // a function body, so build the IR directly.
        let mut module = naga::Module::default();
        let f32_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            },
            naga::Span::UNDEFINED,
        );
        let bool_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::BOOL),
            },
            naga::Span::UNDEFINED,
        );
        let _ = f32_ty; // silence unused if naga ever changes the API

        let mut func = naga::Function::default();
        func.arguments.push(naga::FunctionArgument {
            name: Some("c".to_string()),
            ty: bool_ty,
            binding: None,
        });
        let cond = func
            .expressions
            .append(naga::Expression::FunctionArgument(0), naga::Span::UNDEFINED);
        // The accept block holds a single Block(empty) - exactly the
        // shape the elision targets.
        let mut accept = naga::Block::new();
        accept.push(
            naga::Statement::Block(naga::Block::new()),
            naga::Span::UNDEFINED,
        );
        func.body.push(
            naga::Statement::If {
                condition: cond,
                accept,
                reject: naga::Block::new(),
            },
            naga::Span::UNDEFINED,
        );
        module.functions.append(func, naga::Span::UNDEFINED);

        // Pre-condition: the hand-built input must satisfy naga's
        // validator.  Without this, a stricter future validator would
        // silently turn this into an invalid-IR exercise.
        crate::io::validate_module(&module)
            .expect("hand-built input must satisfy naga's validator");

        let mut pass = DeadBranchPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };
        let changed = pass.run(&mut module, &ctx).expect("pass should run");
        assert!(
            changed,
            "empty nested Block elision must report `changed = true`"
        );

        // Post-condition every IR pass must uphold.
        crate::io::validate_module(&module).expect("module must remain valid after the elision");

        // Walk the resulting body and verify no `Block(empty)` remains.
        fn contains_empty_block(block: &naga::Block) -> bool {
            block.iter().any(|stmt| match stmt {
                naga::Statement::Block(inner) if inner.is_empty() => true,
                naga::Statement::Block(inner) => contains_empty_block(inner),
                naga::Statement::If { accept, reject, .. } => {
                    contains_empty_block(accept) || contains_empty_block(reject)
                }
                naga::Statement::Switch { cases, .. } => {
                    cases.iter().any(|c| contains_empty_block(&c.body))
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => contains_empty_block(body) || contains_empty_block(continuing),
                _ => false,
            })
        }

        let (_, f) = module.functions.iter().next().unwrap();
        assert!(
            !contains_empty_block(&f.body),
            "empty Block(_) statements must be elided after the pass"
        );
    }

    // MARK: Generalized redundant store elimination tests

    #[test]
    fn redundant_store_same_literal_i32() {
        // Storing the same i32 literal a variable already holds is a no-op.
        let src = r#"
fn f(a: bool) -> i32 {
    var x: i32 = 42;
    if a { x = 42; } else { x = 42; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, module) = run_pass(src);
        assert!(changed, "redundant i32 stores should be eliminated");
        for (_, func) in module.functions.iter() {
            assert_eq!(
                count_ifs(&func.body),
                0,
                "entire if should be removed after both branches cleared"
            );
        }
    }

    #[test]
    fn redundant_store_after_explicit_store() {
        // After storing a literal, a subsequent if-else storing the same
        // value should be eliminated.
        let src = r#"
fn f(a: bool) -> f32 {
    var x: f32;
    x = 3.0;
    if a { x = 3.0; } else { x = 3.0; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f(true)); }
"#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "redundant stores after explicit store should be eliminated"
        );
        for (_, func) in module.functions.iter() {
            assert_eq!(count_ifs(&func.body), 0);
        }
    }

    #[test]
    fn preserves_store_of_different_literal() {
        // Storing a different literal should be preserved.
        let src = r#"
fn f(a: bool) -> i32 {
    var x: i32 = 42;
    if a { x = 99; } else { x = 42; }
    return x;
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(f32(f(true))); }
"#;
        let (changed, module) = run_pass(src);
        // reject stores 42 which matches init -> cleared.  But accept stores
        // 99 which differs -> preserved.  The if should still exist.
        assert!(changed);
        for (_, func) in module.functions.iter() {
            assert!(
                count_ifs(&func.body) >= 1,
                "if with non-redundant accept store must be preserved"
            );
        }
    }

    // Regression: dead_branch_elimination must not treat a switch where all
    // cases end with `break` as a terminator of the outer block.  Pattern: phi
    // variables are assigned inside the switch cases, then captured with
    // `let ev = phi;` after the switch.  The continuing block stores `phi = ev`
    // (phi-assignment), forcing naga to emit the `ev` expressions in the loop
    // body scope (not lazily in continuing).  Treating the switch as a
    // terminator drops the Emit covering those let-bindings, causing
    // `Expression NotInScope` in the continuing block on naga validation.
    #[test]
    fn switch_with_all_break_cases_does_not_drop_continuing_emits() {
        let src = r#"
@compute @workgroup_size(1) fn main() {
    var phi_a: u32;
    var phi_b: bool;
    loop {
        switch phi_a {
            case 10u: { phi_b = false; break; }
            default: { phi_b = true; break; }
        }
        let ev_b = phi_b;
        let ev_a = phi_a + 1u;
        continue;
        continuing {
            phi_a = ev_a;
            break if !ev_b;
        }
    }
}
"#;
        // run_pass validates after the pass; panics with NotInScope if bug still present.
        let (_, module) = run_pass(src);
        let ep = &module.entry_points[0];
        let has_loop = ep
            .function
            .body
            .iter()
            .any(|s| matches!(s, naga::Statement::Loop { .. }));
        assert!(has_loop, "loop must be preserved");
    }

    // Regression: code after a switch-with-all-break-cases must not be dead-code-removed.
    #[test]
    fn switch_break_cases_not_terminator_of_outer_block() {
        let src = r#"
@compute @workgroup_size(1) fn main() {
    var x: u32 = 0u;
    var y: u32 = 0u;
    switch x {
        case 0u: { y = 1u; break; }
        default: { y = 2u; break; }
    }
    x = y + 1u;
}
"#;
        let (_, module) = run_pass(src);
        let ep = &module.entry_points[0];
        let store_count = ep
            .function
            .body
            .iter()
            .filter(|s| matches!(s, naga::Statement::Store { .. }))
            .count();
        assert!(store_count >= 1, "stores after switch must not be removed");
    }

    // MARK: Switch fall-through edge in definitely_terminates
    //
    // Round-3 review flagged that the `definitely_terminates` fix
    // for the case `cases.last().fall_through == true` had no
    // regression test because the unsafe shape can't be produced via
    // a WGSL source - naga's frontend never emits a final case with
    // `fall_through: true`.  Hand-build the IR directly to assert
    // the fix's behaviour.

    /// Build a single-Default-case `Switch` whose body terminates.
    /// Helper for the fall-through tests below.
    fn build_terminating_default_switch(fall_through: bool) -> naga::Statement {
        let mut body = naga::Block::new();
        body.push(
            naga::Statement::Return { value: None },
            naga::Span::UNDEFINED,
        );
        naga::Statement::Switch {
            // Selector handle is irrelevant - `definitely_terminates`
            // looks only at the case structure.  We construct a fake
            // expression arena just to obtain one Handle.
            selector: {
                let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(0)),
                    naga::Span::UNDEFINED,
                )
            },
            cases: vec![naga::SwitchCase {
                value: naga::SwitchValue::Default,
                body,
                fall_through,
            }],
        }
    }

    #[test]
    fn switch_with_default_terminator_and_no_fallthrough_definitely_terminates() {
        let stmt = build_terminating_default_switch(/*fall_through=*/ false);
        assert!(
            definitely_terminates(&stmt),
            "Default case with terminating body and no fall-through must terminate"
        );
    }

    #[test]
    fn switch_with_last_case_fallthrough_does_not_terminate() {
        // The fall_through-on-last-case shape is the regression
        // target: fall-through past the last case is Break-equivalent
        // (execution resumes after the switch), so the switch as a
        // whole does NOT terminate the outer block.  naga's WGSL
        // frontend won't emit this shape, but the inliner / CSE
        // could, and the previous version of `definitely_terminates`
        // would have mis-classified it as terminating.
        let stmt = build_terminating_default_switch(/*fall_through=*/ true);
        assert!(
            !definitely_terminates(&stmt),
            "last-case fall-through must not classify the switch as terminating \
             (fall-through past the last case is Break-equivalent and execution \
             resumes after the switch)"
        );
    }

    /// `break if NAMED_CONST` must fold the same sweep as `if
    /// NAMED_CONST` / `switch NAMED_CONST` - all three flow through
    /// `resolve_to_literal`.  Pre-fix the break_if arm only matched
    /// raw `Expression::Literal`, so the fold lagged until
    /// `const_fold` inlined the constant on a later sweep.
    #[test]
    fn break_if_with_named_const_true_unwraps_loop() {
        let src = r#"
            const STOP: bool = true;
            fn f() -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    continuing {
                        break if STOP;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "break_if with a named-const-true selector must unwrap the loop"
        );
        // Confirm the loop is gone in the helper function.
        let helper = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .expect("helper function `f` survives the pass");
        fn has_loop(block: &naga::Block) -> bool {
            block.iter().any(|stmt| match stmt {
                naga::Statement::Loop { .. } => true,
                naga::Statement::Block(inner) => has_loop(inner),
                naga::Statement::If { accept, reject, .. } => has_loop(accept) || has_loop(reject),
                naga::Statement::Switch { cases, .. } => cases.iter().any(|c| has_loop(&c.body)),
                _ => false,
            })
        }
        assert!(
            !has_loop(&helper.1.body),
            "loop with `break if true` selector must have been unwrapped"
        );
    }

    /// `break if true` must not unwrap a loop whose body has a bare
    /// `break`/`continue` targeting that loop - unwrapping would
    /// re-target the bare statement at the surrounding scope.  The
    /// `contains_bare_loop_control` guard must also fire on
    /// named-const-true selectors, not just literal `true`.
    #[test]
    fn break_if_with_named_const_true_preserves_loop_with_bare_break() {
        let src = r#"
            const STOP: bool = true;
            fn f(c: bool) -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    if c {
                        break;
                    }
                    continuing {
                        break if STOP;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(true); }
        "#;
        let (_, module) = run_pass(src);
        fn has_loop(block: &naga::Block) -> bool {
            block.iter().any(|stmt| match stmt {
                naga::Statement::Loop { .. } => true,
                naga::Statement::Block(inner) => has_loop(inner),
                naga::Statement::If { accept, reject, .. } => has_loop(accept) || has_loop(reject),
                naga::Statement::Switch { cases, .. } => cases.iter().any(|c| has_loop(&c.body)),
                _ => false,
            })
        }
        let helper = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .expect("helper function `f` survives the pass");
        assert!(
            has_loop(&helper.1.body),
            "loop with bare `break` in body must NOT be unwrapped even when \
             `break if STOP` selects a named-const-true (the bare break would \
             mis-target the surrounding scope after splice)"
        );
    }

    /// Counterpart: a `const NEVER: bool = false;` selector must
    /// rewrite the loop's `break_if` to `None` (the loop is
    /// non-terminating via that path but otherwise preserved).
    #[test]
    fn break_if_with_named_const_false_drops_break_if() {
        let src = r#"
            const NEVER: bool = false;
            fn f() -> i32 {
                var i: i32 = 0;
                loop {
                    i = i + 1;
                    if i > 10 {
                        break;
                    }
                    continuing {
                        break if NEVER;
                    }
                }
                return i;
            }
            @compute @workgroup_size(1) fn main() { _ = f(); }
        "#;
        let (changed, module) = run_pass(src);
        assert!(
            changed,
            "break_if with a named-const-false selector must be dropped"
        );
        // Confirm the surviving loop has no break_if.
        fn first_loop_break_if(
            block: &naga::Block,
        ) -> Option<Option<naga::Handle<naga::Expression>>> {
            for stmt in block.iter() {
                match stmt {
                    naga::Statement::Loop { break_if, .. } => return Some(*break_if),
                    naga::Statement::Block(inner) => {
                        if let Some(bi) = first_loop_break_if(inner) {
                            return Some(bi);
                        }
                    }
                    naga::Statement::If { accept, reject, .. } => {
                        if let Some(bi) = first_loop_break_if(accept) {
                            return Some(bi);
                        }
                        if let Some(bi) = first_loop_break_if(reject) {
                            return Some(bi);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        let helper = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("f"))
            .expect("helper function `f` survives the pass");
        let bi = first_loop_break_if(&helper.1.body)
            .expect("loop should survive (only the break_if is dropped)");
        assert!(
            bi.is_none(),
            "break_if with a named-const-false selector must be rewritten to None"
        );
    }

    /// Same fall-through gate must apply at the NESTED level via
    /// `case_body_terminates_beyond_switch`.  Build an outer switch
    /// whose default case body contains *another* switch whose last
    /// case falls through.  The inner switch falls past its cases
    /// (Break-equivalent), so the OUTER switch's default case body
    /// does NOT terminate beyond the switch - it falls through to
    /// whatever the outer block does after the switch.
    ///
    /// Pre-fix: `case_body_terminates_beyond_switch` missed the
    /// `!last_falls_through` gate and would classify the outer
    /// switch's case body as terminating, letting upstream callers
    /// drop reachable statements after the outer switch.
    #[test]
    fn nested_switch_with_last_case_fallthrough_does_not_terminate_beyond() {
        let inner = build_terminating_default_switch(/*fall_through=*/ true);
        let mut outer_body = naga::Block::new();
        outer_body.push(inner, naga::Span::UNDEFINED);
        // The outer switch wraps the inner switch in its default case.
        // If `case_body_terminates_beyond_switch` correctly applies
        // the fall-through gate to nested switches, the outer switch
        // should NOT be classified as terminating either.
        let outer = naga::Statement::Switch {
            selector: {
                let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
                arena.append(
                    naga::Expression::Literal(naga::Literal::U32(0)),
                    naga::Span::UNDEFINED,
                )
            },
            cases: vec![naga::SwitchCase {
                value: naga::SwitchValue::Default,
                body: outer_body,
                fall_through: false,
            }],
        };
        assert!(
            !definitely_terminates(&outer),
            "nested switch whose inner switch has last-case fall-through must \
             propagate that non-termination through \
             `case_body_terminates_beyond_switch`, NOT classify the outer \
             switch as terminating"
        );
    }
}
