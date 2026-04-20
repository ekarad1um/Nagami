//! Statement-level WGSL emission.
//!
//! Drives the body of every function by walking its
//! [`naga::Block`] in source order and dispatching to the matching
//! emitter.  The file also hosts the "does expression `h` still have
//! a pending use inside this block?" helpers used by the inliner
//! heuristic that decides whether a `Call` result can be folded into
//! its single caller.

use crate::error::Error;

use super::core::{FunctionCtx, Generator};

// MARK: Expression use detection

/// `true` when any statement in `block` still references `target`.
/// Used by the call-inlining heuristic to confirm a `Call`'s result
/// is consumed exactly once and nothing between the `Call` and its
/// use would see a stale value.
fn block_uses_expr(block: &naga::Block, target: naga::Handle<naga::Expression>) -> bool {
    block.iter().any(|s| stmt_uses_expr(s, target))
}

/// Slice variant of [`block_uses_expr`] used when scanning a subset
/// of statements (for example the tail of a block).
fn stmts_use_expr(stmts: &[&naga::Statement], target: naga::Handle<naga::Expression>) -> bool {
    stmts.iter().any(|s| stmt_uses_expr(s, target))
}

/// `true` when `stmt` references `target` in any of its operand
/// positions, recursing into nested blocks.  Exhaustive per statement
/// variant so a new naga statement forces an explicit decision.
fn stmt_uses_expr(stmt: &naga::Statement, target: naga::Handle<naga::Expression>) -> bool {
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => range.clone().any(|h| h == target),
        S::Store { pointer, value } => *pointer == target || *value == target,
        S::Call { arguments, .. } => arguments.contains(&target),
        S::Return { value: Some(v) } => *v == target,
        S::Atomic { pointer, value, .. } => *pointer == target || *value == target,
        S::WorkGroupUniformLoad { pointer, result } => *pointer == target || *result == target,
        S::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            *image == target
                || *coordinate == target
                || array_index.is_some_and(|a| a == target)
                || *value == target
        }
        S::If {
            condition,
            accept,
            reject,
        } => {
            *condition == target
                || block_uses_expr(accept, target)
                || block_uses_expr(reject, target)
        }
        S::Switch { selector, cases } => {
            *selector == target || cases.iter().any(|c| block_uses_expr(&c.body, target))
        }
        S::Loop {
            body,
            continuing,
            break_if,
        } => {
            break_if.is_some_and(|b| b == target)
                || block_uses_expr(body, target)
                || block_uses_expr(continuing, target)
        }
        S::Block(inner) => block_uses_expr(inner, target),
        _ => false,
    }
}

// MARK: Block emission

impl<'a> Generator<'a> {
    /// Emit every statement in `block` in source order.
    pub(super) fn generate_block(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, false)
    }

    /// Emit a function body, dropping a trailing `return;` on
    /// void-returning functions because WGSL makes it optional.
    pub(super) fn generate_block_elide_trailing_return(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        self.generate_block_inner(block, ctx, true)
    }

    /// Shared driver for [`generate_block`] and
    /// [`generate_block_elide_trailing_return`].
    fn generate_block_inner(
        &mut self,
        block: &naga::Block,
        ctx: &mut FunctionCtx<'a, '_>,
        elide_trailing_void_return: bool,
    ) -> Result<(), Error> {
        let mut stmts: Vec<_> = block.iter().collect();
        // Optionally drop trailing `return;` (void return) that WGSL does not require.
        if elide_trailing_void_return {
            if let Some(naga::Statement::Return { value: None }) = stmts.last() {
                stmts.pop();
            }
        }
        self.emit_stmts(&stmts, ctx)
    }

    /// Emit a slice of statement references while attempting to
    /// reconstruct a `for` loop from the `Loop` / `Store` init pattern
    /// whenever possible.  Invoked from
    /// [`generate_block_inner`](Self::generate_block_inner) and from
    /// the for-loop body emission in `try_emit_for_loop`.
    fn emit_stmts(
        &mut self,
        stmts: &[&naga::Statement],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        let len = stmts.len();
        let mut i = 0;
        while i < len {
            let stmt = stmts[i];

            // Try to reconstruct a for-loop from a Loop statement
            // (optionally preceded by a deferred-var Store as the init).
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = stmt
            {
                if self.try_emit_for_loop(body, continuing, break_if, None, ctx)? {
                    i += 1;
                    continue;
                }
            }
            // Look ahead: if stmt[i] is a deferred-var Store and stmt[i+1]
            // is a for-loop-shaped Loop, absorb the Store as the for-init.
            if i + 1 < len {
                if let naga::Statement::Store { pointer, value } = stmt {
                    if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
                        if ctx.deferred_vars[lh.index()] {
                            // Safety check: the for-init scopes `lh` inside the
                            // loop body.  If `lh` is referenced after the Loop
                            // (stmts[i+2..]), absorbing it would leave those
                            // later uses out of scope -> skip the absorption.
                            let safe = i + 2 >= len
                                || !super::module_emit::local_var_in_stmts(
                                    &stmts[i + 2..],
                                    lh,
                                    &ctx.func.expressions,
                                );
                            if safe {
                                if let naga::Statement::Loop {
                                    body,
                                    continuing,
                                    break_if,
                                } = stmts[i + 1]
                                {
                                    if self.try_emit_for_loop(
                                        body,
                                        continuing,
                                        break_if,
                                        Some((*pointer, *value)),
                                        ctx,
                                    )? {
                                        // Mark deferred var as emitted.
                                        ctx.deferred_vars[lh.index()] = false;
                                        i += 2; // skip both Store and Loop
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let before = self.out.len();
            self.push_indent();
            let after_indent = self.out.len();
            self.generate_statement(stmt, ctx)?;
            if self.out.len() > after_indent {
                // Statement produced output; terminate the line.
                self.push_newline();
            } else {
                // Statement produced nothing (e.g. all-single-use Emit);
                // remove the indent we speculatively pushed.
                self.out.truncate(before);
            }
            i += 1;
        }
        Ok(())
    }

    fn generate_statement(
        &mut self,
        stmt: &naga::Statement,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        use naga::Statement as S;
        match stmt {
            S::Emit(range) => {
                let mut emitted_any = false;
                for h in range.clone() {
                    if !self.should_bind_expression(h, ctx) {
                        continue;
                    }
                    // Skip binding when the expression has too few references
                    // to justify the `let X=...;` overhead.  For trivially short
                    // expressions (e.g. `-x`, `v.y`, `a*b`) the threshold is
                    // higher because inlining them at each use site is cheaper.
                    let refs = ctx.ref_counts[h.index()];
                    if refs < self.min_binding_refs(h, ctx) {
                        continue;
                    }
                    // For subsequent bindings, start a new indented line.
                    if emitted_any {
                        self.push_newline();
                        self.push_indent();
                    }
                    emitted_any = true;
                    let name = ctx.next_expr_name();
                    let value = self.emit_expr_uncached(h, ctx)?;
                    self.out.push_str("let ");
                    self.out.push_str(&name);
                    self.push_assign();
                    self.out.push_str(&value);
                    self.out.push(';');
                    ctx.expr_names.insert(h, name);
                }
            }
            S::Block(block) => {
                self.out.push('{');
                self.push_newline();
                self.indent_depth += 1;
                self.generate_block(block, ctx)?;
                self.close_brace();
            }
            S::If {
                condition,
                accept,
                reject,
            } => {
                if accept.is_empty() && !reject.is_empty() {
                    // Empty true-branch: flip the condition and emit only
                    // the false-branch, e.g. `if c{}else{break;}` -> `if c>=32{break;}`
                    self.out.push_str("if ");
                    self.out
                        .push_str(&self.emit_negated_condition(*condition, ctx)?);
                    self.open_brace();
                    self.generate_block(reject, ctx)?;
                    self.close_brace();
                } else {
                    self.out.push_str("if ");
                    self.out.push_str(&self.emit_expr(*condition, ctx)?);
                    self.open_brace();
                    self.generate_block(accept, ctx)?;
                    self.close_brace();
                    if !reject.is_empty() {
                        self.push_else();
                        self.open_brace();
                        self.generate_block(reject, ctx)?;
                        self.close_brace();
                    }
                }
            }
            S::Switch { selector, cases } => {
                self.out.push_str("switch ");
                self.out.push_str(&self.emit_expr(*selector, ctx)?);
                self.open_brace();
                let mut new_case = true;
                for case in cases {
                    if case.fall_through && !case.body.is_empty() {
                        return Err(Error::Emit(format!(
                            "fall-through switch case with non-empty body \
                             in function '{}' is not representable in WGSL",
                            ctx.display_name,
                        )));
                    }
                    if new_case {
                        self.push_indent();
                    }
                    match case.value {
                        naga::SwitchValue::I32(v) => {
                            if new_case {
                                self.out.push_str("case ");
                            }
                            self.out.push_str(&v.to_string());
                        }
                        naga::SwitchValue::U32(v) => {
                            if new_case {
                                self.out.push_str("case ");
                            }
                            self.out.push_str(&v.to_string());
                            self.out.push('u');
                        }
                        naga::SwitchValue::Default => {
                            if new_case && case.fall_through {
                                self.out.push_str("case ");
                            }
                            self.out.push_str("default");
                        }
                    }
                    new_case = !case.fall_through;
                    if case.fall_through {
                        self.out.push_str(self.comma_sep());
                    } else {
                        self.open_brace();
                        self.generate_block(&case.body, ctx)?;
                        self.close_brace();
                        self.push_newline();
                    }
                }
                self.close_brace();
            }
            S::Loop {
                body,
                continuing,
                break_if,
            } => {
                self.out.push_str("loop");
                self.open_brace();
                self.generate_block(body, ctx)?;
                if !continuing.is_empty() || break_if.is_some() {
                    self.push_indent();
                    self.out.push_str("continuing");
                    self.open_brace();
                    if !continuing.is_empty() {
                        self.generate_block(continuing, ctx)?;
                    }
                    // `break if` must be the last statement inside `continuing`.
                    if let Some(expr) = break_if {
                        self.push_indent();
                        self.out.push_str("break if ");
                        self.out.push_str(&self.emit_expr(*expr, ctx)?);
                        self.out.push(';');
                        self.push_newline();
                    }
                    self.close_brace();
                    self.push_newline();
                }
                self.close_brace();
            }
            S::Break => self.out.push_str("break;"),
            S::Continue => self.out.push_str("continue;"),
            S::Return { value } => {
                self.out.push_str("return");
                if let Some(v) = value {
                    self.out.push(' ');
                    self.out.push_str(&self.emit_expr(*v, ctx)?);
                }
                self.out.push(';');
            }
            S::Kill => self.out.push_str("discard;"),
            S::ControlBarrier(flags) => {
                // Only emit barriers for the scopes explicitly requested.
                // An empty flag set produces no output (consistent with
                // MemoryBarrier handling).
                self.emit_barrier_calls(*flags);
            }
            S::MemoryBarrier(flags) => {
                // Only emit barriers for the scopes explicitly requested.
                self.emit_barrier_calls(*flags);
            }
            S::Store { pointer, value } => {
                if let Some(atomic_scalar) = self.atomic_scalar_for_expr(*pointer, ctx) {
                    let sep = self.comma_sep();
                    self.out.push_str("atomicStore(&");
                    self.out.push_str(&self.emit_expr(*pointer, ctx)?);
                    self.out.push_str(sep);
                    self.out
                        .push_str(&self.emit_expr_for_atomic(*value, atomic_scalar, ctx)?);
                    self.out.push_str(");");
                    return Ok(());
                }

                // Check if this is the first store to a deferred local var.
                let deferred_local =
                    if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
                        if ctx.deferred_vars[lh.index()] {
                            Some(lh)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                if let Some(lh) = deferred_local {
                    ctx.deferred_vars[lh.index()] = false;
                    self.out.push_str("var ");
                    self.out.push_str(&ctx.local_names[&lh]);
                    self.push_assign();
                    // When the value is an uncached Literal, use the typed
                    // suffix so WGSL infers the correct concrete type.
                    if !ctx.expr_names.contains_key(value) {
                        if let naga::Expression::Literal(lit) = ctx.func.expressions[*value] {
                            self.out.push_str(&super::syntax::literal_to_wgsl(
                                lit,
                                self.options.max_precision,
                            ));
                        } else {
                            self.out.push_str(&self.emit_expr(*value, ctx)?);
                        }
                    } else {
                        self.out.push_str(&self.emit_expr(*value, ctx)?);
                    }
                    self.out.push(';');
                } else if let Some((cop, other)) = self.try_compound_assign(*pointer, *value, ctx) {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    let sp = self.bin_op_sep();
                    self.out.push_str(sp);
                    self.out.push_str(cop);
                    self.out.push_str(sp);
                    self.out
                        .push_str(&self.emit_compound_assign_rhs(cop, other, ctx)?);
                    self.out.push(';');
                } else {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    self.push_assign();
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
                    self.out.push(';');
                }
            }
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                self.out.push_str("textureStore(");
                self.out.push_str(&self.emit_expr(*image, ctx)?);
                let sep = self.comma_sep();
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*index, ctx)?);
                }
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*value, ctx)?);
                self.out.push_str(");");
            }
            S::Call {
                function,
                arguments,
                result,
            } => {
                let call = self.emit_call(*function, arguments, ctx)?;
                self.emit_call_result(&call, *result, ctx);
            }
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                let atomic_scalar = self.atomic_scalar_for_expr(*pointer, ctx);
                let sep = self.comma_sep();
                let fn_name = match *fun {
                    naga::AtomicFunction::Add => "atomicAdd",
                    naga::AtomicFunction::Subtract => "atomicSub",
                    naga::AtomicFunction::And => "atomicAnd",
                    naga::AtomicFunction::ExclusiveOr => "atomicXor",
                    naga::AtomicFunction::InclusiveOr => "atomicOr",
                    naga::AtomicFunction::Min => "atomicMin",
                    naga::AtomicFunction::Max => "atomicMax",
                    naga::AtomicFunction::Exchange { compare: None } => "atomicExchange",
                    naga::AtomicFunction::Exchange {
                        compare: Some(compare),
                    } => {
                        let mut call = String::from("atomicCompareExchangeWeak(&");
                        call.push_str(&self.emit_expr(*pointer, ctx)?);
                        call.push_str(sep);
                        if let Some(scalar) = atomic_scalar {
                            call.push_str(&self.emit_expr_for_atomic(compare, scalar, ctx)?);
                        } else {
                            call.push_str(&self.emit_expr(compare, ctx)?);
                        }
                        call.push_str(sep);
                        if let Some(scalar) = atomic_scalar {
                            call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                        } else {
                            call.push_str(&self.emit_expr(*value, ctx)?);
                        }
                        call.push(')');
                        self.emit_call_result(&call, *result, ctx);
                        return Ok(());
                    }
                };
                let mut call = String::from(fn_name);
                call.push_str("(&");
                call.push_str(&self.emit_expr(*pointer, ctx)?);
                call.push_str(sep);
                if let Some(scalar) = atomic_scalar {
                    call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                } else {
                    call.push_str(&self.emit_expr(*value, ctx)?);
                }
                call.push(')');
                self.emit_call_result(&call, *result, ctx);
            }
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                let image_atomic_scalar = self.image_atomic_scalar_for_expr(*image, ctx);
                let fn_name = match *fun {
                    naga::AtomicFunction::Add => "textureAtomicAdd",
                    naga::AtomicFunction::Subtract => "textureAtomicSub",
                    naga::AtomicFunction::And => "textureAtomicAnd",
                    naga::AtomicFunction::ExclusiveOr => "textureAtomicXor",
                    naga::AtomicFunction::InclusiveOr => "textureAtomicOr",
                    naga::AtomicFunction::Min => "textureAtomicMin",
                    naga::AtomicFunction::Max => "textureAtomicMax",
                    naga::AtomicFunction::Exchange { compare: None } => "textureAtomicExchange",
                    naga::AtomicFunction::Exchange {
                        compare: Some(compare),
                    } => {
                        let sep = self.comma_sep();
                        let mut call = String::from("textureAtomicCompareExchangeWeak(");
                        call.push_str(&self.emit_expr(*image, ctx)?);
                        call.push_str(sep);
                        call.push_str(&self.emit_expr(*coordinate, ctx)?);
                        if let Some(index) = array_index {
                            call.push_str(sep);
                            call.push_str(&self.emit_expr(*index, ctx)?);
                        }
                        call.push_str(sep);
                        if let Some(scalar) = image_atomic_scalar {
                            call.push_str(&self.emit_expr_for_atomic(compare, scalar, ctx)?);
                        } else {
                            call.push_str(&self.emit_expr(compare, ctx)?);
                        }
                        call.push_str(sep);
                        if let Some(scalar) = image_atomic_scalar {
                            call.push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                        } else {
                            call.push_str(&self.emit_expr(*value, ctx)?);
                        }
                        call.push(')');
                        self.out.push_str(&call);
                        self.out.push(';');
                        return Ok(());
                    }
                };
                let sep = self.comma_sep();
                self.out.push_str(fn_name);
                self.out.push('(');
                self.out.push_str(&self.emit_expr(*image, ctx)?);
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*index, ctx)?);
                }
                self.out.push_str(sep);
                if let Some(scalar) = image_atomic_scalar {
                    self.out
                        .push_str(&self.emit_expr_for_atomic(*value, scalar, ctx)?);
                } else {
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
                }
                self.out.push_str(");");
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("workgroupUniformLoad(&");
                self.out.push_str(&self.emit_expr(*pointer, ctx)?);
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupBallot { result, predicate } => {
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str("subgroupBallot(");
                if let Some(pred) = predicate {
                    self.out.push_str(&self.emit_expr(*pred, ctx)?);
                }
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                let fn_name = subgroup_collective_name(*op, *collective_op)?;
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                self.out.push_str(fn_name);
                self.out.push('(');
                let arg_hint = self.expr_scalar_hint(*result, ctx);
                self.out
                    .push_str(&self.emit_expr_with_scalar_hint(*argument, arg_hint, ctx)?);
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                let sep = self.comma_sep();
                let name = ctx.next_expr_name();
                self.out.push_str("let ");
                self.out.push_str(&name);
                self.push_assign();
                let (fn_name, index) = subgroup_gather_name_and_index(mode);
                self.out.push_str(fn_name);
                self.out.push('(');
                let arg_hint = self.expr_scalar_hint(*result, ctx);
                self.out
                    .push_str(&self.emit_expr_with_scalar_hint(*argument, arg_hint, ctx)?);
                if let Some(idx) = index {
                    self.out.push_str(sep);
                    let u32_hint = Some(naga::Scalar {
                        kind: naga::ScalarKind::Uint,
                        width: 4,
                    });
                    self.out
                        .push_str(&self.emit_expr_with_scalar_hint(idx, u32_hint, ctx)?);
                }
                self.out.push_str(");");
                ctx.expr_names.insert(*result, name);
            }
            S::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    let sep = self.comma_sep();
                    self.out.push_str("traceRay(");
                    self.out
                        .push_str(&self.emit_expr(*acceleration_structure, ctx)?);
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*descriptor, ctx)?);
                    self.out.push_str(sep);
                    self.out.push('&');
                    self.out.push_str(&self.emit_expr(*payload, ctx)?);
                    self.out.push_str(");");
                }
            },
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported statement in function '{}': {:?}",
                    ctx.display_name, stmt,
                )));
            }
        }
        Ok(())
    }

    fn emit_barrier_calls(&mut self, flags: naga::Barrier) {
        let mut emitted = false;
        let barriers: &[(naga::Barrier, &str)] = &[
            (naga::Barrier::WORK_GROUP, "workgroupBarrier();"),
            (naga::Barrier::STORAGE, "storageBarrier();"),
            (naga::Barrier::TEXTURE, "textureBarrier();"),
            (naga::Barrier::SUB_GROUP, "subgroupBarrier();"),
        ];
        for &(flag, call) in barriers {
            if flags.contains(flag) {
                if emitted {
                    self.push_newline();
                    self.push_indent();
                }
                self.out.push_str(call);
                emitted = true;
            }
        }
    }

    fn should_bind_expression(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            E::CallResult(_)
            | E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult
            | E::RayQueryGetIntersection { .. } => return false,
            // Load from a bare local/global variable emits as just the
            // variable name (1-2 chars after rename).  A `let` binding
            // would introduce overhead (`let X=Y;`) with zero savings
            // per use since both names come from the same short-name pool.
            E::Load { pointer } => {
                if matches!(
                    ctx.func.expressions[*pointer],
                    E::LocalVariable(_) | E::GlobalVariable(_)
                ) {
                    return false;
                }
            }
            _ => {}
        }

        let inner = ctx.info[h].ty.inner_with(&self.module.types);
        !matches!(
            inner,
            naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. }
        )
    }

    /// Minimum reference count before a `let` binding pays for itself.
    ///
    /// A `let X = EXPR;` costs roughly `textLen + 7` characters (in compact
    /// mode), saving `textLen - 1` characters per use compared to inlining.
    /// For very short expressions the overhead exceeds the savings at low
    /// reference counts, so we require more references before introducing a
    /// binding.
    ///
    /// Thresholds are derived from:
    ///   binding wins when  `refs > (textLen + 7) / (textLen - 1)`
    fn min_binding_refs(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> usize {
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            // `-x` ~= 2 chars -> need 10+ refs to break even
            E::Unary { expr: child, .. } => {
                if self.expr_resolves_to_name(*child, ctx) {
                    return 10;
                }
                2
            }
            // `v.x` on a vector ~= 3 chars -> need 6+ refs
            E::AccessIndex { base, .. } => {
                if self.expr_resolves_to_name(*base, ctx) {
                    let inner = ctx.info[*base].ty.inner_with(&self.module.types);
                    if matches!(inner, naga::TypeInner::Vector { .. }) {
                        return 6;
                    }
                }
                2
            }
            // `a*b` with both operands named ~= 3 chars -> need 6+ refs
            E::Binary { left, right, .. } => {
                if self.expr_resolves_to_name(*left, ctx) && self.expr_resolves_to_name(*right, ctx)
                {
                    return 6;
                }
                2
            }
            _ => 2,
        }
    }

    /// Whether an expression handle will be emitted as a short name (1-2
    /// characters) rather than a sub-expression.
    fn expr_resolves_to_name(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        if ctx.expr_names.contains_key(&h) {
            return true;
        }
        use naga::Expression as E;
        match &ctx.func.expressions[h] {
            E::FunctionArgument(_) | E::Constant(_) | E::Override(_) => true,
            // Load from a bare local/global emits as the variable name.
            E::Load { pointer } => matches!(
                ctx.func.expressions[*pointer],
                E::LocalVariable(_) | E::GlobalVariable(_)
            ),
            _ => false,
        }
    }

    /// Try to emit a `Loop` as a WGSL `for(init; cond; update)` statement.
    ///
    /// Returns `Ok(true)` if a for-loop was emitted, `Ok(false)` if the
    /// loop doesn't match the pattern and should use the regular `loop`
    /// emission path.
    ///
    /// Pattern recognised:
    /// - `break_if` is `None`
    /// - `body` starts with optional `WorkGroupUniformLoad` preloads then
    ///   `If { condition, accept: [], reject: [Break] }`
    /// - `continuing` has at most one statement (the update)
    fn try_emit_for_loop(
        &mut self,
        body: &naga::Block,
        continuing: &naga::Block,
        break_if: &Option<naga::Handle<naga::Expression>>,
        init: Option<(
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        )>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<bool, Error> {
        // Only convert loops without `break if` in continuing.
        if break_if.is_some() {
            return Ok(false);
        }

        // Find the If-break guard, skipping leading Emit statements.
        // naga always places Emit ranges before the expressions they cover,
        // so the condition's Emit precedes the If statement.
        let body_stmts: Vec<_> = body.iter().collect();
        // Optional leading `WorkGroupUniformLoad` statements feeding the guard.
        let mut guard_preloads: Vec<(
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        )> = Vec::new();
        let mut guard_preload_stmt_indices: Vec<usize> = Vec::new();
        let mut guard_idx = 0;
        while guard_idx < body_stmts.len() {
            match body_stmts[guard_idx] {
                naga::Statement::Emit(_) => {
                    guard_idx += 1;
                }
                naga::Statement::WorkGroupUniformLoad { pointer, result } => {
                    guard_preloads.push((*pointer, *result));
                    guard_preload_stmt_indices.push(guard_idx);
                    guard_idx += 1;
                }
                _ => break,
            }
        }
        if guard_idx >= body_stmts.len() {
            return Ok(false);
        }

        let condition = match body_stmts[guard_idx] {
            // Pattern 1: `if cond {} else { break; }` - cond is the continue condition.
            naga::Statement::If {
                condition,
                accept,
                reject,
            } if accept.is_empty()
                && reject.len() == 1
                && matches!(reject.iter().next(), Some(naga::Statement::Break)) =>
            {
                (*condition, false) // (handle, needs_negation)
            }
            // Pattern 2: `if cond { break; }` - cond is the exit condition (negate).
            naga::Statement::If {
                condition,
                accept,
                reject,
            } if reject.is_empty()
                && accept.len() == 1
                && matches!(accept.iter().next(), Some(naga::Statement::Break)) =>
            {
                (*condition, true) // (handle, needs_negation)
            }
            _ => return Ok(false),
        };

        // If we are going to drop leading preload `let` bindings and inline
        // them in the `for` condition, their result handles must not be used
        // anywhere after the guard.
        if !guard_preloads.is_empty() {
            let tail = if guard_idx + 1 < body_stmts.len() {
                &body_stmts[guard_idx + 1..]
            } else {
                &[]
            };
            for (_, result) in &guard_preloads {
                if stmts_use_expr(tail, *result) {
                    return Ok(false);
                }
            }
        }

        // Continuing may start with `WorkGroupUniformLoad` preloads and then
        // have at most one core update statement.
        let mut update_preloads: Vec<(
            naga::Handle<naga::Expression>,
            naga::Handle<naga::Expression>,
        )> = Vec::new();
        let update_stmt = {
            let mut found = None;
            for s in continuing.iter() {
                match s {
                    naga::Statement::Emit(_) => continue,
                    naga::Statement::WorkGroupUniformLoad { pointer, result }
                        if found.is_none() =>
                    {
                        update_preloads.push((*pointer, *result));
                    }
                    _ => {
                        if found.is_some() {
                            // More than 1 core update statement - can't fit in for-header.
                            return Ok(false);
                        }
                        found = Some(s);
                    }
                }
            }
            found
        };

        // Pre-validate: Store / Call / ImageStore are supported in the for-loop
        // update slot.  Bail out *before* writing any output so the caller
        // can fall back to normal `loop` emission cleanly.
        if let Some(stmt) = update_stmt {
            if !matches!(
                stmt,
                naga::Statement::Store { .. }
                    | naga::Statement::Call { .. }
                    | naga::Statement::ImageStore { .. }
            ) {
                return Ok(false);
            }
        }

        // If no init was provided (from deferred-var absorption), try to
        // absorb an initialized local whose refs are confined to this loop.

        // Track a for-loop counter consumed without an init value, so we
        // can emit a bare `var name:type` declaration in the init clause.
        let mut decl_only_local: Option<naga::Handle<naga::LocalVariable>> = None;

        let init = if init.is_some() {
            // An external init occupies the for-loop init slot.  If the
            // loop counter is a for_loop_var it would be left undeclared
            // (its `var` was suppressed).  Bail out so the Store is emitted
            // as a normal statement and the Loop falls through to path 1,
            // which can absorb the for_loop_var.
            if let Some(naga::Statement::Store { pointer, .. }) = update_stmt {
                if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
                    if ctx.for_loop_vars[lh.index()] {
                        return Ok(false);
                    }
                }
            }
            init
        } else if let Some(naga::Statement::Store { pointer, .. }) = update_stmt {
            if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[*pointer] {
                if ctx.for_loop_vars[lh.index()] {
                    // Mark as consumed so try_emit_for_loop won't absorb again.
                    ctx.for_loop_vars[lh.index()] = false;
                    if let Some(init_handle) = ctx.func.local_variables[lh].init {
                        Some((*pointer, init_handle))
                    } else {
                        // Variable has no explicit init (zero-initialised by
                        // WGSL semantics, e.g. after dead-init removal).
                        // Record it for a bare declaration in the init clause.
                        decl_only_local = Some(lh);
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Emit the for-loop
        self.push_indent();
        self.push_for_open();

        // Init clause.
        if let Some((pointer, value)) = init {
            if let naga::Expression::LocalVariable(lh) = ctx.func.expressions[pointer] {
                self.out.push_str("var ");
                self.out.push_str(&ctx.local_names[&lh]);
                self.push_assign();
                if !ctx.expr_names.contains_key(&value) {
                    if let naga::Expression::Literal(lit) = ctx.func.expressions[value] {
                        self.out.push_str(&super::syntax::literal_to_wgsl(
                            lit,
                            self.options.max_precision,
                        ));
                    } else {
                        self.out.push_str(&self.emit_expr(value, ctx)?);
                    }
                } else {
                    self.out.push_str(&self.emit_expr(value, ctx)?);
                }
            } else {
                self.out.push_str(&self.emit_lvalue(pointer, ctx)?);
                self.push_assign();
                self.out.push_str(&self.emit_expr(value, ctx)?);
            }
        } else if let Some(lh) = decl_only_local {
            // For-loop counter with no explicit init (WGSL zero-initialises).
            // Emit a bare `var name:type` declaration so the variable is in scope.
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&lh]);
            let ty = ctx.func.local_variables[lh].ty;
            self.out.push(':');
            self.out.push_str(&self.type_ref(ty)?);
        }
        self.push_for_sep();

        // Condition clause (inlined - Emits haven't been processed yet).
        let mut preload_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
            Vec::new();
        for (pointer, result) in &guard_preloads {
            let mut preload = String::from("workgroupUniformLoad(&");
            preload.push_str(&self.emit_expr(*pointer, ctx)?);
            preload.push(')');
            let old = ctx.expr_names.insert(*result, preload);
            preload_old_bindings.push((*result, old));
        }

        let (cond_handle, needs_negation) = condition;
        if needs_negation {
            self.out
                .push_str(&self.emit_negated_condition(cond_handle, ctx)?);
        } else {
            self.out.push_str(&self.emit_expr(cond_handle, ctx)?);
        }

        for (result, old) in preload_old_bindings {
            if let Some(name) = old {
                ctx.expr_names.insert(result, name);
            } else {
                ctx.expr_names.remove(&result);
            }
        }

        self.push_for_sep();

        // Update clause.
        if let Some(stmt) = update_stmt {
            let mut update_old_bindings: Vec<(naga::Handle<naga::Expression>, Option<String>)> =
                Vec::new();
            for (pointer, result) in &update_preloads {
                let mut preload = String::from("workgroupUniformLoad(&");
                preload.push_str(&self.emit_expr(*pointer, ctx)?);
                preload.push(')');
                let old = ctx.expr_names.insert(*result, preload);
                update_old_bindings.push((*result, old));
            }
            self.emit_statement_inline(stmt, ctx)?;
            for (result, old) in update_old_bindings {
                if let Some(name) = old {
                    ctx.expr_names.insert(result, name);
                } else {
                    ctx.expr_names.remove(&result);
                }
            }
        }

        self.out.push(')');
        self.open_brace();

        // The condition (and update) were emitted inline into the for-header,
        // bypassing the normal Emit -> let-binding path.  Decrement ref counts
        // for children of expressions in the guard's Emit ranges so that
        // expressions shared between the condition and body (e.g. loads
        // deduplicated by load_dedup) get correct effective ref counts and
        // are inlined rather than bound to unnecessary `let` variables.
        for s in &body_stmts[..guard_idx] {
            if let naga::Statement::Emit(range) = s {
                for h in range.clone() {
                    super::module_emit::visit_expr_children(&ctx.func.expressions[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }
        for stmt in continuing.iter() {
            if let naga::Statement::Emit(range) = stmt {
                for h in range.clone() {
                    super::module_emit::visit_expr_children(&ctx.func.expressions[h], |child| {
                        ctx.ref_counts[child.index()] =
                            ctx.ref_counts[child.index()].saturating_sub(1);
                    });
                }
            }
        }

        // Emit all body statements except the If-break guard,
        // using the shared helper so nested for-loops are also reconstructed.
        let remaining: Vec<_> = body_stmts
            .iter()
            .enumerate()
            .filter_map(|(j, s)| {
                if j == guard_idx || guard_preload_stmt_indices.contains(&j) {
                    None
                } else {
                    Some(*s)
                }
            })
            .collect();

        // Unwrap a single trailing Block statement to avoid double-braces.
        // naga wraps the user's for-body in `Statement::Block`, producing
        // `for(...){ { body } }` - this removes the inner braces.
        //
        // `non_emit` collects the indices of non-Emit statements in the
        // remaining body.  When there is exactly one and it is a
        // `Statement::Block`, we emit its contents directly (unwrapping
        // the inner braces).  Non-`Block` single-statement bodies
        // (e.g. `for(...){ i+=1; }`) don't have a double-brace problem,
        // so they fall through to the normal `emit_stmts` path below.
        let mut non_emit: Vec<usize> = Vec::new();
        for (k, s) in remaining.iter().enumerate() {
            if !matches!(s, naga::Statement::Emit(_)) {
                non_emit.push(k);
            }
        }
        if non_emit.len() == 1 {
            if let naga::Statement::Block(inner) = remaining[non_emit[0]] {
                // Emit leading Emits, then the inner block contents directly.
                for (k, s) in remaining.iter().enumerate() {
                    if k == non_emit[0] {
                        continue;
                    }
                    let before = self.out.len();
                    self.push_indent();
                    let after_indent = self.out.len();
                    self.generate_statement(s, ctx)?;
                    if self.out.len() > after_indent {
                        self.push_newline();
                    } else {
                        self.out.truncate(before);
                    }
                }
                self.generate_block(inner, ctx)?;
                self.close_brace();
                self.push_newline();
                return Ok(true);
            }
        }

        self.emit_stmts(&remaining, ctx)?;

        self.close_brace();
        self.push_newline();
        Ok(true)
    }

    /// Emit a single statement's text without indent/newline/trailing semicolon.
    /// Used for for-loop update clauses.
    fn emit_statement_inline(
        &mut self,
        stmt: &naga::Statement,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        match stmt {
            naga::Statement::Store { pointer, value } => {
                if let Some((cop, other)) = self.try_compound_assign(*pointer, *value, ctx) {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    let sp = self.bin_op_sep();
                    self.out.push_str(sp);
                    self.out.push_str(cop);
                    self.out.push_str(sp);
                    self.out
                        .push_str(&self.emit_compound_assign_rhs(cop, other, ctx)?);
                } else {
                    self.out.push_str(&self.emit_lvalue(*pointer, ctx)?);
                    self.push_assign();
                    self.out.push_str(&self.emit_expr(*value, ctx)?);
                }
            }
            naga::Statement::Call {
                function,
                arguments,
                result: _,
            } => {
                self.out
                    .push_str(&self.emit_call(*function, arguments, ctx)?);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                self.out.push_str("textureStore(");
                self.out.push_str(&self.emit_expr(*image, ctx)?);
                let sep = self.comma_sep();
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(index) = array_index {
                    self.out.push_str(sep);
                    self.out.push_str(&self.emit_expr(*index, ctx)?);
                }
                self.out.push_str(sep);
                self.out.push_str(&self.emit_expr(*value, ctx)?);
                self.out.push(')');
            }
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported statement in for-loop update clause \
                     in function '{}': {:?}",
                    ctx.display_name, stmt,
                )));
            }
        }
        Ok(())
    }

    /// Try to recognise `Store(ptr, Binary(op, Load(ptr), rhs))` or the
    /// commutative mirror and return the compound-assign operator token
    /// plus the "other" operand handle.
    fn try_compound_assign(
        &self,
        pointer: naga::Handle<naga::Expression>,
        value: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<(&'static str, naga::Handle<naga::Expression>)> {
        // The value expression must not already be cached (bound to a let).
        if ctx.expr_names.contains_key(&value) {
            return None;
        }
        let naga::Expression::Binary { op, left, right } = &ctx.func.expressions[value] else {
            return None;
        };
        let (cop, commutative) = compound_assign_info(*op)?;

        // Check if left is Load of a structurally-equivalent pointer.
        let left_is_self = !ctx.expr_names.contains_key(left) && {
            if let naga::Expression::Load { pointer: p } = &ctx.func.expressions[*left] {
                ptrs_structurally_equal(*p, pointer, &ctx.func.expressions)
            } else {
                false
            }
        };
        if left_is_self {
            return Some((cop, *right));
        }

        // For commutative ops, also check the right side.
        if commutative && !ctx.expr_names.contains_key(right) {
            if let naga::Expression::Load { pointer: p } = &ctx.func.expressions[*right] {
                if ptrs_structurally_equal(*p, pointer, &ctx.func.expressions) {
                    return Some((cop, *left));
                }
            }
        }

        None
    }

    /// Emit the right-hand side of a compound assignment, applying splat
    /// elision when the operand is a Splat / splat-Compose and the compound
    /// operator is arithmetic (`+=`, `-=`, `*=`, `/=`, `%=`).
    fn emit_compound_assign_rhs(
        &self,
        cop: &str,
        other: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if matches!(cop, "+=" | "-=" | "*=" | "/=" | "%=") {
            let cached = ctx.expr_names.contains_key(&other);
            if let Some(scalar) = self.try_splat_scalar(other, &ctx.func.expressions, cached) {
                return self.emit_constructor_arg(scalar, ctx);
            }
        }
        self.emit_expr(other, ctx)
    }

    /// Emit `let <name> = <call>;` when there is a result handle, or
    /// just `<call>;` otherwise.  When the result is used exactly once
    /// and no side-effecting statements can intervene (pre-computed in
    /// `inlineable_calls`), the call text is stored directly so it can
    /// be emitted inline at the use site, avoiding the `let` binding.
    fn emit_call_result(
        &mut self,
        call: &str,
        result: Option<naga::Handle<naga::Expression>>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) {
        if let Some(handle) = result {
            if ctx.inlineable_calls.contains(&handle) {
                // Single-use in a safe zone: store call text for inline emission
                ctx.expr_names.insert(handle, call.to_string());
                return;
            }
            let name = ctx.next_expr_name();
            self.out.push_str("let ");
            self.out.push_str(&name);
            self.push_assign();
            self.out.push_str(call);
            self.out.push(';');
            ctx.expr_names.insert(handle, name);
        } else {
            self.out.push_str(call);
            self.out.push(';');
        }
    }
}

impl<'a> Generator<'a> {
    fn atomic_scalar_for_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        match ctx.info[expr].ty.inner_with(&self.module.types) {
            naga::TypeInner::Atomic(s) => Some(*s),
            naga::TypeInner::Pointer { base, .. } => match &self.module.types[*base].inner {
                naga::TypeInner::Atomic(s) => Some(*s),
                _ => None,
            },
            _ => None,
        }
    }

    fn emit_expr_for_atomic(
        &self,
        expr: naga::Handle<naga::Expression>,
        target: naga::Scalar,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        use naga::Literal as L;

        let literal = match &ctx.func.expressions[expr] {
            E::Literal(lit) => Some(*lit),
            E::Constant(h) => {
                let c = &self.module.constants[*h];
                if c.name.is_none() {
                    if let E::Literal(lit) = self.module.global_expressions[c.init] {
                        Some(lit)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(lit) = literal {
            let forced = match (lit, target.kind, target.width) {
                (L::U32(v), naga::ScalarKind::Uint, 4) => Some(format!("{v}u")),
                (L::U64(v), naga::ScalarKind::Uint, 8) => Some(format!("{v}lu")),
                (L::I32(v), naga::ScalarKind::Sint, 4) => Some(format!("{v}i")),
                (L::I64(v), naga::ScalarKind::Sint, 8) => Some(format!("{v}li")),
                (L::AbstractInt(v), naga::ScalarKind::Sint, 4) => Some(format!("{v}i")),
                (L::AbstractInt(v), naga::ScalarKind::Uint, 4) => Some(format!("{v}u")),
                (L::AbstractInt(v), naga::ScalarKind::Sint, 8) => Some(format!("{v}li")),
                (L::AbstractInt(v), naga::ScalarKind::Uint, 8) => Some(format!("{v}lu")),
                _ => None,
            };
            if let Some(v) = forced {
                return Ok(v);
            }
        }

        self.emit_expr(expr, ctx)
    }

    fn image_atomic_scalar_for_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        fn resolve_global_image(
            expr: naga::Handle<naga::Expression>,
            exprs: &naga::Arena<naga::Expression>,
        ) -> Option<naga::Handle<naga::GlobalVariable>> {
            match exprs[expr] {
                naga::Expression::GlobalVariable(h) => Some(h),
                naga::Expression::Access { base, .. }
                | naga::Expression::AccessIndex { base, .. } => resolve_global_image(base, exprs),
                _ => None,
            }
        }

        let image_inner = if let Some(gh) = resolve_global_image(expr, &ctx.func.expressions) {
            &self.module.types[self.module.global_variables[gh].ty].inner
        } else {
            let inner = ctx.info[expr].ty.inner_with(&self.module.types);
            match inner {
                naga::TypeInner::Image { .. } => inner,
                naga::TypeInner::Pointer { base, .. } => &self.module.types[*base].inner,
                _ => return None,
            }
        };

        let naga::TypeInner::Image {
            class: naga::ImageClass::Storage { format, .. },
            ..
        } = image_inner
        else {
            return None;
        };

        use naga::StorageFormat as F;
        match format {
            F::R64Uint => Some(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 8,
            }),
            F::R32Uint | F::Rg32Uint | F::Rgba32Uint => Some(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 4,
            }),
            F::R32Sint | F::Rg32Sint | F::Rgba32Sint => Some(naga::Scalar {
                kind: naga::ScalarKind::Sint,
                width: 4,
            }),
            _ => None,
        }
    }
}

// MARK: Statement-level helpers

/// Return the WGSL compound-assignment token (`+=`, `*=`, and so on)
/// for a binary operator, along with a boolean flagging operators
/// whose compound form preserves NaN-safety only for integer types.
/// Returns `None` when the operator has no compound form.
fn compound_assign_info(op: naga::BinaryOperator) -> Option<(&'static str, bool)> {
    use naga::BinaryOperator as B;
    match op {
        B::Add => Some(("+=", true)),
        B::Subtract => Some(("-=", false)),
        B::Multiply => Some(("*=", true)),
        B::Divide => Some(("/=", false)),
        B::Modulo => Some(("%=", false)),
        B::And => Some(("&=", true)),
        B::ExclusiveOr => Some(("^=", true)),
        B::InclusiveOr => Some(("|=", true)),
        B::ShiftLeft => Some(("<<=", false)),
        B::ShiftRight => Some((">>=", false)),
        _ => None,
    }
}

/// `true` when two pointer expressions walk identically through
/// `Access` and `AccessIndex` chains and share the same root
/// (`GlobalVariable`, `LocalVariable`, or `FunctionArgument`).
/// Used to detect `lhs = lhs <op> rhs` patterns safe to fold into
/// `lhs <op>= rhs`.
fn ptrs_structurally_equal(
    a: naga::Handle<naga::Expression>,
    b: naga::Handle<naga::Expression>,
    exprs: &naga::Arena<naga::Expression>,
) -> bool {
    if a == b {
        return true;
    }
    use naga::Expression as E;
    match (&exprs[a], &exprs[b]) {
        (E::GlobalVariable(ga), E::GlobalVariable(gb)) => ga == gb,
        (E::LocalVariable(la), E::LocalVariable(lb)) => la == lb,
        (E::FunctionArgument(ia), E::FunctionArgument(ib)) => ia == ib,
        (
            E::AccessIndex {
                base: ba,
                index: ia,
            },
            E::AccessIndex {
                base: bb,
                index: ib,
            },
        ) => ia == ib && ptrs_structurally_equal(*ba, *bb, exprs),
        (
            E::Access {
                base: ba,
                index: ia,
            },
            E::Access {
                base: bb,
                index: ib,
            },
        ) => {
            // Only match when both indices are identical literals.
            if ia == ib {
                return ptrs_structurally_equal(*ba, *bb, exprs);
            }
            if let (E::Literal(la), E::Literal(lb)) = (&exprs[*ia], &exprs[*ib]) {
                la == lb && ptrs_structurally_equal(*ba, *bb, exprs)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Render a `(SubgroupOperation, CollectiveOperation)` pair as its
/// WGSL built-in name (`subgroupInclusiveAdd`, and so on).
fn subgroup_collective_name(
    op: naga::SubgroupOperation,
    collective_op: naga::CollectiveOperation,
) -> Result<&'static str, Error> {
    use naga::CollectiveOperation as C;
    use naga::SubgroupOperation as S;
    match (collective_op, op) {
        (C::Reduce, S::All) => Ok("subgroupAll"),
        (C::Reduce, S::Any) => Ok("subgroupAny"),
        (C::Reduce, S::Add) => Ok("subgroupAdd"),
        (C::Reduce, S::Mul) => Ok("subgroupMul"),
        (C::Reduce, S::Min) => Ok("subgroupMin"),
        (C::Reduce, S::Max) => Ok("subgroupMax"),
        (C::Reduce, S::And) => Ok("subgroupAnd"),
        (C::Reduce, S::Or) => Ok("subgroupOr"),
        (C::Reduce, S::Xor) => Ok("subgroupXor"),
        (C::InclusiveScan, S::Add) => Ok("subgroupInclusiveAdd"),
        (C::InclusiveScan, S::Mul) => Ok("subgroupInclusiveMul"),
        (C::ExclusiveScan, S::Add) => Ok("subgroupExclusiveAdd"),
        (C::ExclusiveScan, S::Mul) => Ok("subgroupExclusiveMul"),
        _ => Err(Error::Emit(format!(
            "unsupported subgroup collective operation: {:?}/{:?}",
            collective_op, op,
        ))),
    }
}

/// Render a subgroup `GatherMode` as the
/// `(builtin_name, Option<index_handle>)` pair the caller needs to
/// build a WGSL call expression for broadcast / shuffle /
/// quad-broadcast variants.
fn subgroup_gather_name_and_index(
    mode: &naga::GatherMode,
) -> (&'static str, Option<naga::Handle<naga::Expression>>) {
    match *mode {
        naga::GatherMode::BroadcastFirst => ("subgroupBroadcastFirst", None),
        naga::GatherMode::Broadcast(h) => ("subgroupBroadcast", Some(h)),
        naga::GatherMode::Shuffle(h) => ("subgroupShuffle", Some(h)),
        naga::GatherMode::ShuffleDown(h) => ("subgroupShuffleDown", Some(h)),
        naga::GatherMode::ShuffleUp(h) => ("subgroupShuffleUp", Some(h)),
        naga::GatherMode::ShuffleXor(h) => ("subgroupShuffleXor", Some(h)),
        naga::GatherMode::QuadBroadcast(h) => ("quadBroadcast", Some(h)),
        naga::GatherMode::QuadSwap(naga::Direction::X) => ("quadSwapX", None),
        naga::GatherMode::QuadSwap(naga::Direction::Y) => ("quadSwapY", None),
        naga::GatherMode::QuadSwap(naga::Direction::Diagonal) => ("quadSwapDiagonal", None),
    }
}
