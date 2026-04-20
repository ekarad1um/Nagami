//! Expression emitters.
//!
//! Every [`naga::Expression`] variant routes through `emit_expr`
//! and, for the uncommon non-cached path, `emit_expr_uncached`.  The
//! expression emitters also drive the splat elision, swizzle collapse,
//! literal extraction, and parenthesis-minimisation logic that gives
//! the custom generator its size advantage over naga's own emitter.

use crate::error::Error;

use super::core::{FunctionCtx, Generator};
use super::syntax::{
    literal_extract_key, literal_to_wgsl, literal_to_wgsl_bare, math_name, scalar_zero,
    type_inner_name, type_resolution_name,
};

impl<'a> Generator<'a> {
    /// Emit a `Call` expression (`name(arg0, arg1, ...)`).  Inserts
    /// an explicit `&` before pointer-typed arguments that WGSL would
    /// otherwise treat as references, skipping the `&` for forwarded
    /// function-argument pointers which are already pointer values.
    pub(super) fn emit_call(
        &self,
        function: naga::Handle<naga::Function>,
        arguments: &[naga::Handle<naga::Expression>],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        let sep = self.comma_sep();
        let callee = &self.module.functions[function];
        let mut s = String::new();
        s.push_str(&self.function_names[function.index()]);
        s.push('(');
        for (i, arg) in arguments.iter().enumerate() {
            if i > 0 {
                s.push_str(sep);
            }
            // If the callee parameter is a pointer type, emit `&` to
            // convert the WGSL reference into a pointer - UNLESS the
            // argument is a forwarded function parameter whose declared
            // type is already a pointer (the only expression kind that
            // is a pointer value, not a reference, in WGSL text scope).
            // Globals, locals, and access chains are all references in
            // WGSL and always need `&`.
            let needs_ref = if let Some(param) = callee.arguments.get(i) {
                matches!(
                    self.module.types[param.ty].inner,
                    naga::TypeInner::Pointer { .. }
                ) && !matches!(
                    ctx.func.expressions[*arg],
                    naga::Expression::FunctionArgument(idx)
                    if matches!(
                        self.module.types[ctx.func.arguments[idx as usize].ty].inner,
                        naga::TypeInner::Pointer { .. }
                    )
                )
            } else {
                false
            };
            if needs_ref {
                s.push('&');
            }
            s.push_str(&self.emit_expr(*arg, ctx)?);
        }
        s.push(')');
        Ok(s)
    }

    /// Emit the logical negation of `condition` in the shortest
    /// form that still preserves semantics:
    ///
    /// 1. For a `Binary` comparison whose operator is safe to flip
    ///    (`<` becomes `>=`, `==` becomes `!=`, etc.), drop the outer
    ///    `!`.  Ordered comparisons on floats are NOT flipped because
    ///    `!(x < y)` diverges from `x >= y` when either operand is
    ///    NaN; equality is safe for every type.
    /// 2. For `Unary(LogicalNot, inner)`, emit `inner` directly.
    /// 3. Otherwise emit `!(expr)`.
    pub(super) fn emit_negated_condition(
        &self,
        cond: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;

        // If the condition has a let-binding, just negate the name.
        if let Some(name) = ctx.expr_names.get(&cond) {
            return Ok(format!("!{name}"));
        }

        match &ctx.func.expressions[cond] {
            // Flip comparison operators for shorter output.
            E::Binary { op, left, right } => {
                if let Some(flipped) = flip_comparison(*op) {
                    // Ordered comparisons on floats are not equivalent
                    // when negated due to NaN (IEEE 754).  Only
                    // equality/inequality flips are NaN-safe.
                    let is_ordered = matches!(
                        op,
                        naga::BinaryOperator::Less
                            | naga::BinaryOperator::LessEqual
                            | naga::BinaryOperator::Greater
                            | naga::BinaryOperator::GreaterEqual
                    );
                    let is_float = is_ordered
                        && ctx.info[*left]
                            .ty
                            .inner_with(&self.module.types)
                            .scalar_kind()
                            == Some(naga::ScalarKind::Float);
                    if !is_float {
                        let left = *left;
                        let right = *right;
                        let op_str = binary_op_str(flipped);
                        let sp = self.bin_op_sep();
                        let arena = &ctx.func.expressions;
                        let lc = ctx.expr_names.contains_key(&left);
                        let rc = ctx.expr_names.contains_key(&right);
                        let wrap_l = child_needs_parens(left, arena, flipped, false, lc);
                        let wrap_r = child_needs_parens(right, arena, flipped, true, rc);
                        let ls = self.emit_expr(left, ctx)?;
                        let rs = self.emit_expr(right, ctx)?;
                        return Ok(assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r));
                    }
                }
            }
            // Double-negation elimination: !(! x) -> x
            E::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr,
            } => {
                return self.emit_expr(*expr, ctx);
            }
            _ => {}
        }

        // Fallback: wrap in !().
        let mut inner = self.emit_expr(cond, ctx)?;
        inner.insert_str(0, "!(");
        inner.push(')');
        Ok(inner)
    }

    /// Emit `expr` treated as an assignable place: globals and
    /// locals render by name, accesses cascade through `.field`,
    /// `.xyz`, or `[index]`, and anything else is dereferenced with
    /// a leading `*`.
    pub(super) fn emit_lvalue(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        Ok(match &ctx.func.expressions[expr] {
            E::GlobalVariable(h) => self.global_names[h.index()].clone(),
            E::LocalVariable(h) => ctx.local_names[h].clone(),
            E::Access { base, index } => {
                let mut s = self.emit_lvalue_or_value(*base, ctx)?;
                s.push('[');
                s.push_str(&self.emit_expr(*index, ctx)?);
                s.push(']');
                s
            }
            E::AccessIndex { base, index } => {
                let mut s = self.emit_lvalue_or_value(*base, ctx)?;
                if let Some(field_name) = self.struct_field_name(*base, *index, ctx) {
                    s.push('.');
                    s.push_str(&field_name);
                } else if let Some(c) = self.vector_component_name(*base, *index, ctx) {
                    s.push('.');
                    s.push(c);
                } else {
                    s.push('[');
                    s.push_str(&index.to_string());
                    s.push(']');
                }
                s
            }
            _ => {
                let mut s = String::with_capacity(1 + 16);
                s.push('*');
                s.push_str(&self.emit_expr(expr, ctx)?);
                s
            }
        })
    }

    /// Emit `expr` as either an lvalue (when the top-level shape is
    /// an `Access`/`AccessIndex` chain rooted in a variable) or the
    /// ordinary value form.  Used when the caller accepts either,
    /// such as the left operand of a store-through access chain.
    pub(super) fn emit_lvalue_or_value(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        Ok(match &ctx.func.expressions[expr] {
            E::GlobalVariable(h) => self.global_names[h.index()].clone(),
            E::LocalVariable(h) => ctx.local_names[h].clone(),
            E::Access { .. } | E::AccessIndex { .. } => self.emit_lvalue(expr, ctx)?,
            _ => self.emit_expr(expr, ctx)?,
        })
    }

    /// Emit `expr` as its WGSL value-text form.  Cached bindings (let
    /// / var / argument names) short-circuit to the bound identifier;
    /// otherwise control flows into [`emit_expr_uncached`] which
    /// handles every expression variant.
    pub(super) fn emit_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if let Some(name) = ctx.expr_names.get(&expr) {
            return Ok(name.clone());
        }
        self.emit_expr_uncached(expr, ctx)
    }

    /// Emit an expression that appears inside a type constructor
    /// (`Compose` / `Splat`).  An uncached literal is rendered in the
    /// suffix-free bare form because the enclosing constructor type
    /// pins the concrete type.
    ///
    /// Bare `<` / `<=` comparisons inside a template-delimited
    /// constructor (for example `vec3<bool>(a < b, ...)`) are wrapped
    /// in parentheses to avoid the WGSL parser mistaking the `<` for
    /// a template-list opener.
    pub(super) fn emit_constructor_arg(
        &self,
        arg: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if !ctx.expr_names.contains_key(&arg) {
            if let naga::Expression::Literal(lit) = &ctx.func.expressions[arg] {
                let bare = literal_to_wgsl_bare(*lit, self.options.max_precision);
                let key = literal_extract_key(*lit, self.options.max_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    return Ok(name.clone());
                }
                // Inside a type constructor the enclosing type provides
                // context, so bare literals (no suffix) are safe and produce
                // the shortest output when the literal is not extracted.
                return Ok(bare);
            }
            if let naga::Expression::Binary { op, .. } = &ctx.func.expressions[arg] {
                if matches!(
                    op,
                    naga::BinaryOperator::Less | naga::BinaryOperator::LessEqual
                ) {
                    let s = self.emit_expr_uncached(arg, ctx)?;
                    return Ok(format!("({s})"));
                }
            }
        }
        self.emit_expr(arg, ctx)
    }

    /// Check whether `handle` is an uncached `Splat` (or splat-like
    /// `Compose` with all identical components) that can be elided to
    /// its bare scalar when used as an operand of an arithmetic
    /// binary expression.  WGSL's scalar-vector broadcasting rules
    /// make the bare scalar a valid, shorter substitute in that
    /// context.  Returns the scalar value handle on success; `None`
    /// when elision is unsafe (for example when both operands are
    /// scalars, which would change the result type).
    pub(super) fn try_splat_scalar(
        &self,
        handle: naga::Handle<naga::Expression>,
        arena: &naga::Arena<naga::Expression>,
        cached: bool,
    ) -> Option<naga::Handle<naga::Expression>> {
        if cached {
            return None;
        }
        match &arena[handle] {
            naga::Expression::Splat { value, .. } => Some(*value),
            naga::Expression::Compose { ty, components } => {
                // Only vector-typed Compose can be elided (not matrix/struct).
                if !matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
                    return None;
                }
                if compose_is_splat(components, arena) {
                    Some(components[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Emit an expression that is the base of a postfix operation (`.member`,
    /// `[index]`, `.xyzw`).  Postfix operators bind tighter than any prefix
    /// or infix operator, so a Binary or Unary base expression must be
    /// wrapped in parentheses to preserve evaluation order.
    ///
    /// Example: naga IR `AccessIndex(Subtract(a, b), .x)` must emit
    /// `(a-b).x`, **not** `a-b.x` (which WGSL parses as `a - (b.x)`).
    fn emit_postfix_base(
        &self,
        base: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        // Named / cached expressions produce a single identifier token;
        // no parentheses needed.
        if ctx.expr_names.contains_key(&base) {
            return self.emit_expr(base, ctx);
        }
        let needs_parens = matches!(
            ctx.func.expressions[base],
            naga::Expression::Binary { .. }
                | naga::Expression::Unary { .. }
                | naga::Expression::Select { .. }
        );
        let s = self.emit_expr(base, ctx)?;
        if needs_parens {
            Ok(format!("({s})"))
        } else {
            Ok(s)
        }
    }

    // MARK: Expression dispatch

    /// Emit an expression that has no cached binding, dispatching on
    /// the [`naga::Expression`] variant.  Central switch for the
    /// emitter: every shape-specific rewrite (splat elision, swizzle
    /// collapse, literal extraction substitution, operator
    /// parenthesisation, and friends) lives inside the relevant arm.
    pub(super) fn emit_expr_uncached(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;

        Ok(match &ctx.func.expressions[expr] {
            E::Literal(lit) => {
                if let Some(concrete) =
                    self.concretize_abstract_literal_for_expr(*lit, expr, ctx)?
                {
                    concrete
                } else {
                    let key = literal_extract_key(*lit, self.options.max_precision);
                    if let Some(name) = self.extracted_literals.get(&key) {
                        name.clone()
                    } else {
                        key.expr_text
                    }
                }
            }
            E::Constant(h) => {
                let c = &self.module.constants[*h];
                if c.name.is_some() {
                    self.constant_names[h.index()].clone()
                } else {
                    if let naga::Expression::Literal(lit) = &self.module.global_expressions[c.init]
                    {
                        if let Some(concrete) =
                            self.concretize_abstract_literal_for_expr(*lit, expr, ctx)?
                        {
                            concrete
                        } else {
                            // Mirror the Literal arm: an unnamed constant
                            // whose init is a literal is emitted as the
                            // literal text at every use site, so the same
                            // shared-const substitution applies here.
                            let key = literal_extract_key(*lit, self.options.max_precision);
                            if let Some(name) = self.extracted_literals.get(&key) {
                                name.clone()
                            } else {
                                self.emit_global_expr(c.init)?
                            }
                        }
                    } else {
                        self.emit_global_expr(c.init)?
                    }
                }
            }
            E::Override(h) => self.override_names[h.index()].clone(),
            E::ZeroValue(ty) => self.zero_value(*ty)?,
            E::Compose { ty, components } => 'compose: {
                // For vector composes, try to emit as a bare swizzle
                // (e.g. vec3f(v.x,v.y,v.z) -> v.xyz), eliminating the
                // type constructor entirely.
                if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
                    if let Some(swizzle) = self.try_compose_as_full_swizzle(components, ctx)? {
                        break 'compose swizzle;
                    }
                }

                let mut s = String::new();
                s.push_str(&self.type_ref(*ty)?);
                s.push('(');
                // Collapse vector Compose with all-identical scalar components
                // into splat form: vec3f(x,x,x) -> vec3f(x).
                let is_splat = matches!(
                    self.module.types[*ty].inner,
                    naga::TypeInner::Vector { size, .. }
                        if components.len() == size as usize
                            && components.len() > 1
                ) && compose_is_splat(components, &ctx.func.expressions);
                if is_splat {
                    s.push_str(&self.emit_constructor_arg(components[0], ctx)?);
                } else if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. })
                    && self.emit_compose_grouped(&mut s, components, ctx)?
                {
                    // Partial swizzle grouping applied
                    // (e.g. vec4f(v.x,v.y,0.,1.) -> vec4f(v.xy,0.,1.))
                } else {
                    let sep = self.comma_sep();
                    for (i, c) in components.iter().enumerate() {
                        if i > 0 {
                            s.push_str(sep);
                        }
                        s.push_str(&self.emit_constructor_arg(*c, ctx)?);
                    }
                }
                s.push(')');
                s
            }
            E::Access { base, index } => {
                let mut s = self.emit_postfix_base(*base, ctx)?;
                s.push('[');
                s.push_str(&self.emit_expr(*index, ctx)?);
                s.push(']');
                s
            }
            E::AccessIndex { base, index } => {
                let mut s = self.emit_postfix_base(*base, ctx)?;
                if let Some(field_name) = self.struct_field_name(*base, *index, ctx) {
                    s.push('.');
                    s.push_str(&field_name);
                } else if let Some(c) = self.vector_component_name(*base, *index, ctx) {
                    s.push('.');
                    s.push(c);
                } else {
                    s.push('[');
                    s.push_str(&index.to_string());
                    s.push(']');
                }
                s
            }
            E::Splat { size: _, value } => {
                let target_ty = self.expr_type_name(expr, ctx)?;
                let lane = self.emit_constructor_arg(*value, ctx)?;
                {
                    let mut s = target_ty;
                    s.push('(');
                    s.push_str(&lane);
                    s.push(')');
                    s
                }
            }
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let mut s = self.emit_postfix_base(*vector, ctx)?;
                s.push('.');
                let n = *size as u8 as usize;
                for c in &pattern[..n] {
                    s.push(match c {
                        naga::SwizzleComponent::X => 'x',
                        naga::SwizzleComponent::Y => 'y',
                        naga::SwizzleComponent::Z => 'z',
                        naga::SwizzleComponent::W => 'w',
                    });
                }
                s
            }
            E::FunctionArgument(i) => ctx.argument_names[*i as usize].clone(),
            E::GlobalVariable(h) => self.global_names[h.index()].clone(),
            E::LocalVariable(h) => ctx.local_names[h].clone(),
            E::Load { pointer } => self.emit_lvalue(*pointer, ctx)?,
            E::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge,
            } => {
                let sep = self.comma_sep();
                let mut s = String::new();

                if let Some(component) = gather {
                    // textureGather / textureGatherCompare
                    let suffix = if depth_ref.is_some() { "Compare" } else { "" };
                    s.push_str("textureGather");
                    s.push_str(suffix);
                    s.push('(');
                    // For non-depth textures, the component index comes first.
                    if depth_ref.is_none() {
                        let is_depth = matches!(
                            ctx.info[*image].ty.inner_with(&self.module.types),
                            naga::TypeInner::Image {
                                class: naga::ImageClass::Depth { .. },
                                ..
                            }
                        );
                        if !is_depth {
                            s.push_str(&(*component as u8).to_string());
                            s.push_str(sep);
                        }
                    }
                    s.push_str(&self.emit_expr(*image, ctx)?);
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*sampler, ctx)?);
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*coordinate, ctx)?);
                    if let Some(ai) = array_index {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*ai, ctx)?);
                    }
                    if let Some(dr) = depth_ref {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*dr, ctx)?);
                    }
                    if let Some(off) = offset {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*off, ctx)?);
                    }
                    s.push(')');
                } else {
                    // textureSample variants
                    let fn_name = match (depth_ref.is_some(), level, clamp_to_edge) {
                        (false, naga::SampleLevel::Zero, true) => "textureSampleBaseClampToEdge",
                        (false, naga::SampleLevel::Auto, _) => "textureSample",
                        (false, naga::SampleLevel::Zero, _) => "textureSampleLevel",
                        (false, naga::SampleLevel::Exact(_), _) => "textureSampleLevel",
                        (false, naga::SampleLevel::Bias(_), _) => "textureSampleBias",
                        (false, naga::SampleLevel::Gradient { .. }, _) => "textureSampleGrad",
                        (true, naga::SampleLevel::Auto, _) => "textureSampleCompare",
                        (true, naga::SampleLevel::Zero, _)
                        | (true, naga::SampleLevel::Exact(_), _) => "textureSampleCompareLevel",
                        _ => {
                            return Err(Error::Emit(format!(
                                "unsupported sampling mode in function '{}' \
                                 (depth_ref={depth_ref:?}, level={level:?})",
                                ctx.display_name,
                            )));
                        }
                    };
                    s.push_str(fn_name);
                    s.push('(');
                    s.push_str(&self.emit_expr(*image, ctx)?);
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*sampler, ctx)?);
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*coordinate, ctx)?);
                    if let Some(ai) = array_index {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*ai, ctx)?);
                    }
                    if let Some(dr) = depth_ref {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*dr, ctx)?);
                    }
                    match level {
                        naga::SampleLevel::Auto => {}
                        naga::SampleLevel::Zero => {
                            // textureSampleLevel needs explicit 0;
                            // textureSampleBaseClampToEdge and
                            // textureSampleCompareLevel do not.
                            if !clamp_to_edge && depth_ref.is_none() {
                                s.push_str(sep);
                                s.push('0');
                            }
                        }
                        naga::SampleLevel::Exact(h) => {
                            // textureSampleCompareLevel doesn't take a level
                            // parameter - it's always level 0 implicitly.
                            if depth_ref.is_none() {
                                s.push_str(sep);
                                s.push_str(&self.emit_expr(*h, ctx)?);
                            }
                        }
                        naga::SampleLevel::Bias(h) => {
                            s.push_str(sep);
                            s.push_str(&self.emit_expr(*h, ctx)?);
                        }
                        naga::SampleLevel::Gradient { x, y } => {
                            s.push_str(sep);
                            s.push_str(&self.emit_expr(*x, ctx)?);
                            s.push_str(sep);
                            s.push_str(&self.emit_expr(*y, ctx)?);
                        }
                    }
                    if let Some(off) = offset {
                        s.push_str(sep);
                        s.push_str(&self.emit_expr(*off, ctx)?);
                    }
                    s.push(')');
                }
                s
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let sep = self.comma_sep();
                let mut s = String::from("textureLoad(");
                s.push_str(&self.emit_expr(*image, ctx)?);
                s.push_str(sep);
                s.push_str(&self.emit_expr(*coordinate, ctx)?);
                if let Some(ai) = array_index {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*ai, ctx)?);
                }
                if let Some(sample) = sample {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*sample, ctx)?);
                } else if let Some(level) = level {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*level, ctx)?);
                }
                s.push(')');
                s
            }
            E::ImageQuery { image, query } => {
                let sep = self.comma_sep();
                let mut s = String::new();
                match query {
                    naga::ImageQuery::Size { level } => {
                        s.push_str("textureDimensions(");
                        s.push_str(&self.emit_expr(*image, ctx)?);
                        if let Some(level) = level {
                            s.push_str(sep);
                            s.push_str(&self.emit_expr(*level, ctx)?);
                        }
                        s.push(')');
                    }
                    naga::ImageQuery::NumLevels => {
                        s.push_str("textureNumLevels(");
                        s.push_str(&self.emit_expr(*image, ctx)?);
                        s.push(')');
                    }
                    naga::ImageQuery::NumLayers => {
                        s.push_str("textureNumLayers(");
                        s.push_str(&self.emit_expr(*image, ctx)?);
                        s.push(')');
                    }
                    naga::ImageQuery::NumSamples => {
                        s.push_str("textureNumSamples(");
                        s.push_str(&self.emit_expr(*image, ctx)?);
                        s.push(')');
                    }
                }
                s
            }
            E::Unary { op, expr } => {
                let op_str = match op {
                    naga::UnaryOperator::Negate => "-",
                    naga::UnaryOperator::LogicalNot => "!",
                    naga::UnaryOperator::BitwiseNot => "~",
                };
                let cached = ctx.expr_names.contains_key(expr);
                let wrap = unary_child_needs_parens(*expr, &ctx.func.expressions, cached);
                let mut s = self.emit_expr(*expr, ctx)?;
                if wrap {
                    s.insert(0, '(');
                    s.push(')');
                }
                s.insert_str(0, op_str);
                s
            }
            E::Binary { op, left, right } => {
                let op_str = binary_op_str(*op);
                let sp = self.bin_op_sep();
                let arena = &ctx.func.expressions;
                let lc = ctx.expr_names.contains_key(left);
                let rc = ctx.expr_names.contains_key(right);

                // Splat elision: in arithmetic binary ops, replace an uncached
                // Splat / splat-Compose operand with the bare scalar value.
                // WGSL defines mixed scalar-vector overloads for +, -, *, /, %.
                // Elide at most one side to keep the result a vector type.
                // Guard: the OTHER operand must be a vector; if both sides
                // are scalar after elision the result type changes.
                let is_arith = is_arithmetic_op(*op);
                let left_scalar = if is_arith {
                    self.try_splat_scalar(*left, arena, lc)
                } else {
                    None
                };
                let right_scalar = if is_arith {
                    self.try_splat_scalar(*right, arena, rc)
                } else {
                    None
                };
                // Check whether each operand resolves to a vector type.
                let left_is_vec = matches!(
                    ctx.info[*left].ty.inner_with(&self.module.types),
                    naga::TypeInner::Vector { .. }
                );
                let right_is_vec = matches!(
                    ctx.info[*right].ty.inner_with(&self.module.types),
                    naga::TypeInner::Vector { .. }
                );
                let (elide_l, elide_r) = match (left_scalar.is_some(), right_scalar.is_some()) {
                    // Both splats: elide right; left (also a splat) stays as the
                    // vector operand so the result type remains a vector.
                    (true, true) => (false, true),
                    // Only left is splat: elide only if right is a vector.
                    (true, false) => (right_is_vec, false),
                    // Only right is splat: elide only if left is a vector.
                    (false, true) => (false, left_is_vec),
                    (false, false) => (false, false),
                };

                let (eff_l, eff_lc) = if elide_l {
                    let h = left_scalar.unwrap();
                    (h, ctx.expr_names.contains_key(&h))
                } else {
                    (*left, lc)
                };
                let (eff_r, eff_rc) = if elide_r {
                    let h = right_scalar.unwrap();
                    (h, ctx.expr_names.contains_key(&h))
                } else {
                    (*right, rc)
                };

                let wrap_l = child_needs_parens(eff_l, arena, *op, false, eff_lc);
                let mut wrap_r = child_needs_parens(eff_r, arena, *op, true, eff_rc);

                let ls = if elide_l {
                    self.emit_constructor_arg(left_scalar.unwrap(), ctx)?
                } else {
                    self.emit_expr(*left, ctx)?
                };
                let rs = if elide_r {
                    let s = self.emit_constructor_arg(right_scalar.unwrap(), ctx)?;
                    // Prevent ambiguous '--' token when subtracting a negative
                    // scalar in minified (no-space) mode.
                    if !wrap_r
                        && sp.is_empty()
                        && matches!(op, naga::BinaryOperator::Subtract)
                        && s.starts_with('-')
                    {
                        wrap_r = true;
                    }
                    s
                } else {
                    self.emit_expr(*right, ctx)?
                };
                assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r)
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let sep = self.comma_sep();
                let mut s = String::from("select(");

                // For select(reject, accept, condition), both reject and accept must have
                // the same concrete type.  If either is a literal, we must use a type-suffixed
                // form (literal_to_wgsl) to ensure type matching with the other argument.
                let reject_str =
                    if let naga::Expression::Literal(lit) = &ctx.func.expressions[*reject] {
                        literal_to_wgsl(*lit, self.options.max_precision)
                    } else {
                        self.emit_expr(*reject, ctx)?
                    };
                s.push_str(&reject_str);
                s.push_str(sep);

                let accept_str =
                    if let naga::Expression::Literal(lit) = &ctx.func.expressions[*accept] {
                        literal_to_wgsl(*lit, self.options.max_precision)
                    } else {
                        self.emit_expr(*accept, ctx)?
                    };
                s.push_str(&accept_str);
                s.push_str(sep);

                s.push_str(&self.emit_expr(*condition, ctx)?);
                s.push(')');
                s
            }
            E::Derivative { axis, ctrl, expr } => {
                let name = match (axis, ctrl) {
                    (naga::DerivativeAxis::X, naga::DerivativeControl::None) => "dpdx",
                    (naga::DerivativeAxis::X, naga::DerivativeControl::Coarse) => "dpdxCoarse",
                    (naga::DerivativeAxis::X, naga::DerivativeControl::Fine) => "dpdxFine",
                    (naga::DerivativeAxis::Y, naga::DerivativeControl::None) => "dpdy",
                    (naga::DerivativeAxis::Y, naga::DerivativeControl::Coarse) => "dpdyCoarse",
                    (naga::DerivativeAxis::Y, naga::DerivativeControl::Fine) => "dpdyFine",
                    (naga::DerivativeAxis::Width, naga::DerivativeControl::None) => "fwidth",
                    (naga::DerivativeAxis::Width, naga::DerivativeControl::Coarse) => {
                        "fwidthCoarse"
                    }
                    (naga::DerivativeAxis::Width, naga::DerivativeControl::Fine) => "fwidthFine",
                };
                {
                    let mut s = String::from(name);
                    s.push('(');
                    // Derivative builtins require float input.  If an uncached
                    // literal reaches here, emit the typed token form (e.g. `1f`)
                    // instead of the bare shortest form (`1`) to avoid i32
                    // inference and validation fallback.
                    if !ctx.expr_names.contains_key(expr) {
                        if let naga::Expression::Literal(lit) = ctx.func.expressions[*expr] {
                            s.push_str(&literal_to_wgsl(lit, self.options.max_precision));
                        } else {
                            s.push_str(&self.emit_expr(*expr, ctx)?);
                        }
                    } else {
                        s.push_str(&self.emit_expr(*expr, ctx)?);
                    }
                    s.push(')');
                    s
                }
            }
            E::Relational { fun, argument } => {
                let name = match fun {
                    naga::RelationalFunction::All => "all",
                    naga::RelationalFunction::Any => "any",
                    naga::RelationalFunction::IsNan => "isNan",
                    naga::RelationalFunction::IsInf => "isInf",
                };
                let mut s = String::from(name);
                s.push('(');
                s.push_str(&self.emit_expr(*argument, ctx)?);
                s.push(')');
                s
            }
            E::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let sep = self.comma_sep();
                let mut s = String::new();
                s.push_str(math_name(*fun));
                s.push('(');
                s.push_str(&self.emit_expr(*arg, ctx)?);
                if let Some(v) = arg1 {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*v, ctx)?);
                }
                if let Some(v) = arg2 {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*v, ctx)?);
                }
                if let Some(v) = arg3 {
                    s.push_str(sep);
                    s.push_str(&self.emit_expr(*v, ctx)?);
                }
                s.push(')');
                s
            }
            E::As {
                expr,
                kind,
                convert,
            } => {
                let src_inner = ctx.info[*expr].ty.inner_with(&self.module.types);
                // Matrix-to-matrix element-type conversion: mat2x2<f16> -> mat2x2<f32>
                // emits as `matCxR<T>(source)` (type-constructor form).
                if let naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar: src_scalar,
                } = src_inner
                {
                    let target_width = convert.unwrap_or(src_scalar.width);
                    let target_inner = naga::TypeInner::Matrix {
                        columns: *columns,
                        rows: *rows,
                        scalar: naga::Scalar {
                            kind: *kind,
                            width: target_width,
                        },
                    };
                    let target = self.type_name_for_inner(&target_inner)?;
                    let source = self.emit_expr(*expr, ctx)?;
                    let mut s = if convert.is_some() {
                        target
                    } else {
                        let mut s = String::from("bitcast<");
                        s.push_str(&target);
                        s.push('>');
                        s
                    };
                    s.push('(');
                    s.push_str(&source);
                    s.push(')');
                    return Ok(s);
                }
                let (vec_size, src_width) = match src_inner {
                    naga::TypeInner::Scalar(s) => (None, s.width),
                    naga::TypeInner::Vector { size, scalar } => (Some(*size), scalar.width),
                    _ => {
                        return Err(Error::Emit(format!(
                            "unsupported cast source type in function '{}': {:?}",
                            ctx.display_name, src_inner,
                        )));
                    }
                };
                let target_width = convert.unwrap_or(src_width);
                let target_inner = match vec_size {
                    Some(size) => naga::TypeInner::Vector {
                        size,
                        scalar: naga::Scalar {
                            kind: *kind,
                            width: target_width,
                        },
                    },
                    None => naga::TypeInner::Scalar(naga::Scalar {
                        kind: *kind,
                        width: target_width,
                    }),
                };
                let target = self.type_name_for_inner(&target_inner)?;
                let source = self.emit_expr(*expr, ctx)?;
                let mut s = if convert.is_some() {
                    target
                } else {
                    let mut s = String::from("bitcast<");
                    s.push_str(&target);
                    s.push('>');
                    s
                };
                s.push('(');
                s.push_str(&source);
                s.push(')');
                s
            }
            E::CallResult(_) => ctx
                .expr_names
                .get(&expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult
            | E::RayQueryGetIntersection { .. } => ctx
                .expr_names
                .get(&expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            E::ArrayLength(e) => {
                let mut s = String::from("arrayLength(&");
                s.push_str(&self.emit_expr(*e, ctx)?);
                s.push(')');
                s
            }
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported expression in function '{}' (expr {}): {:?}",
                    ctx.display_name,
                    expr.index(),
                    ctx.func.expressions[expr],
                )));
            }
        })
    }

    fn concretize_abstract_literal_for_expr(
        &self,
        lit: naga::Literal,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        let inner = ctx.info[expr].ty.inner_with(&self.module.types);
        Ok(match concretize_abstract_literal_via_inner(lit, inner) {
            // The pure projection produced a regular concrete literal.
            // Both `count_literals` (scan side) and this emission path
            // key on the *concrete* form via [`literal_extract_key`], so
            // shared-literal extraction substitutes when the concretized
            // form was hot enough across the module.
            //
            // Falls back to the typed form (`literal_to_wgsl`) rather than
            // the bare form: a free-standing abstract literal that gets
            // concretized must keep its type pinned at the use site,
            // otherwise downstream WGSL coercion could re-derive a
            // different concrete type.
            Some(ConcretizedAbstract::Lit(concrete)) => {
                let key = literal_extract_key(concrete, self.options.max_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    Some(name.clone())
                } else {
                    Some(literal_to_wgsl(concrete, self.options.max_precision))
                }
            }
            // Pre-built text form (e.g. `f16(0.5f)`, `i32(<huge>)`) cannot
            // be substituted via `extracted_literals`.  `count_literals`
            // mirrors this branch and skips counting these on the scan
            // side, so the alignment is preserved.
            Some(ConcretizedAbstract::Text(text)) => Some(text),
            // Not an abstract literal (or no projection possible from the
            // resolved type): caller falls back to its existing
            // non-abstract path.
            None => None,
        })
    }

    // MARK: Global expression emission

    /// Emit a global (module-scope) expression.  Reuses most of the
    /// [`emit_expr_uncached`] logic but resolves handles through the
    /// module's global-expression arena and emits `Constant` handles
    /// as their declared name when mangling is in effect.
    pub(super) fn emit_global_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
    ) -> Result<String, Error> {
        // If this expression handle is the init of a previously-emitted named
        // constant, emit the constant's name instead of re-inlining the value.
        // This is populated incrementally during constant emission (see
        // module_emit.rs), so only earlier constants are in the map -
        // preventing self-referential `const X = X;` cycles.
        if let Some(&ch) = self.expr_to_const.get(&expr) {
            return Ok(self.constant_names[ch.index()].clone());
        }
        use naga::Expression as E;
        let arena = &self.module.global_expressions;
        Ok(match &arena[expr] {
            E::Literal(lit) => literal_to_wgsl(*lit, self.options.max_precision),
            E::Constant(h) => {
                let c = &self.module.constants[*h];
                if c.name.is_some() {
                    self.constant_names[h.index()].clone()
                } else {
                    self.emit_global_expr(c.init)?
                }
            }
            E::Override(h) => self.override_names[h.index()].clone(),
            E::ZeroValue(ty) => self.zero_value(*ty)?,
            E::Splat { size, value } => {
                // Determine the scalar type from the value expression.
                let scalar = match &arena[*value] {
                    E::Literal(lit) => lit.scalar(),
                    E::Constant(h) => {
                        match &self.module.types[self.module.constants[*h].ty].inner {
                            naga::TypeInner::Scalar(s) => *s,
                            _ => {
                                return Err(Error::Emit(
                                    "splat value in global expression must be scalar".into(),
                                ));
                            }
                        }
                    }
                    E::Override(h) => {
                        match &self.module.types[self.module.overrides[*h].ty].inner {
                            naga::TypeInner::Scalar(s) => *s,
                            _ => {
                                return Err(Error::Emit(
                                    "splat value in global expression must be scalar".into(),
                                ));
                            }
                        }
                    }
                    E::ZeroValue(ty) => match &self.module.types[*ty].inner {
                        naga::TypeInner::Scalar(s) => *s,
                        _ => {
                            return Err(Error::Emit(
                                "splat value in global expression must be scalar".into(),
                            ));
                        }
                    },
                    other => {
                        return Err(Error::Emit(format!(
                            "unsupported splat value in global expression: {other:?}",
                        )));
                    }
                };
                // Look up the vector type in the arena to pick up any alias.
                let type_name = self.type_name_for_inner(&naga::TypeInner::Vector {
                    size: *size,
                    scalar,
                })?;
                let mut s = format!("{type_name}(");
                if let E::Literal(lit) = &arena[*value] {
                    s.push_str(&literal_to_wgsl_bare(*lit, self.options.max_precision));
                } else {
                    s.push_str(&self.emit_global_expr(*value)?);
                }
                s.push(')');
                s
            }
            E::Compose { ty, components } => {
                let mut s = String::new();
                s.push_str(&self.type_ref(*ty)?);
                s.push('(');
                // Collapse vector Compose with all-identical scalar components
                // into splat form: vec3f(x,x,x) -> vec3f(x).
                let is_splat = matches!(
                    self.module.types[*ty].inner,
                    naga::TypeInner::Vector { size, .. }
                        if components.len() == size as usize
                            && components.len() > 1
                ) && compose_is_splat(components, arena);
                if is_splat {
                    if let naga::Expression::Literal(lit) = &arena[components[0]] {
                        s.push_str(&literal_to_wgsl_bare(*lit, self.options.max_precision));
                    } else {
                        s.push_str(&self.emit_global_expr(components[0])?);
                    }
                } else {
                    let sep = self.comma_sep();
                    for (i, c) in components.iter().enumerate() {
                        if i > 0 {
                            s.push_str(sep);
                        }
                        if let naga::Expression::Literal(lit) = &arena[*c] {
                            s.push_str(&literal_to_wgsl_bare(*lit, self.options.max_precision));
                        } else {
                            s.push_str(&self.emit_global_expr(*c)?);
                        }
                    }
                }
                s.push(')');
                s
            }
            E::Unary { op, expr } => {
                let op_str = match op {
                    naga::UnaryOperator::Negate => "-",
                    naga::UnaryOperator::LogicalNot => "!",
                    naga::UnaryOperator::BitwiseNot => "~",
                };
                let wrap = unary_child_needs_parens(*expr, arena, false);
                let mut s = self.emit_global_expr(*expr)?;
                if wrap {
                    s.insert(0, '(');
                    s.push(')');
                }
                s.insert_str(0, op_str);
                s
            }
            E::Binary { op, left, right } => {
                let op_str = binary_op_str(*op);
                let sp = self.bin_op_sep();
                let wrap_l = child_needs_parens(*left, arena, *op, false, false);
                let wrap_r = child_needs_parens(*right, arena, *op, true, false);
                let ls = self.emit_global_expr(*left)?;
                let rs = self.emit_global_expr(*right)?;
                assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r)
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let sep = self.comma_sep();
                let mut s = String::from("select(");

                // For select(reject, accept, condition), both reject and accept must have
                // the same concrete type.  If either is a literal, we must use a type-suffixed
                // form (literal_to_wgsl) to ensure type matching with the other argument.
                let reject_str = if let naga::Expression::Literal(lit) = &arena[*reject] {
                    literal_to_wgsl(*lit, self.options.max_precision)
                } else {
                    self.emit_global_expr(*reject)?
                };
                s.push_str(&reject_str);
                s.push_str(sep);

                let accept_str = if let naga::Expression::Literal(lit) = &arena[*accept] {
                    literal_to_wgsl(*lit, self.options.max_precision)
                } else {
                    self.emit_global_expr(*accept)?
                };
                s.push_str(&accept_str);
                s.push_str(sep);

                s.push_str(&self.emit_global_expr(*condition)?);
                s.push(')');
                s
            }
            E::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let sep = self.comma_sep();
                let mut s = String::new();
                s.push_str(math_name(*fun));
                s.push('(');
                s.push_str(&self.emit_global_expr(*arg)?);
                if let Some(v) = arg1 {
                    s.push_str(sep);
                    s.push_str(&self.emit_global_expr(*v)?);
                }
                if let Some(v) = arg2 {
                    s.push_str(sep);
                    s.push_str(&self.emit_global_expr(*v)?);
                }
                if let Some(v) = arg3 {
                    s.push_str(sep);
                    s.push_str(&self.emit_global_expr(*v)?);
                }
                s.push(')');
                s
            }
            E::As {
                expr: inner,
                kind,
                convert,
            } => {
                let src_inner = self.info[*inner].inner_with(&self.module.types);
                // Matrix-to-matrix element-type conversion: mat2x2<f16> -> mat2x2<f32>
                // emits as `matCxR<T>(source)` (type-constructor form).
                if let naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar: src_scalar,
                } = src_inner
                {
                    let target_width = convert.unwrap_or(src_scalar.width);
                    let target_inner = naga::TypeInner::Matrix {
                        columns: *columns,
                        rows: *rows,
                        scalar: naga::Scalar {
                            kind: *kind,
                            width: target_width,
                        },
                    };
                    let target = self.type_name_for_inner(&target_inner)?;
                    let source = self.emit_global_expr(*inner)?;
                    let mut s = if convert.is_some() {
                        target
                    } else {
                        let mut s = String::from("bitcast<");
                        s.push_str(&target);
                        s.push('>');
                        s
                    };
                    s.push('(');
                    s.push_str(&source);
                    s.push(')');
                    s
                } else {
                    let (vec_size, src_width) = match src_inner {
                        naga::TypeInner::Scalar(s) => (None, s.width),
                        naga::TypeInner::Vector { size, scalar } => (Some(*size), scalar.width),
                        _ => {
                            return Err(Error::Emit(format!(
                                "unsupported cast source type in global expression {}: {:?}",
                                inner.index(),
                                src_inner,
                            )));
                        }
                    };
                    let target_width = convert.unwrap_or(src_width);
                    let target_inner = match vec_size {
                        Some(size) => naga::TypeInner::Vector {
                            size,
                            scalar: naga::Scalar {
                                kind: *kind,
                                width: target_width,
                            },
                        },
                        None => naga::TypeInner::Scalar(naga::Scalar {
                            kind: *kind,
                            width: target_width,
                        }),
                    };
                    let target = self.type_name_for_inner(&target_inner)?;
                    let source = self.emit_global_expr(*inner)?;
                    let mut s = if convert.is_some() {
                        target
                    } else {
                        let mut s = String::from("bitcast<");
                        s.push_str(&target);
                        s.push('>');
                        s
                    };
                    s.push('(');
                    s.push_str(&source);
                    s.push(')');
                    s
                }
            }
            E::Access { base, index } => {
                let mut s = self.emit_global_postfix_base(*base)?;
                s.push('[');
                s.push_str(&self.emit_global_expr(*index)?);
                s.push(']');
                s
            }
            E::AccessIndex { base, index } => {
                let mut s = self.emit_global_postfix_base(*base)?;
                if let Some(name) = self.global_struct_field_name(*base, *index) {
                    s.push('.');
                    s.push_str(&name);
                } else if let Some(c) = self.global_vector_component_name(*base, *index) {
                    s.push('.');
                    s.push(c);
                } else {
                    s.push('[');
                    s.push_str(&index.to_string());
                    s.push(']');
                }
                s
            }
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let mut s = self.emit_global_postfix_base(*vector)?;
                s.push('.');
                let n = *size as u8 as usize;
                for c in &pattern[..n] {
                    s.push(match c {
                        naga::SwizzleComponent::X => 'x',
                        naga::SwizzleComponent::Y => 'y',
                        naga::SwizzleComponent::Z => 'z',
                        naga::SwizzleComponent::W => 'w',
                    });
                }
                s
            }
            E::Relational { fun, argument } => {
                let name = match fun {
                    naga::RelationalFunction::All => "all",
                    naga::RelationalFunction::Any => "any",
                    naga::RelationalFunction::IsNan => "isNan",
                    naga::RelationalFunction::IsInf => "isInf",
                };
                let mut s = String::from(name);
                s.push('(');
                s.push_str(&self.emit_global_expr(*argument)?);
                s.push(')');
                s
            }
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported global expression (expr {}): {:?}",
                    expr.index(),
                    arena[expr],
                )));
            }
        })
    }

    // MARK: Type helpers

    /// Render a [`naga::TypeInner`] to its WGSL type-name string via
    /// [`super::syntax::type_inner_name`], consulting the generator's
    /// alias table along the way.
    pub(super) fn type_name_for_inner(&self, inner: &naga::TypeInner) -> Result<String, Error> {
        let res = naga::proc::TypeResolution::Value(inner.clone());
        type_resolution_name(&res, self.module, &self.type_names, &self.override_names)
    }

    /// Render the declared WGSL type of `expr` using the module's
    /// cached type-resolution info.
    pub(super) fn expr_type_name(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        let res = &ctx.info[expr].ty;
        type_resolution_name(res, self.module, &self.type_names, &self.override_names)
    }

    /// Look up the WGSL type-name string for a type handle,
    /// honouring alias substitution when enabled.
    pub(super) fn type_ref(&self, ty: naga::Handle<naga::Type>) -> Result<String, Error> {
        if let Some(name) = self.type_names.get(&ty) {
            return Ok(name.clone());
        }
        type_inner_name(
            &self.module.types[ty].inner,
            self.module,
            &self.type_names,
            &self.override_names,
        )
    }

    /// Emit the shortest WGSL expression that evaluates to the
    /// zero value of `ty`.  Prefers scalar literals (`0`, `false`)
    /// over wrapper syntax (`vec3f()`, `T()`) when both are legal.
    pub(super) fn zero_value(&self, ty: naga::Handle<naga::Type>) -> Result<String, Error> {
        Ok(match &self.module.types[ty].inner {
            naga::TypeInner::Scalar(s) => scalar_zero(s.kind, s.width).to_string(),
            naga::TypeInner::Vector { .. }
            | naga::TypeInner::Matrix { .. }
            | naga::TypeInner::Array { .. }
            | naga::TypeInner::Struct { .. } => format!("{}()", self.type_ref(ty)?),
            _ => format!("{}()", self.type_ref(ty)?),
        })
    }

    /// Emit an expression that is the base of a postfix operation in the
    /// global expression arena (`.member`, `[index]`, `.xyzw`).
    fn emit_global_postfix_base(
        &self,
        base: naga::Handle<naga::Expression>,
    ) -> Result<String, Error> {
        let needs_parens = matches!(
            self.module.global_expressions[base],
            naga::Expression::Binary { .. }
                | naga::Expression::Unary { .. }
                | naga::Expression::Select { .. }
        );
        let s = self.emit_global_expr(base)?;
        if needs_parens {
            Ok(format!("({s})"))
        } else {
            Ok(s)
        }
    }

    /// Resolve a struct field name for an `AccessIndex` in a global expression.
    fn global_struct_field_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
    ) -> Option<String> {
        use naga::proc::TypeResolution;

        let resolution = &self.info[base];
        let (ty_handle, members) = match resolution {
            TypeResolution::Handle(h) => match &self.module.types[*h].inner {
                naga::TypeInner::Struct { members, .. } => (Some(*h), members),
                naga::TypeInner::Pointer { base: bty, .. } => {
                    match &self.module.types[*bty].inner {
                        naga::TypeInner::Struct { members, .. } => (Some(*bty), members),
                        _ => return None,
                    }
                }
                _ => return None,
            },
            TypeResolution::Value(inner) => match inner {
                naga::TypeInner::Pointer { base: bty, .. } => {
                    match &self.module.types[*bty].inner {
                        naga::TypeInner::Struct { members, .. } => (Some(*bty), members),
                        _ => return None,
                    }
                }
                _ => return None,
            },
        };

        if let Some(h) = ty_handle {
            if let Some(mangled) = self.member_names.get(&(h, index)) {
                return Some(mangled.clone());
            }
        }

        members
            .get(index as usize)
            .map(|m| m.name.clone().unwrap_or_else(|| format!("m{}", index)))
    }

    /// If the base expression is a vector (or pointer-to-vector) and the
    /// index is 0-3, return the corresponding WGSL component letter.
    fn global_vector_component_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
    ) -> Option<char> {
        const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];
        if index > 3 {
            return None;
        }
        let inner = self.info[base].inner_with(&self.module.types);
        let is_vec = matches!(
            inner,
            naga::TypeInner::Vector { .. } | naga::TypeInner::ValuePointer { size: Some(_), .. }
        ) || matches!(inner, naga::TypeInner::Pointer { base: bty, .. } if matches!(
            self.module.types[*bty].inner,
            naga::TypeInner::Vector { .. }
        ));
        if is_vec {
            Some(COMPONENTS[index as usize])
        } else {
            None
        }
    }

    /// Return the mangled (or preserved) member name at `(base, index)`
    /// when `base` is a struct-typed expression, or `None` when the
    /// access is actually a vector swizzle position.
    pub(super) fn struct_field_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<String> {
        use naga::proc::TypeResolution;

        // Extract the struct handle and members directly from the
        // TypeResolution, avoiding an O(n) linear scan of the type arena.
        let resolution = &ctx.info[base].ty;
        let (ty_handle, members) = match resolution {
            TypeResolution::Handle(h) => match &self.module.types[*h].inner {
                naga::TypeInner::Struct { members, .. } => (Some(*h), members),
                naga::TypeInner::Pointer { base: bty, .. } => {
                    match &self.module.types[*bty].inner {
                        naga::TypeInner::Struct { members, .. } => (Some(*bty), members),
                        _ => return None,
                    }
                }
                _ => return None,
            },
            TypeResolution::Value(inner) => match inner {
                naga::TypeInner::Pointer { base: bty, .. } => {
                    match &self.module.types[*bty].inner {
                        naga::TypeInner::Struct { members, .. } => (Some(*bty), members),
                        _ => return None,
                    }
                }
                _ => return None,
            },
        };

        // If we have a mangled member name, use it.
        if let Some(h) = ty_handle {
            if let Some(mangled) = self.member_names.get(&(h, index)) {
                return Some(mangled.clone());
            }
        }

        members
            .get(index as usize)
            .map(|m| m.name.clone().unwrap_or_else(|| format!("m{}", index)))
    }

    /// If the base expression is a vector (or pointer-to-vector) and the
    /// index is 0-3, return the corresponding WGSL component letter
    /// (`x`, `y`, `z`, `w`).  This produces `.x` instead of `[0]`,
    /// saving one byte per access.
    fn vector_component_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<char> {
        const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];
        if index > 3 {
            return None;
        }
        let inner = ctx.info[base].ty.inner_with(&self.module.types);
        let is_vec = matches!(
            inner,
            naga::TypeInner::Vector { .. } | naga::TypeInner::ValuePointer { size: Some(_), .. }
        ) || matches!(inner, naga::TypeInner::Pointer { base: bty, .. } if matches!(
            self.module.types[*bty].inner,
            naga::TypeInner::Vector { .. }
        ));
        if is_vec {
            Some(COMPONENTS[index as usize])
        } else {
            None
        }
    }

    /// WGSL component letters for swizzle patterns.
    const SWIZZLE_LETTERS: [char; 4] = ['x', 'y', 'z', 'w'];

    /// Check if a Compose component can participate in swizzle grouping.
    ///
    /// Recognises two patterns:
    ///   A. `AccessIndex { base: <vector_value>, index }` - function args, etc.
    ///   B. `Load { pointer: AccessIndex { base: <ptr_to_vector>, index } }` -
    ///      local/global variables accessed through a pointer.
    ///
    /// Returns `(base_handle, component_index)` on success.
    fn swizzle_component(
        &self,
        handle: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<(naga::Handle<naga::Expression>, u32)> {
        if ctx.expr_names.contains_key(&handle) {
            return None;
        }
        let arena = &ctx.func.expressions;

        // Pattern A: direct AccessIndex on a vector value.
        if let naga::Expression::AccessIndex { base, index } = arena[handle] {
            if index <= 3 {
                let inner = ctx.info[base].ty.inner_with(&self.module.types);
                if matches!(inner, naga::TypeInner::Vector { .. }) {
                    return Some((base, index));
                }
            }
        }

        // Pattern B: Load { pointer: AccessIndex { base: ptr_to_vec, index } }
        if let naga::Expression::Load { pointer } = arena[handle] {
            if let naga::Expression::AccessIndex { base, index } = arena[pointer] {
                if index <= 3 {
                    let inner = ctx.info[base].ty.inner_with(&self.module.types);
                    let is_ptr_to_vec =
                        matches!(
                            inner,
                            naga::TypeInner::Pointer { base: bty, .. }
                                if matches!(
                                    self.module.types[*bty].inner,
                                    naga::TypeInner::Vector { .. }
                                )
                        ) || matches!(inner, naga::TypeInner::ValuePointer { size: Some(_), .. });
                    if is_ptr_to_vec {
                        return Some((base, index));
                    }
                }
            }
        }

        None
    }

    /// Try to emit a vector Compose entirely as a bare swizzle expression,
    /// eliminating the type constructor.
    ///
    /// `vec3f(v.x, v.y, v.z)` -> `v.xyz`
    /// `vec4f(v.x, v.y, v.z, v.w)` -> `v` (identity on same-size vector)
    fn try_compose_as_full_swizzle(
        &self,
        components: &[naga::Handle<naga::Expression>],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        if components.len() < 2 || components.len() > 4 {
            return Ok(None);
        }

        let mut common_base: Option<naga::Handle<naga::Expression>> = None;
        let mut pattern: Vec<u32> = Vec::with_capacity(components.len());

        for &comp in components {
            if let Some((base, idx)) = self.swizzle_component(comp, ctx) {
                match common_base {
                    None => common_base = Some(base),
                    Some(b) if b == base => {}
                    _ => return Ok(None),
                }
                pattern.push(idx);
            } else {
                return Ok(None);
            }
        }

        let base = common_base.unwrap();

        // Identity check: pattern is [0, 1, ..., N-1] and source vector
        // has exactly N components -> emit just the base expression.
        let source_size = self.vector_size_of(base, ctx);
        if let Some(src_n) = source_size {
            if pattern.len() == src_n && pattern.iter().enumerate().all(|(i, &idx)| idx == i as u32)
            {
                return Ok(Some(self.emit_expr(base, ctx)?));
            }
        }

        // Emit as `base.xyzw`
        let mut s = self.emit_expr(base, ctx)?;
        s.push('.');
        for &idx in &pattern {
            s.push(Self::SWIZZLE_LETTERS[idx as usize]);
        }
        Ok(Some(s))
    }

    /// Emit Compose components with swizzle grouping.
    ///
    /// Consecutive uncached AccessIndex/Load-of-AccessIndex components that
    /// share the same vector base are collapsed into a single swizzle argument
    /// (e.g. `v.x,v.y` -> `v.xy`).
    ///
    /// Returns `Ok(true)` if at least one group was formed and the output
    /// was written to `s`; `Ok(false)` if no grouping was possible.
    fn emit_compose_grouped(
        &self,
        s: &mut String,
        components: &[naga::Handle<naga::Expression>],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<bool, Error> {
        // Build groups of consecutive same-base components.
        let mut groups: Vec<ComposeGroup> = Vec::new();
        let mut i = 0;
        while i < components.len() {
            if let Some((base, idx)) = self.swizzle_component(components[i], ctx) {
                let mut indices = vec![idx];
                let mut j = i + 1;
                while j < components.len() {
                    if let Some((b2, idx2)) = self.swizzle_component(components[j], ctx) {
                        if b2 == base {
                            indices.push(idx2);
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }
                if indices.len() >= 2 {
                    groups.push(ComposeGroup::Swizzle { base, indices });
                    i = j;
                    continue;
                }
            }
            groups.push(ComposeGroup::Single(components[i]));
            i += 1;
        }

        if !groups
            .iter()
            .any(|g| matches!(g, ComposeGroup::Swizzle { .. }))
        {
            return Ok(false);
        }

        let sep = self.comma_sep();
        let mut first = true;
        for group in &groups {
            if !first {
                s.push_str(sep);
            }
            first = false;
            match group {
                ComposeGroup::Swizzle { base, indices } => {
                    s.push_str(&self.emit_expr(*base, ctx)?);
                    s.push('.');
                    for &idx in indices {
                        s.push(Self::SWIZZLE_LETTERS[idx as usize]);
                    }
                }
                ComposeGroup::Single(handle) => {
                    s.push_str(&self.emit_constructor_arg(*handle, ctx)?);
                }
            }
        }

        Ok(true)
    }

    /// Return the number of scalar components in the vector type of `expr`.
    /// Works for both value vectors and pointer-to-vector expressions.
    fn vector_size_of(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<usize> {
        let inner = ctx.info[expr].ty.inner_with(&self.module.types);
        match inner {
            naga::TypeInner::Vector { size, .. } => Some(*size as usize),
            naga::TypeInner::Pointer { base: bty, .. } => {
                if let naga::TypeInner::Vector { size, .. } = self.module.types[*bty].inner {
                    Some(size as usize)
                } else {
                    None
                }
            }
            naga::TypeInner::ValuePointer {
                size: Some(size), ..
            } => Some(*size as usize),
            _ => None,
        }
    }

    // MARK: Scalar hinting

    /// Walk a small prefix of `expr`'s definition and return the
    /// scalar `(kind, width)` the emitter should use when a literal
    /// child needs to be concretised.  Returns `None` when no hint
    /// can be inferred without running a full type resolution.
    pub(super) fn expr_scalar_hint(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        match ctx.info[expr].ty.inner_with(&self.module.types) {
            naga::TypeInner::Scalar(s) => Some(*s),
            naga::TypeInner::Vector { scalar, .. } => Some(*scalar),
            _ => None,
        }
    }

    /// Emit `expr`, but if it's an uncached abstract literal and a scalar
    /// hint is provided, emit the literal concretized to that scalar type.
    /// Emit `expr` while forcing any bare literal it contains to the
    /// supplied scalar hint.  Used when the surrounding operator
    /// requires a specific concrete type (for example `Derivative`
    /// on a bare literal, where naga would otherwise infer `i32`).
    pub(super) fn emit_expr_with_scalar_hint(
        &self,
        expr: naga::Handle<naga::Expression>,
        hint: Option<naga::Scalar>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if !ctx.expr_names.contains_key(&expr) {
            if let Some(scalar) = hint {
                match ctx.func.expressions[expr] {
                    naga::Expression::Binary { op, left, right }
                        if matches!(
                            op,
                            naga::BinaryOperator::Add | naga::BinaryOperator::Subtract
                        ) =>
                    {
                        let arena = &ctx.func.expressions;
                        let lc = ctx.expr_names.contains_key(&left);
                        let rc = ctx.expr_names.contains_key(&right);
                        let wrap_l = child_needs_parens(left, arena, op, false, lc);
                        let wrap_r = child_needs_parens(right, arena, op, true, rc);
                        let ls = self.emit_expr_with_scalar_hint(left, Some(scalar), ctx)?;
                        let rs = self.emit_expr_with_scalar_hint(right, Some(scalar), ctx)?;
                        let op_str = binary_op_str(op);
                        let sp = self.bin_op_sep();
                        return Ok(assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r));
                    }
                    naga::Expression::Literal(lit) => {
                        if let Some(concrete) =
                            self.concretize_abstract_literal_for_scalar(lit, scalar)
                        {
                            return Ok(concrete);
                        }
                    }
                    naga::Expression::Constant(h) => {
                        let c = &self.module.constants[h];
                        if c.name.is_none() {
                            if let naga::Expression::Literal(lit) =
                                self.module.global_expressions[c.init]
                            {
                                if let Some(concrete) =
                                    self.concretize_abstract_literal_for_scalar(lit, scalar)
                                {
                                    return Ok(concrete);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        self.emit_expr(expr, ctx)
    }

    fn concretize_abstract_literal_for_scalar(
        &self,
        lit: naga::Literal,
        target: naga::Scalar,
    ) -> Option<String> {
        use naga::Literal as L;

        let concrete = match lit {
            L::I32(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Uint, 4) if v >= 0 => Some(L::U32(v as u32)),
                (naga::ScalarKind::Sint, 4) => Some(L::I32(v)),
                _ => None,
            },
            L::I64(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Uint, 8) if v >= 0 => Some(L::U64(v as u64)),
                (naga::ScalarKind::Sint, 8) => Some(L::I64(v)),
                _ => None,
            },
            L::U32(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Sint, 4) if v <= i32::MAX as u32 => Some(L::I32(v as i32)),
                (naga::ScalarKind::Uint, 4) => Some(L::U32(v)),
                _ => None,
            },
            L::U64(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Sint, 8) if v <= i64::MAX as u64 => Some(L::I64(v as i64)),
                (naga::ScalarKind::Uint, 8) => Some(L::U64(v)),
                _ => None,
            },
            L::AbstractInt(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Sint, 4) => i32::try_from(v).ok().map(L::I32),
                (naga::ScalarKind::Sint, 8) => Some(L::I64(v)),
                (naga::ScalarKind::Uint, 4) => u32::try_from(v).ok().map(L::U32),
                (naga::ScalarKind::Uint, 8) => u64::try_from(v).ok().map(L::U64),
                (naga::ScalarKind::Float, 2) => return Some(format!("f16({}f)", v as f32)),
                (naga::ScalarKind::Float, 4) => Some(L::F32(v as f32)),
                (naga::ScalarKind::Float, 8) => Some(L::F64(v as f64)),
                _ => None,
            },
            L::AbstractFloat(v) => match (target.kind, target.width) {
                (naga::ScalarKind::Float, 2) => return Some(format!("f16({}f)", v as f32)),
                (naga::ScalarKind::Float, 4) => Some(L::F32(v as f32)),
                (naga::ScalarKind::Float, 8) => Some(L::F64(v)),
                (naga::ScalarKind::Sint, 4) => Some(L::I32(v as i32)),
                (naga::ScalarKind::Sint, 8) => Some(L::I64(v as i64)),
                (naga::ScalarKind::Uint, 4) => Some(L::U32(v as u32)),
                (naga::ScalarKind::Uint, 8) => Some(L::U64(v as u64)),
                _ => None,
            },
            _ => return None,
        }?;

        Some(literal_to_wgsl(concrete, self.options.max_precision))
    }
}

/// Intermediate representation for grouped Compose arguments during emission.
enum ComposeGroup {
    /// A run of >=2 AccessIndex/Load-of-AccessIndex components on the same base.
    Swizzle {
        base: naga::Handle<naga::Expression>,
        indices: Vec<u32>,
    },
    /// A single component emitted normally.
    Single(naga::Handle<naga::Expression>),
}

// WGSL operator precedence definitions, used for determining when to insert parentheses
// around child expressions of a Binary operator.
// Source: <https://www.w3.org/TR/WGSL/#operator-precedence>

const PREC_ADDITIVE: u8 = 9;
const PREC_MULTIPLICATIVE: u8 = 10;
const PREC_UNARY: u8 = 11;

// MARK: Binary-operator rendering

/// WGSL precedence level for a binary operator, higher binds tighter.
/// Used by the parenthesis minimiser: a child binary is parenthesised
/// only when its precedence is below the parent's (with ties and
/// non-associative operators handled explicitly by the caller).
fn binary_precedence(op: naga::BinaryOperator) -> u8 {
    use naga::BinaryOperator as B;
    match op {
        B::LogicalOr => 1,
        B::LogicalAnd => 2,
        B::InclusiveOr => 3,
        B::ExclusiveOr => 4,
        B::And => 5,
        B::Equal | B::NotEqual => 6,
        B::Less | B::LessEqual | B::Greater | B::GreaterEqual => 7,
        B::ShiftLeft | B::ShiftRight => 8,
        B::Add | B::Subtract => PREC_ADDITIVE,
        B::Multiply | B::Divide | B::Modulo => PREC_MULTIPLICATIVE,
    }
}

/// Map each binary operator to its WGSL token (`+`, `*`, `&&`, etc.).
fn binary_op_str(op: naga::BinaryOperator) -> &'static str {
    use naga::BinaryOperator as B;
    match op {
        B::Add => "+",
        B::Subtract => "-",
        B::Multiply => "*",
        B::Divide => "/",
        B::Modulo => "%",
        B::Equal => "==",
        B::NotEqual => "!=",
        B::Less => "<",
        B::LessEqual => "<=",
        B::Greater => ">",
        B::GreaterEqual => ">=",
        B::And => "&",
        B::ExclusiveOr => "^",
        B::InclusiveOr => "|",
        B::LogicalAnd => "&&",
        B::LogicalOr => "||",
        B::ShiftLeft => "<<",
        B::ShiftRight => ">>",
    }
}

/// Check whether a child expression must be wrapped in parentheses when
/// appearing as an operand of `parent_op`.
/// `true` when a binary child needs parentheses for the enclosing
/// parent shape.  Accounts for operator precedence, associativity,
/// and the handful of cases WGSL treats specially (for example
/// `a - (b - c)`).
fn child_needs_parens(
    child: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    parent_op: naga::BinaryOperator,
    is_right: bool,
    is_cached: bool,
) -> bool {
    if is_cached {
        return false;
    }
    let child_op = match &arena[child] {
        naga::Expression::Binary { op, .. } => Some(*op),
        _ => None,
    };
    let child_prec = match &arena[child] {
        naga::Expression::Binary { op, .. } => binary_precedence(*op),
        naga::Expression::Unary { .. } => PREC_UNARY,
        _ => return false,
    };
    let parent_prec = binary_precedence(parent_op);

    // Keep arithmetic children parenthesized under bitwise operators.
    // This preserves explicit tree shape and avoids parser/type-inference
    // ambiguities on patterns like `a^(b-1u)` when emitted in compact form.
    if matches!(
        parent_op,
        naga::BinaryOperator::And
            | naga::BinaryOperator::ExclusiveOr
            | naga::BinaryOperator::InclusiveOr
    ) {
        if let Some(op) = child_op {
            if matches!(
                op,
                naga::BinaryOperator::Add | naga::BinaryOperator::Subtract
            ) {
                return true;
            }
        }
    }

    // WGSL shift operators require `unary_expression` on both sides.
    if matches!(
        parent_op,
        naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight
    ) {
        return child_prec < PREC_UNARY;
    }

    // WGSL relational operators are non-chainable.
    if matches!(
        parent_op,
        naga::BinaryOperator::Less
            | naga::BinaryOperator::LessEqual
            | naga::BinaryOperator::Greater
            | naga::BinaryOperator::GreaterEqual
    ) {
        return child_prec <= parent_prec;
    }

    // Left-associative: left child needs parens if strictly lower,
    // right child if lower-or-equal (to preserve tree structure).
    if is_right {
        child_prec <= parent_prec
    } else {
        child_prec < parent_prec
    }
}

/// A Unary's operand needs parentheses only when it is a Binary
/// expression (all Binary precedences are below Unary).
/// `true` when a unary child needs parentheses.  Only operators that
/// themselves lower to WGSL binary or postfix shapes qualify; atoms
/// and calls never do.
fn unary_child_needs_parens(
    child: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    is_cached: bool,
) -> bool {
    !is_cached && matches!(arena[child], naga::Expression::Binary { .. })
}

/// Assemble a binary expression string from pre-computed operand strings,
/// operator token, spacing, and parenthesisation flags.
/// Assemble `left op right` with the correct parenthesisation and
/// operator spacing for the current beautify mode.  All binary
/// emission paths funnel through here.
fn assemble_binary(
    ls: &str,
    rs: &str,
    op_str: &str,
    sp: &str,
    wrap_l: bool,
    wrap_r: bool,
) -> String {
    let mut s = String::new();
    if wrap_l {
        s.push('(');
    }
    s.push_str(ls);
    if wrap_l {
        s.push(')');
    }
    s.push_str(sp);
    s.push_str(op_str);
    if !sp.is_empty() {
        s.push_str(sp);
    } else if !wrap_r {
        // Disambiguate `a--b` and `a//b` which WGSL would parse as
        // decrement / line-comment.
        if let (Some(&oc), Some(&rc)) = (op_str.as_bytes().last(), rs.as_bytes().first()) {
            if oc == rc && (oc == b'-' || oc == b'/') {
                s.push(' ');
            }
        }
    }
    if wrap_r {
        s.push('(');
    }
    s.push_str(rs);
    if wrap_r {
        s.push(')');
    }
    s
}

// MARK: Abstract literal concretisation

/// Outcome of projecting an *abstract* literal (`AbstractInt` /
/// `AbstractFloat`) to its concrete form given the resolved type at
/// the use site.
///
/// Single source of truth shared by two consumers:
///
/// - `expr_emit::Generator::concretize_abstract_literal_for_expr`
///   on the emission side, producing the textual literal.
/// - `literal_extract::scan_and_extract_literals::count_literals`
///   on the scan side, counting textual emissions to decide which
///   literals to extract into a shared `const`.
///
/// Both sides MUST agree on the projection so the
/// [`literal_extract_key`] computed during the scan matches the key
/// looked up during emission.  Splitting the two would re-introduce
/// the historical drift where a hot abstract literal was extracted
/// under its abstract key but the emission path bypassed the lookup
/// by computing the typed-form text directly, leaving the extracted
/// `const` unreferenced.
pub(super) enum ConcretizedAbstract {
    /// The concrete `naga::Literal` form.  Callers should consult their
    /// `extracted_literals` map keyed via [`literal_extract_key`] before
    /// falling back to [`literal_to_wgsl`] for the typed text.
    Lit(naga::Literal),
    /// A pre-built text wrapper that bypasses extraction (`f16(...)`
    /// constructor calls; `i32(...)` / `u32(...)` / `u64(...)` casts
    /// for out-of-range `AbstractInt`s).  Both sides must skip
    /// extraction here: the emission text is not a single literal
    /// token, and the matching `LiteralExtractKey` cannot be
    /// reconstructed.
    Text(String),
}

/// Pure projection of an abstract literal to its concrete form,
/// given the resolved type at the use site.  Returns `None` for
/// non-abstract literals or when the type does not pin a
/// scalar/vector kind+width.
///
/// Mirrors WGSL's abstract-numeric coercion rules for the cases naga
/// actually emits in `Expression::Literal` (vector/scalar contexts).
/// Out-of-range `AbstractInt` projections fall back to an explicit
/// cast wrapper text.  `AbstractInt` / `AbstractFloat` at `f16` always
/// wraps in `f16(...)f` since WGSL has no `AbstractInt -> f16`
/// coercion shorthand.
pub(super) fn concretize_abstract_literal_via_inner(
    lit: naga::Literal,
    inner: &naga::TypeInner,
) -> Option<ConcretizedAbstract> {
    use naga::Literal as L;

    let (kind, width) = match lit {
        L::AbstractInt(_) | L::AbstractFloat(_) => match inner {
            naga::TypeInner::Scalar(s) => (s.kind, s.width),
            naga::TypeInner::Vector { scalar, .. } => (scalar.kind, scalar.width),
            _ => return None,
        },
        _ => return None,
    };

    let concrete = match lit {
        L::AbstractInt(v) => match (kind, width) {
            (naga::ScalarKind::Sint, 4) => match i32::try_from(v) {
                Ok(x) => L::I32(x),
                Err(_) => return Some(ConcretizedAbstract::Text(format!("i32({v})"))),
            },
            (naga::ScalarKind::Sint, 8) => L::I64(v),
            (naga::ScalarKind::Uint, 4) => match u32::try_from(v) {
                Ok(x) => L::U32(x),
                Err(_) => return Some(ConcretizedAbstract::Text(format!("u32({v})"))),
            },
            (naga::ScalarKind::Uint, 8) => match u64::try_from(v) {
                Ok(x) => L::U64(x),
                Err(_) => return Some(ConcretizedAbstract::Text(format!("u64({v})"))),
            },
            (naga::ScalarKind::Float, 2) => {
                return Some(ConcretizedAbstract::Text(format!("f16({}f)", v as f32)));
            }
            (naga::ScalarKind::Float, 4) => L::F32(v as f32),
            (naga::ScalarKind::Float, 8) => L::F64(v as f64),
            _ => return None,
        },
        L::AbstractFloat(v) => match (kind, width) {
            (naga::ScalarKind::Float, 2) => {
                return Some(ConcretizedAbstract::Text(format!("f16({}f)", v as f32)));
            }
            (naga::ScalarKind::Float, 4) => L::F32(v as f32),
            (naga::ScalarKind::Float, 8) => L::F64(v),
            (naga::ScalarKind::Sint, 4) => L::I32(v as i32),
            (naga::ScalarKind::Sint, 8) => L::I64(v as i64),
            (naga::ScalarKind::Uint, 4) => L::U32(v as u32),
            (naga::ScalarKind::Uint, 8) => L::U64(v as u64),
            _ => return None,
        },
        _ => return None,
    };

    Some(ConcretizedAbstract::Lit(concrete))
}

// MARK: Splat detection

/// Return `true` when every component in a vector `Compose` resolves
/// to the same value, so the constructor can be emitted in single-arg
/// splat form (e.g. `vec3f(x)` instead of `vec3f(x,x,x)`).
///
/// Only considers cases guaranteed correct:
///
/// - all component handles are identical
/// - all components are the same `Literal` / `Constant` / `Override`
///   / `ZeroValue`.
pub(super) fn compose_is_splat(
    components: &[naga::Handle<naga::Expression>],
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    debug_assert!(components.len() > 1);
    let first = components[0];
    // Fast path: all handles point to the same expression.
    if components[1..].iter().all(|&c| c == first) {
        return true;
    }
    // Slow path: compare underlying expression values.
    let first_expr = &arena[first];
    components[1..]
        .iter()
        .all(|&c| exprs_splat_eq(first_expr, &arena[c]))
}

/// Bit-exact comparison of two expressions for splat-collapse purposes.
/// Structural equality check tailored for splat detection.  Allows
/// literals to compare by bit pattern (handling `-0.0` vs `+0.0`
/// correctly) while falling back to `==` for non-literals.
fn exprs_splat_eq(a: &naga::Expression, b: &naga::Expression) -> bool {
    use naga::Expression as E;
    match (a, b) {
        (E::Literal(la), E::Literal(lb)) => literal_bit_eq(la, lb),
        (E::Constant(ha), E::Constant(hb)) => ha == hb,
        (E::Override(ha), E::Override(hb)) => ha == hb,
        (E::ZeroValue(ta), E::ZeroValue(tb)) => ta == tb,
        _ => false,
    }
}

/// Bit-exact literal comparison.  Standard `PartialEq` for floats treats
/// `-0.0 == 0.0`, but they produce different textual output, so we compare
/// via `to_bits()` for float variants.
/// Bit-pattern equality for literals.  Unlike `PartialEq`, treats
/// `+0.0` and `-0.0` as distinct and NaN values as equal when their
/// bits match.
fn literal_bit_eq(a: &naga::Literal, b: &naga::Literal) -> bool {
    use naga::Literal as L;
    match (a, b) {
        (L::F64(x), L::F64(y)) | (L::AbstractFloat(x), L::AbstractFloat(y)) => {
            x.to_bits() == y.to_bits()
        }
        (L::F32(x), L::F32(y)) => x.to_bits() == y.to_bits(),
        (L::F16(x), L::F16(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Return the logically-negated comparison operator, if `op` is a
/// comparison.
/// Map a comparison operator to the equivalent negated operator
/// when negation is semantically safe (equality and inequality are
/// safe for every scalar type; ordered comparisons are deferred to
/// the caller, which checks for float operands).
fn flip_comparison(op: naga::BinaryOperator) -> Option<naga::BinaryOperator> {
    use naga::BinaryOperator as B;
    match op {
        B::Less => Some(B::GreaterEqual),
        B::LessEqual => Some(B::Greater),
        B::Greater => Some(B::LessEqual),
        B::GreaterEqual => Some(B::Less),
        B::Equal => Some(B::NotEqual),
        B::NotEqual => Some(B::Equal),
        _ => None,
    }
}

/// Returns `true` for arithmetic binary operators where WGSL defines
/// mixed scalar-vector overloads (`scalar OP vector` and `vector OP scalar`).
/// `true` when the binary operator is one of the arithmetic operators
/// that participate in scalar-vector broadcasting (`+`, `-`, `*`,
/// `/`, `%`).  Used by splat elision to decide whether a vector-typed
/// operand can safely collapse to its bare scalar.
pub(super) fn is_arithmetic_op(op: naga::BinaryOperator) -> bool {
    use naga::BinaryOperator as B;
    matches!(
        op,
        B::Add | B::Subtract | B::Multiply | B::Divide | B::Modulo
    )
}

#[cfg(test)]
mod tests {
    use super::{concretize_abstract_literal_via_inner, ConcretizedAbstract};
    use naga::Literal as L;

    fn scalar_inner(kind: naga::ScalarKind, width: u8) -> naga::TypeInner {
        naga::TypeInner::Scalar(naga::Scalar { kind, width })
    }

    fn vector_inner(size: naga::VectorSize, kind: naga::ScalarKind, width: u8) -> naga::TypeInner {
        naga::TypeInner::Vector {
            size,
            scalar: naga::Scalar { kind, width },
        }
    }

    fn assert_lit(out: Option<ConcretizedAbstract>, expected: naga::Literal) {
        match out {
            Some(ConcretizedAbstract::Lit(c)) => assert_eq!(c, expected),
            Some(ConcretizedAbstract::Text(t)) => {
                panic!("expected Lit({expected:?}), got Text({t})")
            }
            None => panic!("expected Lit({expected:?}), got None"),
        }
    }

    fn assert_text(out: Option<ConcretizedAbstract>, expected: &str) {
        match out {
            Some(ConcretizedAbstract::Text(t)) => assert_eq!(t, expected),
            Some(ConcretizedAbstract::Lit(c)) => {
                panic!("expected Text({expected}), got Lit({c:?})")
            }
            None => panic!("expected Text({expected}), got None"),
        }
    }

    #[test]
    fn concretize_returns_none_for_non_abstract_literal() {
        // Concrete literals already carry their typed form; the emission
        // path skips the helper and counts them under their own key.
        let inner = scalar_inner(naga::ScalarKind::Sint, 4);
        assert!(concretize_abstract_literal_via_inner(L::I32(5), &inner).is_none());
        assert!(concretize_abstract_literal_via_inner(L::F32(0.5), &inner).is_none());
        assert!(concretize_abstract_literal_via_inner(L::Bool(true), &inner).is_none());
    }

    #[test]
    fn concretize_returns_none_for_unsupported_inner() {
        // No projection target (matrix, array, struct, etc.).  Caller
        // falls back to its non-abstract path; emission re-enters the
        // tree via the constructor, which pins types per slot.
        let mat_inner = naga::TypeInner::Matrix {
            columns: naga::VectorSize::Bi,
            rows: naga::VectorSize::Bi,
            scalar: naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 4,
            },
        };
        assert!(concretize_abstract_literal_via_inner(L::AbstractInt(5), &mat_inner).is_none());
        assert!(concretize_abstract_literal_via_inner(L::AbstractFloat(0.5), &mat_inner).is_none());
    }

    #[test]
    fn concretize_abstract_int_to_concrete_signed_unsigned() {
        let i32_inner = scalar_inner(naga::ScalarKind::Sint, 4);
        let i64_inner = scalar_inner(naga::ScalarKind::Sint, 8);
        let u32_inner = scalar_inner(naga::ScalarKind::Uint, 4);
        let u64_inner = scalar_inner(naga::ScalarKind::Uint, 8);
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(42), &i32_inner),
            L::I32(42),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(-7), &i64_inner),
            L::I64(-7),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(99), &u32_inner),
            L::U32(99),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(123), &u64_inner),
            L::U64(123),
        );
    }

    #[test]
    fn concretize_abstract_int_overflow_falls_back_to_text() {
        // Out-of-range AbstractInts cannot survive `try_from` to the
        // narrower concrete type, so the helper emits an explicit cast
        // wrapper.  Both sides skip extraction here; pin the exact text
        // so any future drift in the formatting fails this test.
        let i32_inner = scalar_inner(naga::ScalarKind::Sint, 4);
        let u32_inner = scalar_inner(naga::ScalarKind::Uint, 4);
        let u64_inner = scalar_inner(naga::ScalarKind::Uint, 8);
        let big = i64::from(i32::MAX) + 1;
        assert_text(
            concretize_abstract_literal_via_inner(L::AbstractInt(big), &i32_inner),
            &format!("i32({big})"),
        );
        assert_text(
            concretize_abstract_literal_via_inner(L::AbstractInt(-1), &u32_inner),
            "u32(-1)",
        );
        assert_text(
            concretize_abstract_literal_via_inner(L::AbstractInt(-1), &u64_inner),
            "u64(-1)",
        );
    }

    #[test]
    fn concretize_abstract_int_to_floats_projects_lit() {
        // f32 / f64: round-trip via lit; f16 always wraps in `f16(...)f`.
        let f32_inner = scalar_inner(naga::ScalarKind::Float, 4);
        let f64_inner = scalar_inner(naga::ScalarKind::Float, 8);
        let f16_inner = scalar_inner(naga::ScalarKind::Float, 2);
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(3), &f32_inner),
            L::F32(3.0),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(3), &f64_inner),
            L::F64(3.0),
        );
        assert_text(
            concretize_abstract_literal_via_inner(L::AbstractInt(2), &f16_inner),
            "f16(2f)",
        );
    }

    #[test]
    fn concretize_abstract_float_to_concrete_floats_and_ints() {
        let f32_inner = scalar_inner(naga::ScalarKind::Float, 4);
        let f64_inner = scalar_inner(naga::ScalarKind::Float, 8);
        let f16_inner = scalar_inner(naga::ScalarKind::Float, 2);
        let i32_inner = scalar_inner(naga::ScalarKind::Sint, 4);
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractFloat(0.5), &f32_inner),
            L::F32(0.5),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractFloat(1.5), &f64_inner),
            L::F64(1.5),
        );
        assert_text(
            concretize_abstract_literal_via_inner(L::AbstractFloat(0.5), &f16_inner),
            "f16(0.5f)",
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractFloat(2.0), &i32_inner),
            L::I32(2),
        );
    }

    #[test]
    fn concretize_uses_vector_scalar_when_inner_is_vector() {
        // `Expression::Literal` resolves to a vector type when used as
        // a `Compose` arg; the helper must extract scalar kind/width
        // from the vector's `scalar` field, not bail.
        let vec3f_inner = vector_inner(naga::VectorSize::Tri, naga::ScalarKind::Float, 4);
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractFloat(0.25), &vec3f_inner),
            L::F32(0.25),
        );
        assert_lit(
            concretize_abstract_literal_via_inner(L::AbstractInt(7), &vec3f_inner),
            L::F32(7.0),
        );
    }
}
