//! Expression emitters.  `emit_expr` short-circuits a let-bound expression to
//! its name; `emit_expr_uncached` renders every [`naga::Expression`] variant
//! and hosts the size rewrites (splat elision, swizzle collapse, extracted-
//! literal substitution, parenthesis minimisation) that beat naga's writer.

use crate::error::Error;
use crate::passes::expr_util::literal_bit_eq;

use super::core::{FunctionCtx, Generator};
use super::syntax::{
    expression_kind, literal_extract_key, literal_to_wgsl, literal_to_wgsl_bare, math_name,
    scalar_zero, type_inner_kind, type_inner_name, type_resolution_name,
};

/// `true` when the bare (suffix-less) spelling of `literal` would re-parse as
/// a different concrete type.  Every top-level `Literal` in valid naga IR sits
/// in a pinning position (Binary partner of known type, argument/return/store
/// slot, constructor) where the abstract value coerces to the local type, and
/// abstract literals default to i32/f32: `I32`/`U32`/`F32`/`Bool` coerce back
/// to the original and drop the suffix byte, while `F16`/`F64`/`I64`/`U64` do
/// not (bare `0.5` defaults f32 and refuses f16/f64 contexts, bare `42`
/// defaults i32 and conflicts with an i64/u64 target).
///
/// A NON-pinning position breaks the `U32` case: a bare switch selector
/// re-parses as i32 while its case labels carry `u`, so every new top-level
/// literal position must force the typed form or hint the scalar through
/// [`Generator::emit_expr_with_scalar_hint`].  A literal PAIR pins nothing
/// either - bare `-1*0` is AbstractInt, not the `f32` the operands were - so
/// an operand that only becomes a literal mid-pipeline must keep whatever
/// left it runtime; see [`i32_binary_widens`] for the arithmetic half.
/// `true` when both operands are `i32` literals whose concrete (wrapping)
/// result differs from the exact one an AbstractInt evaluation would give -
/// exactly the cases the checked operation rejects.  `u32` literals keep
/// their suffix already, and shifts type as their left operand.
fn i32_binary_widens(op: naga::BinaryOperator, left: naga::Literal, right: naga::Literal) -> bool {
    use naga::BinaryOperator as B;
    let (naga::Literal::I32(l), naga::Literal::I32(r)) = (left, right) else {
        return false;
    };
    match op {
        B::Add => l.checked_add(r).is_none(),
        B::Subtract => l.checked_sub(r).is_none(),
        B::Multiply => l.checked_mul(r).is_none(),
        B::Divide => l.checked_div(r).is_none(),
        B::Modulo => l.checked_rem(r).is_none(),
        _ => false,
    }
}

/// `true` when both operands are float literals whose bare spellings are
/// integer-shaped (`-1`, `0`), so the pair re-parses as AbstractInt: `-1*0` is
/// the integer 0, not the float `-0.0` the operands held.  The float analogue
/// of [`i32_binary_widens`] - a pair pins nothing, and here it loses the type
/// family, not just the width.  A float-shaped partner (`1.5`, `1e3`) already
/// pins AbstractFloat, so only an all-integer-shaped pair qualifies.
fn float_pair_reinfers_int(
    left: naga::Literal,
    right: naga::Literal,
    precision: &crate::config::FloatPrecision,
) -> bool {
    let int_shaped = |l: naga::Literal| {
        matches!(l, naga::Literal::F32(_) | naga::Literal::AbstractFloat(_)) && {
            let bare = literal_to_wgsl_bare(l, precision);
            !bare.is_empty() && bare.bytes().all(|b| b.is_ascii_digit() || b == b'-')
        }
    };
    int_shaped(left) && int_shaped(right)
}

fn literal_needs_typed_form_outside_constructor(literal: naga::Literal) -> bool {
    matches!(
        literal,
        naga::Literal::F16(_)
            | naga::Literal::F64(_)
            | naga::Literal::I64(_)
            | naga::Literal::U64(_)
    )
}

/// Bare form re-infers a different type in a position nothing pins (shift
/// left operand, `extractBits` value): only `I32` / `F32` / `Bool` match
/// the abstract defaults. Stricter than
/// [`literal_needs_typed_form_outside_constructor`], whose positions still
/// coerce u32. `literal_extract` subtracts exactly the occurrences this
/// forces typed.
pub(super) fn literal_bare_form_changes_type(literal: naga::Literal) -> bool {
    !matches!(
        literal,
        naga::Literal::I32(_) | naga::Literal::F32(_) | naga::Literal::Bool(_)
    )
}

/// A bitcast operand always keeps its suffix (bits); a conversion's only
/// when the abstract value converts differently: a u32 above i32::MAX or a
/// negative i32 into the other integer type (rejected where the typed form
/// wraps), a width with no abstract spelling, or an f16 target (`f16(.1)`
/// double-rounds). Shared with `literal_extract`.
pub(super) fn as_operand_keeps_suffix(literal: naga::Literal, convert: Option<u8>) -> bool {
    let differs = match literal {
        naga::Literal::U32(v) => v > i32::MAX as u32,
        naga::Literal::I32(v) => v < 0,
        naga::Literal::F32(_) | naga::Literal::Bool(_) => false,
        _ => true,
    };
    convert.is_none() || differs || convert == Some(2)
}

/// `true` when `literal`, in the bare form constructor components use,
/// re-infers exactly `scalar` and so can pin an elided `vecN(...)`'s element
/// type.  An integer-form token is AbstractInt (defaults i32) and a float-form
/// token AbstractFloat (defaults f32): any integer literal pins i32, and a
/// literal whose bare rendering keeps a float shape (`.5`, `1e3`, `0x1p2`, not
/// a whole number, which renders as a bare int) pins f32; u32/f16/f64/16-bit
/// elements are never literal-pinned.  Mixed abstract components keep the
/// pinned default (`vec4(.5,1,1,1)` is AbstractFloat throughout -> f32).  The
/// float-shape test inspects the rendering the emitter ships (precision
/// rounding included), so the decision cannot drift from the emitted token.
fn literal_bare_form_pins_scalar(
    literal: naga::Literal,
    scalar: naga::Scalar,
    precision: &crate::config::FloatPrecision,
) -> bool {
    match scalar {
        naga::Scalar::I32 => matches!(
            literal,
            naga::Literal::I32(_) | naga::Literal::AbstractInt(_)
        ),
        naga::Scalar::F32 => {
            if !matches!(
                literal,
                naga::Literal::F32(_) | naga::Literal::AbstractFloat(_)
            ) {
                return false;
            }
            let bare = literal_to_wgsl_bare(literal, precision);
            // Integer-shaped text (`1`, `-2`) re-parses as AbstractInt.
            !bare.bytes().all(|b| b.is_ascii_digit() || b == b'-')
        }
        _ => false,
    }
}

/// Width-8 numeric literals (`f64`/`u64`/`i64`), the only ones the vector
/// narrowing fold accepts; `literal_extract`'s pre-pass must mirror this set
/// or its count diverges from what the emitter prints.
pub(super) fn literal_is_width8(l: naga::Literal) -> bool {
    matches!(
        l,
        naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_)
    )
}

/// Numeric zero, `-0.0` included (F16 compares bit patterns: its `==` needs
/// `half`).
fn literal_is_zero(lit: naga::Literal) -> bool {
    match lit {
        naga::Literal::F16(v) => v.to_bits() == 0 || v.to_bits() == 0x8000,
        naga::Literal::F32(v) => v == 0.0,
        naga::Literal::F64(v) => v == 0.0,
        naga::Literal::AbstractFloat(v) => v == 0.0,
        naga::Literal::I16(v) => v == 0,
        naga::Literal::U16(v) => v == 0,
        naga::Literal::I32(v) => v == 0,
        naga::Literal::U32(v) => v == 0,
        naga::Literal::I64(v) => v == 0,
        naga::Literal::U64(v) => v == 0,
        naga::Literal::AbstractInt(v) => v == 0,
        naga::Literal::Bool(v) => !v,
    }
}

impl<'a> Generator<'a> {
    /// Statically provable zero (literal zero, `ZeroValue`, or a constant
    /// initialised by one), for emission paths whose WGSL signature can only
    /// encode zero - `textureSampleCompareLevel` has no level parameter.
    pub(super) fn expression_is_provable_zero(
        &self,
        handle: naga::Handle<naga::Expression>,
        ctx: &super::core::FunctionCtx<'_, '_>,
    ) -> bool {
        match ctx.exprs[handle] {
            naga::Expression::Literal(lit) => literal_is_zero(lit),
            naga::Expression::ZeroValue(_) => true,
            naga::Expression::Constant(c) => {
                let constant = &self.module.constants[c];
                match self.module.global_expressions[constant.init] {
                    naga::Expression::Literal(lit) => literal_is_zero(lit),
                    naga::Expression::ZeroValue(_) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// `true` when `handle` renders with a top-level bare `<`: an uncached
    /// `Less`, or an uncached `&&`/`||` whose right spine ends in one (those
    /// operators do not parenthesise a comparison child: `x&&a<b`).  In a
    /// comma-delimited argument list a later argument's top-level `>` pairs
    /// with it in WGSL's template-list scanner (`f(a<b,c>d)` scans as the
    /// template `a<b,c>` applied to `d`) and strict parsers reject the file,
    /// so such arguments are parenthesised.  Only bare `<` opens a candidate
    /// (`<=`/`<<` never do), a cached expression renders as its name, no other
    /// uncached root renders a top-level `<`, and the LAST argument never needs
    /// the wrap because the closing `)` discards the candidate.
    fn renders_as_bare_less(
        &self,
        handle: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> bool {
        if ctx.expr_names.contains_key(handle) {
            return false;
        }
        match &ctx.exprs[handle] {
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                ..
            } => true,
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr,
                right,
                ..
            } => self.renders_as_bare_less(*right, ctx),
            _ => false,
        }
    }

    /// `name(args)`, prefixing `&` to an argument bound to a pointer
    /// parameter: globals, locals and access chains are WGSL references,
    /// while a forwarded pointer-typed function argument is already a
    /// pointer value.
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
            let needs_ref = if let Some(param) = callee.arguments.get(i) {
                matches!(
                    self.module.types[param.ty].inner,
                    naga::TypeInner::Pointer { .. }
                ) && !matches!(
                    ctx.exprs[*arg],
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
            let arg_text = self.emit_expr(*arg, ctx)?;
            if i + 1 < arguments.len() && self.renders_as_bare_less(*arg, ctx) {
                s.push('(');
                s.push_str(&arg_text);
                s.push(')');
            } else {
                s.push_str(&arg_text);
            }
        }
        s.push(')');
        Ok(s)
    }

    /// Shortest negation of `cond`: a let-bound condition is `!name`, a
    /// comparison flips its operator (`<` -> `>=`, `==` -> `!=`) except an
    /// ordered float comparison, where `!(x<y)` and `x>=y` differ under NaN,
    /// `!!x` collapses to `x`, and anything else is `!(expr)`.
    pub(super) fn emit_negated_condition(
        &self,
        cond: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;

        if let Some(name) = ctx.expr_names.get(cond) {
            return Ok(format!("!{name}"));
        }

        match &ctx.exprs[cond] {
            E::Binary { op, left, right } => {
                if let Some(flipped) = flip_comparison(*op) {
                    let is_ordered = matches!(
                        op,
                        naga::BinaryOperator::Less
                            | naga::BinaryOperator::LessEqual
                            | naga::BinaryOperator::Greater
                            | naga::BinaryOperator::GreaterEqual
                    );
                    let is_float = is_ordered
                        && ctx.ty(*left).inner_with(&self.module.types).scalar_kind()
                            == Some(naga::ScalarKind::Float);
                    if !is_float {
                        let left = *left;
                        let right = *right;
                        let op_str = binary_op_str(flipped);
                        let sp = self.bin_op_sep();
                        let arena = &ctx.exprs;
                        let lc = ctx.expr_names.contains_key(left);
                        let rc = ctx.expr_names.contains_key(right);
                        let wrap_l = child_needs_parens(left, arena, flipped, false, lc);
                        let wrap_r = child_needs_parens(right, arena, flipped, true, rc);
                        let ls = self.emit_expr(left, ctx)?;
                        let rs = self.emit_expr(right, ctx)?;
                        return Ok(assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r));
                    }
                }
            }
            E::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr,
            } => {
                return self.emit_expr(*expr, ctx);
            }
            _ => {}
        }

        let mut inner = self.emit_expr(cond, ctx)?;
        inner.insert_str(0, "!(");
        inner.push(')');
        Ok(inner)
    }

    /// `expr` as an assignable place; a root that is neither a variable nor
    /// an access chain renders dereferenced (`*expr`).
    pub(super) fn emit_lvalue(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        Ok(match &ctx.exprs[expr] {
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
                self.push_access_index(&mut s, *base, *index, ctx);
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

    /// Place form for a variable or access chain, value form otherwise, for
    /// callers that accept either (the base of a store-through access chain).
    pub(super) fn emit_lvalue_or_value(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;
        Ok(match &ctx.exprs[expr] {
            E::GlobalVariable(h) => self.global_names[h.index()].clone(),
            E::LocalVariable(h) => ctx.local_names[h].clone(),
            E::Access { .. } | E::AccessIndex { .. } => self.emit_lvalue(expr, ctx)?,
            // A function-argument `ptr<...>` is a pointer value, not a
            // reference, so as the root of an lvalue chain it needs the
            // explicit `(*p)`.
            _ if self.pointer_is_ptr_value(expr, ctx) => {
                format!("(*{})", self.emit_expr(expr, ctx)?)
            }
            _ => self.emit_expr(expr, ctx)?,
        })
    }

    /// Value text of `expr`; a let/var/argument-bound expression renders as
    /// its name.
    pub(super) fn emit_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if let Some(name) = ctx.expr_names.get(expr) {
            return Ok(name.clone());
        }
        self.emit_expr_uncached(expr, ctx)
    }

    /// A ray-query builtin's `query` operand: naga admits any
    /// `ptr<function, ray_query>`, so a `ray_query` LOCAL (a reference) takes
    /// `&`, while the only other legal producer, a pointer-typed function
    /// argument (`ray_query` cannot live in composites, so no access chain
    /// yields one), is already a pointer value.
    fn emit_ray_query_arg(
        &self,
        query: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        let text = self.emit_expr(query, ctx)?;
        Ok(match ctx.exprs[query] {
            naga::Expression::LocalVariable(_) => format!("&{text}"),
            _ => text,
        })
    }

    /// A `Compose`/`Splat` component.  An uncached literal renders bare
    /// because the constructor pins its type; an argument rendering with a
    /// top-level `<` is parenthesised (template-list guard).
    pub(super) fn emit_constructor_arg(
        &self,
        arg: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if !ctx.expr_names.contains_key(arg) {
            if let naga::Expression::Literal(lit) = &ctx.exprs[arg] {
                let bare = literal_to_wgsl_bare(*lit, &self.options.float_precision);
                let key = literal_extract_key(*lit, &self.options.float_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    return Ok(name.clone());
                }
                return Ok(bare);
            }
            // Template-list guard; the `<=` wrap is redundant but harmless.
            if self.renders_as_bare_less(arg, ctx)
                || matches!(
                    &ctx.exprs[arg],
                    naga::Expression::Binary {
                        op: naga::BinaryOperator::LessEqual,
                        ..
                    }
                )
            {
                let s = self.emit_expr_uncached(arg, ctx)?;
                return Ok(format!("({s})"));
            }
            // An identity-swizzle `Compose` (`vecN(b.0,..,b.N-1)`) collapses to
            // its bare base.  A constructor argument has nothing appended, so
            // an operator base needs none of the parens the postfix path adds
            // (`mat3x3(l*m,..)`, not `mat3x3((l*m),..)`).
            if let Some(base) = self.compose_identity_collapse_base(arg, ctx) {
                // A pointer-value base collapses to `(*p)`, not the bare
                // pointer.
                if self.pointer_is_ptr_value(base, ctx) {
                    return Ok(format!("(*{})", self.emit_expr(base, ctx)?));
                }
                let s = self.emit_expr(base, ctx)?;
                // A collapsed `Less`/`LessEqual` base takes the template-list
                // guard; other operator bases stay bare.
                if !ctx.expr_names.contains_key(base)
                    && matches!(
                        ctx.exprs[base],
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::Less | naga::BinaryOperator::LessEqual,
                            ..
                        }
                    )
                {
                    return Ok(format!("({s})"));
                }
                return Ok(s);
            }
        }
        self.emit_expr(arg, ctx)
    }

    /// The scalar an uncached `Splat` (or scalar-per-lane splat `Compose`)
    /// elides to as an arithmetic binary operand, where WGSL's scalar-vector
    /// overloads make it a shorter equivalent; the caller must keep the other
    /// operand a vector or the result type changes.
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
                // `vec4(v2, v2)` also satisfies `compose_is_splat`, but its
                // component is a VECTOR and would emit a type-mismatched
                // operand (`v2 * v4`); require one scalar per lane.
                let is_splat = matches!(
                    self.module.types[*ty].inner,
                    naga::TypeInner::Vector { size, .. }
                        if components.len() == size as usize
                            && components.len() > 1
                ) && compose_is_splat(components, arena);
                if is_splat { Some(components[0]) } else { None }
            }
            _ => None,
        }
    }

    /// Base of a postfix `.member` / `[index]` / `.xyzw`: postfix binds
    /// tighter than any prefix or infix operator, so a Binary/Unary/Select
    /// base is parenthesised (`(a-b).x`, since `a-b.x` parses as `a-(b.x)`).
    fn emit_postfix_base(
        &self,
        base: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        // A pointer VALUE (function-argument `ptr<...>`) does not auto-deref:
        // naga accepts `p.x` on it but tint/Dawn reject it, so it renders
        // `(*p)`; references (globals, locals, chains rooted in them) do
        // auto-deref.  Checked before the cached path because an argument
        // renders by name yet still needs the deref.
        if self.pointer_is_ptr_value(base, ctx) {
            return Ok(format!("(*{})", self.emit_expr(base, ctx)?));
        }
        if ctx.expr_names.contains_key(base) {
            return self.emit_expr(base, ctx);
        }
        let needs_parens = matches!(
            ctx.exprs[base],
            naga::Expression::Binary { .. }
                | naga::Expression::Unary { .. }
                | naga::Expression::Select { .. }
        );
        let s = self.emit_expr(base, ctx)?;
        // An inlined whole-pointee `Load` of a pointer value renders `*p`, and
        // a bare `*p.field` parses as `*(p.field)` ("operand of `*` must be a
        // pointer"), so it is wrapped too; a let-bound one already rendered
        // as a name.
        if needs_parens || s.starts_with('*') {
            Ok(format!("({s})"))
        } else {
            Ok(s)
        }
    }

    // MARK: Expression dispatch

    /// Every [`naga::Expression`] variant of an expression with no cached
    /// binding; each shape rewrite lives in its arm.
    pub(super) fn emit_expr_uncached(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        use naga::Expression as E;

        Ok(match &ctx.exprs[expr] {
            E::Literal(lit) => {
                if let Some(concrete) =
                    self.concretize_abstract_literal_for_expr(*lit, expr, ctx)?
                {
                    concrete
                } else {
                    let key = literal_extract_key(*lit, &self.options.float_precision);
                    if let Some(name) = self.extracted_literals.get(&key) {
                        name.clone()
                    } else if literal_needs_typed_form_outside_constructor(*lit) {
                        literal_to_wgsl(*lit, &self.options.float_precision)
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
                            // An unnamed literal-init constant renders as its
                            // literal at every use: same extraction lookup and
                            // bare-form gate as the Literal arm, one suffix
                            // byte cheaper than the typed global-expr path.
                            let key = literal_extract_key(*lit, &self.options.float_precision);
                            if let Some(name) = self.extracted_literals.get(&key) {
                                name.clone()
                            } else if literal_needs_typed_form_outside_constructor(*lit) {
                                literal_to_wgsl(*lit, &self.options.float_precision)
                            } else {
                                key.expr_text
                            }
                        }
                    } else {
                        self.emit_global_expr(c.init, false)?
                    }
                }
            }
            E::Override(h) => self.override_names[h.index()].clone(),
            E::ZeroValue(ty) => self.zero_value(*ty)?,
            E::Compose { ty, components } => 'compose: {
                // All-zero vector/matrix -> `vec2f()`, but ONLY when inlined
                // (ref count 1): naga re-parses `vec2f()` as a non-emittable
                // `ZeroValue` that can never be re-bound, so a let-bound one
                // would re-inline at every use - a non-idempotent size blow-up
                // - whereas the splat `vec2f(0)` re-parses bindable.  Strict
                // `+0` only (`-0.0` is rejected); never array/struct.
                if ctx.ref_counts[expr.index()] <= 1
                    && matches!(
                        self.module.types[*ty].inner,
                        naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. }
                    )
                    && components
                        .iter()
                        .all(|&c| compose_is_all_zero(c, ctx.exprs))
                {
                    break 'compose self.zero_value(*ty)?;
                }

                if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. })
                    && let Some(swizzle) = self.try_compose_as_full_swizzle(components, ctx)?
                {
                    break 'compose swizzle;
                }

                let mut s = String::new();
                // A pinned root must spell its type; both elisions infer it.
                let pinned = ctx.pinned_root == Some(expr);
                let ctor_name = if pinned {
                    self.type_ref(*ty)?
                } else {
                    self.vector_ctor_name(*ty, components, ctx)?
                };
                let bare = (!pinned && ctx.elide_array_ctor)
                    .then(|| self.array_ctor_name(*ty, components, &ctor_name, ctx.exprs))
                    .flatten();
                match bare {
                    Some(bare) => s.push_str(bare),
                    None => s.push_str(&ctor_name),
                }
                s.push('(');
                let is_splat = matches!(
                    self.module.types[*ty].inner,
                    naga::TypeInner::Vector { size, .. }
                        if components.len() == size as usize
                            && components.len() > 1
                ) && compose_is_splat(components, ctx.exprs);
                if is_splat {
                    s.push_str(&self.emit_constructor_arg(components[0], ctx)?);
                } else if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. })
                    && self.emit_compose_grouped(&mut s, components, ctx)?
                {
                    // Grouped swizzle runs already written to `s`.
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
                // A matrix of explicit scalar columns also has the flat form
                // `mat2x2f(a,b,c,d)`; keep whichever is shorter - the column
                // form wins when a column is let-bound (`mat3x3f(a,a,a)`) or
                // splat-collapses.
                if let Some(flat) =
                    matrix_flatten_scalars(*ty, components, &self.module.types, ctx.exprs)
                {
                    let mut sf = self.vector_ctor_name(*ty, components, ctx)?;
                    sf.push('(');
                    let sep = self.comma_sep();
                    for (i, c) in flat.iter().enumerate() {
                        if i > 0 {
                            sf.push_str(sep);
                        }
                        sf.push_str(&self.emit_constructor_arg(*c, ctx)?);
                    }
                    sf.push(')');
                    if sf.len() < s.len() {
                        s = sf;
                    }
                }
                // Equal-scalar runs may collapse to sub-vector splats
                // (`vec4f(0,0,0,2)` -> `vec4f(vec3f(),2)`); kept only when
                // strictly shorter, since the sub-vector type often lacks a
                // short alias.
                if let Some(sub) = self.try_subsplat_compose(*ty, components, ctx)?
                    && sub.len() < s.len()
                {
                    s = sub;
                }
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
                self.push_access_index(&mut s, *base, *index, ctx);
                s
            }
            E::Splat { size: _, value } => 'splat: {
                let target_ty = self.expr_type_name(expr, ctx)?;
                // All-zero splat -> `vec3f()`, gated on ref count <= 1: a bound
                // `vec3f()` re-parses to a non-emittable `ZeroValue` that
                // re-inlines at every use, a non-idempotent size blow-up.
                if ctx.ref_counts[expr.index()] <= 1 && compose_is_all_zero(*value, ctx.exprs) {
                    break 'splat format!("{target_ty}()");
                }
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
            } => 'swizzle: {
                let n = *size as u8 as usize;
                // An identity swizzle over a base of exactly `n` lanes is a
                // no-op, so emit the base alone - but only a base the postfix
                // path would not wrap (not Binary/Unary/Select): the parent's
                // parenthesisation, computed for a postfix `Swizzle`, must
                // stay valid for the substituted base, and a kept swizzle over
                // such a base re-minifies identically (eliding a `Select`
                // would drift to a parenthesised re-parse).
                let is_identity = pattern[..n].iter().enumerate().all(|(i, c)| {
                    let idx = match c {
                        naga::SwizzleComponent::X => 0,
                        naga::SwizzleComponent::Y => 1,
                        naga::SwizzleComponent::Z => 2,
                        naga::SwizzleComponent::W => 3,
                    };
                    idx == i
                });
                let base_is_full = match ctx.ty(*vector).inner_with(&self.module.types) {
                    naga::TypeInner::Vector { size: bs, .. } => *bs as u8 as usize == n,
                    _ => false,
                };
                let base_paren_free = !matches!(
                    ctx.exprs[*vector],
                    naga::Expression::Binary { .. }
                        | naga::Expression::Unary { .. }
                        | naga::Expression::Select { .. }
                );
                if is_identity && base_is_full && base_paren_free {
                    break 'swizzle self.emit_expr(*vector, ctx)?;
                }

                let mut s = self.emit_postfix_base(*vector, ctx)?;
                s.push('.');
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
            E::Load { pointer } => {
                // naga lowers `atomicLoad(&p)` and a direct read to the same
                // `Load`, but the spec and tint/Dawn reject reading `atomic<T>`
                // directly.
                if self.atomic_scalar_for_expr(*pointer, ctx).is_some() {
                    format!("atomicLoad({})", self.emit_pointer_operand(*pointer, ctx)?)
                } else {
                    self.emit_lvalue(*pointer, ctx)?
                }
            }
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
                    let suffix = if depth_ref.is_some() { "Compare" } else { "" };
                    s.push_str("textureGather");
                    s.push_str(suffix);
                    s.push('(');
                    // Only a non-depth gather takes the leading component index.
                    if depth_ref.is_none() {
                        let is_depth = matches!(
                            ctx.ty(*image).inner_with(&self.module.types),
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
                            // Only `textureSampleLevel` takes the explicit `0`.
                            if !clamp_to_edge && depth_ref.is_none() {
                                s.push_str(sep);
                                s.push('0');
                            }
                        }
                        naga::SampleLevel::Exact(h) => {
                            if depth_ref.is_none() {
                                s.push_str(sep);
                                s.push_str(&self.emit_expr(*h, ctx)?);
                            } else {
                                // `textureSampleCompareLevel` has no level
                                // slot (always level 0); naga's frontend
                                // encodes it as `Exact(Literal(0))`, so a
                                // provable zero drops and anything else is
                                // unrepresentable.
                                if !self.expression_is_provable_zero(*h, ctx) {
                                    return Err(Error::Emit(format!(
                                        "textureSampleCompareLevel cannot represent a \
                                         non-zero sample level in function '{}'",
                                        ctx.display_name,
                                    )));
                                }
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
                // Deliberately no comparison flip in value position (unlike
                // if/break_if conditions): every parenthesisation and
                // template-list guard classifies a child by its ARENA variant,
                // so a Unary rendering as a bare comparison would ship
                // atom-tight into comparison/bitwise parents (`a<b==c`,
                // tint-rejected) and slip every `renders_as_bare_less` guard.
                let op_str = match op {
                    naga::UnaryOperator::Negate => "-",
                    naga::UnaryOperator::LogicalNot => "!",
                    naga::UnaryOperator::BitwiseNot => "~",
                };
                let cached = ctx.expr_names.contains_key(expr);
                let wrap = unary_child_needs_parens(*expr, ctx.exprs, cached);
                let mut s = self.emit_expr(*expr, ctx)?;
                // A float zero renders bare as `0` where a sibling pins the
                // type, and `-0` re-parses as the ABSTRACT INTEGER zero, which
                // has no sign bit (`1.0 / -0` is `+inf`).  Abstract floats
                // count: this emitter also renders module-scope
                // const-expressions, which naga has not concretised.
                if matches!(op, naga::UnaryOperator::Negate)
                    && s == "0"
                    && matches!(
                        ctx.ty(*expr).inner_with(&self.module.types).scalar_kind(),
                        Some(naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat)
                    )
                {
                    s.push('.');
                }
                if wrap {
                    s.insert(0, '(');
                    s.push(')');
                }
                // WGSL reserves `--`: a `Negate` over a child rendering with a
                // leading `-` takes a space, one byte cheaper than parens
                // (`!`/`~` form no reserved adjacency).
                if matches!(op, naga::UnaryOperator::Negate) && !wrap && s.starts_with('-') {
                    s.insert(0, ' ');
                }
                s.insert_str(0, op_str);
                s
            }
            E::Binary { op, left, right } => {
                let op_str = binary_op_str(*op);
                let sp = self.bin_op_sep();
                let arena = &ctx.exprs;
                let lc = ctx.expr_names.contains_key(left);
                let rc = ctx.expr_names.contains_key(right);

                // Splat elision: WGSL's mixed scalar-vector overloads for
                // + - * / % let an uncached splat operand render as its bare
                // scalar, on at most one side and only while the other stays
                // a vector, or the result type changes.
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
                let left_is_vec = matches!(
                    ctx.ty(*left).inner_with(&self.module.types),
                    naga::TypeInner::Vector { .. }
                );
                let right_is_vec = matches!(
                    ctx.ty(*right).inner_with(&self.module.types),
                    naga::TypeInner::Vector { .. }
                );
                let (elide_l, elide_r) = match (left_scalar.is_some(), right_scalar.is_some()) {
                    // Both splats: the left one stays as the vector operand.
                    (true, true) => (false, true),
                    (true, false) => (right_is_vec, false),
                    (false, true) => (false, left_is_vec),
                    (false, false) => (false, false),
                };

                let (eff_l, eff_lc) = if elide_l {
                    let h = left_scalar.unwrap();
                    (h, ctx.expr_names.contains_key(h))
                } else {
                    (*left, lc)
                };
                let (eff_r, eff_rc) = if elide_r {
                    let h = right_scalar.unwrap();
                    (h, ctx.expr_names.contains_key(h))
                } else {
                    (*right, rc)
                };

                let wrap_l = child_needs_parens(eff_l, arena, *op, false, eff_lc);
                let mut wrap_r = child_needs_parens(eff_r, arena, *op, true, eff_rc);

                // A shift types as its left operand (`e2` is always u32), so a
                // bare literal there stays abstract: tint concretizes it to
                // i32 (`4294967295>>x` rejected, `100u>>x` retyped) while naga
                // takes the consumer's type.  Typed form, bypassing
                // `extracted_literals` (a hoisted bare const is abstract too).
                // Two bare literals pin nothing, so the operation evaluates
                // as AbstractInt: 64-bit and exact where the concrete i32 form
                // wraps, which both changes the value and turns an overflow
                // into a shader-creation error, and for a float pair spelled
                // `-1*0` it is the wrong type family outright.  Typing the
                // left operand pins the pair back.
                let widening_left = (!elide_l && !elide_r)
                    .then(|| {
                        let l = self.inline_scalar_literal(*left, ctx)?;
                        let r = self.inline_scalar_literal(*right, ctx)?;
                        (i32_binary_widens(*op, l, r)
                            || float_pair_reinfers_int(l, r, &self.options.float_precision))
                        .then_some(l)
                    })
                    .flatten();
                let ls = if let Some(lit) = widening_left {
                    literal_to_wgsl(lit, &self.options.float_precision)
                } else if !elide_l
                    && matches!(
                        op,
                        naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight
                    )
                    && let Some(lit) = self.inline_scalar_literal(*left, ctx)
                    && literal_bare_form_changes_type(lit)
                {
                    literal_to_wgsl(lit, &self.options.float_precision)
                } else if elide_l {
                    self.emit_expr(left_scalar.unwrap(), ctx)?
                } else {
                    self.emit_expr(*left, ctx)?
                };
                let rs = if elide_r {
                    let s = self.emit_expr(right_scalar.unwrap(), ctx)?;
                    // `a--b` would lex as a decrement in no-space mode.
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

                // `select`'s value operands must share one concrete type, so
                // a literal takes its typed form.
                let reject_str = if let naga::Expression::Literal(lit) = &ctx.exprs[*reject] {
                    literal_to_wgsl(*lit, &self.options.float_precision)
                } else {
                    self.emit_expr(*reject, ctx)?
                };
                // Template-list guard on the two non-final arguments; the
                // closing `)` covers the condition.
                if self.renders_as_bare_less(*reject, ctx) {
                    s.push('(');
                    s.push_str(&reject_str);
                    s.push(')');
                } else {
                    s.push_str(&reject_str);
                }
                s.push_str(sep);

                let accept_str = if let naga::Expression::Literal(lit) = &ctx.exprs[*accept] {
                    literal_to_wgsl(*lit, &self.options.float_precision)
                } else {
                    self.emit_expr(*accept, ctx)?
                };
                if self.renders_as_bare_less(*accept, ctx) {
                    s.push('(');
                    s.push_str(&accept_str);
                    s.push(')');
                } else {
                    s.push_str(&accept_str);
                }
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
                    // Derivatives take floats only; a bare `1` would infer i32.
                    if !ctx.expr_names.contains_key(expr) {
                        if let naga::Expression::Literal(lit) = ctx.exprs[*expr] {
                            s.push_str(&literal_to_wgsl(lit, &self.options.float_precision));
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
                    // `isNan`/`isInf` have no WGSL spelling and naga's WGSL
                    // frontend never produces them; refuse rather than emit
                    // an identifier no consumer recognises.
                    naga::RelationalFunction::IsNan | naga::RelationalFunction::IsInf => {
                        return Err(Error::Emit(format!(
                            "relational function {fun:?} has no WGSL spelling"
                        )));
                    }
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
                // `extractBits` / `insertBits` type as their value
                // operand(s) alone, so `offset` / `count` pin nothing;
                // every other builtin shares one type across its arguments
                // and any runtime sibling pins a literal.
                let pins_alone = matches!(
                    fun,
                    naga::MathFunction::ExtractBits | naga::MathFunction::InsertBits
                );
                // Only a float SCALAR loses its type this way: a vector renders
                // through a constructor naming the element type, and an integer
                // literal's bare form re-infers the integer it is.
                let float_scalar = match ctx.ty(*arg).inner_with(&self.module.types) {
                    naga::TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Float => Some(*s),
                    _ => None,
                };
                // The runtime sibling above pins nothing when EVERY argument is
                // a whole-valued float literal: `mix(1.f,2.f,1.f)` renders
                // `mix(1,2,1)`, three AbstractInt tokens typing the call
                // AbstractInt.  tint accepts that, naga does not, and naga is
                // the self-check, so one such call drops the WHOLE module to
                // naga's emitter; typing one argument re-pins the float.  Every
                // shared-type builtin, not just the two that fail today (5
                // corpus files at -110 bytes, 10 at +1).
                //
                // An extracted literal renders as its `const` name, which
                // carries a declared type and so pins on its own.  Treating
                // that as pinning is also what keeps the forced text off
                // `literal_extract`'s books: it only ever replaces a literal
                // that was going to render bare.
                let float_needs_pin = !pins_alone
                    && float_scalar.is_some_and(|scalar| {
                        [Some(*arg), *arg1, *arg2, *arg3]
                            .into_iter()
                            .flatten()
                            .all(|h| {
                                self.inline_scalar_literal(h, ctx).is_some_and(|lit| {
                                    let key =
                                        literal_extract_key(lit, &self.options.float_precision);
                                    !self.extracted_literals.contains_key(&key)
                                        && !literal_bare_form_pins_scalar(
                                            lit,
                                            scalar,
                                            &self.options.float_precision,
                                        )
                                })
                            })
                    });
                let typed_if_literal = |g: &Self,
                                        h: naga::Handle<naga::Expression>,
                                        ctx: &mut FunctionCtx<'a, '_>|
                 -> Result<String, Error> {
                    if pins_alone
                        && let Some(lit) = g.inline_scalar_literal(h, ctx)
                        && literal_bare_form_changes_type(lit)
                    {
                        Ok(literal_to_wgsl(lit, &g.options.float_precision))
                    } else {
                        g.emit_expr(h, ctx)
                    }
                };
                if float_needs_pin && let Some(lit) = self.inline_scalar_literal(*arg, ctx) {
                    s.push_str(&literal_to_wgsl(lit, &self.options.float_precision));
                } else {
                    s.push_str(&typed_if_literal(self, *arg, ctx)?);
                }
                if let Some(v) = arg1 {
                    s.push_str(sep);
                    if *fun == naga::MathFunction::InsertBits {
                        s.push_str(&typed_if_literal(self, *v, ctx)?);
                    } else {
                        s.push_str(&self.emit_expr(*v, ctx)?);
                    }
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
                let src_inner = ctx.ty(*expr).inner_with(&self.module.types);
                if let naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar: src_scalar,
                } = src_inner
                {
                    // `bitcast<T>` is restricted to numeric scalars and
                    // vectors, so a matrix `As { convert: None }` has no WGSL
                    // spelling; refuse and let the pipeline fall back.
                    if convert.is_none() {
                        return Err(Error::Emit(format!(
                            "matrix bitcast (As {{ convert: None }}) is not representable in WGSL \
                             in function '{}' (expr {}): source type {}",
                            ctx.display_name,
                            expr.index(),
                            type_inner_kind(src_inner),
                        )));
                    }
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
                    let mut s = target;
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
                            "unsupported cast source type in function '{}': {}",
                            ctx.display_name,
                            type_inner_kind(src_inner),
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
                // `bitcast<T>` excludes `bool` and abstract scalars
                // (WGSL #bit-reinterp-builtin-functions); refuse so the
                // fallback emitter handles the cast.
                if convert.is_none()
                    && matches!(
                        kind,
                        naga::ScalarKind::Bool
                            | naga::ScalarKind::AbstractInt
                            | naga::ScalarKind::AbstractFloat
                    )
                {
                    return Err(Error::Emit(format!(
                        "bitcast (As {{ convert: None }}) is not representable for \
                         scalar kind {:?} in function '{}'",
                        kind, ctx.display_name,
                    )));
                }
                // Narrowing a CONST width-8 vector (`vec2<f32>(vec2<f64>(.5lf,..))`)
                // is rejected by naga's frontend on re-parse, and naga's own
                // backend emits the same token, so the run() fallback cannot
                // save it; const_fold handles only the scalar case.  Fold an
                // inlined const Compose/Splat of width-8 literals to an
                // f32/i32/u32/bool target here; everything else (runtime
                // vectors, named operands, other targets, non-finite results)
                // takes the verbatim path.
                if convert.is_some()
                    && vec_size.is_some()
                    && src_width == 8
                    && matches!(
                        src_inner,
                        naga::TypeInner::Vector { scalar, .. }
                            if matches!(
                                scalar.kind,
                                naga::ScalarKind::Float
                                    | naga::ScalarKind::Sint
                                    | naga::ScalarKind::Uint
                            )
                    )
                    && !ctx.expr_names.contains_key(expr)
                    && let Some(folded) = self.try_emit_const_width8_vector_narrow(
                        *expr,
                        naga::Scalar {
                            kind: *kind,
                            width: target_width,
                        },
                        &target_inner,
                        ctx,
                    )?
                {
                    return Ok(folded);
                }
                let target = self.type_name_for_inner(&target_inner)?;
                let source = if let Some(lit) = self.inline_scalar_literal(*expr, ctx)
                    && as_operand_keeps_suffix(lit, *convert)
                {
                    literal_to_wgsl(lit, &self.options.float_precision)
                } else {
                    self.emit_expr(*expr, ctx)?
                };
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
                .get(expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult => ctx
                .expr_names
                .get(expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            // Free-standing builtin calls with no binding statement (unlike
            // `RayQueryProceedResult`, bound by its `RayQuery` statement) that
            // read the query's CURRENT traversal state, so
            // `compute_must_bind_loads` tracks them like loads and force-binds
            // a read that would otherwise re-evaluate past a query-mutating
            // statement.
            E::RayQueryGetIntersection { query, committed } => {
                let mut s = String::from(if *committed {
                    "rayQueryGetCommittedIntersection("
                } else {
                    "rayQueryGetCandidateIntersection("
                });
                s.push_str(&self.emit_ray_query_arg(*query, ctx)?);
                s.push(')');
                s
            }
            E::RayQueryVertexPositions { query, committed } => {
                let mut s = String::from(if *committed {
                    "getCommittedHitVertexPositions("
                } else {
                    "getCandidateHitVertexPositions("
                });
                s.push_str(&self.emit_ray_query_arg(*query, ctx)?);
                s.push(')');
                s
            }
            E::ArrayLength(e) => format!("arrayLength({})", self.emit_pointer_operand(*e, ctx)?),
            _ => {
                return Err(Error::Emit(format!(
                    "unsupported expression in function '{}' (expr {}): {}",
                    ctx.display_name,
                    expr.index(),
                    expression_kind(&ctx.exprs[expr]),
                )));
            }
        })
    }

    /// The concrete scalar literal `h` inlines as (a `Literal` or an
    /// unnamed `Constant` over one) unless already `let`-bound; every
    /// position that pins a literal's type resolves through this so the
    /// emitter and `literal_extract` agree.
    fn inline_scalar_literal(
        &self,
        h: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Literal> {
        if ctx.expr_names.contains_key(h) {
            return None;
        }
        let lit = match ctx.exprs[h] {
            naga::Expression::Literal(lit) => lit,
            naga::Expression::Constant(c) if self.module.constants[c].name.is_none() => {
                match self.module.global_expressions[self.module.constants[c].init] {
                    naga::Expression::Literal(lit) => lit,
                    _ => return None,
                }
            }
            _ => return None,
        };
        (!matches!(
            lit,
            naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
        ))
        .then_some(lit)
    }

    /// `(annotation, initializer)` of a hazard `let`. A `let` initializer
    /// pins nothing (`let a=5;` is i32), so a concrete literal takes its
    /// typed form and everything else (extracted or named constants,
    /// constructors) an explicit type; abstract operands bind bare since
    /// the default is their type.
    pub(super) fn const_hazard_binding_value(
        &self,
        operand: naga::Handle<naga::Expression>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(Option<String>, String), Error> {
        if let naga::Expression::Literal(lit) = ctx.exprs[operand]
            && !matches!(
                lit,
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
            )
        {
            let key = literal_extract_key(lit, &self.options.float_precision);
            if !self.extracted_literals.contains_key(&key) {
                return Ok((None, literal_to_wgsl(lit, &self.options.float_precision)));
            }
        }
        let inner = ctx.ty(operand).inner_with(&self.module.types);
        let abstract_typed = matches!(
            inner.scalar(),
            Some(naga::Scalar {
                kind: naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat,
                ..
            })
        );
        let annotation = if abstract_typed {
            None
        } else {
            Some(self.type_name_for_inner(inner)?)
        };
        Ok((annotation, self.emit_expr(operand, ctx)?))
    }

    fn concretize_abstract_literal_for_expr(
        &self,
        lit: naga::Literal,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        let inner = ctx.ty(expr).inner_with(&self.module.types);
        Ok(match concretize_abstract_literal_via_inner(lit, inner) {
            // Scan (`count_literals`) and emission both key on the CONCRETE
            // form, so a hot concretized literal substitutes its extracted
            // name; otherwise the typed form keeps the type pinned where
            // coercion could re-derive a different one.
            Some(ConcretizedAbstract::Lit(concrete)) => {
                let key = literal_extract_key(concrete, &self.options.float_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    Some(name.clone())
                } else {
                    Some(literal_to_wgsl(concrete, &self.options.float_precision))
                }
            }
            // Wrapper text (`f16(0.5f)`, `i32(<huge>)`) cannot be
            // substituted; the scan side skips counting it too.
            Some(ConcretizedAbstract::Text(text)) => Some(text),
            None => None,
        })
    }

    // MARK: Global expression emission

    /// A module-scope expression (constant, override or global initializer)
    /// through the one emitter, under [`Generator::module_ctx`].  A root
    /// literal keeps its suffix unless the bare spelling infers the same
    /// concrete type: nothing pins a declaration's initializer, so `256`
    /// would turn a `u32` constant abstract and `i32` at its next `let`.
    /// `self_typed` additionally demands the concrete form of a root literal
    /// or constructor - the `const` emitter drops `: T` on exactly those
    /// shapes, and an abstract `7` or `vec2(42,43)` is then a different type
    /// that naga's front end folds away instead of declaring.
    pub(super) fn emit_global_expr(
        &self,
        expr: naga::Handle<naga::Expression>,
        self_typed: bool,
    ) -> Result<String, Error> {
        let mut ctx = self.module_ctx();
        self.emit_global_expr_in(expr, self_typed, &mut ctx)
    }

    /// [`Generator::emit_global_expr`] against a caller-owned context.  Building
    /// one costs a clone of every constant name, so the declaration sections
    /// share a single context and name each constant in it as they go instead
    /// of rebuilding per declaration.
    pub(super) fn emit_global_expr_in(
        &self,
        expr: naga::Handle<naga::Expression>,
        self_typed: bool,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        // A shared initializer already declared under a name emits the name.
        if !ctx.expr_names.contains_key(expr)
            && let naga::Expression::Literal(lit) = self.module.global_expressions[expr]
            && (self_typed || literal_bare_form_changes_type(lit))
        {
            return Ok(literal_to_wgsl(lit, &self.options.float_precision));
        }
        ctx.pinned_root = self_typed.then_some(expr);
        let text = self.emit_expr(expr, ctx);
        ctx.pinned_root = None;
        text
    }

    /// Fold a narrowing cast of a const width-8 vector into a converted
    /// constructor (`vec2<f32>(vec2<f64>(.5lf,1.5lf))` -> `vec2f(.5,1.5)`).
    /// `None` (verbatim `target(source)` path) unless the operand is a
    /// Compose/Splat of width-8 literals that all convert to finite target
    /// literals - an f64->f32 overflow stays a diagnosable hard error rather
    /// than a silent `inf` token.
    fn try_emit_const_width8_vector_narrow(
        &self,
        operand: naga::Handle<naga::Expression>,
        target: naga::Scalar,
        target_inner: &naga::TypeInner,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        let lits: Vec<naga::Literal> = match &ctx.exprs[operand] {
            naga::Expression::Compose { components, .. } => {
                let mut out = Vec::with_capacity(components.len());
                for &c in components.iter() {
                    match ctx.exprs[c] {
                        naga::Expression::Literal(l) if literal_is_width8(l) => out.push(l),
                        _ => return Ok(None),
                    }
                }
                out
            }
            naga::Expression::Splat { size, value } => match ctx.exprs[*value] {
                naga::Expression::Literal(l) if literal_is_width8(l) => vec![l; *size as usize],
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let mut converted = Vec::with_capacity(lits.len());
        for l in lits {
            match crate::passes::expr_util::cast_width8_to(l, target) {
                Some(lit) => converted.push(lit),
                None => return Ok(None),
            }
        }
        let mut s = self.type_name_for_inner(target_inner)?;
        s.push('(');
        // Splat form when all components are bit-equal (`-0` stays distinct
        // from `0`).
        let all_same =
            converted.len() > 1 && converted.iter().all(|l| literal_bit_eq(l, &converted[0]));
        if all_same {
            s.push_str(&literal_to_wgsl_bare(
                converted[0],
                &self.options.float_precision,
            ));
        } else {
            let sep = self.comma_sep();
            for (i, lit) in converted.iter().enumerate() {
                if i > 0 {
                    s.push_str(sep);
                }
                s.push_str(&literal_to_wgsl_bare(*lit, &self.options.float_precision));
            }
        }
        s.push(')');
        Ok(Some(s))
    }

    // MARK: Type helpers

    /// WGSL name of `inner`, through the alias table.
    pub(super) fn type_name_for_inner(&self, inner: &naga::TypeInner) -> Result<String, Error> {
        let res = naga::proc::TypeResolution::Value(inner.clone());
        type_resolution_name(
            &res,
            self.module,
            &self.type_names,
            &self.override_names,
            &self.shadowed_type_aliases,
        )
    }

    /// WGSL name of `expr`'s resolved type.
    pub(super) fn expr_type_name(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        let res = &ctx.ty(expr);
        type_resolution_name(
            res,
            self.module,
            &self.type_names,
            &self.override_names,
            &self.shadowed_type_aliases,
        )
    }

    /// WGSL name of a type handle, alias first.
    pub(super) fn type_ref(&self, ty: naga::Handle<naga::Type>) -> Result<String, Error> {
        if let Some(name) = self.type_names.get(ty) {
            return Ok(name.clone());
        }
        type_inner_name(
            &self.module.types[ty].inner,
            self.module,
            &self.type_names,
            &self.override_names,
            &self.shadowed_type_aliases,
        )
    }

    /// Shortest zero of `ty`: a scalar literal (`0i`, `false`) or `T()`.
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

    /// Append `:<type>` (WGSL zero-initialises an uninitialised local) or
    /// `=<zero-literal>` after `var <name>`, no trailing `;`, whichever
    /// renders shorter: `=0i` beats `:i32`, `:bool` beats `=false`,
    /// composites favour the annotation, and a short alias (`:j`) can
    /// undercut even `=0i`, so lengths are compared rather than assumed.
    /// Every zero-init `var` routes here so the choice never drifts.
    pub(super) fn emit_zero_init_tail(
        &mut self,
        ty: naga::Handle<naga::Type>,
    ) -> Result<(), Error> {
        let type_str = self.type_ref(ty)?;
        if let naga::TypeInner::Scalar(s) = self.module.types[ty].inner {
            let zlit = scalar_zero(s.kind, s.width);
            if zlit.len() < type_str.len() {
                self.push_assign();
                self.out.push_str(zlit);
                return Ok(());
            }
        }
        self.push_colon();
        self.out.push_str(&type_str);
        Ok(())
    }

    /// Constructor name for a vector `Compose`: the rendered type name, or
    /// bare `vecN` when that is strictly shorter AND a component pins the
    /// element type - a non-literal whose concrete scalar equals it, or a
    /// literal whose bare form re-infers it (float-shaped -> f32, any integer
    /// -> i32, per [`literal_bare_form_pins_scalar`]); `vec2u(4,1)` cannot
    /// drop its `u`.
    ///
    /// The non-literal arm relies on the pinning component rendering TYPED.
    /// An unnamed literal-init `Constant` resolves to the right scalar yet
    /// emits its bare token, so it cannot be the sole pinner - and never is,
    /// because `const_fold` runs first and folds every all-constant
    /// `Compose`, so a surviving vector has a runtime component (`ZeroValue`
    /// emits typed either way).  A pass that leaves an unnamed const as the
    /// only non-literal component must revisit this gate.
    fn vector_ctor_name(
        &self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        let type_str = self.type_ref(ty)?;
        if let naga::TypeInner::Vector { size, scalar } = self.module.types[ty].inner {
            let bare = format!("vec{}", super::syntax::vector_size_num(size));
            if bare.len() < type_str.len()
                && components.iter().any(|&c| match &ctx.exprs[c] {
                    naga::Expression::Literal(lit) => {
                        literal_bare_form_pins_scalar(*lit, scalar, &self.options.float_precision)
                    }
                    _ => {
                        type_inner_scalar(ctx.ty(c).inner_with(&self.module.types)) == Some(scalar)
                    }
                })
            {
                return Ok(bare);
            }
        }
        Ok(type_str)
    }

    /// `array(...)` for an array `Compose` when shorter than the full/aliased
    /// `array<T,N>` and inference is guaranteed to reproduce `base`: naga
    /// takes the element type from the FIRST component (after leaf-scalar
    /// consensus), and a bare abstract literal there would re-infer
    /// (`array<u32,2>(1,2)` -> `array<i32,2>`, a silent retype), so that
    /// component must be a `Compose`/`ZeroValue`/`Constant`/`Override` of
    /// exactly `base`, which always emits concretely typed.  Rewrites the
    /// constructor name only, never a type annotation.
    fn array_ctor_name(
        &self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
        full_name: &str,
        arena: &naga::Arena<naga::Expression>,
    ) -> Option<&'static str> {
        let naga::TypeInner::Array { base, .. } = self.module.types[ty].inner else {
            return None;
        };
        if components.is_empty() || "array".len() >= full_name.len() {
            return None;
        }
        let pins = components.first().is_some_and(|&c| match &arena[c] {
            naga::Expression::Compose { ty: cty, .. } => *cty == base,
            naga::Expression::ZeroValue(zty) => *zty == base,
            naga::Expression::Constant(h) => self.module.constants[*h].ty == base,
            naga::Expression::Override(h) => self.module.overrides[*h].ty == base,
            _ => false,
        });
        pins.then_some("array")
    }

    /// Collapse maximal runs of >=2 equal adjacent scalar components into
    /// sub-vector splats (`vec4f(0,0,0,2)` -> `vec4f(vec3f(),2)`); `None`
    /// without a run of length `2..N`, and the caller keeps the result only
    /// when strictly shorter.  Run members are equal under [`exprs_splat_eq`]
    /// (bit pattern / handle identity), so one shared value reproduces the
    /// same lanes, and a sub-vector constructor is postfix-safe.
    fn try_subsplat_compose(
        &self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        let naga::TypeInner::Vector { size, scalar } = self.module.types[ty].inner else {
            return Ok(None);
        };
        let n = size as usize;
        if components.len() != n {
            return Ok(None);
        }
        // A run forms `vecK(scalar)` only from scalar components.
        if !components.iter().all(|&c| {
            matches!(
                ctx.ty(c).inner_with(&self.module.types),
                naga::TypeInner::Scalar(_)
            )
        }) {
            return Ok(None);
        }
        // Runs are found under an immutable borrow before emission needs
        // `ctx` mutably.
        let (runs, zero): (Vec<(usize, usize)>, Vec<bool>) = {
            let arena = &ctx.exprs;
            let mut runs = Vec::new();
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && exprs_splat_eq(&arena[components[i]], &arena[components[j]]) {
                    j += 1;
                }
                runs.push((i, j - i));
                i = j;
            }
            let zero = runs
                .iter()
                .map(|&(s, _)| compose_is_all_zero(components[s], arena))
                .collect();
            (runs, zero)
        };
        if !runs.iter().any(|&(_, k)| k >= 2 && k < n) {
            return Ok(None);
        }
        let mut parts: Vec<String> = Vec::with_capacity(n);
        for (ri, &(s, k)) in runs.iter().enumerate() {
            if k >= 2 && k < n {
                let sub_size = match k {
                    2 => naga::VectorSize::Bi,
                    3 => naga::VectorSize::Tri,
                    _ => naga::VectorSize::Quad,
                };
                let sub_name = self.type_name_for_inner(&naga::TypeInner::Vector {
                    size: sub_size,
                    scalar,
                })?;
                if zero[ri] {
                    parts.push(format!("{sub_name}()"));
                } else {
                    parts.push(format!(
                        "{sub_name}({})",
                        self.emit_constructor_arg(components[s], ctx)?
                    ));
                }
            } else {
                for &c in &components[s..s + k] {
                    parts.push(self.emit_constructor_arg(c, ctx)?);
                }
            }
        }
        let name = self.vector_ctor_name(ty, components, ctx)?;
        let sep = self.comma_sep();
        Ok(Some(format!("{name}({})", parts.join(sep))))
    }

    /// Member name at `index` of the struct `resolution` names (directly or
    /// through a pointer): mangled, else source, else positional.
    fn field_name_of(&self, resolution: &naga::proc::TypeResolution, index: u32) -> Option<String> {
        use naga::proc::TypeResolution;
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
            TypeResolution::Value(naga::TypeInner::Pointer { base: bty, .. }) => {
                match &self.module.types[*bty].inner {
                    naga::TypeInner::Struct { members, .. } => (Some(*bty), members),
                    _ => return None,
                }
            }
            TypeResolution::Value(_) => return None,
        };
        if let Some(h) = ty_handle
            && let Some(mangled) = self.member_names.get(&(h, index))
        {
            return Some(mangled.clone());
        }
        members
            .get(index as usize)
            .map(|m| m.name.clone().unwrap_or_else(|| format!("m{}", index)))
    }

    /// The WGSL component letter (`x`..`w`) for `index` when `resolution`
    /// is a vector or a pointer to one; `.x` is one byte shorter than `[0]`.
    fn component_letter_of(
        &self,
        resolution: &naga::proc::TypeResolution,
        index: u32,
    ) -> Option<char> {
        const COMPONENTS: [char; 4] = ['x', 'y', 'z', 'w'];
        if index > 3 {
            return None;
        }
        let inner = resolution.inner_with(&self.module.types);
        let is_vec = matches!(
            inner,
            naga::TypeInner::Vector { .. } | naga::TypeInner::ValuePointer { size: Some(_), .. }
        ) || matches!(inner, naga::TypeInner::Pointer { base: bty, .. } if matches!(
            self.module.types[*bty].inner,
            naga::TypeInner::Vector { .. }
        ));
        is_vec.then(|| COMPONENTS[index as usize])
    }

    /// Append an `AccessIndex`'s suffix - struct field, vector component or
    /// numeric index - to an already-rendered base.  Shared by the lvalue and
    /// rvalue emitters: a place and a read of it must spell the same access,
    /// or a store lands where its matching load does not.
    fn push_access_index(
        &self,
        out: &mut String,
        base: naga::Handle<naga::Expression>,
        index: u32,
        ctx: &FunctionCtx<'a, '_>,
    ) {
        if let Some(field_name) = self.struct_field_name(base, index, ctx) {
            out.push('.');
            out.push_str(&field_name);
        } else if let Some(c) = self.vector_component_name(base, index, ctx) {
            out.push('.');
            out.push(c);
        } else {
            out.push('[');
            out.push_str(&index.to_string());
            out.push(']');
        }
    }

    pub(super) fn struct_field_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<String> {
        self.field_name_of(ctx.ty(base), index)
    }

    fn vector_component_name(
        &self,
        base: naga::Handle<naga::Expression>,
        index: u32,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<char> {
        self.component_letter_of(ctx.ty(base), index)
    }

    const SWIZZLE_LETTERS: [char; 4] = ['x', 'y', 'z', 'w'];

    /// `(base, component)` of a swizzle-groupable Compose component: an
    /// uncached `AccessIndex` on a vector value, or a `Load` of one on a
    /// pointer to a vector.
    fn swizzle_component(
        &self,
        handle: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<(naga::Handle<naga::Expression>, u32)> {
        if ctx.expr_names.contains_key(handle) {
            return None;
        }
        let arena = &ctx.exprs;

        if let naga::Expression::AccessIndex { base, index } = arena[handle]
            && index <= 3
        {
            let inner = ctx.ty(base).inner_with(&self.module.types);
            if matches!(inner, naga::TypeInner::Vector { .. }) {
                return Some((base, index));
            }
        }

        if let naga::Expression::Load { pointer } = arena[handle]
            && let naga::Expression::AccessIndex { base, index } = arena[pointer]
            && index <= 3
        {
            let inner = ctx.ty(base).inner_with(&self.module.types);
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

        None
    }

    /// The base an uncached identity-swizzle `Compose` (`vecN(b.0,..,b.N-1)`)
    /// collapses to, for loose positions (comma-delimited arguments with
    /// nothing appended) that emit it without the postfix wrap the general
    /// collapse applies.  Must agree with the identity branch of
    /// [`Self::try_compose_as_full_swizzle`].
    fn compose_identity_collapse_base(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Handle<naga::Expression>> {
        if ctx.expr_names.contains_key(expr) {
            return None;
        }
        let naga::Expression::Compose { ty, components } = &ctx.exprs[expr] else {
            return None;
        };
        if !matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
            return None;
        }
        if components.len() < 2 || components.len() > 4 {
            return None;
        }
        let mut common_base: Option<naga::Handle<naga::Expression>> = None;
        let mut pattern: Vec<u32> = Vec::with_capacity(components.len());
        for &comp in components.iter() {
            let (base, idx) = self.swizzle_component(comp, ctx)?;
            match common_base {
                None => common_base = Some(base),
                Some(b) if b == base => {}
                _ => return None,
            }
            pattern.push(idx);
        }
        let base = common_base.unwrap();
        let src_n = self.vector_size_of(base, ctx)?;
        if pattern.len() == src_n && pattern.iter().enumerate().all(|(i, &idx)| idx == i as u32) {
            Some(base)
        } else {
            None
        }
    }

    /// A vector `Compose` as a bare swizzle: `vec3f(v.x,v.y,v.z)` -> `v.xyz`,
    /// and the identity over a same-size vector -> `v`.
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

        let source_size = self.vector_size_of(base, ctx);
        if let Some(src_n) = source_size
            && pattern.len() == src_n
            && pattern.iter().enumerate().all(|(i, &idx)| idx == i as u32)
        {
            // The substituted text replaces the whole Compose, so the parent
            // cannot see an operator expression; an uncached
            // Binary/Unary/Select base keeps its parens.
            return Ok(Some(self.emit_postfix_base(base, ctx)?));
        }

        let mut s = self.emit_postfix_base(base, ctx)?;
        s.push('.');
        for &idx in &pattern {
            s.push(Self::SWIZZLE_LETTERS[idx as usize]);
        }
        Ok(Some(s))
    }

    /// Write the components to `s`, collapsing consecutive same-base swizzle
    /// components into one swizzle (`v.x,v.y` -> `v.xy`); `false`, with
    /// nothing written, when no group forms.
    fn emit_compose_grouped(
        &self,
        s: &mut String,
        components: &[naga::Handle<naga::Expression>],
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<bool, Error> {
        let mut groups: Vec<ComposeGroup> = Vec::new();
        let mut i = 0;
        while i < components.len() {
            if let Some((base, idx)) = self.swizzle_component(components[i], ctx) {
                let mut indices = vec![idx];
                let mut j = i + 1;
                while j < components.len() {
                    if let Some((b2, idx2)) = self.swizzle_component(components[j], ctx)
                        && b2 == base
                    {
                        indices.push(idx2);
                        j += 1;
                        continue;
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
                    s.push_str(&self.emit_postfix_base(*base, ctx)?);
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

    /// Lane count of `expr`'s vector or pointer-to-vector type.
    fn vector_size_of(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<usize> {
        let inner = ctx.ty(expr).inner_with(&self.module.types);
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

    /// Scalar of `expr`'s scalar/vector type, for concretising a literal
    /// child.
    pub(super) fn expr_scalar_hint(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Scalar> {
        match ctx.ty(expr).inner_with(&self.module.types) {
            naga::TypeInner::Scalar(s) => Some(*s),
            naga::TypeInner::Vector { scalar, .. } => Some(*scalar),
            _ => None,
        }
    }

    /// `expr` with every bare literal in it forced to `hint`'s TYPED
    /// spelling, for positions naga's lowerer gives no abstract coercion
    /// (the `rayQueryGenerateIntersection` hit_t slot, subgroup-op operands),
    /// where a bare literal would re-concretize to i32.
    pub(super) fn emit_expr_with_scalar_hint(
        &self,
        expr: naga::Handle<naga::Expression>,
        hint: Option<naga::Scalar>,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<String, Error> {
        if !ctx.expr_names.contains_key(expr)
            && let Some(scalar) = hint
        {
            match ctx.exprs[expr] {
                naga::Expression::Binary { op, left, right }
                    if matches!(
                        op,
                        naga::BinaryOperator::Add | naga::BinaryOperator::Subtract
                    ) =>
                {
                    let arena = &ctx.exprs;
                    let lc = ctx.expr_names.contains_key(left);
                    let rc = ctx.expr_names.contains_key(right);
                    let wrap_l = child_needs_parens(left, arena, op, false, lc);
                    let wrap_r = child_needs_parens(right, arena, op, true, rc);
                    let ls = self.emit_expr_with_scalar_hint(left, Some(scalar), ctx)?;
                    let rs = self.emit_expr_with_scalar_hint(right, Some(scalar), ctx)?;
                    let op_str = binary_op_str(op);
                    let sp = self.bin_op_sep();
                    return Ok(assemble_binary(&ls, &rs, op_str, sp, wrap_l, wrap_r));
                }
                naga::Expression::Literal(lit) => {
                    if let Some(concrete) = self.concretize_abstract_literal_for_scalar(lit, scalar)
                    {
                        return Ok(concrete);
                    }
                }
                naga::Expression::Constant(h) => {
                    let c = &self.module.constants[h];
                    if c.name.is_none()
                        && let naga::Expression::Literal(lit) =
                            self.module.global_expressions[c.init]
                        && let Some(concrete) =
                            self.concretize_abstract_literal_for_scalar(lit, scalar)
                    {
                        return Ok(concrete);
                    }
                }
                _ => {}
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
            // A concrete float matching the hint still needs the TYPED
            // spelling: the default path renders `10.0` as `10`, which
            // re-parses as a float only under abstract coercion, and a hinted
            // position has none.
            L::F32(v) if target == naga::Scalar::F32 => Some(L::F32(v)),
            L::F64(v) if target == naga::Scalar::F64 => Some(L::F64(v)),
            L::F16(v) if target == naga::Scalar::F16 => Some(L::F16(v)),
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

        Some(literal_to_wgsl(concrete, &self.options.float_precision))
    }
}

/// Swizzle-grouped Compose components.
enum ComposeGroup {
    /// >=2 consecutive components indexing the same vector base.
    Swizzle {
        base: naga::Handle<naga::Expression>,
        indices: Vec<u32>,
    },
    Single(naga::Handle<naga::Expression>),
}

// WGSL operator precedence (<https://www.w3.org/TR/WGSL/#operator-precedence>),
// higher binds tighter.

const PREC_SHIFT: u8 = 8;
const PREC_ADDITIVE: u8 = 9;
const PREC_MULTIPLICATIVE: u8 = 10;
const PREC_UNARY: u8 = 11;

// MARK: Binary-operator rendering

/// Precedence level, higher binds tighter; ties and the non-associative
/// grammar levels are the caller's job.
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
        B::ShiftLeft | B::ShiftRight => PREC_SHIFT,
        B::Add | B::Subtract => PREC_ADDITIVE,
        B::Multiply | B::Divide | B::Modulo => PREC_MULTIPLICATIVE,
    }
}

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

/// `true` when a binary or unary child needs parentheses as an operand of
/// `parent_op`, per operator precedence, associativity, and the WGSL grammar
/// levels that forbid bare operands (bitwise, shift, comparison, and the
/// no-relative-precedence `&&`/`||` mix).
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

    // Bitwise `&`/`|`/`^` take `unary_expression` on both sides, so ANY
    // binary child (`a^b-1u`, `a|b<<c`) is ill-formed even where precedence
    // alone would group it: naga's permissive parser round-trips it, so no
    // fallback fires, but Tint/Dawn reject it.  The grammar's left-recursion
    // (`binary_and_expression '&' unary_expression`) keeps a same-operator
    // LEFT child bare (`a&b&c`).
    if matches!(
        parent_op,
        naga::BinaryOperator::And
            | naga::BinaryOperator::ExclusiveOr
            | naga::BinaryOperator::InclusiveOr
    ) {
        return match child_op {
            Some(op) => is_right || op != parent_op,
            // Unary / atom children already satisfy `unary_expression`.
            None => false,
        };
    }

    // WGSL shift operators require `unary_expression` on both sides.
    if matches!(
        parent_op,
        naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight
    ) {
        return child_prec < PREC_UNARY;
    }

    // All six comparison operators share one non-associative grammar level
    // whose operands are `shift_expression`s, so any comparison child is
    // wrapped: Dawn/Tint reject `a<b==c<d` ("mixing '<' and '==' requires
    // parenthesis") though naga round-trips it.  Hence `< PREC_SHIFT` rather
    // than `<= parent_prec`: `==`/`!=` (6) and the relational quartet (7)
    // differ in this table yet share the grammar level.  Always
    // meaning-preserving - the parenthesised grouping is the only well-typed
    // one.
    if matches!(
        parent_op,
        naga::BinaryOperator::Less
            | naga::BinaryOperator::LessEqual
            | naga::BinaryOperator::Greater
            | naga::BinaryOperator::GreaterEqual
            | naga::BinaryOperator::Equal
            | naga::BinaryOperator::NotEqual
    ) {
        // A bare `<` opens a template candidate in WGSL's scanner (`<=`/`<<`
        // never do) and a top-level `>>` to its right closes it: `a<b>>c`
        // scans as the template `a<b>` plus `>c`, rejected by strict parsers
        // (naga's self-check then forces a whole-file fallback).  `>>` is
        // top-level in the right operand only as the child's ROOT; deeper it
        // is already parenthesised.  Greater-family children cannot appear
        // here (a bool result is untypeable under a comparison).
        if parent_op == naga::BinaryOperator::Less
            && is_right
            && child_op == Some(naga::BinaryOperator::ShiftRight)
        {
            return true;
        }
        return child_prec < PREC_SHIFT;
    }

    // `&&`/`||` operands are each a `relational_expression`: comparisons,
    // shifts and arithmetic are bare, another logical or a bitwise expression
    // is not; naga round-trips the bare forms, Tint/Dawn reject them.
    if matches!(
        parent_op,
        naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
    ) {
        return match child_op {
            // Mixing `&&` and `||` bare is illegal; left-recursion keeps a
            // same-operator LEFT child bare, while a same-operator RIGHT child
            // stays wrapped to preserve the IR's tree shape (re-grouping a
            // right-leaning chain would perturb idempotence).
            Some(naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr) => {
                child_op != Some(parent_op) || is_right
            }
            // A bitwise expression is not a `relational_expression` (Tint:
            // "mixing '|' and '&&' requires parenthesis"); `bool | bool` is
            // legal and the short-circuit re-sugar collapses `(a | b) && c`
            // into one logical Binary, so this shape is reachable.  (`bool ^
            // bool` is invalid WGSL and never arrives; listing it is harmless.)
            Some(
                naga::BinaryOperator::InclusiveOr
                | naga::BinaryOperator::ExclusiveOr
                | naga::BinaryOperator::And,
            ) => true,
            _ => false,
        };
    }

    // Left-associative: a left child needs parens below the parent, a right
    // child at or below it (preserving the tree shape).
    if is_right {
        child_prec <= parent_prec
    } else {
        child_prec < parent_prec
    }
}

/// Every Binary precedence is below Unary, so only an uncached Binary
/// operand is wrapped.
fn unary_child_needs_parens(
    child: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    is_cached: bool,
) -> bool {
    !is_cached && matches!(arena[child], naga::Expression::Binary { .. })
}

/// `left op right` with the requested wraps and beautify spacing; every
/// binary emission funnels through here.
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
        // No-space mode: keep the lexer from fusing `-` and `-b` into the
        // reserved decrement, `/` and `/` into a line comment (impossible
        // from valid IR, guarded for symmetry) or `/` and `*p` (a pointer
        // deref) into a block-comment opener.
        if let (Some(&oc), Some(&rc)) = (op_str.as_bytes().last(), rs.as_bytes().first())
            && ((oc == rc && (oc == b'-' || oc == b'/')) || (oc == b'/' && rc == b'*'))
        {
            s.push(' ');
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

/// Projection of an abstract literal to its concrete form at a use site.  The
/// emitter and the extraction scan (`literal_extract`'s `count_literals`)
/// MUST share it so the [`literal_extract_key`] computed during the scan
/// matches the key looked up at emission; otherwise a hot literal is
/// extracted under one key while emission bypasses the lookup, leaving the
/// `const` unreferenced.
pub(super) enum ConcretizedAbstract {
    /// Consult `extracted_literals` by [`literal_extract_key`] before falling
    /// back to [`literal_to_wgsl`].
    Lit(naga::Literal),
    /// Wrapper text (`f16(...)`; `i32(...)`/`u32(...)`/`u64(...)` casts for
    /// out-of-range `AbstractInt`s) that both sides must skip for extraction:
    /// it is not one literal token and no `LiteralExtractKey` reconstructs it.
    Text(String),
}

/// Pure projection of an abstract literal through the resolved scalar/vector
/// type at its use site; `None` for a concrete literal or a type that pins
/// no scalar.  An out-of-range `AbstractInt` becomes explicit cast text, and
/// any abstract at `f16` becomes `f16(...f)` because WGSL has no abstract ->
/// f16 literal shorthand.
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

/// Every component of a vector `Compose` is provably the same value
/// (identical handles, or equal under [`exprs_splat_eq`]), so it can render
/// as the splat `vec3f(x)`.
pub(super) fn compose_is_splat(
    components: &[naga::Handle<naga::Expression>],
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    debug_assert!(components.len() > 1);
    let first = components[0];
    if components[1..].iter().all(|&c| c == first) {
        return true;
    }
    let first_expr = &arena[first];
    components[1..]
        .iter()
        .all(|&c| exprs_splat_eq(first_expr, &arena[c]))
}

/// Column-major scalar handles of a matrix `Compose` whose every column is a
/// vector `Compose` of exactly `rows` components, for the flat form
/// `mat2x2f(a,b,c,d)`; `None` keeps the column form.  A `vecR` built from
/// `R` components is necessarily `R` scalars (a vector sub-component would
/// lower the count), so the structural test needs no per-component type
/// lookup and works for both function and global arenas.  Splat, variable,
/// let-bound and swizzle columns are excluded, keeping the rewrite a regroup
/// of the scalar leaves the column form already emits; both forms lower to
/// byte-identical IR (f16 and negative leaves included).
pub(super) fn matrix_flatten_scalars(
    ty: naga::Handle<naga::Type>,
    components: &[naga::Handle<naga::Expression>],
    types: &naga::UniqueArena<naga::Type>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<Vec<naga::Handle<naga::Expression>>> {
    let naga::TypeInner::Matrix { columns, rows, .. } = types[ty].inner else {
        return None;
    };
    if components.len() != columns as usize {
        return None;
    }
    let rows = rows as usize;
    let mut flat = Vec::with_capacity(columns as usize * rows);
    for &col in components {
        let naga::Expression::Compose {
            ty: col_ty,
            components: sub,
        } = &arena[col]
        else {
            return None;
        };
        let naga::TypeInner::Vector { size, .. } = types[*col_ty].inner else {
            return None;
        };
        if size as usize != rows || sub.len() != rows {
            return None;
        }
        flat.extend_from_slice(sub);
    }
    Some(flat)
}

/// Bit-exact `+0` (or integer `0`), stricter than `literal_is_zero`: folding
/// `-0.0` into a zero-value constructor would flip the sign bit
/// (`1.0/-0.0 == -inf`), and `Bool` is excluded so the fold never removes a
/// `false` a constructor needs.
fn literal_is_strict_numeric_zero(l: naga::Literal) -> bool {
    use naga::Literal as L;
    match l {
        L::F16(v) => v.to_bits() == 0,
        L::F32(v) => v.to_bits() == 0,
        L::F64(v) => v.to_bits() == 0,
        L::AbstractFloat(v) => v.to_bits() == 0,
        L::I16(v) => v == 0,
        L::U16(v) => v == 0,
        L::I32(v) => v == 0,
        L::U32(v) => v == 0,
        L::I64(v) => v == 0,
        L::U64(v) => v == 0,
        L::AbstractInt(v) => v == 0,
        L::Bool(_) => false,
    }
}

/// Provably all `+0` (strict-zero `Literal`, `ZeroValue`, or `Splat`/`Compose`
/// of those); drives the `vec2f(0,0)` -> `vec2f()` fold.
fn compose_is_all_zero(
    h: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Expression as E;
    match &arena[h] {
        E::Literal(l) => literal_is_strict_numeric_zero(*l),
        E::ZeroValue(_) => true,
        E::Splat { value, .. } => compose_is_all_zero(*value, arena),
        E::Compose { components, .. } => components.iter().all(|&c| compose_is_all_zero(c, arena)),
        _ => false,
    }
}

/// Conservative value equality for splat/run collapse: two `Literal`s with
/// identical bit patterns (`-0.0` differs from `+0.0`) or the same
/// `Constant`/`Override`/`ZeroValue` handle; every other pair (`Load`,
/// `CallResult`, `Binary`, ...) is `false`, so impure or reorder-sensitive
/// components never share one value.
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

/// Scalar of a scalar or vector type.
fn type_inner_scalar(inner: &naga::TypeInner) -> Option<naga::Scalar> {
    match inner {
        naga::TypeInner::Scalar(s) => Some(*s),
        naga::TypeInner::Vector { scalar, .. } => Some(*scalar),
        _ => None,
    }
}

/// Negated comparison operator; an ordered one is NaN-unsafe on floats,
/// which the caller must check.
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

/// Operators with WGSL scalar-vector overloads (`+ - * / %`), where a splat
/// operand may collapse to its scalar.
pub(super) fn is_arithmetic_op(op: naga::BinaryOperator) -> bool {
    use naga::BinaryOperator as B;
    matches!(
        op,
        B::Add | B::Subtract | B::Multiply | B::Divide | B::Modulo
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ConcretizedAbstract, concretize_abstract_literal_via_inner, literal_bare_form_pins_scalar,
        literal_needs_typed_form_outside_constructor,
    };
    use naga::Literal as L;

    #[test]
    fn literal_pin_rules_follow_abstract_defaults() {
        let precision = crate::config::FloatPrecision::default();
        let pins = |lit, scalar| literal_bare_form_pins_scalar(lit, scalar, &precision);
        // Float-form f32 tokens pin f32; whole numbers render bare-int and
        // re-infer i32, so they must not.
        assert!(pins(L::F32(1.5), naga::Scalar::F32));
        assert!(pins(L::F32(1e-6), naga::Scalar::F32));
        assert!(!pins(L::F32(1.0), naga::Scalar::F32));
        assert!(!pins(L::AbstractFloat(2.0), naga::Scalar::F32));
        // Any integer literal pins i32 (AbstractInt's default).
        assert!(pins(L::I32(7), naga::Scalar::I32));
        assert!(pins(L::AbstractInt(-3), naga::Scalar::I32));
        // Scalars whose abstract default differs never literal-pin.
        assert!(!pins(L::U32(4), naga::Scalar::U32));
        assert!(!pins(L::F16(half::f16::from_f32(0.5)), naga::Scalar::F16));
        assert!(!pins(L::F64(0.5), naga::Scalar::F64));
    }

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
        let inner = scalar_inner(naga::ScalarKind::Sint, 4);
        assert!(concretize_abstract_literal_via_inner(L::I32(5), &inner).is_none());
        assert!(concretize_abstract_literal_via_inner(L::F32(0.5), &inner).is_none());
        assert!(concretize_abstract_literal_via_inner(L::Bool(true), &inner).is_none());
    }

    #[test]
    fn concretize_returns_none_for_unsupported_inner() {
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
        // Both sides skip extraction for cast text, so the exact spelling is
        // pinned.
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
        // A `Compose` argument literal resolves to the vector type.
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

    // MARK: Typed-form gate for non-AbstractInt/AbstractFloat default
    // types (F16 / F64 / I64 / U64).  These cases would silently
    // re-type to f32 / i32 under WGSL's abstract-coercion default if
    // emitted bare outside a constructor.

    #[test]
    fn typed_form_gate_flags_f16_f64_i64_u64() {
        use naga::Literal as L;
        assert!(literal_needs_typed_form_outside_constructor(L::F16(
            half::f16::from_f32(0.5)
        )));
        assert!(literal_needs_typed_form_outside_constructor(L::F64(0.5)));
        assert!(literal_needs_typed_form_outside_constructor(L::I64(7)));
        assert!(literal_needs_typed_form_outside_constructor(L::U64(7)));
    }

    #[test]
    fn typed_form_gate_passes_safe_types_through() {
        use naga::Literal as L;
        assert!(!literal_needs_typed_form_outside_constructor(L::I32(7)));
        assert!(!literal_needs_typed_form_outside_constructor(L::U32(7)));
        assert!(!literal_needs_typed_form_outside_constructor(L::F32(0.5)));
        assert!(!literal_needs_typed_form_outside_constructor(L::Bool(true)));
        assert!(!literal_needs_typed_form_outside_constructor(
            L::AbstractInt(7)
        ));
        assert!(!literal_needs_typed_form_outside_constructor(
            L::AbstractFloat(0.5)
        ));
    }
}
