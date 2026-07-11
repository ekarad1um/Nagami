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

/// `true` when emitting `literal` in its bare (suffix-less) form at a
/// non-constructor, non-extracted-const position would re-parse as a
/// different concrete type than the original.
///
/// WGSL's abstract-type coercion defaults `AbstractInt` to `i32` and
/// `AbstractFloat` to `f32` when no enclosing context pins another
/// type.  Every reachable top-level `Literal` in valid naga IR sits
/// in a pinning context (Binary partner of known type, function-arg
/// slot, return slot, Store target, constructor), and the abstract
/// value coerces into that local `(kind, width)`.  So bare emission
/// is sound for `I32`/`U32`/`F32`/`Bool` (the coercion target
/// matches the original) and saves a suffix byte.
///
/// For `F16`/`F64`/`I64`/`U64` the abstract default does NOT match:
/// * `F16` / `F64`: bare `0.5` parses as `AbstractFloat` ->
///   default-f32, refusing to coerce into f16/f64 contexts.
/// * `I64` / `U64`: bare `42` parses as `AbstractInt` -> default i32,
///   not i64/u64; even in-range values conflict with the
///   i64/u64-typed coercion target.
///
/// Counter-example: a NEW non-pinning context (e.g. a switch
/// selector that is itself a `Literal::U32`) DOES break the safe-bare
/// assumption for `U32`.  Switch dispatch in [`super::stmt_emit`]
/// already special-cases literal selectors via [`literal_to_wgsl`]
/// because the case label carries `u` while a bare selector would
/// parse as AbstractInt -> i32.  Any new top-level Literal-bearing
/// position must apply the same special-case OR hint the scalar kind
/// through [`super::expr_emit::Generator::emit_expr_with_scalar_hint`].
fn literal_needs_typed_form_outside_constructor(literal: naga::Literal) -> bool {
    matches!(
        literal,
        naga::Literal::F16(_)
            | naga::Literal::F64(_)
            | naga::Literal::I64(_)
            | naga::Literal::U64(_)
    )
}

/// `true` when `literal`, rendered in the BARE form constructor components
/// use, re-infers exactly `scalar` as a `vecN(...)` component - i.e. it can
/// pin the elided constructor's element type.
///
/// WGSL's abstract defaults decide: an integer-form token is `AbstractInt`
/// (defaults i32), a float-form token `AbstractFloat` (defaults f32).  So an
/// i32 element is pinned by ANY integer literal, and an f32 element by a
/// literal whose bare rendering keeps a float shape (`.5`, `1e3`, `0x1p2` -
/// but NOT a whole number, which renders as a bare int and would re-infer
/// i32).  u32/f16/f64/16-bit elements are never literal-pinned: their
/// abstract default is a different type.  Mixed abstract components stay
/// compatible - the constructor's common type keeps the pinned default
/// (`vec4(.5,1,1,1)` is AbstractFloat throughout -> f32).
///
/// The float-form test inspects the same rendering the emitter ships
/// (per-type precision rounding included), so the decision cannot drift
/// from the emitted token.
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

/// Convert a constant **width-8** literal (`F64` / `U64` / `I64`) to `target`
/// (one of f32 / i32 / u32 / bool), matching naga's value-conversion
/// semantics, or `None` for any other source/target (or a non-finite f32).
///
/// Mirror of [`super::super::passes::const_fold::cast_width8_to`] (kept in
/// sync; any divergence is caught by the round-trip tests).  Used only by
/// [`Generator::try_emit_const_width8_vector_narrow`] to fold a *vector*
/// width-8 narrowing cast that const_fold cannot (`materialize_vector` would
/// need the converted component literals to already exist as arena handles).
/// f64 sources CLAMP on int narrowing; u64/i64 sources WRAP (`as`) - naga only
/// clamps float sources.
fn cast_width8_to_literal(src: naga::Literal, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    if let L::F64(v) = src {
        return match (target.kind, target.width) {
            (K::Float, 4) => {
                let r = v as f32;
                r.is_finite().then_some(L::F32(r))
            }
            (K::Sint, 4) => Some(L::I32(v.clamp(i32::MIN as f64, i32::MAX as f64) as i32)),
            (K::Uint, 4) => Some(L::U32(v.clamp(u32::MIN as f64, u32::MAX as f64) as u32)),
            (K::Bool, _) => Some(L::Bool(v != 0.0)),
            _ => None,
        };
    }
    let v: i128 = match src {
        L::U64(v) => v as i128,
        L::I64(v) => v as i128,
        _ => return None,
    };
    match (target.kind, target.width) {
        (K::Float, 4) => Some(L::F32(v as f32)),
        (K::Sint, 4) => Some(L::I32(v as i32)),
        (K::Uint, 4) => Some(L::U32(v as u32)),
        (K::Bool, _) => Some(L::Bool(v != 0)),
        _ => None,
    }
}

/// `true` when `l` is a width-8 numeric literal (`f64`/`u64`/`i64`) - the
/// only literals [`Generator::try_emit_const_width8_vector_narrow`] folds.
/// `literal_extract`'s narrow-fold pre-pass mirrors this set, so both must
/// agree or the extraction count diverges from what the emitter prints.
pub(super) fn literal_is_width8(l: naga::Literal) -> bool {
    matches!(
        l,
        naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_)
    )
}

/// `true` when the literal numerically equals zero.  Both `+0.0` and
/// `-0.0` qualify (IEEE `==` already conflates them; F16 is matched on
/// bit-pattern because its `==` is not available without `half`).
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
    /// `true` when `handle` resolves to a value that is statically
    /// provably zero (literal zero, `ZeroValue`, or a constant whose
    /// init is one of those).  Used by emission paths that must reject
    /// IR with an unrepresentable non-zero argument - e.g.,
    /// `textureSampleCompareLevel`, whose WGSL signature has no level
    /// parameter and therefore can only encode level 0.
    pub(super) fn expression_is_provable_zero(
        &self,
        handle: naga::Handle<naga::Expression>,
        ctx: &super::core::FunctionCtx<'_, '_>,
    ) -> bool {
        match ctx.func.expressions[handle] {
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
            // A function-argument `ptr<...>` used as the base of an lvalue
            // access chain (`(*p).field = ..`, `(*p)[i] = ..`) is a pointer
            // value, not a reference, so it needs the explicit `(*p)` deref;
            // see `emit_postfix_base` for the value-context counterpart.
            _ if self.pointer_is_ptr_value(expr, ctx) => {
                format!("(*{})", self.emit_expr(expr, ctx)?)
            }
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
                let bare = literal_to_wgsl_bare(*lit, &self.options.float_precision);
                let key = literal_extract_key(*lit, &self.options.float_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    return Ok(name.clone());
                }
                // Inside a type constructor the enclosing type provides
                // context, so bare literals (no suffix) are safe and produce
                // the shortest output when the literal is not extracted.
                return Ok(bare);
            }
            if let naga::Expression::Binary { op, .. } = &ctx.func.expressions[arg]
                && matches!(
                    op,
                    naga::BinaryOperator::Less | naga::BinaryOperator::LessEqual
                )
            {
                let s = self.emit_expr_uncached(arg, ctx)?;
                return Ok(format!("({s})"));
            }
            // An identity-swizzle `Compose` (`vecN(b.0,..,b.N-1)`) collapses to
            // its bare base `b`.  A constructor argument is a complete,
            // comma-delimited expression with nothing appended, so the base
            // needs no parentheses here even when it is an operator expression
            // (`mat3x3(l*m, ..)`, not `mat3x3((l*m), ..)`).  Emitting it through
            // the general path would route the collapse through
            // `emit_postfix_base`, which conservatively wraps a Binary/Unary/
            // Select base for the *postfix* context it cannot rule out - bytes
            // that are redundant in this loose position.
            if let Some(base) = self.compose_identity_collapse_base(arg, ctx) {
                // A pointer-value base collapses to the dereferenced value
                // `(*p)`, not the bare pointer; see `emit_postfix_base`.
                if self.pointer_is_ptr_value(base, ctx) {
                    return Ok(format!("(*{})", self.emit_expr(base, ctx)?));
                }
                let s = self.emit_expr(base, ctx)?;
                // Same template-ambiguity guard as the direct-Binary case
                // above: a bare leading `<` / `<=` right after a constructor's
                // `(` can be mis-scanned as a template list.  If the collapsed
                // base is itself an (uncached) `Less`/`LessEqual` comparison,
                // wrap it.  Other operator bases (`*`, `+`, ...) stay bare.
                if !ctx.expr_names.contains_key(&base)
                    && matches!(
                        ctx.func.expressions[base],
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
                // A real scalar splat has exactly `size` scalar lanes -
                // `vecN(s, s, .., s)`.  A vector built from SUB-VECTORS
                // (`vec4(v2, v2)`) can also satisfy `compose_is_splat` when its
                // parts are identical handles, but then `components[0]` is a
                // VECTOR, and substituting it into the binary splat-elision
                // path emits a type-mismatched operand (`v2 * v4`).  Require the
                // scalar-per-lane shape, mirroring the sibling splat-collapse
                // guards (`components.len() == size`).
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
        // A pointer VALUE (a function-argument `ptr<...>`) does NOT
        // auto-deref in WGSL, so a postfix `.field` / `.xyz` / `[i]` on it
        // - or its use as a collapsed identity value - requires an explicit
        // `(*base)`.  naga's permissive frontend accepts `p.x` on a pointer,
        // but strict parsers (tint/Dawn/browsers, nagami's target) reject it.
        // References (globals, locals, access chains rooted in them)
        // auto-deref and emit bare.  Checked before the cached path because
        // a function argument renders by name yet still needs the deref.
        if self.pointer_is_ptr_value(base, ctx) {
            return Ok(format!("(*{})", self.emit_expr(base, ctx)?));
        }
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
        // A base that renders with a leading `*` - a whole-pointee `Load` of a
        // pointer value (`let _ = (*p)`) inlined into a postfix position, which
        // emits `*p` via the `E::Load`/`emit_lvalue` deref path - must be
        // parenthesised: a bare `*p.field` / `*p[i]` parses as `*(p.field)` and
        // is rejected ("operand of `*` must be a pointer").  Wrap so the deref
        // binds first: `(*p).field`.  (A let-bound such Load short-circuits via
        // the cached path above and renders as a bare name, needing no wrap.)
        if needs_parens || s.starts_with('*') {
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
                    let key = literal_extract_key(*lit, &self.options.float_precision);
                    if let Some(name) = self.extracted_literals.get(&key) {
                        name.clone()
                    } else if literal_needs_typed_form_outside_constructor(*lit) {
                        // Bare form would re-parse as the wrong type
                        // (see `literal_needs_typed_form_outside_constructor`).
                        // Force the typed form here even though we are
                        // not strictly inside one of the two sanctioned
                        // bare-emit positions, because the bare form
                        // changes the inferred concrete type.
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
                            // Mirror the Literal arm: an unnamed constant
                            // whose init is a literal is emitted as the
                            // literal text at every use site, so the
                            // same shared-const substitution applies
                            // here.  Same bare-emit safety gate.
                            let key = literal_extract_key(*lit, &self.options.float_precision);
                            if let Some(name) = self.extracted_literals.get(&key) {
                                name.clone()
                            } else if literal_needs_typed_form_outside_constructor(*lit) {
                                literal_to_wgsl(*lit, &self.options.float_precision)
                            } else {
                                // Bare-safe literal: emit the bare token like
                                // the `Literal` arm does (one suffix byte
                                // cheaper than the typed `emit_global_expr`).
                                key.expr_text
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
                // All-zero vector / matrix -> zero-value constructor
                // (`vec2f(0,0)` -> `vec2f()`), but ONLY when this Compose is
                // INLINED (ref count 1).  A multi-use Compose is `let`-bound, so
                // folding its decl to `vec2f()` would round-trip badly: naga
                // re-parses `vec2f()` as a `ZeroValue`, which is non-emittable
                // and so can NEVER be re-bound, forcing the generator to inline
                // it at every use (`vec2f()` x N) - a non-idempotent size blow-up.
                // Leaving the bound case as the splat `vec2f(0)` re-parses to a
                // bindable `Splat` and stays bound.  Strict `+0` only
                // (compose_is_all_zero rejects -0.0); never array/struct.
                if ctx.ref_counts[expr.index()] <= 1
                    && matches!(
                        self.module.types[*ty].inner,
                        naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. }
                    )
                    && components
                        .iter()
                        .all(|&c| compose_is_all_zero(c, &ctx.func.expressions))
                {
                    break 'compose self.zero_value(*ty)?;
                }

                // For vector composes, try to emit as a bare swizzle
                // (e.g. vec3f(v.x,v.y,v.z) -> v.xyz), eliminating the
                // type constructor entirely.
                if matches!(self.module.types[*ty].inner, naga::TypeInner::Vector { .. })
                    && let Some(swizzle) = self.try_compose_as_full_swizzle(components, ctx)?
                {
                    break 'compose swizzle;
                }

                let mut s = String::new();
                let ctor_name = self.vector_ctor_name(*ty, components, ctx)?;
                match self.array_ctor_name(*ty, components, &ctor_name, &ctx.func.expressions) {
                    Some(bare) => s.push_str(bare),
                    None => s.push_str(&ctor_name),
                }
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
                // A matrix built from explicit scalar columns can also be emitted
                // in flat all-scalar form: mat2x2f(vec2f(a,b),vec2f(c,d)) ->
                // mat2x2f(a,b,c,d).  Build it and keep whichever is shorter - the
                // column form wins when a column is shared/let-bound (emitted as a
                // short name, e.g. mat3x3f(a,a,a)) or splat-collapses to vecR(x).
                if let Some(flat) = matrix_flatten_scalars(
                    *ty,
                    components,
                    &self.module.types,
                    &ctx.func.expressions,
                ) {
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
                // A vector built from scalar runs can collapse runs into
                // sub-vector splats: vec4f(0,0,0,2) -> vec4f(vec3f(),2).  Kept
                // only when strictly shorter (the sub-vector type often lacks a
                // short alias, in which case it loses).
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
            E::Splat { size: _, value } => 'splat: {
                let target_ty = self.expr_type_name(expr, ctx)?;
                // All-zero splat -> zero-value constructor (`vec3f(0)` ->
                // `vec3f()`), gated on ref count <= 1: a bound `vec3f()`
                // re-parses to a non-emittable `ZeroValue` that must re-inline
                // at every use, a non-idempotent size blow-up.
                if ctx.ref_counts[expr.index()] <= 1
                    && compose_is_all_zero(*value, &ctx.func.expressions)
                {
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
                // Identity swizzle: `.xy` on a vec2, `.xyz` on a vec3, `.xyzw`
                // on a vec4 - selects component `i` at position `i` for all `i`
                // AND the base vector has exactly `n` lanes (nothing dropped or
                // reordered) - is a no-op, so emit just the base.  Only elide a
                // base that `emit_postfix_base` would NOT wrap - i.e. not
                // `Binary`/`Unary`/`Select` (the operand kinds it parenthesises):
                // the parent's parenthesisation, computed for a postfix
                // `Swizzle`, stays valid for the substituted base, AND a kept
                // swizzle over such a base re-minifies identically (eliding a
                // `Select` would drift to a parenthesised re-parse).
                let is_identity = pattern[..n].iter().enumerate().all(|(i, c)| {
                    let idx = match c {
                        naga::SwizzleComponent::X => 0,
                        naga::SwizzleComponent::Y => 1,
                        naga::SwizzleComponent::Z => 2,
                        naga::SwizzleComponent::W => 3,
                    };
                    idx == i
                });
                let base_is_full = match ctx.info[*vector].ty.inner_with(&self.module.types) {
                    naga::TypeInner::Vector { size: bs, .. } => *bs as u8 as usize == n,
                    _ => false,
                };
                let base_paren_free = !matches!(
                    ctx.func.expressions[*vector],
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
                // Reading an atomic requires the `atomicLoad` builtin: a bare
                // atomic identifier is non-portable - naga lowers `atomicLoad(&p)`
                // and a direct read to the SAME `Load`, so it accepts either, but
                // the WGSL spec and strict consumers (tint/Dawn) reject reading
                // `atomic<T>` directly.  Mirror `emit_atomic_store` exactly.
                if self.atomic_scalar_for_expr(*pointer, ctx).is_some() {
                    if self.pointer_is_ptr_value(*pointer, ctx) {
                        // Already a `ptr<>` value (a pointer parameter): emit it
                        // BY VALUE - `emit_lvalue` would deref it to `*p`, giving
                        // `atomicLoad(*p)`.  (Latent today: naga rejects atomic
                        // pointer parameters, so this branch is unreachable - kept
                        // correct and consistent with `emit_atomic_store`.)
                        format!("atomicLoad({})", self.emit_expr(*pointer, ctx)?)
                    } else {
                        format!("atomicLoad(&{})", self.emit_lvalue(*pointer, ctx)?)
                    }
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
                            if depth_ref.is_none() {
                                s.push_str(sep);
                                s.push_str(&self.emit_expr(*h, ctx)?);
                            } else {
                                // `textureSampleCompareLevel` always
                                // samples at level 0; the WGSL signature
                                // has no level slot.  naga's WGSL front
                                // encodes that as `Exact(Literal(0))`,
                                // so a provable zero is safe to drop -
                                // any other value is unrepresentable
                                // and must defer to the fallback emitter.
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
                // WGSL reserves `--` and `++`, so a `Negate` over a child that
                // already renders with a leading `-` (a nested `-(-x)`, or a
                // negative literal) would lex as a forbidden decrement token.
                // A single space disambiguates and is one byte cheaper than
                // wrapping in parens.  (`!`/`~` form no reserved adjacency.)
                if matches!(op, naga::UnaryOperator::Negate) && !wrap && s.starts_with('-') {
                    s.insert(0, ' ');
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

                // Splat-elision drops the scalar out of its
                // type-pinning constructor context.  Use `emit_expr`
                // (not `emit_constructor_arg`): the latter's
                // defensive comparison-wrap would double-paren
                // operands (`vec3(a<b) * v` -> `((a<b)) * v`) and its
                // bare-literal path would emit a literal whose
                // abstract default might mismatch the vector operand's
                // scalar type.  `child_needs_parens` (above) already
                // handles precedence-driven parenthesisation.
                //
                // The literal-side mismatch is exactly the case
                // [`literal_needs_typed_form_outside_constructor`]
                // exists for: F16/F64/I64/U64 literals get typed
                // suffixes (`0h`/`0lf`/`0li`/`0lu`) when emitted via
                // `emit_expr`'s Literal arm.  Splat-elision is the
                // primary site that creates the "outside a
                // constructor" condition, so the two are interdependent.
                let ls = if elide_l {
                    self.emit_expr(left_scalar.unwrap(), ctx)?
                } else {
                    self.emit_expr(*left, ctx)?
                };
                let rs = if elide_r {
                    let s = self.emit_expr(right_scalar.unwrap(), ctx)?;
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
                        literal_to_wgsl(*lit, &self.options.float_precision)
                    } else {
                        self.emit_expr(*reject, ctx)?
                    };
                s.push_str(&reject_str);
                s.push_str(sep);

                let accept_str =
                    if let naga::Expression::Literal(lit) = &ctx.func.expressions[*accept] {
                        literal_to_wgsl(*lit, &self.options.float_precision)
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
                    // `isNan`/`isInf` are not WGSL builtins; naga's WGSL
                    // front-end never produces these (only `all`/`any`), so
                    // this is unreachable from a WGSL->IR->WGSL pipeline.
                    // Refuse rather than emit an identifier no WGSL consumer
                    // recognises (which would slip past round-trip checks).
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
                    // WGSL forbids `bitcast` on matrices: the
                    // `bitcast<T>` operator is restricted to numeric
                    // scalars and vectors of numeric scalars.  Naga
                    // should never lower a matrix `As` with
                    // `convert: None`, but if it does, refuse to emit
                    // invalid WGSL - the pipeline can fall back to
                    // naga's emitter (which would presumably also
                    // refuse, but consistently).
                    if convert.is_none() {
                        return Err(Error::Emit(format!(
                            "matrix bitcast (As {{ convert: None }}) is not representable in WGSL \
                             in function '{}' (expr {}): source type {:?}",
                            ctx.display_name,
                            expr.index(),
                            src_inner,
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
                // WGSL (https://www.w3.org/TR/WGSL/#bit-reinterp-builtin-functions)
                // limits `bitcast<T>` to numeric scalars and their vectors;
                // `bool` and abstract scalars are explicitly excluded.
                // Refuse before emission so the pipeline's fallback
                // emitter handles the cast.
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
                // Narrowing a CONST width-8 vector (`vec2<f32>(vec2<f64>(.5lf,..))`,
                // `vec2<u32>(vec2<u64>(..lu..))`) is rejected by naga's frontend
                // on re-parse - and naga's own backend emits the same invalid
                // token, so the run() fallback can't save it.  const_fold
                // handles the scalar case but not the vector one
                // (`materialize_vector` needs the converted component literals
                // to already exist as arena handles).  When the (inlined,
                // unnamed) operand is a const Compose/Splat of width-8 literals
                // and the target is f32/i32/u32/bool, fold it here to a valid
                // converted constructor.  Everything else - runtime vectors,
                // named operands, f16/f64/i64/u64 targets, non-finite results -
                // falls through to the verbatim path below.
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
                // A scalar `bitcast<T>` reinterprets its operand's BITS, so an
                // inline literal operand must be emitted in TYPED (suffixed) form
                // to pin its concrete type.  The bare form drops the suffix and
                // re-parses as an abstract literal that materialises to a
                // different concrete type than the source had: a whole-number
                // float collapses to an int token (`1.0f` -> `1` -> `AbstractInt`
                // -> default i32), so `bitcast<u32>(1.0f)` reinterprets `0x1` and
                // not the float's `0x3F800000` (a silent VALUE miscompile); and a
                // `u32` above `i32::MAX` (`bitcast<f32>(3212836864u)` ->
                // `3212836864`) overflows the i32 default and is rejected outright
                // by spec-conformant consumers.  Vector bitcasts and
                // `convert.is_some()` conversions pin the operand via their
                // constructor, so only the scalar-bitcast case is forced.
                //
                // This also bypasses `extracted_literals`: a hoisted
                // `const N = <bare>;` is abstract-typed (the bare form merges
                // types, so `F32(1024.0)` and `I32(1024)` share one
                // `const N=1024;`), and `bitcast<u32>(N)` would reinterpret the
                // abstract-int default - the same miscompile via a reference.  The
                // extraction counter MUST subtract this same occurrence (the
                // `As { convert: None }` arm in `literal_extract`) or it leaves a
                // dangling const.  Abstract literals are excluded: `emit_expr`'s
                // concretization already yields the correct typed form.
                let source = if convert.is_none()
                    && vec_size.is_none()
                    && !ctx.expr_names.contains_key(expr)
                    && let E::Literal(lit) = &ctx.func.expressions[*expr]
                    && !matches!(
                        lit,
                        naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                    ) {
                    literal_to_wgsl(*lit, &self.options.float_precision)
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
                .get(&expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::RayQueryProceedResult => ctx
                .expr_names
                .get(&expr)
                .cloned()
                .unwrap_or_else(|| format!("_e{}", expr.index())),
            // Unlike `RayQueryProceedResult` above (bound by the enclosing
            // `Statement::RayQuery { fun: Proceed }`), these two are
            // free-standing builtin calls with no binding statement, so they
            // render inline.  `query` is a `ray_query` local per naga's
            // validator (`InvalidRayQueryExpression` otherwise), spelled with
            // an explicit `&` exactly like the `Statement::RayQuery` emitter.
            E::RayQueryGetIntersection { query, committed } => {
                let mut s = String::from(if *committed {
                    "rayQueryGetCommittedIntersection(&"
                } else {
                    "rayQueryGetCandidateIntersection(&"
                });
                s.push_str(&self.emit_expr(*query, ctx)?);
                s.push(')');
                s
            }
            E::RayQueryVertexPositions { query, committed } => {
                let mut s = String::from(if *committed {
                    "getCommittedHitVertexPositions(&"
                } else {
                    "getCandidateHitVertexPositions(&"
                });
                s.push_str(&self.emit_expr(*query, ctx)?);
                s.push(')');
                s
            }
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
                let key = literal_extract_key(concrete, &self.options.float_precision);
                if let Some(name) = self.extracted_literals.get(&key) {
                    Some(name.clone())
                } else {
                    Some(literal_to_wgsl(concrete, &self.options.float_precision))
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
            E::Literal(lit) => literal_to_wgsl(*lit, &self.options.float_precision),
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
                // All-zero global splat -> zero-value constructor (emitted once
                // in a const/var initializer, so no over-inline risk).
                if compose_is_all_zero(*value, arena) {
                    format!("{type_name}()")
                } else {
                    let mut s = format!("{type_name}(");
                    if let E::Literal(lit) = &arena[*value] {
                        s.push_str(&literal_to_wgsl_bare(*lit, &self.options.float_precision));
                    } else {
                        s.push_str(&self.emit_global_expr(*value)?);
                    }
                    s.push(')');
                    s
                }
            }
            E::Compose { ty, components } => 'global_compose: {
                // All-zero vector / matrix -> zero-value constructor.  No
                // ref-count gate: a global expression is a const/var initializer
                // emitted exactly ONCE (uses reference the const NAME), so the
                // re-parse over-inline that gates the function-local arm cannot
                // occur here.
                if matches!(
                    self.module.types[*ty].inner,
                    naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. }
                ) && components.iter().all(|&c| compose_is_all_zero(c, arena))
                {
                    break 'global_compose self.zero_value(*ty)?;
                }
                let emit_scalar =
                    |this: &Self, c: naga::Handle<naga::Expression>| -> Result<String, Error> {
                        if let naga::Expression::Literal(lit) = &arena[c] {
                            Ok(literal_to_wgsl_bare(*lit, &this.options.float_precision))
                        } else {
                            this.emit_global_expr(c)
                        }
                    };
                let mut s = String::new();
                // NB: no `array(...)` elision in the GLOBAL arm.  A global
                // expression is a const/override/var initializer, which carries
                // a declared type annotation; naga's text front-end rejects an
                // elided constructor whose inferred array type must match a
                // declared annotation that resolves through a type alias
                // ("expected array<T,N> but got array<T,N>").  Elision is safe
                // only for the unannotated function-local constructors.
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
                    s.push_str(&emit_scalar(self, components[0])?);
                } else {
                    let sep = self.comma_sep();
                    for (i, c) in components.iter().enumerate() {
                        if i > 0 {
                            s.push_str(sep);
                        }
                        s.push_str(&emit_scalar(self, *c)?);
                    }
                }
                s.push(')');
                // Matrix built from explicit scalar columns -> flat all-scalar
                // form, kept only when strictly shorter (a shared / let-bound
                // column emitted as a short name can beat the flat form).
                if let Some(flat) =
                    matrix_flatten_scalars(*ty, components, &self.module.types, arena)
                {
                    let mut sf = self.type_ref(*ty)?;
                    sf.push('(');
                    let sep = self.comma_sep();
                    for (i, c) in flat.iter().enumerate() {
                        if i > 0 {
                            sf.push_str(sep);
                        }
                        sf.push_str(&emit_scalar(self, *c)?);
                    }
                    sf.push(')');
                    if sf.len() < s.len() {
                        s = sf;
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
                let wrap = unary_child_needs_parens(*expr, arena, false);
                let mut s = self.emit_global_expr(*expr)?;
                if wrap {
                    s.insert(0, '(');
                    s.push(')');
                }
                // Mirror the function-local arm: a `Negate` over a child that
                // already renders with a leading `-` would lex as the reserved
                // `--` token; a single space disambiguates (cheaper than parens).
                if matches!(op, naga::UnaryOperator::Negate) && !wrap && s.starts_with('-') {
                    s.insert(0, ' ');
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
                    literal_to_wgsl(*lit, &self.options.float_precision)
                } else {
                    self.emit_global_expr(*reject)?
                };
                s.push_str(&reject_str);
                s.push_str(sep);

                let accept_str = if let naga::Expression::Literal(lit) = &arena[*accept] {
                    literal_to_wgsl(*lit, &self.options.float_precision)
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
                    // WGSL forbids matrix bitcast; the function-local
                    // arm enforces the same rule.
                    if convert.is_none() {
                        return Err(Error::Emit(format!(
                            "matrix bitcast (As {{ convert: None }}) is not representable in WGSL \
                             in global expression {}: source type {:?}",
                            inner.index(),
                            src_inner,
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
                    let source = self.emit_global_expr(*inner)?;
                    let mut s = target;
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
                    // Same `bitcast<T>` restriction as the function-local
                    // arm: T must be a numeric scalar or vector of one.
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
                             scalar kind {:?} in global expression {}",
                            kind,
                            inner.index(),
                        )));
                    }
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
                    // `isNan`/`isInf` are not WGSL builtins; naga's WGSL
                    // front-end never produces these (only `all`/`any`), so
                    // this is unreachable from a WGSL->IR->WGSL pipeline.
                    // Refuse rather than emit an identifier no WGSL consumer
                    // recognises (which would slip past round-trip checks).
                    naga::RelationalFunction::IsNan | naga::RelationalFunction::IsInf => {
                        return Err(Error::Emit(format!(
                            "relational function {fun:?} has no WGSL spelling"
                        )));
                    }
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

    /// Fold a narrowing cast of a *const width-8 vector* (`vecN<f64/u64/i64>`)
    /// into a directly-emitted converted constructor (e.g.
    /// `vec2<f32>(vec2<f64>(.5lf,1.5lf))` -> `vec2f(.5,1.5)`).  Returns `None`
    /// (so the caller falls through to the verbatim `target(source)` path)
    /// when the operand is not a Compose/Splat whose components are *all*
    /// width-8 literals, or when any component does not convert to a finite,
    /// representable target literal - the latter keeps the unrepresentable
    /// f64->f32-overflow case as a diagnosable hard error rather than silently
    /// emitting an `inf` token.
    fn try_emit_const_width8_vector_narrow(
        &self,
        operand: naga::Handle<naga::Expression>,
        target: naga::Scalar,
        target_inner: &naga::TypeInner,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Result<Option<String>, Error> {
        let lits: Vec<naga::Literal> = match &ctx.func.expressions[operand] {
            naga::Expression::Compose { components, .. } => {
                let mut out = Vec::with_capacity(components.len());
                for &c in components.iter() {
                    match ctx.func.expressions[c] {
                        naga::Expression::Literal(l) if literal_is_width8(l) => out.push(l),
                        _ => return Ok(None),
                    }
                }
                out
            }
            naga::Expression::Splat { size, value } => match ctx.func.expressions[*value] {
                naga::Expression::Literal(l) if literal_is_width8(l) => vec![l; *size as usize],
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let mut converted = Vec::with_capacity(lits.len());
        for l in lits {
            match cast_width8_to_literal(l, target) {
                Some(lit) => converted.push(lit),
                None => return Ok(None),
            }
        }
        let mut s = self.type_name_for_inner(target_inner)?;
        s.push('(');
        // Collapse to splat form when every converted component is identical
        // (bit-equal, so `-0` stays distinct from `0`), matching the
        // generator's normal splat-elision.
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

    /// Emit the declaration tail of a zero-initialized `var`: either
    /// `:<type>` (relying on WGSL's guarantee that an uninitialized local
    /// is zero) or `=<zero-literal>`, whichever is the shorter source
    /// form.  Callers emit `var <name>` first; this appends the chosen
    /// tail with no trailing `;`.  Every zero-init `var` emit site routes
    /// through here so the `:T`-vs-`=0i` choice never drifts between paths.
    ///
    /// A scalar's typed-suffix zero literal (`0i`/`0u`/`0f`/`0h`) is one
    /// byte shorter than its `:i32`/`:u32`/`:f32`/`:f16` annotation, while
    /// `:bool` beats `=false` and every composite (`:vec3f` vs `=vec3f(0)`)
    /// favours the annotation.  A short type alias (`alias j=i32;` -> `:j`)
    /// can undercut even `=0i`, so compare the rendered lengths instead of
    /// assuming a winner.
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

    /// The constructor name for a vector `Compose`.  Normally the type's
    /// rendered name (`vec2u`, `vec3f`, or a type alias), but the bare
    /// `vecN` form when BOTH (a) it is strictly shorter than the
    /// suffixed/aliased name, and (b) at least one component is a
    /// non-literal whose concrete scalar type already equals the element
    /// type - so WGSL infers the same element type from that component and
    /// converts the others.
    ///
    /// Literals inside a constructor render in BARE (abstract) form, so most
    /// cannot pin: `vec2u(4, 1)` cannot lose its `u` - dropping it would
    /// re-infer `vec2i`.  Two literal shapes DO pin, though (see
    /// [`literal_bare_form_pins_scalar`]): a float-form token pins f32 and
    /// any integer literal pins i32, because WGSL's abstract defaults land
    /// exactly there and mixing with other abstract components can only
    /// stay abstract-compatible (`vec4f(.5, 1, 1, 1)` -> `vec4(.5,1,1,1)`
    /// still infers f32).  Otherwise only a typed non-literal component
    /// (e.g. a `u32` field access in `vec2u(p.k, 4)`) pins the element
    /// type.
    ///
    /// Soundness of the non-literal arm rests on the pin component
    /// rendering in TYPED form.  A non-literal `ZeroValue` or unnamed
    /// `Constant` would resolve to the right scalar yet still emit a bare
    /// token, so it cannot be the sole pinner - but it never is:
    /// `const_fold` runs first and folds away any all-constant `Compose`,
    /// so a vector reaching the generator always has a runtime (typed)
    /// component.  A future pass that leaves an unnamed const / `ZeroValue`
    /// as a Compose's only non-literal component must revisit this gate.
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
                && components.iter().any(|&c| match &ctx.func.expressions[c] {
                    naga::Expression::Literal(lit) => {
                        literal_bare_form_pins_scalar(*lit, scalar, &self.options.float_precision)
                    }
                    _ => {
                        type_inner_scalar(ctx.info[c].ty.inner_with(&self.module.types))
                            == Some(scalar)
                    }
                })
            {
                return Ok(bare);
            }
        }
        Ok(type_str)
    }

    /// Constructor name for an array `Compose`, eliding the template parameters
    /// to the bare inferring form `array(...)` when provably safe and shorter.
    ///
    /// `array(...)` infers its element type from the components, so eliding is
    /// sound only when inference is GUARANTEED to reproduce the declared element
    /// type `base`.  A bare abstract literal would re-infer
    /// (`array<u32,2>(1,2)` -> `array<i32,2>`, `array<f16,2>(1,2)` ->
    /// `array<f32,2>`) - a silent retype both engines reject/miscompile - so the
    /// gate requires at least one component that is a `Compose` / `ZeroValue` /
    /// `Constant` / `Override` whose type is exactly `base`; such a component
    /// always emits as a concretely-`base`-typed expression and pins inference
    /// to `base`.  Returns `None` (keep the explicit / aliased `array<T,N>`
    /// name) otherwise, including when the bare `array` is not shorter than the
    /// full/aliased name (an aliased short array type must win).  This rewrites
    /// the CONSTRUCTOR name only; type annotations are never touched.
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
        let pins = components.iter().any(|&c| match &arena[c] {
            naga::Expression::Compose { ty: cty, .. } => *cty == base,
            naga::Expression::ZeroValue(zty) => *zty == base,
            naga::Expression::Constant(h) => self.module.constants[*h].ty == base,
            naga::Expression::Override(h) => self.module.overrides[*h].ty == base,
            _ => false,
        });
        pins.then_some("array")
    }

    /// Collapse maximal runs of >=2 identical adjacent SCALAR components in a
    /// vector `Compose` into sub-vector splats - `vec4f(0,0,0,2)` ->
    /// `vec4f(vec3f(0),2)`, which folds to `vec4f(vec3f(),2)` for the zero run.
    /// Returns the collapsed text only when at least one run (of length `2..N`)
    /// was collapsed; the caller keeps it ONLY when it is strictly shorter, so
    /// this never grows the output (the sub-vector type often has no short
    /// alias, in which case the collapsed form loses and is discarded).
    ///
    /// Value-safe: each run's components compare equal under [`exprs_splat_eq`]
    /// (literal bit pattern / structural identity), so emitting one shared value
    /// `vecK(c)` reproduces the same lanes; sub-vector constructors are postfix,
    /// so there is no parenthesisation hazard.
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
        // Every component must be a plain scalar so that a run forms a valid
        // sub-vector `vecK(scalar)`.
        if !components.iter().all(|&c| {
            matches!(
                ctx.info[c].ty.inner_with(&self.module.types),
                naga::TypeInner::Scalar(_)
            )
        }) {
            return Ok(None);
        }
        // Phase 1 (immutable): split components into maximal equal-value runs,
        // and note which runs are all-zero.
        let (runs, zero): (Vec<(usize, usize)>, Vec<bool>) = {
            let arena = &ctx.func.expressions;
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
        // Phase 2 (mutable): emit.
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

        if let Some(h) = ty_handle
            && let Some(mangled) = self.member_names.get(&(h, index))
        {
            return Some(mangled.clone());
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
        if let Some(h) = ty_handle
            && let Some(mangled) = self.member_names.get(&(h, index))
        {
            return Some(mangled.clone());
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
        if let naga::Expression::AccessIndex { base, index } = arena[handle]
            && index <= 3
        {
            let inner = ctx.info[base].ty.inner_with(&self.module.types);
            if matches!(inner, naga::TypeInner::Vector { .. }) {
                return Some((base, index));
            }
        }

        // Pattern B: Load { pointer: AccessIndex { base: ptr_to_vec, index } }
        if let naga::Expression::Load { pointer } = arena[handle]
            && let naga::Expression::AccessIndex { base, index } = arena[pointer]
            && index <= 3
        {
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

        None
    }

    /// If `expr` is an uncached vector `Compose` that
    /// [`try_compose_as_full_swizzle`] would reduce to its *bare* base (the
    /// identity case `vecN(b.0, .., b.N-1)` -> `b`), return that base handle;
    /// otherwise `None`.
    ///
    /// Callers in a *loose* position (a comma-delimited constructor/call
    /// argument, where nothing is appended to the result) use this to emit the
    /// base directly via `emit_expr`, skipping the `emit_postfix_base` wrap
    /// that the general collapse path applies for the tight postfix context it
    /// cannot rule out.  Mirrors the identity branch of
    /// `try_compose_as_full_swizzle` exactly so the two never disagree.
    fn compose_identity_collapse_base(
        &self,
        expr: naga::Handle<naga::Expression>,
        ctx: &FunctionCtx<'a, '_>,
    ) -> Option<naga::Handle<naga::Expression>> {
        if ctx.expr_names.contains_key(&expr) {
            return None;
        }
        let naga::Expression::Compose { ty, components } = &ctx.func.expressions[expr] else {
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
        // Identity: pattern is [0,1,..,N-1] over a same-size source vector.
        let src_n = self.vector_size_of(base, ctx)?;
        if pattern.len() == src_n && pattern.iter().enumerate().all(|(i, &idx)| idx == i as u32) {
            Some(base)
        } else {
            None
        }
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
        if let Some(src_n) = source_size
            && pattern.len() == src_n
            && pattern.iter().enumerate().all(|(i, &idx)| idx == i as u32)
        {
            // Collapsing to the bare base: route through the postfix-aware
            // path so an uncached Binary/Unary/Select base keeps the parens
            // its consuming context needs (the substituted text stands in
            // for the whole Compose node, whose own shape no longer signals
            // "this is an operator expression" to the parent).
            return Ok(Some(self.emit_postfix_base(base, ctx)?));
        }

        // Emit as `base.xyzw`.  The base carries a `.swizzle` suffix, so it
        // must be parenthesised when it is a Binary/Unary/Select - otherwise
        // `(a-b).yx` degrades to `a-b.yx`, which parses as `a-(b.yx)`.
        let mut s = self.emit_postfix_base(base, ctx)?;
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
                    // Postfix-aware: a Binary/Unary/Select base before the
                    // `.swizzle` suffix must be parenthesised (`(a-b).xy`).
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
        if !ctx.expr_names.contains_key(&expr)
            && let Some(scalar) = hint
        {
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
            // Already-concrete floats matching the hint still need the TYPED
            // spelling: the default emit path renders whole-number floats in
            // bare-int form (`10.0` -> `10`), which only re-parses as a float
            // in positions where naga applies abstract coercion.  A hinted
            // position is by definition one where it does not (e.g. the
            // `rayQueryGenerateIntersection` hit_t slot or a `Derivative`
            // argument, which concretize a bare literal to i32 instead).
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

const PREC_SHIFT: u8 = 8;
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
        B::ShiftLeft | B::ShiftRight => PREC_SHIFT,
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

    // Bitwise operators (`&`/`|`/`^`) require `unary_expression` on
    // both sides per WGSL's grammar (https://www.w3.org/TR/WGSL/#operator-precedence-associativity)
    // so ANY binary child (e.g. `a^b-1u`, `a&b*c`, `a|b<<c`) is
    // grammatically ill-formed even when precedence alone would group it
    // correctly.  naga's own parser is permissive, but the strict
    // recursive-descent parsers nagami targets (Tint/Dawn/browsers) reject
    // it, and naga round-trips the malformed text so no fallback fires.
    // The one exception is the grammar's left-recursion
    // (`binary_and_expression '&' unary_expression`): a *left* child that
    // uses the *same* bitwise operator is legal unparenthesised, so keep it
    // bare to avoid needless parens on `a&b&c`.
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

    // WGSL puts all six comparison operators (`< <= > >= == !=`) at a single
    // non-associative grammar level whose operands must each be a
    // `shift_expression` (WGSL https://www.w3.org/TR/WGSL/#operator-precedence-associativity
    // and https://www.w3.org/TR/WGSL/#syntax-relational_expression),
    // so any comparison child needs parens here - `a<b==c<d` is the ill-formed
    // shape Dawn/Tint reject ("mixing '<' and '==' requires parenthesis") even
    // though naga's permissive frontend round-trips it.  Hence `< PREC_SHIFT`
    // rather than the more obvious `<= parent_prec`: `==`/`!=` (6) and the
    // relational quartet (7) differ in this table yet share one grammar level,
    // so a relational child of an equality parent must still be wrapped.
    // Always meaning-preserving - the parenthesised grouping is the only
    // well-typed one.
    if matches!(
        parent_op,
        naga::BinaryOperator::Less
            | naga::BinaryOperator::LessEqual
            | naga::BinaryOperator::Greater
            | naga::BinaryOperator::GreaterEqual
            | naga::BinaryOperator::Equal
            | naga::BinaryOperator::NotEqual
    ) {
        return child_prec < PREC_SHIFT;
    }

    // A `&&` / `||` parent's operands are each a `relational_expression`
    // (WGSL https://www.w3.org/TR/WGSL/#syntax-expression): the grammar
    // admits comparisons, shifts and arithmetic bare but NOT another
    // logical or a bitwise expression.  naga's permissive frontend
    // round-trips the bare forms, but Tint/Dawn (and browsers) reject them,
    // so wrap exactly the children the grammar forbids:
    if matches!(
        parent_op,
        naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
    ) {
        return match child_op {
            // Mixing `&&` and `||` is illegal bare ("mixing '&&' and '||'
            // requires parenthesis").  The grammar's own left-recursion
            // (`a && b && c`) keeps a SAME-operator LEFT child bare; a
            // same-operator RIGHT child is wrapped to preserve the IR's tree
            // shape (associativity makes the value identical either way, but
            // re-grouping a right-leaning chain would perturb idempotence).
            Some(naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr) => {
                child_op != Some(parent_op) || is_right
            }
            // A bitwise expression (`|` / `^` / `&`) is not a
            // `relational_expression`, so it can never be a bare `&&` / `||`
            // operand (Tint rejects "mixing '|' and '&&' requires
            // parenthesis").  `bool | bool` / `bool & bool` are legal and
            // the short-circuit re-sugar now collapses `(a | b) && c` into a
            // single logical Binary, so this child shape is reachable and
            // MUST be wrapped.  (`bool ^ bool` is itself invalid WGSL, so
            // `^` never reaches here, but listing it is harmless.)
            Some(
                naga::BinaryOperator::InclusiveOr
                | naga::BinaryOperator::ExclusiveOr
                | naga::BinaryOperator::And,
            ) => true,
            // Comparisons, shifts, arithmetic, unary and atoms are all valid
            // `relational_expression` operands - keep them bare.
            _ => false,
        };
    }

    // Left-associative: left child needs parens if strictly lower,
    // right child if lower-or-equal (to preserve tree structure).
    if is_right {
        child_prec <= parent_prec
    } else {
        child_prec < parent_prec
    }
}

/// `true` only when the operand is an uncached Binary expression: every
/// Binary precedence is below Unary, so it must be parenthesised; a cached
/// operand (emitted as a name), atom, call, or nested unary never is.
fn unary_child_needs_parens(
    child: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    is_cached: bool,
) -> bool {
    !is_cached && matches!(arena[child], naga::Expression::Binary { .. })
}

/// Assemble `left op right` from pre-computed operand strings with the
/// correct parenthesisation and operator spacing for the current beautify
/// mode.  All binary emission paths funnel through here.
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
        // Disambiguate against three WGSL trigraphs the WGSL lexer would
        // otherwise misread:
        //   `--`  -> decrement (e.g. `a - -b` minified to `a--b`)
        //   `//`  -> line comment start (impossible from valid IR but
        //            guarded for symmetry)
        //   `/*`  -> block comment start (e.g. `a / *p` where `*p` is a
        //            pointer dereference rendered by `emit_lvalue`).
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

/// When a matrix `Compose` is built from one explicit scalar-column `Compose`
/// per column (`mat2x2f(vec2f(a,b), vec2f(c,d))`), return the flattened scalar
/// component handles in column-major order so the matrix can be emitted in the
/// shorter all-scalar form `mat2x2f(a,b,c,d)`.
///
/// Returns `None` (keep column form) unless EVERY column is an
/// `Expression::Compose` of a `Vector` type whose component count equals the
/// matrix row count.  A `vecR` built from exactly `R` Compose-components is
/// necessarily `R` scalars - any vector sub-component (`vec3(v.xy, z)`) would
/// lower the count below `R` - so this structural test alone guarantees scalar
/// columns with no per-component type lookup, and it works for both the
/// function-local and global-constant arenas.  Splat columns (`Splat`) and
/// variable / let-bound / swizzle columns are deliberately excluded (not a
/// scalar `Compose`), keeping the rewrite a strict value-preserving regroup of
/// the same scalar leaves the column form already emits.  Both forms lower to
/// byte-identical naga IR (incl. f16 and negative leaves).
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

/// `true` only for a literal whose bit pattern is exactly `+0` (or integer
/// `0`).  STRICTER than `literal_is_zero`: `-0.0` (bits `0x8000_0000` / `0x8000`)
/// returns `false`, because folding it to a zero-value constructor (`vec2f()`)
/// would silently flip the sign bit (`1.0/-0.0 == -inf` vs `+inf`).  `Bool`
/// excluded so the fold never removes a `false` literal a constructor needs.
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

/// `true` when `h` is a vector/matrix component tree that is provably all `+0`
/// (strict-zero `Literal`, `ZeroValue`, or `Splat` / `Compose` of those).
/// Drives the zero-value-constructor fold `vec2f(0,0)` -> `vec2f()`.
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

/// Value-equality of two expressions for splat/run-collapse purposes.
/// Deliberately conservative: `true` ONLY for two `Literal`s with identical
/// bit patterns (so `-0.0` and `+0.0` differ) or two `Constant`/`Override`/
/// `ZeroValue`s with the same handle/type.  EVERY other pair - `Load`,
/// `CallResult`, `Binary`, `FunctionArgument`, ... - returns `false`, so
/// impure or reorder-sensitive components can never be collapsed into a single
/// shared value.
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

/// The scalar element of a scalar or vector type, or `None` for any
/// other type.  Used by [`Generator::vector_ctor_name`] to test whether
/// a component's concrete scalar pins a vector constructor's element type.
fn type_inner_scalar(inner: &naga::TypeInner) -> Option<naga::Scalar> {
    match inner {
        naga::TypeInner::Scalar(s) => Some(*s),
        naga::TypeInner::Vector { scalar, .. } => Some(*scalar),
        _ => None,
    }
}

/// Map a comparison operator to the equivalent negated operator when
/// negation is semantically safe (equality and inequality are safe for
/// every scalar type; ordered comparisons are deferred to the caller,
/// which checks for float operands).
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
