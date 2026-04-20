//! Constant folding.
//!
//! Rewrites expressions whose operands are statically known values
//! into `Literal`, `ZeroValue`, or `Compose` nodes that the generator
//! can emit directly.  The pass operates in three arenas:
//!
//! * `module.global_expressions` for `const` initializers and global
//!   splats, seeded with every other constant via
//!   `build_constant_literal_cache`.
//! * each function's `expressions` arena, folded with a per-function
//!   pass that tracks which handles must drop out of their `Emit`
//!   range (folded literals are declarative, not emittable).
//! * the `Emit` ranges themselves, rebuilt in place via
//!   `rebuild_emit_ranges_after_removal` after each function fold.
//!
//! Structural folding covers scalar arithmetic, binary and unary
//! operators, casts, relational built-ins, and a subset of math
//! built-ins; composite folding handles vector construction, swizzle
//! collapse, and splat elision.  NaN- and overflow-sensitive math
//! built-ins are intentionally not folded because the emitted WGSL
//! must match the runtime's behaviour bit-for-bit.

use std::collections::{HashMap, HashSet};

use naga::Handle;

use crate::error::Error;
use crate::passes::expr_util::rebuild_emit_ranges_after_removal;
use crate::pipeline::{Pass, PassContext};

/// Constant folding across globals, functions, and entry points.
#[derive(Debug, Default)]
pub struct ConstFoldPass;

impl Pass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn run(&mut self, module: &mut naga::Module, _ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        changed += fold_global_expressions(module);
        let const_literals = build_constant_literal_cache(module);

        for (_, function) in module.functions.iter_mut() {
            let (folded, simplified) =
                fold_local_expressions(&mut function.expressions, &const_literals, &module.types);
            changed += simplified;
            if !folded.is_empty() {
                changed += folded.len();
                rebuild_emit_ranges_after_removal(&mut function.body, &folded);
            }
        }
        for entry in module.entry_points.iter_mut() {
            let (folded, simplified) = fold_local_expressions(
                &mut entry.function.expressions,
                &const_literals,
                &module.types,
            );
            changed += simplified;
            if !folded.is_empty() {
                changed += folded.len();
                rebuild_emit_ranges_after_removal(&mut entry.function.body, &folded);
            }
        }

        Ok(changed > 0)
    }
}

// MARK: Global-expression folding

/// Fold `module.global_expressions`, converting composite
/// sub-expressions (vectors, swizzles, splats) into the most compact
/// form the generator can emit.  The literal and vector-type caches
/// are built once per invocation so the inner materialisation pass
/// stays `O(1)` per handle.
fn fold_global_expressions(module: &mut naga::Module) -> usize {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HashMap<_, _>>();

    let handles = module
        .global_expressions
        .iter()
        .map(|(h, _)| h)
        .collect::<Vec<_>>();
    let mut changed = 0usize;

    // O(N) one-time setup for O(1) materialize_vector lookups (E2).
    let mut literal_cache = build_literal_cache(&module.global_expressions);
    let vector_type_cache = build_vector_type_cache(&module.types);

    for handle in handles {
        // Try composite-aware resolution first (covers vectors, swizzle, etc.).
        let value = {
            let ctx = GlobalConstFoldContext {
                arena: &module.global_expressions,
                types: &module.types,
                const_inits: &const_inits,
            };
            let mut visiting = HashSet::new();
            resolve_const_value(handle, &ctx, &mut visiting)
        };

        if let Some(ConstValue::Scalar(literal)) = value {
            if !matches!(module.global_expressions[handle], naga::Expression::Literal(existing) if existing == literal)
            {
                module.global_expressions[handle] = naga::Expression::Literal(literal);
                // Keep the literal cache in sync so later materialize_vector
                // calls can discover the new literal at `handle`.
                note_literal_in_cache(&mut literal_cache, handle, literal);
                changed += 1;
            }
            continue;
        }

        if let Some(ConstValue::Vector {
            ref components,
            size,
            scalar,
        }) = value
        {
            if let Some(new_expr) = materialize_vector(
                handle,
                components,
                size,
                scalar,
                &literal_cache,
                &vector_type_cache,
            ) {
                if module.global_expressions[handle] != new_expr {
                    module.global_expressions[handle] = new_expr;
                    changed += 1;
                }
            }
        }
    }

    changed
}

/// Map every `Constant` whose initializer resolves to a `Literal` to
/// that literal.  Local-expression folding consults this cache
/// whenever it needs to treat a `Constant(handle)` operand as a
/// concrete value.
fn build_constant_literal_cache(
    module: &naga::Module,
) -> HashMap<naga::Handle<naga::Constant>, naga::Literal> {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HashMap<_, _>>();
    let context = GlobalLiteralContext {
        arena: &module.global_expressions,
        const_inits: &const_inits,
    };

    let mut out = HashMap::new();
    for (ch, c) in module.constants.iter() {
        let mut visiting = HashSet::new();
        if let Some(lit) = resolve_literal(c.init, &context, &mut visiting) {
            out.insert(ch, lit);
        }
    }
    out
}

// MARK: Per-function folding

/// Fold `arena` in place.  Returns two outputs: the set of handles
/// that must leave their original `Emit` ranges (folded-to-literal
/// expressions lose their emit requirement) and the number of
/// simplifications performed, so the caller can roll both into the
/// module-wide change counter.
fn fold_local_expressions(
    arena: &mut naga::Arena<naga::Expression>,
    const_literals: &HashMap<naga::Handle<naga::Constant>, naga::Literal>,
    types: &naga::UniqueArena<naga::Type>,
) -> (HashSet<naga::Handle<naga::Expression>>, usize) {
    let handles = arena.iter().map(|(h, _)| h).collect::<Vec<_>>();
    let mut folded = HashSet::new();

    // O(N) one-time setup for O(1) materialize_vector lookups (E2).
    // The literal cache maps each scalar literal -> smallest handle carrying
    // it.  We keep it in sync as the scalar-fold branch writes new literals
    // into the arena; see `note_literal_in_cache`.
    let mut literal_cache = build_literal_cache(arena);
    let vector_type_cache = build_vector_type_cache(types);

    for handle in handles.iter().copied() {
        let value = {
            let ctx = LocalConstFoldContext {
                arena: &*arena,
                types,
                const_literals,
            };
            let mut visiting = HashSet::new();
            resolve_const_value(handle, &ctx, &mut visiting)
        };

        match value {
            Some(ConstValue::Scalar(literal)) => {
                if !matches!(arena[handle], naga::Expression::Literal(existing) if existing == literal)
                {
                    arena[handle] = naga::Expression::Literal(literal);
                    // Keep the literal cache in sync so subsequent
                    // materialize_vector calls in this pass can use the new
                    // literal at `handle`.
                    note_literal_in_cache(&mut literal_cache, handle, literal);
                    folded.insert(handle);
                }
            }
            Some(ConstValue::Vector {
                ref components,
                size,
                scalar,
            }) => {
                // Only replace with Compose if the original expression is
                // already emittable.  Replacing a non-emittable expression
                // (ZeroValue, Constant, etc.) with an emittable Compose would
                // produce invalid IR because the handle isn't in any Emit range.
                if needs_emit(&arena[handle]) {
                    if let Some(new_expr) = materialize_vector(
                        handle,
                        components,
                        size,
                        scalar,
                        &literal_cache,
                        &vector_type_cache,
                    ) {
                        if arena[handle] != new_expr {
                            arena[handle] = new_expr;
                            // Compose is emittable - do NOT add to folded set.
                        }
                    }
                }
            }
            None => {}
        }
    }

    // Identity / absorbing / involution / select elimination.
    //
    // Identity:   `0 + x -> x`, `x * 1 -> x`, `x | 0 -> x`, etc.
    // Absorbing:  `x * 0 -> 0`, `x & 0 -> 0`, `x | all_ones -> all_ones`, etc.
    // Involution: `-(-x) -> x`, `!(!x) -> x`, `~(~x) -> x`.
    // Select:     `select(x, x, cond) -> x` when accept == reject.
    let mut simplify_count = 0usize;
    for handle in handles {
        match arena[handle] {
            naga::Expression::Binary { op, left, right } => {
                // Try absorbing first (result is always a literal -> non-emittable).
                //
                // SAFETY GATE: absorbing replaces the `Binary` expression with
                // a clone of the scalar-literal operand that matched
                // `is_zero`/`is_all_ones`/`is_bool_*`.  The `Binary`'s result
                // type, however, follows naga's broadcasting rules: a
                // `vec3<f32> * 0.0` is a vec3, not a scalar.  Replacing such a
                // Binary with the scalar `0.0` operand corrupts the IR.
                //
                // Gate by operator class:
                //   * `LogicalAnd` / `LogicalOr` are strictly `bool x bool ->
                //     bool` in valid WGSL IR (short-circuit operators; naga's
                //     validator rejects vector operands).  Absorbing is
                //     therefore always type-safe for these, regardless of
                //     what the other operand is.
                //   * `Multiply` / `And` / `InclusiveOr` can broadcast a
                //     scalar operand against a vector to produce a vector
                //     result.  Absorbing to the scalar literal would corrupt
                //     the type, so we require BOTH operands be scalar
                //     `Literal`.  (In practice `resolve_const_value` folds
                //     the both-literal case upstream; this rewrite is a
                //     safety net.  A broader rule would need a type
                //     resolver.
                let both_literal = matches!(arena[left], naga::Expression::Literal(_))
                    && matches!(arena[right], naga::Expression::Literal(_));
                let is_logical_op = matches!(
                    op,
                    naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
                );
                if is_logical_op || both_literal {
                    if let Some(absorb) = check_absorbing_operand(op, left, right, arena) {
                        arena[handle] = arena[absorb].clone();
                        simplify_count += 1;
                        // Absorbing result is always a Literal -> must leave Emit ranges.
                        folded.insert(handle);
                        continue;
                    }
                }
                // Then try identity (result is the other operand).
                //
                // Identity is type-safe without any additional gate: it returns
                // the *other* (non-literal) operand, whose type IS the
                // broadcast result type of the Binary by definition (the
                // identity literal is by construction the scalar "neutral"
                // element, so the other operand carries the result type).
                if let Some(other) = check_identity_operand(op, left, right, arena) {
                    arena[handle] = arena[other].clone();
                    simplify_count += 1;
                    if !needs_emit(&arena[handle]) {
                        folded.insert(handle);
                    }
                    continue;
                }
            }
            naga::Expression::Unary { op, expr } => {
                // Involution: op(op(x)) -> x
                if let naga::Expression::Unary {
                    op: inner_op,
                    expr: inner,
                } = arena[expr]
                {
                    if op == inner_op {
                        arena[handle] = arena[inner].clone();
                        simplify_count += 1;
                        if !needs_emit(&arena[handle]) {
                            folded.insert(handle);
                        }
                        continue;
                    }
                }
            }
            naga::Expression::Select { accept, reject, .. } => {
                // select(x, x, cond) -> x when both arms are the same handle.
                if accept == reject {
                    arena[handle] = arena[accept].clone();
                    simplify_count += 1;
                    if !needs_emit(&arena[handle]) {
                        folded.insert(handle);
                    }
                    continue;
                }
            }
            _ => {}
        }
    }

    (folded, simplify_count)
}

trait LiteralContext {
    fn arena(&self) -> &naga::Arena<naga::Expression>;
    fn resolve_constant(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<naga::Handle<naga::Expression>>,
    ) -> Option<naga::Literal>;
}

struct GlobalLiteralContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    const_inits: &'a HashMap<naga::Handle<naga::Constant>, naga::Handle<naga::Expression>>,
}

impl LiteralContext for GlobalLiteralContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }

    fn resolve_constant(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<naga::Handle<naga::Expression>>,
    ) -> Option<naga::Literal> {
        let init = *self.const_inits.get(&handle)?;
        resolve_literal(init, self, visiting)
    }
}

// MARK: Constant value resolution

// A fully-resolved constant value: either a scalar literal or a vector
// of scalar literals.  Matrices are intentionally excluded; they rarely
// appear in constant contexts and the folding effort would not justify
// the added case analysis.
#[derive(Debug, Clone, PartialEq)]
enum ConstValue {
    Scalar(naga::Literal),
    /// A constant vector whose components are all known scalar literals.
    Vector {
        components: Vec<naga::Literal>,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    },
}

impl ConstValue {
    /// Return `Some(literal)` if this is a scalar.
    fn as_scalar(&self) -> Option<naga::Literal> {
        match self {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }
}

/// Composite-resolution context.  Extends [`LiteralContext`] with
/// access to the type arena because `Compose` / `Splat` rewrites need
/// to discover the component type and vector size.
trait ConstFoldContext {
    fn arena(&self) -> &naga::Arena<naga::Expression>;
    fn types(&self) -> &naga::UniqueArena<naga::Type>;
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<Handle<naga::Expression>>,
    ) -> Option<ConstValue>;
}

struct GlobalConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    const_inits: &'a HashMap<naga::Handle<naga::Constant>, naga::Handle<naga::Expression>>,
}

impl ConstFoldContext for GlobalConstFoldContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }
    fn types(&self) -> &naga::UniqueArena<naga::Type> {
        self.types
    }
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HashSet<Handle<naga::Expression>>,
    ) -> Option<ConstValue> {
        let init = *self.const_inits.get(&handle)?;
        resolve_const_value(init, self, visiting)
    }
}

struct LocalConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    const_literals: &'a HashMap<naga::Handle<naga::Constant>, naga::Literal>,
}

impl ConstFoldContext for LocalConstFoldContext<'_> {
    fn arena(&self) -> &naga::Arena<naga::Expression> {
        self.arena
    }
    fn types(&self) -> &naga::UniqueArena<naga::Type> {
        self.types
    }
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        _visiting: &mut HashSet<Handle<naga::Expression>>,
    ) -> Option<ConstValue> {
        self.const_literals
            .get(&handle)
            .copied()
            .map(ConstValue::Scalar)
    }
}

// MARK: Resolver entry points

/// Resolve `handle` to a fully-constant [`ConstValue`], threading
/// `visiting` to short-circuit cyclic expression graphs.  This is the
/// composite-aware generalisation of [`resolve_literal`]; it handles
/// `Compose`, `Splat`, `ZeroValue`, `Swizzle`, `AccessIndex`, and
/// component-wise `Binary` / `Unary` on vectors.
fn resolve_const_value<C: ConstFoldContext>(
    handle: Handle<naga::Expression>,
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
) -> Option<ConstValue> {
    if !visiting.insert(handle) {
        return None;
    }

    let expr = &ctx.arena()[handle];
    let out = match expr {
        // Scalar leaves
        naga::Expression::Literal(lit) => Some(ConstValue::Scalar(*lit)),
        naga::Expression::Constant(ch) => ctx.resolve_constant_value(*ch, visiting),

        // Composite constructors
        naga::Expression::ZeroValue(ty) => resolve_zero_value(*ty, ctx),

        naga::Expression::Splat { size, value } => {
            let inner = resolve_const_value(*value, ctx, visiting)?;
            let lit = inner.as_scalar()?;
            Some(ConstValue::Vector {
                scalar: lit.scalar(),
                size: *size,
                components: vec![lit; *size as usize],
            })
        }

        naga::Expression::Compose { ty, components } => {
            resolve_compose(*ty, components, ctx, visiting)
        }

        // Indexing / swizzle
        naga::Expression::AccessIndex { base, index } => {
            let base_val = resolve_const_value(*base, ctx, visiting)?;
            match base_val {
                ConstValue::Vector { ref components, .. } => components
                    .get(*index as usize)
                    .copied()
                    .map(ConstValue::Scalar),
                _ => None,
            }
        }

        naga::Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let vec_val = resolve_const_value(*vector, ctx, visiting)?;
            match vec_val {
                ConstValue::Vector {
                    ref components,
                    scalar,
                    ..
                } => {
                    let n = *size as usize;
                    let mut out = Vec::with_capacity(n);
                    for &sw in &pattern[..n] {
                        let idx = sw as usize;
                        out.push(*components.get(idx)?);
                    }
                    Some(ConstValue::Vector {
                        components: out,
                        size: *size,
                        scalar,
                    })
                }
                _ => None,
            }
        }

        // Scalar ops (delegate to existing evaluators)
        naga::Expression::Unary { op, expr } => {
            let inner = resolve_const_value(*expr, ctx, visiting)?;
            eval_const_unary(*op, inner)
        }

        naga::Expression::Binary { op, left, right } => {
            let l = resolve_const_value(*left, ctx, visiting)?;
            let r = resolve_const_value(*right, ctx, visiting)?;
            eval_const_binary(*op, l, r)
        }

        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_const_value(*arg, ctx, visiting)?;
            let b = match arg1 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting)?),
                None => None,
            };
            eval_const_math(*fun, a, b, c)
        }

        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_const_value(*condition, ctx, visiting)?;
            match cond {
                ConstValue::Scalar(naga::Literal::Bool(true)) => {
                    resolve_const_value(*accept, ctx, visiting)
                }
                ConstValue::Scalar(naga::Literal::Bool(false)) => {
                    resolve_const_value(*reject, ctx, visiting)
                }
                _ => None,
            }
        }

        _ => None,
    };

    visiting.remove(&handle);
    out
}

/// Resolve a `ZeroValue(ty)` to a `ConstValue`.
/// Materialise `ZeroValue(ty)` as a [`ConstValue`] when the type is
/// scalar or a zeroable vector; returns `None` for any type the rest
/// of the pass does not understand (matrices, structs, arrays, etc.).
fn resolve_zero_value<C: ConstFoldContext>(ty: Handle<naga::Type>, ctx: &C) -> Option<ConstValue> {
    match ctx.types()[ty].inner {
        naga::TypeInner::Scalar(s) => naga::Literal::zero(s).map(ConstValue::Scalar),
        naga::TypeInner::Vector { size, scalar } => {
            let z = naga::Literal::zero(scalar)?;
            Some(ConstValue::Vector {
                components: vec![z; size as usize],
                size,
                scalar,
            })
        }
        _ => None,
    }
}

/// Resolve a `Compose { ty, components }` to a `ConstValue`.
/// Resolve a `Compose { ty, components }` expression into a
/// [`ConstValue::Vector`] when every component folds to a scalar of a
/// homogeneous type.  Mixed component kinds, non-vector composites,
/// and unresolved operands all short-circuit to `None`.
fn resolve_compose<C: ConstFoldContext>(
    ty: Handle<naga::Type>,
    components: &[Handle<naga::Expression>],
    ctx: &C,
    visiting: &mut HashSet<Handle<naga::Expression>>,
) -> Option<ConstValue> {
    let inner = &ctx.types()[ty].inner;
    match inner {
        naga::TypeInner::Vector { size, scalar } => {
            let expected = *size as usize;
            // Two forms: N scalar components, or a mix of scalars/vectors
            // that flatten to N scalar components.
            let mut out = Vec::with_capacity(expected);
            for &c in components {
                let val = resolve_const_value(c, ctx, visiting)?;
                match val {
                    ConstValue::Scalar(l) => out.push(l),
                    ConstValue::Vector { components: v, .. } => out.extend(v),
                }
            }
            if out.len() != expected {
                return None;
            }
            Some(ConstValue::Vector {
                components: out,
                size: *size,
                scalar: *scalar,
            })
        }
        _ => None,
    }
}

/// Evaluate a unary operation on a `ConstValue`.
/// Apply a unary operator to a fully-resolved [`ConstValue`],
/// broadcasting over vector components.  Delegates per-scalar
/// evaluation to [`eval_unary`].
fn eval_const_unary(op: naga::UnaryOperator, val: ConstValue) -> Option<ConstValue> {
    match val {
        ConstValue::Scalar(lit) => eval_unary(op, lit).map(ConstValue::Scalar),
        ConstValue::Vector {
            components,
            size,
            scalar,
        } => {
            let folded: Option<Vec<_>> =
                components.into_iter().map(|l| eval_unary(op, l)).collect();
            Some(ConstValue::Vector {
                components: folded?,
                size,
                scalar,
            })
        }
    }
}

/// Returns `true` when `op` is a relational operator whose result type is
/// `bool` (or `vec<bool>`) regardless of the operand type.
/// `true` for binary operators that produce a boolean result
/// (relational or equality).  Used by the vector binary folder to
/// decide whether the result type is a bool vector.
fn is_relational_op(op: naga::BinaryOperator) -> bool {
    matches!(
        op,
        naga::BinaryOperator::Equal
            | naga::BinaryOperator::NotEqual
            | naga::BinaryOperator::Less
            | naga::BinaryOperator::LessEqual
            | naga::BinaryOperator::Greater
            | naga::BinaryOperator::GreaterEqual
    )
}

/// Apply a binary operator pair-wise over two [`ConstValue`]
/// operands.  Handles three shapes:
///
/// - scalar `<op>` scalar, delegating to [`eval_binary`];
/// - vector `<op>` vector, component-wise, same size; and
/// - scalar `<op>` vector / vector `<op>` scalar broadcast for
///   arithmetic ops.
fn eval_const_binary(
    op: naga::BinaryOperator,
    lhs: ConstValue,
    rhs: ConstValue,
) -> Option<ConstValue> {
    match (lhs, rhs) {
        (ConstValue::Scalar(l), ConstValue::Scalar(r)) => {
            eval_binary(op, l, r).map(ConstValue::Scalar)
        }
        (
            ConstValue::Vector {
                components: lc,
                size: ls,
                scalar: lscalar,
            },
            ConstValue::Vector {
                components: rc,
                size: rs,
                ..
            },
        ) if ls == rs => {
            let folded: Option<Vec<_>> = lc
                .into_iter()
                .zip(rc)
                .map(|(l, r)| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                lscalar
            };
            Some(ConstValue::Vector {
                components,
                size: ls,
                scalar: out_scalar,
            })
        }
        // scalar <op> vector -> broadcast scalar then component-wise
        (
            ConstValue::Scalar(l),
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
        ) => {
            let folded: Option<Vec<_>> = components
                .into_iter()
                .map(|r| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                scalar
            };
            Some(ConstValue::Vector {
                components,
                size,
                scalar: out_scalar,
            })
        }
        // vector <op> scalar -> broadcast scalar then component-wise
        (
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
            ConstValue::Scalar(r),
        ) => {
            let folded: Option<Vec<_>> = components
                .into_iter()
                .map(|l| eval_binary(op, l, r))
                .collect();
            let components = folded?;
            let out_scalar = if is_relational_op(op) {
                naga::Scalar::BOOL
            } else {
                scalar
            };
            Some(ConstValue::Vector {
                components,
                size,
                scalar: out_scalar,
            })
        }
        _ => None,
    }
}

// MARK: Materialisation helpers

/// Try to build a `Compose` expression for a folded vector by
/// finding existing `Literal` handles in the arena for each component.
///
/// Returns `Some(new_expression)` when every component literal handle
/// is found at a position BEFORE `target` (topological-order safe),
/// and `None` when any component has no matching literal handle.
///
/// The caller supplies prebuilt O(1) lookup caches:
///
/// - `literal_cache` maps each scalar `Literal` to the SMALLEST handle
///   in the arena carrying that literal.  The "smallest handle"
///   invariant is what lets the topological-safety check below
///   succeed as often as possible.
/// - `vector_type_cache` maps each `(VectorSize, Scalar)` pair to a
///   `Handle<Type>` for the corresponding `TypeInner::Vector` in the
///   type arena.
///
/// Prior to [E2], this function did two linear scans per call:
/// `arena.iter()` for every component literal and `types.iter()` for
/// the vector type.  With K materializations over an arena of size
/// N, the total cost was O(N*K).  The caches reduce this to O(N)
/// one-time setup plus O(1) per component.
fn materialize_vector(
    target: Handle<naga::Expression>,
    literals: &[naga::Literal],
    size: naga::VectorSize,
    scalar: naga::Scalar,
    literal_cache: &HashMap<LiteralKey, Handle<naga::Expression>>,
    vector_type_cache: &HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> Option<naga::Expression> {
    let ty = *vector_type_cache.get(&(size, scalar))?;

    // For each component literal, find a matching `Literal` expression at an
    // index strictly less than `target` so that the resulting Compose respects
    // topological ordering.
    let mut handles = Vec::with_capacity(literals.len());
    for lit in literals {
        let h = *literal_cache.get(&literal_key(*lit))?;
        if h.index() >= target.index() {
            return None;
        }
        handles.push(h);
    }

    Some(naga::Expression::Compose {
        ty,
        components: handles,
    })
}

/// Canonical hash key for a `naga::Literal`.
///
/// Float variants use `to_bits` so that `NaN` payloads are preserved and
/// `-0.0` compares distinct from `+0.0`.  Using `f32`/`f64`/`f16` directly as
/// map keys is impossible anyway (they don't implement `Eq`/`Hash`).
#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
enum LiteralKey {
    F32(u32),
    F64(u64),
    F16(u16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    AbstractInt(i64),
    AbstractFloat(u64),
}

fn literal_key(lit: naga::Literal) -> LiteralKey {
    match lit {
        naga::Literal::F32(v) => LiteralKey::F32(v.to_bits()),
        naga::Literal::F64(v) => LiteralKey::F64(v.to_bits()),
        naga::Literal::F16(v) => LiteralKey::F16(v.to_bits()),
        naga::Literal::U32(v) => LiteralKey::U32(v),
        naga::Literal::I32(v) => LiteralKey::I32(v),
        naga::Literal::U64(v) => LiteralKey::U64(v),
        naga::Literal::I64(v) => LiteralKey::I64(v),
        naga::Literal::Bool(v) => LiteralKey::Bool(v),
        naga::Literal::AbstractInt(v) => LiteralKey::AbstractInt(v),
        naga::Literal::AbstractFloat(v) => LiteralKey::AbstractFloat(v.to_bits()),
    }
}

/// Build a cache mapping each scalar `Literal` in the expression arena to the
/// SMALLEST handle carrying it.  The "smallest" policy maximizes the
/// likelihood that `materialize_vector`'s topological-order check passes.
/// Build the `literal -> smallest handle carrying it` lookup table
/// that [`materialize_vector`] and [`note_literal_in_cache`] use to
/// keep composite rewrites `O(1)` per component.
fn build_literal_cache(
    arena: &naga::Arena<naga::Expression>,
) -> HashMap<LiteralKey, Handle<naga::Expression>> {
    let mut cache: HashMap<LiteralKey, Handle<naga::Expression>> = HashMap::new();
    for (h, expr) in arena.iter() {
        if let naga::Expression::Literal(lit) = expr {
            cache
                .entry(literal_key(*lit))
                .and_modify(|cur| {
                    if h.index() < cur.index() {
                        *cur = h;
                    }
                })
                .or_insert(h);
        }
    }
    cache
}

/// Build a cache mapping each `(VectorSize, Scalar)` pair to the
/// `Handle<Type>` for the corresponding `TypeInner::Vector`.
/// Index every vector `Type` by its `(size, scalar)` shape so
/// [`materialize_vector`] can synthesize a `Compose` with the right
/// result type without scanning the type arena each call.
fn build_vector_type_cache(
    types: &naga::UniqueArena<naga::Type>,
) -> HashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>> {
    let mut cache = HashMap::new();
    for (h, t) in types.iter() {
        if let naga::TypeInner::Vector { size, scalar } = t.inner {
            cache.entry((size, scalar)).or_insert(h);
        }
    }
    cache
}

/// Record a newly-created `Literal` at `handle` into the literal cache,
/// preserving the smallest-handle invariant.
/// Record `handle` as the canonical carrier for `lit` if no earlier
/// handle already holds it, keeping later `materialize_vector` calls
/// aware of literals the scalar fold branch just wrote into the arena.
fn note_literal_in_cache(
    cache: &mut HashMap<LiteralKey, Handle<naga::Expression>>,
    handle: Handle<naga::Expression>,
    literal: naga::Literal,
) {
    cache
        .entry(literal_key(literal))
        .and_modify(|cur| {
            if handle.index() < cur.index() {
                *cur = handle;
            }
        })
        .or_insert(handle);
}

// MARK: Scalar evaluation

/// Resolve `handle` to a bare scalar literal, cycling through
/// `Constant`, `ZeroValue`, and simple unary/binary/cast nodes whose
/// operands themselves resolve.  Shallower than
/// [`resolve_const_value`]: bails out on any vector or composite.
fn resolve_literal<C: LiteralContext>(
    handle: naga::Handle<naga::Expression>,
    context: &C,
    visiting: &mut HashSet<naga::Handle<naga::Expression>>,
) -> Option<naga::Literal> {
    if !visiting.insert(handle) {
        return None;
    }

    let expr = &context.arena()[handle];
    let out = match expr {
        naga::Expression::Literal(lit) => Some(*lit),
        naga::Expression::Constant(ch) => context.resolve_constant(*ch, visiting),
        naga::Expression::Unary { op, expr } => {
            let rhs = resolve_literal(*expr, context, visiting)?;
            eval_unary(*op, rhs)
        }
        naga::Expression::Binary { op, left, right } => {
            let l = resolve_literal(*left, context, visiting)?;
            let r = resolve_literal(*right, context, visiting)?;
            eval_binary(*op, l, r)
        }
        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_literal(*arg, context, visiting)?;
            let b = match arg1 {
                Some(h) => Some(resolve_literal(*h, context, visiting)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_literal(*h, context, visiting)?),
                None => None,
            };
            eval_math_scalar(*fun, a, b, c)
        }
        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_literal(*condition, context, visiting)?;
            match cond {
                naga::Literal::Bool(true) => resolve_literal(*accept, context, visiting),
                naga::Literal::Bool(false) => resolve_literal(*reject, context, visiting),
                _ => None,
            }
        }
        _ => None,
    };

    visiting.remove(&handle);
    out
}

/// Per-scalar unary evaluator.  Returns `None` for operator/type
/// pairs that would change observable behaviour (e.g. negation of
/// `u32::MIN`); those cases fall through to runtime evaluation.
fn eval_unary(op: naga::UnaryOperator, rhs: naga::Literal) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::UnaryOperator as U;

    match (op, rhs) {
        (U::Negate, L::F32(v)) if (-v).is_finite() => Some(L::F32(-v)),
        (U::Negate, L::F64(v)) if (-v).is_finite() => Some(L::F64(-v)),
        (U::Negate, L::I32(v)) => v.checked_neg().map(L::I32),
        (U::Negate, L::I64(v)) => v.checked_neg().map(L::I64),
        (U::Negate, L::AbstractInt(v)) => v.checked_neg().map(L::AbstractInt),
        (U::Negate, L::AbstractFloat(v)) if (-v).is_finite() => Some(L::AbstractFloat(-v)),
        (U::LogicalNot, L::Bool(v)) => Some(L::Bool(!v)),
        (U::BitwiseNot, L::U32(v)) => Some(L::U32(!v)),
        (U::BitwiseNot, L::U64(v)) => Some(L::U64(!v)),
        (U::BitwiseNot, L::I32(v)) => Some(L::I32(!v)),
        (U::BitwiseNot, L::I64(v)) => Some(L::I64(!v)),
        (U::BitwiseNot, L::AbstractInt(v)) => Some(L::AbstractInt(!v)),
        _ => None,
    }
}

/// Per-scalar binary evaluator.  Scalar-type specific; NaN, overflow,
/// and divide-by-zero behaviours match the runtime's contract so
/// folding never changes observable output.
fn eval_binary(
    op: naga::BinaryOperator,
    lhs: naga::Literal,
    rhs: naga::Literal,
) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    use naga::Literal as L;

    match (op, lhs, rhs) {
        (B::Add, L::F32(a), L::F32(b)) if (a + b).is_finite() => Some(L::F32(a + b)),
        (B::Subtract, L::F32(a), L::F32(b)) if (a - b).is_finite() => Some(L::F32(a - b)),
        (B::Multiply, L::F32(a), L::F32(b)) if (a * b).is_finite() => Some(L::F32(a * b)),
        (B::Divide, L::F32(a), L::F32(b)) if b != 0.0 && (a / b).is_finite() => Some(L::F32(a / b)),
        (B::Modulo, L::F32(a), L::F32(b)) if b != 0.0 && (a % b).is_finite() => Some(L::F32(a % b)),

        (B::Add, L::F64(a), L::F64(b)) if (a + b).is_finite() => Some(L::F64(a + b)),
        (B::Subtract, L::F64(a), L::F64(b)) if (a - b).is_finite() => Some(L::F64(a - b)),
        (B::Multiply, L::F64(a), L::F64(b)) if (a * b).is_finite() => Some(L::F64(a * b)),
        (B::Divide, L::F64(a), L::F64(b)) if b != 0.0 && (a / b).is_finite() => Some(L::F64(a / b)),
        (B::Modulo, L::F64(a), L::F64(b)) if b != 0.0 && (a % b).is_finite() => Some(L::F64(a % b)),

        // Signed integer arithmetic uses checked_* so that overflow (which is a
        // shader-creation/execution error in WGSL) is declined rather than
        // silently wrapped.  Folding `i32::MAX + 1` to `-2147483648` would
        // turn an error-producing program into a defined-value program.
        (B::Add, L::I32(a), L::I32(b)) => a.checked_add(b).map(L::I32),
        (B::Subtract, L::I32(a), L::I32(b)) => a.checked_sub(b).map(L::I32),
        (B::Multiply, L::I32(a), L::I32(b)) => a.checked_mul(b).map(L::I32),
        (B::Divide, L::I32(a), L::I32(b)) if b != 0 => a.checked_div(b).map(L::I32),
        (B::Modulo, L::I32(a), L::I32(b)) if b != 0 => a.checked_rem(b).map(L::I32),

        (B::Add, L::I64(a), L::I64(b)) => a.checked_add(b).map(L::I64),
        (B::Subtract, L::I64(a), L::I64(b)) => a.checked_sub(b).map(L::I64),
        (B::Multiply, L::I64(a), L::I64(b)) => a.checked_mul(b).map(L::I64),
        (B::Divide, L::I64(a), L::I64(b)) if b != 0 => a.checked_div(b).map(L::I64),
        (B::Modulo, L::I64(a), L::I64(b)) if b != 0 => a.checked_rem(b).map(L::I64),

        (B::Add, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_add(b))),
        (B::Subtract, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_sub(b))),
        (B::Multiply, L::U32(a), L::U32(b)) => Some(L::U32(a.wrapping_mul(b))),
        (B::Divide, L::U32(a), L::U32(b)) if b != 0 => Some(L::U32(a / b)),
        (B::Modulo, L::U32(a), L::U32(b)) if b != 0 => Some(L::U32(a % b)),

        (B::Add, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_add(b))),
        (B::Subtract, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_sub(b))),
        (B::Multiply, L::U64(a), L::U64(b)) => Some(L::U64(a.wrapping_mul(b))),
        (B::Divide, L::U64(a), L::U64(b)) if b != 0 => Some(L::U64(a / b)),
        (B::Modulo, L::U64(a), L::U64(b)) if b != 0 => Some(L::U64(a % b)),

        (B::Add, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_add(b).map(L::AbstractInt),
        (B::Subtract, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_sub(b).map(L::AbstractInt),
        (B::Multiply, L::AbstractInt(a), L::AbstractInt(b)) => a.checked_mul(b).map(L::AbstractInt),
        (B::Divide, L::AbstractInt(a), L::AbstractInt(b)) if b != 0 => {
            a.checked_div(b).map(L::AbstractInt)
        }
        (B::Modulo, L::AbstractInt(a), L::AbstractInt(b)) if b != 0 => {
            a.checked_rem(b).map(L::AbstractInt)
        }

        (B::Add, L::AbstractFloat(a), L::AbstractFloat(b)) if (a + b).is_finite() => {
            Some(L::AbstractFloat(a + b))
        }
        (B::Subtract, L::AbstractFloat(a), L::AbstractFloat(b)) if (a - b).is_finite() => {
            Some(L::AbstractFloat(a - b))
        }
        (B::Multiply, L::AbstractFloat(a), L::AbstractFloat(b)) if (a * b).is_finite() => {
            Some(L::AbstractFloat(a * b))
        }
        (B::Divide, L::AbstractFloat(a), L::AbstractFloat(b))
            if b != 0.0 && (a / b).is_finite() =>
        {
            Some(L::AbstractFloat(a / b))
        }
        (B::Modulo, L::AbstractFloat(a), L::AbstractFloat(b))
            if b != 0.0 && (a % b).is_finite() =>
        {
            Some(L::AbstractFloat(a % b))
        }

        (B::Equal, a, b) => Some(L::Bool(a == b)),
        (B::NotEqual, a, b) => Some(L::Bool(a != b)),

        (B::Less, L::F32(a), L::F32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::F32(a), L::F32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::F32(a), L::F32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::F32(a), L::F32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::F64(a), L::F64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::F64(a), L::F64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::F64(a), L::F64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::F64(a), L::F64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::I32(a), L::I32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::I32(a), L::I32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::I32(a), L::I32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::I32(a), L::I32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::I64(a), L::I64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::I64(a), L::I64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::I64(a), L::I64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::I64(a), L::I64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::U32(a), L::U32(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::U32(a), L::U32(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::U32(a), L::U32(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::U32(a), L::U32(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::U64(a), L::U64(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::U64(a), L::U64(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::U64(a), L::U64(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::U64(a), L::U64(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::Bool(a >= b)),

        (B::Less, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a < b)),
        (B::LessEqual, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a <= b)),
        (B::Greater, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a > b)),
        (B::GreaterEqual, L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::Bool(a >= b)),

        (B::LogicalAnd, L::Bool(a), L::Bool(b)) => Some(L::Bool(a && b)),
        (B::LogicalOr, L::Bool(a), L::Bool(b)) => Some(L::Bool(a || b)),

        (B::And, L::U32(a), L::U32(b)) => Some(L::U32(a & b)),
        (B::ExclusiveOr, L::U32(a), L::U32(b)) => Some(L::U32(a ^ b)),
        (B::InclusiveOr, L::U32(a), L::U32(b)) => Some(L::U32(a | b)),
        (B::ShiftLeft, L::U32(a), L::U32(b)) if b < 32 => Some(L::U32(a.wrapping_shl(b))),
        (B::ShiftRight, L::U32(a), L::U32(b)) if b < 32 => Some(L::U32(a.wrapping_shr(b))),

        (B::And, L::U64(a), L::U64(b)) => Some(L::U64(a & b)),
        (B::ExclusiveOr, L::U64(a), L::U64(b)) => Some(L::U64(a ^ b)),
        (B::InclusiveOr, L::U64(a), L::U64(b)) => Some(L::U64(a | b)),
        (B::ShiftLeft, L::U64(a), L::U64(b)) if b < 64 => Some(L::U64(a.wrapping_shl(b as u32))),
        (B::ShiftRight, L::U64(a), L::U64(b)) if b < 64 => Some(L::U64(a.wrapping_shr(b as u32))),

        (B::And, L::I32(a), L::I32(b)) => Some(L::I32(a & b)),
        (B::ExclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a ^ b)),
        (B::InclusiveOr, L::I32(a), L::I32(b)) => Some(L::I32(a | b)),
        (B::ShiftLeft, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shl(b))),
        (B::ShiftRight, L::I32(a), L::U32(b)) if b < 32 => Some(L::I32(a.wrapping_shr(b))),

        (B::And, L::I64(a), L::I64(b)) => Some(L::I64(a & b)),
        (B::ExclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a ^ b)),
        (B::InclusiveOr, L::I64(a), L::I64(b)) => Some(L::I64(a | b)),
        (B::ShiftLeft, L::I64(a), L::U64(b)) if b < 64 => Some(L::I64(a.wrapping_shl(b as u32))),
        (B::ShiftRight, L::I64(a), L::U64(b)) if b < 64 => Some(L::I64(a.wrapping_shr(b as u32))),

        (B::And, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a & b)),
        (B::ExclusiveOr, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a ^ b)),
        (B::InclusiveOr, L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a | b)),
        (B::ShiftLeft, L::AbstractInt(a), L::AbstractInt(b)) if (0..64).contains(&b) => {
            Some(L::AbstractInt(a.wrapping_shl(b as u32)))
        }
        (B::ShiftRight, L::AbstractInt(a), L::AbstractInt(b)) if (0..64).contains(&b) => {
            Some(L::AbstractInt(a.wrapping_shr(b as u32)))
        }

        _ => None,
    }
}

/// Evaluate a scalar math built-in function on constant literal arguments.
///
/// Covers Tier 1 (comparison, decomposition, computational), Tier 2
/// (trigonometric, exponential), and Tier 3 (integer bit) functions.
/// Returns `None` for unsupported functions, type mismatches, or domain errors.
/// Fold a single-argument WGSL math built-in over a scalar literal.
/// Only IEEE-stable functions (no NaN-sensitive branches, no
/// environment lookups) are covered; anything else returns `None` so
/// the call survives to runtime.
fn eval_math_scalar(
    fun: naga::MathFunction,
    arg: naga::Literal,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::MathFunction as M;

    /// Finite-check wrapper: returns `Some(v)` only if `v` is finite.
    fn finite_f32(v: f32) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::F32(v))
    }
    fn finite_f64(v: f64) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::F64(v))
    }
    fn finite_af(v: f64) -> Option<naga::Literal> {
        v.is_finite().then_some(naga::Literal::AbstractFloat(v))
    }

    match fun {
        // Tier 1: comparison
        M::Abs => match arg {
            L::F32(v) => Some(L::F32(v.abs())),
            L::F64(v) => Some(L::F64(v.abs())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.abs())),
            L::I32(v) => v.checked_abs().map(L::I32),
            L::I64(v) => v.checked_abs().map(L::I64),
            L::AbstractInt(v) => v.checked_abs().map(L::AbstractInt),
            L::U32(v) => Some(L::U32(v)),
            L::U64(v) => Some(L::U64(v)),
            _ => None,
        },
        M::Min => match (arg, arg1?) {
            (L::F32(a), L::F32(b)) => Some(L::F32(a.min(b))),
            (L::F64(a), L::F64(b)) => Some(L::F64(a.min(b))),
            (L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::AbstractFloat(a.min(b))),
            (L::I32(a), L::I32(b)) => Some(L::I32(a.min(b))),
            (L::I64(a), L::I64(b)) => Some(L::I64(a.min(b))),
            (L::U32(a), L::U32(b)) => Some(L::U32(a.min(b))),
            (L::U64(a), L::U64(b)) => Some(L::U64(a.min(b))),
            (L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a.min(b))),
            _ => None,
        },
        M::Max => match (arg, arg1?) {
            (L::F32(a), L::F32(b)) => Some(L::F32(a.max(b))),
            (L::F64(a), L::F64(b)) => Some(L::F64(a.max(b))),
            (L::AbstractFloat(a), L::AbstractFloat(b)) => Some(L::AbstractFloat(a.max(b))),
            (L::I32(a), L::I32(b)) => Some(L::I32(a.max(b))),
            (L::I64(a), L::I64(b)) => Some(L::I64(a.max(b))),
            (L::U32(a), L::U32(b)) => Some(L::U32(a.max(b))),
            (L::U64(a), L::U64(b)) => Some(L::U64(a.max(b))),
            (L::AbstractInt(a), L::AbstractInt(b)) => Some(L::AbstractInt(a.max(b))),
            _ => None,
        },
        M::Clamp => {
            let lo = arg1?;
            let hi = arg2?;
            match (arg, lo, hi) {
                (L::F32(v), L::F32(lo), L::F32(hi)) if lo <= hi => Some(L::F32(v.clamp(lo, hi))),
                (L::F64(v), L::F64(lo), L::F64(hi)) if lo <= hi => Some(L::F64(v.clamp(lo, hi))),
                (L::AbstractFloat(v), L::AbstractFloat(lo), L::AbstractFloat(hi)) if lo <= hi => {
                    Some(L::AbstractFloat(v.clamp(lo, hi)))
                }
                (L::I32(v), L::I32(lo), L::I32(hi)) if lo <= hi => Some(L::I32(v.clamp(lo, hi))),
                (L::I64(v), L::I64(lo), L::I64(hi)) if lo <= hi => Some(L::I64(v.clamp(lo, hi))),
                (L::U32(v), L::U32(lo), L::U32(hi)) if lo <= hi => Some(L::U32(v.clamp(lo, hi))),
                (L::U64(v), L::U64(lo), L::U64(hi)) if lo <= hi => Some(L::U64(v.clamp(lo, hi))),
                (L::AbstractInt(v), L::AbstractInt(lo), L::AbstractInt(hi)) if lo <= hi => {
                    Some(L::AbstractInt(v.clamp(lo, hi)))
                }
                _ => None,
            }
        }
        M::Saturate => match arg {
            L::F32(v) => Some(L::F32(v.clamp(0.0, 1.0))),
            L::F64(v) => Some(L::F64(v.clamp(0.0, 1.0))),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.clamp(0.0, 1.0))),
            _ => None,
        },

        // Tier 1: computational
        M::Sign => match arg {
            L::F32(v) => Some(L::F32(if v == 0.0 { 0.0 } else { v.signum() })),
            L::F64(v) => Some(L::F64(if v == 0.0 { 0.0 } else { v.signum() })),
            L::AbstractFloat(v) => Some(L::AbstractFloat(if v == 0.0 { 0.0 } else { v.signum() })),
            L::I32(v) => Some(L::I32(v.signum())),
            L::I64(v) => Some(L::I64(v.signum())),
            L::AbstractInt(v) => Some(L::AbstractInt(v.signum())),
            _ => None,
        },

        // Tier 1: decomposition
        M::Floor => match arg {
            L::F32(v) => Some(L::F32(v.floor())),
            L::F64(v) => Some(L::F64(v.floor())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.floor())),
            _ => None,
        },
        M::Ceil => match arg {
            L::F32(v) => Some(L::F32(v.ceil())),
            L::F64(v) => Some(L::F64(v.ceil())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.ceil())),
            _ => None,
        },
        M::Round => match arg {
            // WGSL specifies ties-to-even; Rust 1.77+ has round_ties_even.
            L::F32(v) => Some(L::F32(v.round_ties_even())),
            L::F64(v) => Some(L::F64(v.round_ties_even())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.round_ties_even())),
            _ => None,
        },
        M::Trunc => match arg {
            L::F32(v) => Some(L::F32(v.trunc())),
            L::F64(v) => Some(L::F64(v.trunc())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v.trunc())),
            _ => None,
        },
        M::Fract => match arg {
            // WGSL fract(e) = e - floor(e), NOT Rust's fract().
            L::F32(v) => Some(L::F32(v - v.floor())),
            L::F64(v) => Some(L::F64(v - v.floor())),
            L::AbstractFloat(v) => Some(L::AbstractFloat(v - v.floor())),
            _ => None,
        },

        // Tier 1: computational (continued)
        M::Step => match (arg, arg1?) {
            (L::F32(edge), L::F32(x)) => Some(L::F32(if edge <= x { 1.0 } else { 0.0 })),
            (L::F64(edge), L::F64(x)) => Some(L::F64(if edge <= x { 1.0 } else { 0.0 })),
            (L::AbstractFloat(edge), L::AbstractFloat(x)) => {
                Some(L::AbstractFloat(if edge <= x { 1.0 } else { 0.0 }))
            }
            _ => None,
        },
        M::Sqrt => match arg {
            L::F32(v) if v >= 0.0 => finite_f32(v.sqrt()),
            L::F64(v) if v >= 0.0 => finite_f64(v.sqrt()),
            L::AbstractFloat(v) if v >= 0.0 => finite_af(v.sqrt()),
            _ => None,
        },
        M::InverseSqrt => match arg {
            L::F32(v) if v > 0.0 => finite_f32(1.0 / v.sqrt()),
            L::F64(v) if v > 0.0 => finite_f64(1.0 / v.sqrt()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(1.0 / v.sqrt()),
            _ => None,
        },
        M::Fma => {
            let b = arg1?;
            let c = arg2?;
            match (arg, b, c) {
                (L::F32(a), L::F32(b), L::F32(c)) => finite_f32(a.mul_add(b, c)),
                (L::F64(a), L::F64(b), L::F64(c)) => finite_f64(a.mul_add(b, c)),
                (L::AbstractFloat(a), L::AbstractFloat(b), L::AbstractFloat(c)) => {
                    finite_af(a.mul_add(b, c))
                }
                _ => None,
            }
        }

        // Tier 2: trigonometric
        M::Cos => match arg {
            L::F32(v) => finite_f32(v.cos()),
            L::F64(v) => finite_f64(v.cos()),
            L::AbstractFloat(v) => finite_af(v.cos()),
            _ => None,
        },
        M::Sin => match arg {
            L::F32(v) => finite_f32(v.sin()),
            L::F64(v) => finite_f64(v.sin()),
            L::AbstractFloat(v) => finite_af(v.sin()),
            _ => None,
        },
        M::Tan => match arg {
            L::F32(v) => finite_f32(v.tan()),
            L::F64(v) => finite_f64(v.tan()),
            L::AbstractFloat(v) => finite_af(v.tan()),
            _ => None,
        },
        M::Cosh => match arg {
            L::F32(v) => finite_f32(v.cosh()),
            L::F64(v) => finite_f64(v.cosh()),
            L::AbstractFloat(v) => finite_af(v.cosh()),
            _ => None,
        },
        M::Sinh => match arg {
            L::F32(v) => finite_f32(v.sinh()),
            L::F64(v) => finite_f64(v.sinh()),
            L::AbstractFloat(v) => finite_af(v.sinh()),
            _ => None,
        },
        M::Tanh => match arg {
            L::F32(v) => finite_f32(v.tanh()),
            L::F64(v) => finite_f64(v.tanh()),
            L::AbstractFloat(v) => finite_af(v.tanh()),
            _ => None,
        },
        M::Acos => match arg {
            L::F32(v) if v.abs() <= 1.0 => finite_f32(v.acos()),
            L::F64(v) if v.abs() <= 1.0 => finite_f64(v.acos()),
            L::AbstractFloat(v) if v.abs() <= 1.0 => finite_af(v.acos()),
            _ => None,
        },
        M::Asin => match arg {
            L::F32(v) if v.abs() <= 1.0 => finite_f32(v.asin()),
            L::F64(v) if v.abs() <= 1.0 => finite_f64(v.asin()),
            L::AbstractFloat(v) if v.abs() <= 1.0 => finite_af(v.asin()),
            _ => None,
        },
        M::Atan => match arg {
            L::F32(v) => finite_f32(v.atan()),
            L::F64(v) => finite_f64(v.atan()),
            L::AbstractFloat(v) => finite_af(v.atan()),
            _ => None,
        },
        M::Atan2 => match (arg, arg1?) {
            (L::F32(y), L::F32(x)) => finite_f32(y.atan2(x)),
            (L::F64(y), L::F64(x)) => finite_f64(y.atan2(x)),
            (L::AbstractFloat(y), L::AbstractFloat(x)) => finite_af(y.atan2(x)),
            _ => None,
        },
        M::Asinh => match arg {
            L::F32(v) => finite_f32(v.asinh()),
            L::F64(v) => finite_f64(v.asinh()),
            L::AbstractFloat(v) => finite_af(v.asinh()),
            _ => None,
        },
        M::Acosh => match arg {
            L::F32(v) if v >= 1.0 => finite_f32(v.acosh()),
            L::F64(v) if v >= 1.0 => finite_f64(v.acosh()),
            L::AbstractFloat(v) if v >= 1.0 => finite_af(v.acosh()),
            _ => None,
        },
        M::Atanh => match arg {
            L::F32(v) if v.abs() < 1.0 => finite_f32(v.atanh()),
            L::F64(v) if v.abs() < 1.0 => finite_f64(v.atanh()),
            L::AbstractFloat(v) if v.abs() < 1.0 => finite_af(v.atanh()),
            _ => None,
        },
        M::Radians => match arg {
            L::F32(v) => finite_f32(v.to_radians()),
            L::F64(v) => finite_f64(v.to_radians()),
            L::AbstractFloat(v) => finite_af(v.to_radians()),
            _ => None,
        },
        M::Degrees => match arg {
            L::F32(v) => finite_f32(v.to_degrees()),
            L::F64(v) => finite_f64(v.to_degrees()),
            L::AbstractFloat(v) => finite_af(v.to_degrees()),
            _ => None,
        },

        // Tier 2: exponential
        M::Exp => match arg {
            L::F32(v) => finite_f32(v.exp()),
            L::F64(v) => finite_f64(v.exp()),
            L::AbstractFloat(v) => finite_af(v.exp()),
            _ => None,
        },
        M::Exp2 => match arg {
            L::F32(v) => finite_f32(v.exp2()),
            L::F64(v) => finite_f64(v.exp2()),
            L::AbstractFloat(v) => finite_af(v.exp2()),
            _ => None,
        },
        M::Log => match arg {
            L::F32(v) if v > 0.0 => finite_f32(v.ln()),
            L::F64(v) if v > 0.0 => finite_f64(v.ln()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(v.ln()),
            _ => None,
        },
        M::Log2 => match arg {
            L::F32(v) if v > 0.0 => finite_f32(v.log2()),
            L::F64(v) if v > 0.0 => finite_f64(v.log2()),
            L::AbstractFloat(v) if v > 0.0 => finite_af(v.log2()),
            _ => None,
        },
        M::Pow => match (arg, arg1?) {
            // WGSL precondition: e1 >= 0.0.
            (L::F32(a), L::F32(b)) if a >= 0.0 => finite_f32(a.powf(b)),
            (L::F64(a), L::F64(b)) if a >= 0.0 => finite_f64(a.powf(b)),
            (L::AbstractFloat(a), L::AbstractFloat(b)) if a >= 0.0 => finite_af(a.powf(b)),
            _ => None,
        },

        // Tier 3: integer bit operations
        M::CountTrailingZeros => match arg {
            L::U32(v) => Some(L::U32(v.trailing_zeros())),
            L::I32(v) => Some(L::I32(v.trailing_zeros() as i32)),
            L::U64(v) => Some(L::U64(v.trailing_zeros() as u64)),
            L::I64(v) => Some(L::I64(v.trailing_zeros() as i64)),
            _ => None,
        },
        M::CountLeadingZeros => match arg {
            L::U32(v) => Some(L::U32(v.leading_zeros())),
            L::I32(v) => Some(L::I32(v.leading_zeros() as i32)),
            L::U64(v) => Some(L::U64(v.leading_zeros() as u64)),
            L::I64(v) => Some(L::I64(v.leading_zeros() as i64)),
            _ => None,
        },
        M::CountOneBits => match arg {
            L::U32(v) => Some(L::U32(v.count_ones())),
            L::I32(v) => Some(L::I32(v.count_ones() as i32)),
            L::U64(v) => Some(L::U64(v.count_ones() as u64)),
            L::I64(v) => Some(L::I64(v.count_ones() as i64)),
            _ => None,
        },
        M::ReverseBits => match arg {
            L::U32(v) => Some(L::U32(v.reverse_bits())),
            L::I32(v) => Some(L::I32(v.reverse_bits())),
            L::U64(v) => Some(L::U64(v.reverse_bits())),
            L::I64(v) => Some(L::I64(v.reverse_bits())),
            _ => None,
        },
        M::FirstTrailingBit => match arg {
            L::U32(v) => Some(L::U32(if v == 0 { u32::MAX } else { v.trailing_zeros() })),
            L::I32(v) => Some(L::I32(if v == 0 {
                -1
            } else {
                v.trailing_zeros() as i32
            })),
            L::U64(v) => Some(L::U64(if v == 0 {
                u64::MAX
            } else {
                v.trailing_zeros() as u64
            })),
            L::I64(v) => Some(L::I64(if v == 0 {
                -1
            } else {
                v.trailing_zeros() as i64
            })),
            _ => None,
        },
        M::FirstLeadingBit => match arg {
            L::U32(v) => Some(L::U32(if v == 0 {
                u32::MAX
            } else {
                31 - v.leading_zeros()
            })),
            L::I32(v) => Some(L::I32(if v == 0 || v == -1 {
                -1
            } else if v > 0 {
                31 - (v.leading_zeros() as i32)
            } else {
                // For negative numbers, find the most significant bit that
                // differs from the sign bit.
                31 - (v.leading_ones() as i32)
            })),
            L::U64(v) => Some(L::U64(if v == 0 {
                u64::MAX
            } else {
                63 - v.leading_zeros() as u64
            })),
            L::I64(v) => Some(L::I64(if v == 0 || v == -1 {
                -1
            } else if v > 0 {
                63 - (v.leading_zeros() as i64)
            } else {
                63 - (v.leading_ones() as i64)
            })),
            _ => None,
        },

        // Tier 4+ / unsupported: graceful fallthrough
        _ => None,
    }
}

/// Evaluate a math built-in on `ConstValue`s (scalar or component-wise vector).
///
/// Mirrors the pattern of [`eval_const_unary`] and [`eval_const_binary`]:
/// scalars delegate to [`eval_math_scalar`], vectors apply it per-component.
/// Fold a math built-in over a [`ConstValue`], broadcasting scalar
/// implementations across vector arguments.  Returns `None` when any
/// argument is non-constant or the math function has no per-scalar
/// implementation registered.
fn eval_const_math(
    fun: naga::MathFunction,
    arg: ConstValue,
    arg1: Option<ConstValue>,
    arg2: Option<ConstValue>,
) -> Option<ConstValue> {
    // Helper: extract the scalar from a ConstValue, or None for vectors.
    fn as_scalar(v: &ConstValue) -> Option<naga::Literal> {
        match v {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }

    // Helper: extract vector components + metadata, or None for scalars.
    fn as_vector(v: &ConstValue) -> Option<(&[naga::Literal], naga::VectorSize, naga::Scalar)> {
        match v {
            ConstValue::Vector {
                components,
                size,
                scalar,
            } => Some((components, *size, *scalar)),
            _ => None,
        }
    }

    // All-scalar case
    if let Some(a) = as_scalar(&arg) {
        let b = match &arg1 {
            Some(v) => Some(as_scalar(v)?),
            None => None,
        };
        let c = match &arg2 {
            Some(v) => Some(as_scalar(v)?),
            None => None,
        };
        return eval_math_scalar(fun, a, b, c).map(ConstValue::Scalar);
    }

    // Vector case: apply per-component
    if let Some((comps, size, scalar)) = as_vector(&arg) {
        let n = comps.len();

        // Resolve optional arg1 components (must be same-size vector or absent).
        let arg1_comps: Option<Vec<naga::Literal>> = match &arg1 {
            Some(v) => {
                let (c1, s1, _) = as_vector(v)?;
                if s1 != size {
                    return None;
                }
                Some(c1.to_vec())
            }
            None => None,
        };
        // Resolve optional arg2 components.
        let arg2_comps: Option<Vec<naga::Literal>> = match &arg2 {
            Some(v) => {
                let (c2, s2, _) = as_vector(v)?;
                if s2 != size {
                    return None;
                }
                Some(c2.to_vec())
            }
            None => None,
        };

        let folded: Option<Vec<naga::Literal>> = (0..n)
            .map(|i| {
                let a = comps[i];
                let b = arg1_comps.as_ref().map(|c| c[i]);
                let c = arg2_comps.as_ref().map(|c| c[i]);
                eval_math_scalar(fun, a, b, c)
            })
            .collect();

        return Some(ConstValue::Vector {
            components: folded?,
            size,
            scalar,
        });
    }

    None
}

// MARK: Identity / absorbing operand detection

/// `true` when `expr` requires an `Emit` statement in naga's IR.
/// Non-emittable expressions are implicitly available and must NOT
/// appear inside `Emit` ranges.  The list must stay in sync with
/// naga's validation rules; a missed variant leaves the handle inside
/// its `Emit` range after identity elimination, producing invalid IR
/// that the pipeline rolls back.
///
/// This is a local mirror of
/// [`crate::passes::expr_util::expression_needs_emit`] kept inline to
/// avoid a cross-module call in the hot identity-folding loop.
fn needs_emit(expr: &naga::Expression) -> bool {
    !matches!(
        expr,
        naga::Expression::Literal(_)
            | naga::Expression::Constant(_)
            | naga::Expression::Override(_)
            | naga::Expression::ZeroValue(_)
            | naga::Expression::FunctionArgument(_)
            | naga::Expression::GlobalVariable(_)
            | naga::Expression::LocalVariable(_)
            | naga::Expression::CallResult(_)
            | naga::Expression::AtomicResult { .. }
            | naga::Expression::WorkGroupUniformLoadResult { .. }
            | naga::Expression::RayQueryProceedResult
            | naga::Expression::SubgroupBallotResult
            | naga::Expression::SubgroupOperationResult { .. }
    )
}

fn is_zero(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F32(v)) if v == 0.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F64(v)) if v == 0.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::I32(0)
                | naga::Literal::U32(0)
                | naga::Literal::I64(0)
                | naga::Literal::U64(0)
                | naga::Literal::AbstractInt(0)
        )
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::AbstractFloat(v)) if v == 0.0
    )
}

fn is_one(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F32(v)) if v == 1.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::F64(v)) if v == 1.0
    ) || matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::I32(1)
                | naga::Literal::U32(1)
                | naga::Literal::I64(1)
                | naga::Literal::U64(1)
                | naga::Literal::AbstractInt(1)
        )
    ) || matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::AbstractFloat(v)) if v == 1.0
    )
}

fn is_all_ones(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(
            naga::Literal::U32(u32::MAX)
                | naga::Literal::I32(-1)
                | naga::Literal::U64(u64::MAX)
                | naga::Literal::I64(-1)
                | naga::Literal::AbstractInt(-1)
        )
    )
}

fn is_bool_true(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(true))
    )
}

fn is_bool_false(arena: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> bool {
    matches!(
        arena[h],
        naga::Expression::Literal(naga::Literal::Bool(false))
    )
}

/// Check if a binary expression has an identity operand that can be eliminated.
/// Returns `Some(other_operand)` when the expression can be replaced by just
/// the non-identity side.
///
/// Identities:  `0 + x = x`, `x + 0 = x`, `x - 0 = x`,
///              `1 * x = x`, `x * 1 = x`, `x / 1 = x`,
///              `x | 0 = x`, `0 | x = x`, `x ^ 0 = x`, `0 ^ x = x`,
///              `x & all_ones = x`, `all_ones & x = x`,
///              `x && true = x`, `true && x = x`,
///              `x || false = x`, `false || x = x`.
/// Detect `x <op> identity` patterns (e.g. `x + 0`, `x * 1`) and
/// return the surviving operand handle.  Only fires for operators
/// whose identity element is well-defined across the scalar type set
/// the emitter supports.
fn check_identity_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Add => {
            if is_zero(arena, left) {
                Some(right)
            } else if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Subtract => {
            if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Multiply => {
            if is_one(arena, left) {
                Some(right)
            } else if is_one(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::Divide => {
            if is_one(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::InclusiveOr => {
            if is_zero(arena, left) {
                Some(right)
            } else if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::ExclusiveOr => {
            if is_zero(arena, left) {
                Some(right)
            } else if is_zero(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::And => {
            if is_all_ones(arena, left) {
                Some(right)
            } else if is_all_ones(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::LogicalAnd => {
            if is_bool_true(arena, left) {
                Some(right)
            } else if is_bool_true(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        B::LogicalOr => {
            if is_bool_false(arena, left) {
                Some(right)
            } else if is_bool_false(arena, right) {
                Some(left)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if a binary expression has an absorbing operand, collapsing the entire
/// expression to that operand's value.
///
/// Absorbing:  `x * 0 = 0`, `0 * x = 0`,
///             `x & 0 = 0`, `0 & x = 0`,
///             `x | all_ones = all_ones`, `all_ones | x = all_ones`,
///             `x && false = false`, `false && x = false`,
///             `x || true = true`, `true || x = true`.
/// Detect `x <op> absorbing` patterns (e.g. `x * 0`, `x && false`)
/// and return the synthesised absorbing result.  Mirrors
/// [`check_identity_operand`] but for annihilator elements.
fn check_absorbing_operand(
    op: naga::BinaryOperator,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::Expression>> {
    use naga::BinaryOperator as B;

    match op {
        B::Multiply => {
            if is_zero(arena, left) {
                Some(left)
            } else if is_zero(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::And => {
            if is_zero(arena, left) {
                Some(left)
            } else if is_zero(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::InclusiveOr => {
            if is_all_ones(arena, left) {
                Some(left)
            } else if is_all_ones(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::LogicalAnd => {
            if is_bool_false(arena, left) {
                Some(left)
            } else if is_bool_false(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        B::LogicalOr => {
            if is_bool_true(arena, left) {
                Some(left)
            } else if is_bool_true(arena, right) {
                Some(right)
            } else {
                None
            }
        }
        _ => None,
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn run_pass(module: &mut naga::Module) -> bool {
        let mut pass = ConstFoldPass;
        let config = Config::default();
        let ctx = PassContext {
            config: &config,
            trace_run_dir: None,
        };

        pass.run(module, &ctx).expect("const fold pass should run")
    }

    fn assert_f32_literal(
        arena: &naga::Arena<naga::Expression>,
        handle: naga::Handle<naga::Expression>,
        expected: f32,
    ) {
        match arena[handle] {
            naga::Expression::Literal(naga::Literal::F32(v)) => {
                assert!((v - expected).abs() < f32::EPSILON)
            }
            ref other => panic!("expected f32 literal {expected}, got {other:?}"),
        }
    }

    #[test]
    fn folds_binary_add_in_local_expression_arena() {
        let mut arena = naga::Arena::new();
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let two = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: one,
                right: two,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(
            changed.len(),
            1,
            "one binary add expression should be folded"
        );
        assert_f32_literal(&arena, add, 3.0);
    }

    #[test]
    fn folds_nested_binary_expressions() {
        let mut arena = naga::Arena::new();
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let two = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let three = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: one,
                right: two,
            },
            Default::default(),
        );
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: add,
                right: three,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(changed.len(), 2, "both nested operations should be folded");
        assert_f32_literal(&arena, add, 3.0);
        assert_f32_literal(&arena, mul, 9.0);
    }

    #[test]
    fn folds_select_expression_with_literal_condition() {
        let mut arena = naga::Arena::new();
        let cond = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let accept = arena.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        let reject = arena.append(
            naga::Expression::Literal(naga::Literal::F32(8.0)),
            Default::default(),
        );
        let select = arena.append(
            naga::Expression::Select {
                condition: cond,
                accept,
                reject,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(
            changed.len(),
            1,
            "select expression should fold to its accepted literal"
        );
        assert_f32_literal(&arena, select, 4.0);
    }

    #[test]
    fn folds_local_constant_reference_using_cache() {
        let module =
            naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
        let (constant_handle, _) = module
            .constants
            .iter()
            .next()
            .expect("expected one constant in parsed module");

        let mut arena = naga::Arena::new();
        let c = arena.append(
            naga::Expression::Constant(constant_handle),
            Default::default(),
        );
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: c,
                right: one,
            },
            Default::default(),
        );

        let mut const_literals = HashMap::new();
        const_literals.insert(constant_handle, naga::Literal::F32(41.0));

        let (changed, _) =
            fold_local_expressions(&mut arena, &const_literals, &naga::UniqueArena::new());
        assert_eq!(
            changed.len(),
            2,
            "constant reference and dependent add should fold"
        );
        assert_f32_literal(&arena, c, 41.0);
        assert_f32_literal(&arena, add, 42.0);
    }

    #[test]
    fn does_not_fold_divide_by_zero() {
        let mut arena = naga::Arena::new();
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let div = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Divide,
                left: one,
                right: zero,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(changed.len(), 0, "division by zero should not be folded");

        match arena[div] {
            naga::Expression::Binary {
                op: naga::BinaryOperator::Divide,
                left,
                right,
            } => {
                assert_eq!(left, one);
                assert_eq!(right, zero);
            }
            ref other => panic!("expected divide expression to remain, got {other:?}"),
        }
    }

    #[test]
    fn unary_negate_rejects_non_finite_result() {
        let result = eval_unary(
            naga::UnaryOperator::Negate,
            naga::Literal::F32(f32::INFINITY),
        );
        assert_eq!(result, None, "non-finite result should not be folded");
    }

    #[test]
    fn abstract_int_divide_min_by_neg1_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::Divide,
            naga::Literal::AbstractInt(i64::MIN),
            naga::Literal::AbstractInt(-1),
        );
        assert_eq!(
            result, None,
            "i64::MIN / -1 overflows AbstractInt and should not be folded"
        );
    }

    #[test]
    fn abstract_int_modulo_min_by_neg1_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::Modulo,
            naga::Literal::AbstractInt(i64::MIN),
            naga::Literal::AbstractInt(-1),
        );
        assert_eq!(
            result, None,
            "i64::MIN %% -1 overflows AbstractInt and should not be folded"
        );
    }

    #[test]
    fn abstract_int_add_overflow_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::Add,
            naga::Literal::AbstractInt(i64::MAX),
            naga::Literal::AbstractInt(1),
        );
        assert_eq!(
            result, None,
            "i64::MAX + 1 overflows and should not be folded"
        );
    }

    #[test]
    fn abstract_int_mul_overflow_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::Multiply,
            naga::Literal::AbstractInt(i64::MAX),
            naga::Literal::AbstractInt(2),
        );
        assert_eq!(
            result, None,
            "i64::MAX * 2 overflows and should not be folded"
        );
    }

    #[test]
    fn i32_add_overflow_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Add,
            naga::Literal::I32(i32::MAX),
            naga::Literal::I32(1),
        );
        assert_eq!(r, None, "i32::MAX + 1 overflows and should not be folded");
    }

    #[test]
    fn i32_sub_overflow_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Subtract,
            naga::Literal::I32(i32::MIN),
            naga::Literal::I32(1),
        );
        assert_eq!(r, None, "i32::MIN - 1 overflows and should not be folded");
    }

    #[test]
    fn i32_mul_overflow_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Multiply,
            naga::Literal::I32(i32::MAX),
            naga::Literal::I32(2),
        );
        assert_eq!(r, None, "i32::MAX * 2 overflows and should not be folded");
    }

    #[test]
    fn i32_divide_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Divide,
            naga::Literal::I32(i32::MIN),
            naga::Literal::I32(-1),
        );
        assert_eq!(r, None, "i32::MIN / -1 overflows and should not be folded");
    }

    #[test]
    fn i32_modulo_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Modulo,
            naga::Literal::I32(i32::MIN),
            naga::Literal::I32(-1),
        );
        assert_eq!(r, None, "i32::MIN %% -1 overflows and should not be folded");
    }

    #[test]
    fn i64_add_overflow_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Add,
            naga::Literal::I64(i64::MAX),
            naga::Literal::I64(1),
        );
        assert_eq!(r, None, "i64::MAX + 1 overflows and should not be folded");
    }

    #[test]
    fn i64_divide_min_by_neg1_not_folded() {
        let r = eval_binary(
            naga::BinaryOperator::Divide,
            naga::Literal::I64(i64::MIN),
            naga::Literal::I64(-1),
        );
        assert_eq!(r, None, "i64::MIN / -1 overflows and should not be folded");
    }

    #[test]
    fn abs_i32_min_not_folded() {
        // abs(i32::MIN) overflows (no positive representation); must not fold.
        let r = eval_math_scalar(
            naga::MathFunction::Abs,
            naga::Literal::I32(i32::MIN),
            None,
            None,
        );
        assert_eq!(r, None);
    }

    #[test]
    fn abs_i32_negative_folds() {
        let r = eval_math_scalar(naga::MathFunction::Abs, naga::Literal::I32(-5), None, None);
        assert_eq!(r, Some(naga::Literal::I32(5)));
    }

    #[test]
    fn i32_add_normal_folds() {
        let r = eval_binary(
            naga::BinaryOperator::Add,
            naga::Literal::I32(100),
            naga::Literal::I32(200),
        );
        assert_eq!(r, Some(naga::Literal::I32(300)));
    }

    #[test]
    fn abstract_int_add_normal_folds() {
        let result = eval_binary(
            naga::BinaryOperator::Add,
            naga::Literal::AbstractInt(100),
            naga::Literal::AbstractInt(200),
        );
        assert_eq!(result, Some(naga::Literal::AbstractInt(300)));
    }

    #[test]
    fn shift_left_u32_in_range_folds() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::U32(1),
            naga::Literal::U32(4),
        );
        assert_eq!(result, Some(naga::Literal::U32(16)));
    }

    #[test]
    fn shift_left_u32_out_of_range_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::U32(1),
            naga::Literal::U32(32),
        );
        assert_eq!(result, None, "shift >= bit_width should not be folded");
    }

    #[test]
    fn shift_right_i32_out_of_range_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftRight,
            naga::Literal::I32(1),
            naga::Literal::U32(32),
        );
        assert_eq!(result, None, "shift >= bit_width should not be folded");
    }

    #[test]
    fn shift_left_u64_out_of_range_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::U64(1),
            naga::Literal::U64(64),
        );
        assert_eq!(result, None, "shift >= bit_width should not be folded");
    }

    #[test]
    fn shift_left_i64_out_of_range_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::I64(1),
            naga::Literal::U64(64),
        );
        assert_eq!(result, None, "shift >= bit_width should not be folded");
    }

    #[test]
    fn shift_abstract_int_negative_amount_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::AbstractInt(1),
            naga::Literal::AbstractInt(-1),
        );
        assert_eq!(result, None, "negative shift amount should not be folded");
    }

    #[test]
    fn shift_abstract_int_out_of_range_not_folded() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::AbstractInt(1),
            naga::Literal::AbstractInt(64),
        );
        assert_eq!(
            result, None,
            "shift >= 64 should not be folded for AbstractInt"
        );
    }

    #[test]
    fn shift_abstract_int_in_range_folds() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::AbstractInt(1),
            naga::Literal::AbstractInt(10),
        );
        assert_eq!(result, Some(naga::Literal::AbstractInt(1024)));
    }

    #[test]
    fn run_folds_globals_functions_and_entry_points() {
        let source = r#"
const C: f32 = 0.0;

fn helper() -> f32 {
    return 0.0;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
"#;

        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");

        let (const_handle, const_init) = module
            .constants
            .iter()
            .next()
            .map(|(h, c)| (h, c.init))
            .expect("expected one constant in module");

        let g_one = module.global_expressions.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let g_two = module.global_expressions.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        module.global_expressions[const_init] = naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: g_one,
            right: g_two,
        };

        let helper_handle = module
            .functions
            .iter()
            .map(|(h, _)| h)
            .next()
            .expect("expected helper function");

        let helper_add = {
            let helper = &mut module.functions[helper_handle];
            let c = helper
                .expressions
                .append(naga::Expression::Constant(const_handle), Default::default());
            let four = helper.expressions.append(
                naga::Expression::Literal(naga::Literal::F32(4.0)),
                Default::default(),
            );
            helper.expressions.append(
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: c,
                    right: four,
                },
                Default::default(),
            )
        };

        let entry_add = {
            let entry = &mut module.entry_points[0];
            let c = entry
                .function
                .expressions
                .append(naga::Expression::Constant(const_handle), Default::default());
            let five = entry.function.expressions.append(
                naga::Expression::Literal(naga::Literal::F32(5.0)),
                Default::default(),
            );
            entry.function.expressions.append(
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: c,
                    right: five,
                },
                Default::default(),
            )
        };

        let changed = run_pass(&mut module);
        assert!(
            changed,
            "run should report changes when foldable expressions are present"
        );

        match module.global_expressions[const_init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => {
                assert!((v - 3.0).abs() < f32::EPSILON)
            }
            ref other => panic!("expected global constant initializer to fold, got {other:?}"),
        }

        let helper = &module.functions[helper_handle];
        assert_f32_literal(&helper.expressions, helper_add, 7.0);
        assert_f32_literal(&module.entry_points[0].function.expressions, entry_add, 8.0);
    }

    #[test]
    fn run_returns_false_when_nothing_is_foldable() {
        let source = r#"
fn helper(x: f32) -> f32 {
    return x;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    let y = helper(1.0);
    return vec4f(y, y, y, 1.0);
}
"#;

        let mut module = naga::front::wgsl::parse_str(source).expect("source should parse");
        let changed = run_pass(&mut module);
        assert!(
            !changed,
            "run should report no changes when expressions are already non-foldable"
        );
    }

    #[test]
    fn abstract_int_shift_left_folds() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftLeft,
            naga::Literal::AbstractInt(1),
            naga::Literal::AbstractInt(3),
        );
        assert_eq!(
            result,
            Some(naga::Literal::AbstractInt(8)),
            "1 << 3 should fold to 8"
        );
    }

    #[test]
    fn abstract_int_shift_right_folds() {
        let result = eval_binary(
            naga::BinaryOperator::ShiftRight,
            naga::Literal::AbstractInt(16),
            naga::Literal::AbstractInt(2),
        );
        assert_eq!(
            result,
            Some(naga::Literal::AbstractInt(4)),
            "16 >> 2 should fold to 4"
        );
    }

    struct IdentityArena {
        arena: naga::Arena<naga::Expression>,
        zero_f32: naga::Handle<naga::Expression>,
        one_f32: naga::Handle<naga::Expression>,
        zero_i32: naga::Handle<naga::Expression>,
        one_i32: naga::Handle<naga::Expression>,
        param: naga::Handle<naga::Expression>,
    }

    fn make_identity_arena() -> IdentityArena {
        let mut arena = naga::Arena::new();
        let zero_f32 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let one_f32 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let zero_i32 = arena.append(
            naga::Expression::Literal(naga::Literal::I32(0)),
            Default::default(),
        );
        let one_i32 = arena.append(
            naga::Expression::Literal(naga::Literal::I32(1)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        IdentityArena {
            arena,
            zero_f32,
            one_f32,
            zero_i32,
            one_i32,
            param,
        }
    }

    #[test]
    fn identity_add_zero_left() {
        let a = make_identity_arena();
        // 0 + param -> param
        let result =
            check_identity_operand(naga::BinaryOperator::Add, a.zero_f32, a.param, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_add_zero_right() {
        let a = make_identity_arena();
        // param + 0 -> param
        let result =
            check_identity_operand(naga::BinaryOperator::Add, a.param, a.zero_f32, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_sub_zero_right() {
        let a = make_identity_arena();
        // param - 0 -> param
        let result = check_identity_operand(
            naga::BinaryOperator::Subtract,
            a.param,
            a.zero_f32,
            &a.arena,
        );
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_sub_zero_left_not_eliminated() {
        let a = make_identity_arena();
        // 0 - param != param (it's negation)
        let result = check_identity_operand(
            naga::BinaryOperator::Subtract,
            a.zero_f32,
            a.param,
            &a.arena,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn identity_mul_one_left() {
        let a = make_identity_arena();
        // 1 * param -> param
        let result =
            check_identity_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_mul_one_right() {
        let a = make_identity_arena();
        // param * 1 -> param
        let result =
            check_identity_operand(naga::BinaryOperator::Multiply, a.param, a.one_f32, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_div_one_right() {
        let a = make_identity_arena();
        // param / 1 -> param
        let result =
            check_identity_operand(naga::BinaryOperator::Divide, a.param, a.one_f32, &a.arena);
        assert_eq!(result, Some(a.param));
    }

    #[test]
    fn identity_div_one_left_not_eliminated() {
        let a = make_identity_arena();
        // 1 / param != param
        let result =
            check_identity_operand(naga::BinaryOperator::Divide, a.one_f32, a.param, &a.arena);
        assert_eq!(result, None);
    }

    #[test]
    fn identity_integer_types() {
        let a = make_identity_arena();
        // 0i + param -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::Add, a.zero_i32, a.param, &a.arena),
            Some(a.param)
        );
        // 1i * param -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::Multiply, a.one_i32, a.param, &a.arena),
            Some(a.param)
        );
    }

    #[test]
    fn identity_no_false_positive_for_other_ops() {
        let a = make_identity_arena();
        // Modulo with 1 is not an identity (it gives remainder),
        // And with zero (not all-ones) is not an identity,
        // ShiftLeft with float zero is not an identity (wrong type).
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::Modulo, a.param, a.one_f32, &a.arena),
            None
        );
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::And, a.param, a.zero_f32, &a.arena),
            None
        );
        assert_eq!(
            check_identity_operand(
                naga::BinaryOperator::ShiftLeft,
                a.param,
                a.zero_f32,
                &a.arena
            ),
            None
        );
    }

    // Identity elimination: integration via fold_local_expressions

    #[test]
    fn identity_fold_non_emittable_added_to_folded() {
        // 0 + FunctionArgument -> FunctionArgument is non-emittable,
        // so it must be in the folded set for Emit removal.
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: zero,
                right: param,
            },
            Default::default(),
        );

        let (folded, identity) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            identity > 0,
            "0 + param should trigger identity elimination"
        );
        assert!(
            folded.contains(&add),
            "FunctionArgument replacement must be in folded set for Emit removal"
        );
        assert!(
            matches!(arena[add], naga::Expression::FunctionArgument(0)),
            "expected FunctionArgument(0), got {:?}",
            arena[add]
        );
    }

    #[test]
    fn identity_fold_emittable_not_in_folded() {
        // 0 + Binary(Mul, a, b) -> Binary(Mul, a, b) is emittable,
        // so it must NOT be in the folded set.
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: param_a,
                right: param_b,
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: zero,
                right: mul,
            },
            Default::default(),
        );

        let (folded, identity) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            identity > 0,
            "0 + (a*b) should trigger identity elimination"
        );
        // The result is Binary(Mul, ...) - emittable - NOT in folded.
        assert!(
            !folded.contains(&add),
            "emittable replacement must NOT be in folded set"
        );
        // The expression at `add` should now be the same as `mul`.
        assert!(
            matches!(
                arena[add],
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    ..
                }
            ),
            "expected Binary(Multiply), got {:?}",
            arena[add]
        );
    }

    #[test]
    fn identity_fold_constant_added_to_folded() {
        let module =
            naga::front::wgsl::parse_str("const C: f32 = 41.0;").expect("source should parse");
        let (constant_handle, _) = module.constants.iter().next().unwrap();

        let mut arena = naga::Arena::new();
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let const_expr = arena.append(
            naga::Expression::Constant(constant_handle),
            Default::default(),
        );
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: one,
                right: const_expr,
            },
            Default::default(),
        );

        // With const in literal cache -> const_expr folds to Literal(41.0) first,
        // then 1 * 41.0 is identity-eliminated.
        let mut const_literals = HashMap::new();
        const_literals.insert(constant_handle, naga::Literal::F32(41.0));
        let (folded, _) =
            fold_local_expressions(&mut arena, &const_literals, &naga::UniqueArena::new());
        assert!(folded.contains(&mul), "result should be in folded set");
        assert_f32_literal(&arena, mul, 41.0);

        // Without const in literal cache -> const_expr stays as Constant,
        // identity elim makes mul = Constant (non-emittable -> in folded).
        let mut arena2 = naga::Arena::new();
        let one2 = arena2.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let const_expr2 = arena2.append(
            naga::Expression::Constant(constant_handle),
            Default::default(),
        );
        let mul2 = arena2.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: one2,
                right: const_expr2,
            },
            Default::default(),
        );
        let (folded2, identity2) =
            fold_local_expressions(&mut arena2, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            identity2 > 0,
            "1 * const should trigger identity elimination"
        );
        assert!(
            folded2.contains(&mul2),
            "Constant replacement must be in folded set (non-emittable)"
        );
        assert!(
            matches!(arena2[mul2], naga::Expression::Constant(_)),
            "expected Constant, got {:?}",
            arena2[mul2]
        );
    }

    // Absorbing operand tests

    #[test]
    fn absorbing_mul_zero_left() {
        let a = make_identity_arena();
        // 0 * param -> 0
        assert_eq!(
            check_absorbing_operand(
                naga::BinaryOperator::Multiply,
                a.zero_f32,
                a.param,
                &a.arena
            ),
            Some(a.zero_f32)
        );
    }

    #[test]
    fn absorbing_mul_zero_right() {
        let a = make_identity_arena();
        // param * 0 -> 0
        assert_eq!(
            check_absorbing_operand(
                naga::BinaryOperator::Multiply,
                a.param,
                a.zero_f32,
                &a.arena
            ),
            Some(a.zero_f32)
        );
    }

    #[test]
    fn absorbing_mul_zero_integer() {
        let a = make_identity_arena();
        // 0i * param -> 0i
        assert_eq!(
            check_absorbing_operand(
                naga::BinaryOperator::Multiply,
                a.zero_i32,
                a.param,
                &a.arena
            ),
            Some(a.zero_i32)
        );
    }

    #[test]
    fn absorbing_and_zero() {
        let mut arena = naga::Arena::new();
        let zero_u32 = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        // param & 0u -> 0u
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::And, param, zero_u32, &arena),
            Some(zero_u32)
        );
        // 0u & param -> 0u
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::And, zero_u32, param, &arena),
            Some(zero_u32)
        );
    }

    #[test]
    fn absorbing_or_all_ones() {
        let mut arena = naga::Arena::new();
        let all_ones = arena.append(
            naga::Expression::Literal(naga::Literal::U32(u32::MAX)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        // param | 0xFFFFFFFF -> 0xFFFFFFFF
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::InclusiveOr, param, all_ones, &arena),
            Some(all_ones)
        );
        // 0xFFFFFFFF | param -> 0xFFFFFFFF
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::InclusiveOr, all_ones, param, &arena),
            Some(all_ones)
        );
    }

    #[test]
    fn absorbing_logical_and_false() {
        let mut arena = naga::Arena::new();
        let f = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(false)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::LogicalAnd, param, f, &arena),
            Some(f)
        );
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::LogicalAnd, f, param, &arena),
            Some(f)
        );
    }

    #[test]
    fn absorbing_logical_or_true() {
        let mut arena = naga::Arena::new();
        let t = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::LogicalOr, param, t, &arena),
            Some(t)
        );
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::LogicalOr, t, param, &arena),
            Some(t)
        );
    }

    #[test]
    fn absorbing_no_false_positive() {
        let a = make_identity_arena();
        // Add with zero is identity, NOT absorbing.
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::Add, a.zero_f32, a.param, &a.arena),
            None
        );
        // Multiply with 1 is identity, NOT absorbing.
        assert_eq!(
            check_absorbing_operand(naga::BinaryOperator::Multiply, a.one_f32, a.param, &a.arena),
            None
        );
    }

    // Extended identity tests (bitwise / logical)

    #[test]
    fn identity_or_zero() {
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        // param | 0 -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::InclusiveOr, param, zero, &arena),
            Some(param)
        );
        // 0 | param -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::InclusiveOr, zero, param, &arena),
            Some(param)
        );
    }

    #[test]
    fn identity_xor_zero() {
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        // param ^ 0 -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::ExclusiveOr, param, zero, &arena),
            Some(param)
        );
        // 0 ^ param -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::ExclusiveOr, zero, param, &arena),
            Some(param)
        );
    }

    #[test]
    fn identity_and_all_ones() {
        let mut arena = naga::Arena::new();
        let all_ones = arena.append(
            naga::Expression::Literal(naga::Literal::U32(u32::MAX)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        // param & 0xFFFFFFFF -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::And, param, all_ones, &arena),
            Some(param)
        );
        // 0xFFFFFFFF & param -> param
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::And, all_ones, param, &arena),
            Some(param)
        );
    }

    #[test]
    fn identity_logical_and_true() {
        let mut arena = naga::Arena::new();
        let t = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::LogicalAnd, param, t, &arena),
            Some(param)
        );
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::LogicalAnd, t, param, &arena),
            Some(param)
        );
    }

    #[test]
    fn identity_logical_or_false() {
        let mut arena = naga::Arena::new();
        let f = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(false)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::LogicalOr, param, f, &arena),
            Some(param)
        );
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::LogicalOr, f, param, &arena),
            Some(param)
        );
    }

    #[test]
    fn identity_and_all_ones_i32() {
        let mut arena = naga::Arena::new();
        // i32 -1 is all ones (0xFFFFFFFF)
        let all_ones = arena.append(
            naga::Expression::Literal(naga::Literal::I32(-1)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        assert_eq!(
            check_identity_operand(naga::BinaryOperator::And, param, all_ones, &arena),
            Some(param)
        );
    }

    // Involution tests

    #[test]
    fn involution_double_negate() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let neg1 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: param,
            },
            Default::default(),
        );
        let neg2 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: neg1,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(count > 0, "-(-x) should be simplified");
        // neg2 should now be FunctionArgument(0) (non-emittable -> in folded)
        assert!(
            matches!(arena[neg2], naga::Expression::FunctionArgument(0)),
            "expected FunctionArgument(0), got {:?}",
            arena[neg2]
        );
        assert!(folded.contains(&neg2));
    }

    #[test]
    fn involution_double_logical_not() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let not1 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr: param,
            },
            Default::default(),
        );
        let not2 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr: not1,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(count > 0, "!(!x) should be simplified");
        assert!(
            matches!(arena[not2], naga::Expression::FunctionArgument(0)),
            "expected FunctionArgument(0), got {:?}",
            arena[not2]
        );
        assert!(folded.contains(&not2));
    }

    #[test]
    fn involution_double_bitwise_not() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let not1 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                expr: param,
            },
            Default::default(),
        );
        let not2 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                expr: not1,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(count > 0, "~(~x) should be simplified");
        assert!(
            matches!(arena[not2], naga::Expression::FunctionArgument(0)),
            "expected FunctionArgument(0), got {:?}",
            arena[not2]
        );
        assert!(folded.contains(&not2));
    }

    #[test]
    fn involution_different_ops_not_simplified() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let neg = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: param,
            },
            Default::default(),
        );
        let not = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                expr: neg,
            },
            Default::default(),
        );

        let (_, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(count, 0, "different unary ops should not be simplified");
        assert!(
            matches!(
                arena[not],
                naga::Expression::Unary {
                    op: naga::UnaryOperator::BitwiseNot,
                    ..
                }
            ),
            "expression should remain unchanged"
        );
    }

    #[test]
    fn involution_emittable_inner_not_in_folded() {
        // -(-Binary(Mul, a, b)) -> Binary(Mul, a, b) which is emittable,
        // so it must NOT be in the folded set.
        let mut arena = naga::Arena::new();
        let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: param_a,
                right: param_b,
            },
            Default::default(),
        );
        let neg1 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: mul,
            },
            Default::default(),
        );
        let neg2 = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: neg1,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(count > 0, "-(-Binary(Mul)) should be simplified");
        assert!(
            matches!(
                arena[neg2],
                naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    ..
                }
            ),
            "expected Binary(Multiply), got {:?}",
            arena[neg2]
        );
        // Binary is emittable -> must NOT be in folded set
        assert!(
            !folded.contains(&neg2),
            "emittable involution result must NOT be in folded set"
        );
    }

    // Select simplification tests

    #[test]
    fn select_same_arms_simplified() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let cond = arena.append(naga::Expression::FunctionArgument(1), Default::default());
        let sel = arena.append(
            naga::Expression::Select {
                condition: cond,
                accept: param,
                reject: param,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(count > 0, "select(x, x, cond) should be simplified");
        assert!(
            matches!(arena[sel], naga::Expression::FunctionArgument(0)),
            "expected FunctionArgument(0), got {:?}",
            arena[sel]
        );
        assert!(folded.contains(&sel));
    }

    #[test]
    fn select_different_arms_not_simplified() {
        let mut arena = naga::Arena::new();
        let param_a = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let param_b = arena.append(naga::Expression::FunctionArgument(1), Default::default());
        let cond = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(false)),
            Default::default(),
        );
        let sel = arena.append(
            naga::Expression::Select {
                condition: cond,
                accept: param_a,
                reject: param_b,
            },
            Default::default(),
        );

        let (folded, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        // With constant condition, resolve_literal will fold this to param_b's value.
        // But param_b is FunctionArgument(1), not a literal, so resolve_literal returns None.
        // Select with different arms and non-foldable result stays as-is.
        // Actually: resolve_literal for Select needs both arms to be literals too.
        // So this should remain a Select.
        assert!(
            matches!(arena[sel], naga::Expression::Select { .. }),
            "select with different arms should not be simplified by the simplify loop, got {:?}",
            arena[sel]
        );
        assert!(!folded.contains(&sel));
    }

    // Absorbing integration test (fold_local_expressions)

    #[test]
    fn absorbing_fold_mul_zero_param_not_rewritten() {
        // Prior to the C1 fix, `param * 0.0` (param = FunctionArgument) was
        // rewritten to the scalar literal `0.0` regardless of param's type.
        // That produced invalid IR whenever param was a vector or matrix.
        //
        // The new gate requires BOTH operands be scalar Literal for the
        // simplify-loop absorbing rewrite to fire.  A non-literal operand
        // (FunctionArgument here) carries unknown type, so absorbing is
        // safely declined.  The Binary stays as a Binary.
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: param,
                right: zero,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(
            count, 0,
            "param * 0 with unknown-type param must not fire absorbing"
        );
        assert!(
            matches!(arena[mul], naga::Expression::Binary { .. }),
            "Binary must be preserved, got {:?}",
            arena[mul]
        );
        assert!(!folded.contains(&mul));
    }

    #[test]
    fn absorbing_fold_and_zero_u32_param_not_rewritten() {
        // Same C1 case for integer `&` absorbing.
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::U32(0)),
            Default::default(),
        );
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let and = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::And,
                left: param,
                right: zero,
            },
            Default::default(),
        );

        let (folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert_eq!(
            count, 0,
            "param & 0u with unknown-type param must not fire absorbing"
        );
        assert!(
            matches!(arena[and], naga::Expression::Binary { .. }),
            "Binary must be preserved, got {:?}",
            arena[and]
        );
        assert!(!folded.contains(&and));
    }

    #[test]
    fn absorbing_fold_mul_zero_both_literal_produces_literal() {
        // When BOTH operands are scalar Literals, absorbing is type-safe
        // (the Binary's result type equals both operand types).  This is
        // the only case the simplify-loop absorbing rewrite fires.
        // (In practice, the preceding `resolve_const_value` stage usually
        // folds it first, but the safety net is still exercised here.)
        let mut arena = naga::Arena::new();
        let zero = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let two = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: two,
                right: zero,
            },
            Default::default(),
        );

        let (_folded, _count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        // Either eval_binary or absorbing must have collapsed this to a literal 0.
        assert_f32_literal(&arena, mul, 0.0);
    }

    #[test]
    fn absorbing_fold_logical_and_false_with_non_literal_rhs_rewritten() {
        // C1-refinement regression: `LogicalAnd`/`LogicalOr` are strictly
        // scalar-bool in valid WGSL IR (no vector broadcasting possible),
        // so absorbing is type-safe even when the other operand is NOT a
        // literal.  This is the pattern produced by dead_branch Phase 0 when
        // re-sugaring `var tmp = false && (j < -8)` from its lowered form;
        // without this rewrite the dead loop `for(;0<0&&j<0-8;){}` cannot be
        // collapsed and the minified output grows rather than shrinks.
        let mut arena = naga::Arena::new();
        let false_lit = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(false)),
            Default::default(),
        );
        // Non-literal RHS (FunctionArgument stands in for `j < -8`, which at
        // the point of absorbing is a Binary, also a non-Literal).
        let rhs = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let and_ = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalAnd,
                left: false_lit,
                right: rhs,
            },
            Default::default(),
        );

        let (_folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());

        assert!(count > 0, "false && rhs must be absorbed to false");
        assert!(
            matches!(
                arena[and_],
                naga::Expression::Literal(naga::Literal::Bool(false))
            ),
            "expected Literal(Bool(false)), got {:?}",
            arena[and_]
        );
    }

    #[test]
    fn absorbing_fold_logical_or_true_with_non_literal_rhs_rewritten() {
        // Symmetric case: `true || rhs` must collapse to `true` even when
        // `rhs` is non-literal.
        let mut arena = naga::Arena::new();
        let true_lit = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let rhs = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let or_ = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left: true_lit,
                right: rhs,
            },
            Default::default(),
        );

        let (_folded, count) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());

        assert!(count > 0, "true || rhs must be absorbed to true");
        assert!(
            matches!(
                arena[or_],
                naga::Expression::Literal(naga::Literal::Bool(true))
            ),
            "expected Literal(Bool(true)), got {:?}",
            arena[or_]
        );
    }

    /// End-to-end regression for the `0<0 && j<-8` pattern inside a dead
    /// `for` loop (chromium/1403752.wgsl).  Before the C1 refinement the
    /// minified output GREW from 66 bytes to 87 bytes because the absorbing
    /// rewrite declined to fold `false && (j < -8)` and the dead loop
    /// survived into the emitter as `loop { var a = false && A<-8; if !(a)
    /// { break; } }`.  After the refinement the loop collapses and the
    /// output stays at or below input size.
    #[test]
    fn e2e_dead_for_loop_with_short_circuit_condition_does_not_grow() {
        let source = "@compute @workgroup_size(1) fn d(){var j:i32;for(;0<0&&j<0-8;){}}\n";
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
        assert!(
            out.source.len() <= source.len(),
            "dead for-loop minification must not grow: input={} bytes, output={} bytes\n  in:  {:?}\n  out: {:?}",
            source.len(),
            out.source.len(),
            source,
            out.source,
        );
    }

    // MARK: Vector constant folding tests

    /// Helper: insert a vec type into the type arena and return its handle.
    fn make_vec_type(
        types: &mut naga::UniqueArena<naga::Type>,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    ) -> naga::Handle<naga::Type> {
        types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector { size, scalar },
            },
            Default::default(),
        )
    }

    /// Helper: assert that expression is a Compose of literal scalars.
    fn assert_compose_of_f32(
        arena: &naga::Arena<naga::Expression>,
        handle: naga::Handle<naga::Expression>,
        expected: &[f32],
    ) {
        match &arena[handle] {
            naga::Expression::Compose { components, .. } => {
                assert_eq!(
                    components.len(),
                    expected.len(),
                    "compose component count mismatch"
                );
                for (i, &c) in components.iter().enumerate() {
                    assert_f32_literal(arena, c, expected[i]);
                }
            }
            other => panic!("expected Compose, got {other:?}"),
        }
    }

    #[test]
    fn vector_splat_folds_to_compose() {
        let mut types = naga::UniqueArena::new();
        let _vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let one = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let splat = arena.append(
            naga::Expression::Splat {
                size: naga::VectorSize::Tri,
                value: one,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // Splat of literal 1.0 -> Compose(vec3f, [1.0, 1.0, 1.0])
        assert_compose_of_f32(&arena, splat, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn vector_compose_binary_add_folds() {
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let a1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let a2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let a3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let b1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(10.0)),
            Default::default(),
        );
        let b2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(20.0)),
            Default::default(),
        );
        let b3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(30.0)),
            Default::default(),
        );
        // Pre-place result literals for materialization
        let _r1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(11.0)),
            Default::default(),
        );
        let _r2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(22.0)),
            Default::default(),
        );
        let _r3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(33.0)),
            Default::default(),
        );

        let va = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![a1, a2, a3],
            },
            Default::default(),
        );
        let vb = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![b1, b2, b3],
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: va,
                right: vb,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // vec3(1,2,3) + vec3(10,20,30) = vec3(11,22,33)
        assert_compose_of_f32(&arena, add, &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn vector_negate_folds() {
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(-2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.5)),
            Default::default(),
        );
        // Pre-place result literals for materialization
        let _rn1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(-1.0)),
            Default::default(),
        );
        let _rn2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let _rn3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(-3.5)),
            Default::default(),
        );
        let v = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c1, c2, c3],
            },
            Default::default(),
        );
        let neg = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: v,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // -vec3(1, -2, 3.5) = vec3(-1, 2, -3.5)
        assert_compose_of_f32(&arena, neg, &[-1.0, 2.0, -3.5]);
    }

    #[test]
    fn vector_access_index_folds_to_scalar() {
        let mut types = naga::UniqueArena::new();
        let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(10.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(20.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(30.0)),
            Default::default(),
        );
        let c4 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(40.0)),
            Default::default(),
        );
        let v = arena.append(
            naga::Expression::Compose {
                ty: vec4f_ty,
                components: vec![c1, c2, c3, c4],
            },
            Default::default(),
        );
        // .y == index 1
        let access = arena.append(
            naga::Expression::AccessIndex { base: v, index: 1 },
            Default::default(),
        );

        let (folded, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // vec4f(10,20,30,40).y -> 20.0
        assert_f32_literal(&arena, access, 20.0);
        assert!(
            folded.contains(&access),
            "scalar result should be in folded set"
        );
    }

    #[test]
    fn vector_swizzle_folds() {
        let mut types = naga::UniqueArena::new();
        let _vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);
        let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let c4 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        let v = arena.append(
            naga::Expression::Compose {
                ty: vec4f_ty,
                components: vec![c1, c2, c3, c4],
            },
            Default::default(),
        );
        // .zw swizzle -> vec2(3.0, 4.0)
        let swiz = arena.append(
            naga::Expression::Swizzle {
                size: naga::VectorSize::Bi,
                vector: v,
                pattern: [
                    naga::SwizzleComponent::Z,
                    naga::SwizzleComponent::W,
                    naga::SwizzleComponent::X, // unused
                    naga::SwizzleComponent::X, // unused
                ],
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert_compose_of_f32(&arena, swiz, &[3.0, 4.0]);
    }

    #[test]
    fn vector_scalar_broadcast_mul() {
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        // Pre-place result literals for materialization
        let _r1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(20.0)),
            Default::default(),
        );
        let _r2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(30.0)),
            Default::default(),
        );
        let _r3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(40.0)),
            Default::default(),
        );
        let v = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c1, c2, c3],
            },
            Default::default(),
        );
        let s = arena.append(
            naga::Expression::Literal(naga::Literal::F32(10.0)),
            Default::default(),
        );
        // vec3(2,3,4) * 10.0
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: v,
                right: s,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert_compose_of_f32(&arena, mul, &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn vector_zero_value_stays_non_emittable() {
        // ZeroValue is non-emittable, so it must NOT be replaced with
        // an emittable Compose - that would produce invalid IR.
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let _z = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let zero = arena.append(naga::Expression::ZeroValue(vec3f_ty), Default::default());

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert!(
            matches!(arena[zero], naga::Expression::ZeroValue(_)),
            "ZeroValue must stay as ZeroValue, got {:?}",
            arena[zero]
        );
    }

    #[test]
    fn vector_add_zero_value_folds() {
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let v = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c1, c2, c3],
            },
            Default::default(),
        );
        let zero = arena.append(naga::Expression::ZeroValue(vec3f_ty), Default::default());
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: v,
                right: zero,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // vec3(1,2,3) + vec3(0,0,0) = vec3(1,2,3)
        assert_compose_of_f32(&arena, add, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn vector_splat_binary_scalar_add() {
        // Splat(2.0) + Splat(3.0) -> Compose([5.0, 5.0])
        let mut types = naga::UniqueArena::new();
        let _vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let two = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let three = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        // Need a literal 5.0 in the arena for materialization
        let _five = arena.append(
            naga::Expression::Literal(naga::Literal::F32(5.0)),
            Default::default(),
        );

        let sp1 = arena.append(
            naga::Expression::Splat {
                size: naga::VectorSize::Bi,
                value: two,
            },
            Default::default(),
        );
        let sp2 = arena.append(
            naga::Expression::Splat {
                size: naga::VectorSize::Bi,
                value: three,
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: sp1,
                right: sp2,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert_compose_of_f32(&arena, add, &[5.0, 5.0]);
    }

    #[test]
    fn vector_nested_chain_folds() {
        // Test: -negate(compose(1, 2, 3)) + compose(10, 20, 30) == compose(9, 18, 27)
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let c10 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(10.0)),
            Default::default(),
        );
        let c20 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(20.0)),
            Default::default(),
        );
        let c30 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(30.0)),
            Default::default(),
        );
        // Pre-place result literals for materialisation
        let _c9 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(9.0)),
            Default::default(),
        );
        let _c18 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(18.0)),
            Default::default(),
        );
        let _c27 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(27.0)),
            Default::default(),
        );

        let va = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c1, c2, c3],
            },
            Default::default(),
        );
        let neg = arena.append(
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: va,
            },
            Default::default(),
        );
        let vb = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c10, c20, c30],
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: neg,
                right: vb,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // -vec3(1,2,3) + vec3(10,20,30) = vec3(9, 18, 27)
        assert_compose_of_f32(&arena, add, &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn vector_integer_types_fold() {
        let mut types = naga::UniqueArena::new();
        let vec2i_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::I32);

        let mut arena = naga::Arena::new();
        let a = arena.append(
            naga::Expression::Literal(naga::Literal::I32(10)),
            Default::default(),
        );
        let b = arena.append(
            naga::Expression::Literal(naga::Literal::I32(20)),
            Default::default(),
        );
        let c = arena.append(
            naga::Expression::Literal(naga::Literal::I32(3)),
            Default::default(),
        );
        let d = arena.append(
            naga::Expression::Literal(naga::Literal::I32(7)),
            Default::default(),
        );
        // Results
        let _r1 = arena.append(
            naga::Expression::Literal(naga::Literal::I32(13)),
            Default::default(),
        );
        let _r2 = arena.append(
            naga::Expression::Literal(naga::Literal::I32(27)),
            Default::default(),
        );

        let va = arena.append(
            naga::Expression::Compose {
                ty: vec2i_ty,
                components: vec![a, b],
            },
            Default::default(),
        );
        let vb = arena.append(
            naga::Expression::Compose {
                ty: vec2i_ty,
                components: vec![c, d],
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: va,
                right: vb,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // vec2i(10,20) + vec2i(3,7) = vec2i(13,27)
        match &arena[add] {
            naga::Expression::Compose { components, .. } => {
                assert_eq!(components.len(), 2);
                assert!(matches!(
                    arena[components[0]],
                    naga::Expression::Literal(naga::Literal::I32(13))
                ));
                assert!(matches!(
                    arena[components[1]],
                    naga::Expression::Literal(naga::Literal::I32(27))
                ));
            }
            other => panic!("expected Compose, got {other:?}"),
        }
    }

    #[test]
    fn vector_no_matching_literal_skips_materialization() {
        // If the result literal doesn't exist in the arena before the target,
        // materialization is skipped and the expression remains unchanged.
        let mut types = naga::UniqueArena::new();
        let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(100.0)),
            Default::default(),
        );
        let c4 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(200.0)),
            Default::default(),
        );
        // Intentionally do NOT add literals 101.0 and 202.0

        let va = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![c1, c2],
            },
            Default::default(),
        );
        let vb = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![c3, c4],
            },
            Default::default(),
        );
        let add = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: va,
                right: vb,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // Result 101.0 and 202.0 are not in the arena, so add stays as Binary
        assert!(
            matches!(arena[add], naga::Expression::Binary { .. }),
            "should remain Binary when materialization fails, got {:?}",
            arena[add]
        );
    }

    #[test]
    fn vector_compose_mixed_scalar_and_vector() {
        // Compose(vec4f, [scalar, vec3f]) should flatten to 4 components.
        let mut types = naga::UniqueArena::new();
        let vec3f_ty = make_vec_type(&mut types, naga::VectorSize::Tri, naga::Scalar::F32);
        let vec4f_ty = make_vec_type(&mut types, naga::VectorSize::Quad, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let c4 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        let v3 = arena.append(
            naga::Expression::Compose {
                ty: vec3f_ty,
                components: vec![c2, c3, c4],
            },
            Default::default(),
        );
        // Compose(vec4f, [scalar(1.0), vec3(2.0, 3.0, 4.0)])
        let v4 = arena.append(
            naga::Expression::Compose {
                ty: vec4f_ty,
                components: vec![c1, v3],
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert_compose_of_f32(&arena, v4, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vector_relational_op_produces_bool_vector() {
        // vec2(1.0, 3.0) < vec2(2.0, 2.0) -> vec2<bool>(true, false)
        let mut types = naga::UniqueArena::new();
        let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);
        let _vec2b_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::BOOL);

        let mut arena = naga::Arena::new();
        let a1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let a2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        let b1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let b2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        // Result literals for materialization
        let _t = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let _f = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(false)),
            Default::default(),
        );

        let va = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![a1, a2],
            },
            Default::default(),
        );
        let vb = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![b1, b2],
            },
            Default::default(),
        );
        let lt = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Less,
                left: va,
                right: vb,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // 1.0 < 2.0 -> true,  3.0 < 2.0 -> false
        match &arena[lt] {
            naga::Expression::Compose { components, .. } => {
                assert_eq!(components.len(), 2);
                assert!(matches!(
                    arena[components[0]],
                    naga::Expression::Literal(naga::Literal::Bool(true))
                ));
                assert!(matches!(
                    arena[components[1]],
                    naga::Expression::Literal(naga::Literal::Bool(false))
                ));
            }
            other => panic!("expected Compose(vec2<bool>), got {other:?}"),
        }
    }

    #[test]
    fn vector_select_constant_true() {
        // select(accept_vec, reject_vec, true) -> accept_vec
        let mut types = naga::UniqueArena::new();
        let vec2f_ty = make_vec_type(&mut types, naga::VectorSize::Bi, naga::Scalar::F32);

        let mut arena = naga::Arena::new();
        let c1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let c2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let c3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(9.0)),
            Default::default(),
        );
        let c4 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(8.0)),
            Default::default(),
        );
        let accept = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![c1, c2],
            },
            Default::default(),
        );
        let reject = arena.append(
            naga::Expression::Compose {
                ty: vec2f_ty,
                components: vec![c3, c4],
            },
            Default::default(),
        );
        let cond = arena.append(
            naga::Expression::Literal(naga::Literal::Bool(true)),
            Default::default(),
        );
        let sel = arena.append(
            naga::Expression::Select {
                condition: cond,
                accept,
                reject,
            },
            Default::default(),
        );

        let (_, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        // select with true condition -> accept = vec2(1.0, 2.0)
        assert_compose_of_f32(&arena, sel, &[1.0, 2.0]);
    }

    #[test]
    fn scalar_zero_value_resolves() {
        // ZeroValue(f32) should fold to Literal(F32(0.0))
        let mut types = naga::UniqueArena::new();
        let f32_ty = types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            },
            Default::default(),
        );

        let mut arena = naga::Arena::new();
        let zv = arena.append(naga::Expression::ZeroValue(f32_ty), Default::default());

        let (folded, _) = fold_local_expressions(&mut arena, &HashMap::new(), &types);
        assert!(
            matches!(arena[zv], naga::Expression::Literal(naga::Literal::F32(v)) if v == 0.0),
            "ZeroValue(f32) should fold to Literal(0.0), got {:?}",
            arena[zv]
        );
        assert!(
            folded.contains(&zv),
            "scalar ZeroValue should be in folded set"
        );
    }

    #[test]
    fn math_abs_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Abs,
                naga::Literal::F32(-3.0),
                None,
                None
            ),
            Some(naga::Literal::F32(3.0))
        );
    }

    #[test]
    fn math_abs_i32() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Abs, naga::Literal::I32(-5), None, None),
            Some(naga::Literal::I32(5))
        );
    }

    #[test]
    fn math_abs_u32_identity() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Abs, naga::Literal::U32(7), None, None),
            Some(naga::Literal::U32(7))
        );
    }

    #[test]
    fn math_abs_i32_min_not_folded() {
        // WGSL: abs(i32::MIN) has no positive representation; folding to
        // wrapping i32::MIN would silently change observable semantics
        // (naga would otherwise produce an execution error), so the folder
        // must decline.
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Abs,
                naga::Literal::I32(i32::MIN),
                None,
                None
            ),
            None
        );
    }

    #[test]
    fn math_min_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Min,
                naga::Literal::F32(3.0),
                Some(naga::Literal::F32(1.0)),
                None
            ),
            Some(naga::Literal::F32(1.0))
        );
    }

    #[test]
    fn math_max_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Max,
                naga::Literal::F32(3.0),
                Some(naga::Literal::F32(7.0)),
                None
            ),
            Some(naga::Literal::F32(7.0))
        );
    }

    #[test]
    fn math_min_i32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Min,
                naga::Literal::I32(10),
                Some(naga::Literal::I32(-2)),
                None
            ),
            Some(naga::Literal::I32(-2))
        );
    }

    #[test]
    fn math_clamp_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Clamp,
                naga::Literal::F32(5.0),
                Some(naga::Literal::F32(0.0)),
                Some(naga::Literal::F32(1.0)),
            ),
            Some(naga::Literal::F32(1.0))
        );
    }

    #[test]
    fn math_clamp_rejects_inverted_range() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Clamp,
                naga::Literal::F32(0.5),
                Some(naga::Literal::F32(1.0)),
                Some(naga::Literal::F32(0.0)),
            ),
            None,
            "clamp with low > high should not fold"
        );
    }

    #[test]
    fn math_saturate_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Saturate,
                naga::Literal::F32(2.0),
                None,
                None
            ),
            Some(naga::Literal::F32(1.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Saturate,
                naga::Literal::F32(-0.5),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_sign_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sign,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sign,
                naga::Literal::F32(-5.0),
                None,
                None
            ),
            Some(naga::Literal::F32(-1.0))
        );
    }

    #[test]
    fn math_floor_ceil_trunc_round() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Floor,
                naga::Literal::F32(1.7),
                None,
                None
            ),
            Some(naga::Literal::F32(1.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Ceil,
                naga::Literal::F32(1.2),
                None,
                None
            ),
            Some(naga::Literal::F32(2.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Trunc,
                naga::Literal::F32(-1.9),
                None,
                None
            ),
            Some(naga::Literal::F32(-1.0))
        );
        // Round uses ties-to-even: 0.5 -> 0.0, 1.5 -> 2.0
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Round,
                naga::Literal::F32(0.5),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Round,
                naga::Literal::F32(1.5),
                None,
                None
            ),
            Some(naga::Literal::F32(2.0))
        );
    }

    #[test]
    fn math_fract_f32() {
        // WGSL fract(e) = e - floor(e)
        let result = eval_math_scalar(
            naga::MathFunction::Fract,
            naga::Literal::F32(1.75),
            None,
            None,
        );
        assert_eq!(result, Some(naga::Literal::F32(0.75)));
        // Negative: fract(-0.25) = -0.25 - floor(-0.25) = -0.25 - (-1.0) = 0.75
        let result = eval_math_scalar(
            naga::MathFunction::Fract,
            naga::Literal::F32(-0.25),
            None,
            None,
        );
        assert_eq!(result, Some(naga::Literal::F32(0.75)));
    }

    #[test]
    fn math_step_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Step,
                naga::Literal::F32(0.5),
                Some(naga::Literal::F32(1.0)),
                None,
            ),
            Some(naga::Literal::F32(1.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Step,
                naga::Literal::F32(0.5),
                Some(naga::Literal::F32(0.3)),
                None,
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_sqrt_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sqrt,
                naga::Literal::F32(4.0),
                None,
                None
            ),
            Some(naga::Literal::F32(2.0))
        );
    }

    #[test]
    fn math_sqrt_negative_not_folded() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sqrt,
                naga::Literal::F32(-1.0),
                None,
                None
            ),
            None,
            "sqrt of negative should not fold"
        );
    }

    #[test]
    fn math_inverse_sqrt_f32() {
        let result = eval_math_scalar(
            naga::MathFunction::InverseSqrt,
            naga::Literal::F32(4.0),
            None,
            None,
        );
        assert_eq!(result, Some(naga::Literal::F32(0.5)));
    }

    #[test]
    fn math_inverse_sqrt_zero_not_folded() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::InverseSqrt,
                naga::Literal::F32(0.0),
                None,
                None,
            ),
            None,
            "inverseSqrt(0) should not fold"
        );
    }

    #[test]
    fn math_fma_f32() {
        // fma(2.0, 3.0, 1.0) = 2.0 * 3.0 + 1.0 = 7.0
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Fma,
                naga::Literal::F32(2.0),
                Some(naga::Literal::F32(3.0)),
                Some(naga::Literal::F32(1.0)),
            ),
            Some(naga::Literal::F32(7.0))
        );
    }

    #[test]
    fn math_sin_cos_zero() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Sin, naga::Literal::F32(0.0), None, None),
            Some(naga::Literal::F32(0.0))
        );
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Cos, naga::Literal::F32(0.0), None, None),
            Some(naga::Literal::F32(1.0))
        );
    }

    #[test]
    fn math_exp_log() {
        // exp(0) = 1
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Exp, naga::Literal::F32(0.0), None, None),
            Some(naga::Literal::F32(1.0))
        );
        // log(1) = 0
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Log, naga::Literal::F32(1.0), None, None),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_log_zero_not_folded() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Log, naga::Literal::F32(0.0), None, None),
            None,
            "log(0) should not fold"
        );
    }

    #[test]
    fn math_log_negative_not_folded() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Log,
                naga::Literal::F32(-1.0),
                None,
                None
            ),
            None,
            "log(negative) should not fold"
        );
    }

    #[test]
    fn math_exp_overflow_not_folded() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Exp,
                naga::Literal::F32(1000.0),
                None,
                None
            ),
            None,
            "exp(1000) overflows and should not fold"
        );
    }

    #[test]
    fn math_pow_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Pow,
                naga::Literal::F32(2.0),
                Some(naga::Literal::F32(3.0)),
                None,
            ),
            Some(naga::Literal::F32(8.0))
        );
    }

    #[test]
    fn math_log2_exp2() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Log2,
                naga::Literal::F32(8.0),
                None,
                None
            ),
            Some(naga::Literal::F32(3.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Exp2,
                naga::Literal::F32(3.0),
                None,
                None
            ),
            Some(naga::Literal::F32(8.0))
        );
    }

    #[test]
    fn math_acos_domain_violation() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Acos,
                naga::Literal::F32(2.0),
                None,
                None
            ),
            None,
            "acos(2.0) is outside [-1, 1] and should not fold"
        );
    }

    #[test]
    fn math_acosh_domain_violation() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Acosh,
                naga::Literal::F32(0.5),
                None,
                None
            ),
            None,
            "acosh(0.5) is below 1.0 and should not fold"
        );
    }

    #[test]
    fn math_atanh_domain_violation() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Atanh,
                naga::Literal::F32(1.0),
                None,
                None
            ),
            None,
            "atanh(1.0) is at boundary and should not fold"
        );
    }

    #[test]
    fn math_radians_degrees_roundtrip() {
        let rad = eval_math_scalar(
            naga::MathFunction::Radians,
            naga::Literal::F32(180.0),
            None,
            None,
        );
        match rad {
            Some(naga::Literal::F32(v)) => {
                assert!((v - std::f32::consts::PI).abs() < 1e-5);
            }
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn math_count_trailing_zeros_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::CountTrailingZeros,
                naga::Literal::U32(8),
                None,
                None
            ),
            Some(naga::Literal::U32(3))
        );
    }

    #[test]
    fn math_count_leading_zeros_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::CountLeadingZeros,
                naga::Literal::U32(1),
                None,
                None
            ),
            Some(naga::Literal::U32(31))
        );
    }

    #[test]
    fn math_count_one_bits_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::CountOneBits,
                naga::Literal::U32(0b1011),
                None,
                None
            ),
            Some(naga::Literal::U32(3))
        );
    }

    #[test]
    fn math_reverse_bits_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::ReverseBits,
                naga::Literal::U32(1),
                None,
                None
            ),
            Some(naga::Literal::U32(1u32 << 31))
        );
    }

    #[test]
    fn math_first_trailing_bit_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstTrailingBit,
                naga::Literal::U32(0),
                None,
                None
            ),
            Some(naga::Literal::U32(u32::MAX))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstTrailingBit,
                naga::Literal::U32(12),
                None,
                None
            ),
            Some(naga::Literal::U32(2))
        );
    }

    #[test]
    fn math_first_leading_bit_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::U32(0),
                None,
                None
            ),
            Some(naga::Literal::U32(u32::MAX))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::U32(8),
                None,
                None
            ),
            Some(naga::Literal::U32(3))
        );
    }

    #[test]
    fn math_first_leading_bit_i32() {
        // 0 and -1 both return -1
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::I32(0),
                None,
                None
            ),
            Some(naga::Literal::I32(-1))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::I32(-1),
                None,
                None
            ),
            Some(naga::Literal::I32(-1))
        );
        // Positive: firstLeadingBit(8) = 3
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::I32(8),
                None,
                None
            ),
            Some(naga::Literal::I32(3))
        );
    }

    #[test]
    fn math_unsupported_returns_none() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Dot, naga::Literal::F32(1.0), None, None),
            None,
            "Dot is Tier 4 and should not fold at scalar level"
        );
    }

    #[test]
    fn math_abstract_float_sqrt() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sqrt,
                naga::Literal::AbstractFloat(9.0),
                None,
                None
            ),
            Some(naga::Literal::AbstractFloat(3.0))
        );
    }

    #[test]
    fn math_pow_negative_base_not_folded() {
        // WGSL precondition: e1 >= 0.0; negative base is undefined.
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Pow,
                naga::Literal::F32(-1.0),
                Some(naga::Literal::F32(2.0)),
                None,
            ),
            None,
            "pow with negative base should not fold"
        );
    }

    #[test]
    fn math_pow_zero_base_negative_exp_not_folded() {
        // pow(0, -1) -> inf -> not finite -> None
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Pow,
                naga::Literal::F32(0.0),
                Some(naga::Literal::F32(-1.0)),
                None,
            ),
            None,
            "pow(0, negative) overflows and should not fold"
        );
    }

    #[test]
    fn math_sign_negative_zero() {
        // WGSL: sign(-0.0) should return 0.0 (or -0.0, impl-defined)
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sign,
                naga::Literal::F32(-0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_sign_i32() {
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(42), None, None),
            Some(naga::Literal::I32(1))
        );
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(0), None, None),
            Some(naga::Literal::I32(0))
        );
        assert_eq!(
            eval_math_scalar(naga::MathFunction::Sign, naga::Literal::I32(-7), None, None),
            Some(naga::Literal::I32(-1))
        );
    }

    #[test]
    fn math_tan_f32() {
        let result = eval_math_scalar(naga::MathFunction::Tan, naga::Literal::F32(0.0), None, None);
        assert_eq!(result, Some(naga::Literal::F32(0.0)));
    }

    #[test]
    fn math_atan2_f32() {
        let result = eval_math_scalar(
            naga::MathFunction::Atan2,
            naga::Literal::F32(1.0),
            Some(naga::Literal::F32(1.0)),
            None,
        );
        match result {
            Some(naga::Literal::F32(v)) => {
                assert!((v - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
            }
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn math_tanh_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Tanh,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_cosh_sinh_f32() {
        // cosh(0) = 1, sinh(0) = 0
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Cosh,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(1.0))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Sinh,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_cosh_overflow_not_folded() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Cosh,
                naga::Literal::F32(1000.0),
                None,
                None
            ),
            None,
            "cosh(1000) overflows and should not fold"
        );
    }

    #[test]
    fn math_asinh_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Asinh,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_asin_f32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Asin,
                naga::Literal::F32(0.0),
                None,
                None
            ),
            Some(naga::Literal::F32(0.0))
        );
    }

    #[test]
    fn math_asin_domain_violation() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Asin,
                naga::Literal::F32(2.0),
                None,
                None
            ),
            None,
            "asin(2.0) is outside [-1, 1] and should not fold"
        );
    }

    #[test]
    fn math_first_leading_bit_negative_i32_edge_cases() {
        // -2 = 0xFFFFFFFE: first differing bit from sign is at position 0
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::I32(-2),
                None,
                None
            ),
            Some(naga::Literal::I32(0))
        );
        // i32::MIN = 0x80000000: first differing bit from sign is at position 30
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstLeadingBit,
                naga::Literal::I32(i32::MIN),
                None,
                None
            ),
            Some(naga::Literal::I32(30))
        );
    }

    #[test]
    fn math_first_trailing_bit_i32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstTrailingBit,
                naga::Literal::I32(0),
                None,
                None
            ),
            Some(naga::Literal::I32(-1))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::FirstTrailingBit,
                naga::Literal::I32(12),
                None,
                None
            ),
            Some(naga::Literal::I32(2))
        );
    }

    #[test]
    fn math_clamp_i32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Clamp,
                naga::Literal::I32(10),
                Some(naga::Literal::I32(0)),
                Some(naga::Literal::I32(5)),
            ),
            Some(naga::Literal::I32(5))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Clamp,
                naga::Literal::I32(-3),
                Some(naga::Literal::I32(0)),
                Some(naga::Literal::I32(5)),
            ),
            Some(naga::Literal::I32(0))
        );
    }

    #[test]
    fn math_min_max_u32() {
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Min,
                naga::Literal::U32(10),
                Some(naga::Literal::U32(3)),
                None
            ),
            Some(naga::Literal::U32(3))
        );
        assert_eq!(
            eval_math_scalar(
                naga::MathFunction::Max,
                naga::Literal::U32(10),
                Some(naga::Literal::U32(3)),
                None
            ),
            Some(naga::Literal::U32(10))
        );
    }

    #[test]
    fn math_abs_vector_component_wise() {
        let v = ConstValue::Vector {
            components: vec![
                naga::Literal::F32(-1.0),
                naga::Literal::F32(2.0),
                naga::Literal::F32(-3.0),
            ],
            size: naga::VectorSize::Tri,
            scalar: naga::Scalar::F32,
        };
        let result = eval_const_math(naga::MathFunction::Abs, v, None, None);
        match result {
            Some(ConstValue::Vector { components, .. }) => {
                assert_eq!(components[0], naga::Literal::F32(1.0));
                assert_eq!(components[1], naga::Literal::F32(2.0));
                assert_eq!(components[2], naga::Literal::F32(3.0));
            }
            other => panic!("expected vector, got {other:?}"),
        }
    }

    #[test]
    fn math_min_vector_component_wise() {
        let a = ConstValue::Vector {
            components: vec![naga::Literal::F32(3.0), naga::Literal::F32(1.0)],
            size: naga::VectorSize::Bi,
            scalar: naga::Scalar::F32,
        };
        let b = ConstValue::Vector {
            components: vec![naga::Literal::F32(1.0), naga::Literal::F32(5.0)],
            size: naga::VectorSize::Bi,
            scalar: naga::Scalar::F32,
        };
        let result = eval_const_math(naga::MathFunction::Min, a, Some(b), None);
        match result {
            Some(ConstValue::Vector { components, .. }) => {
                assert_eq!(components[0], naga::Literal::F32(1.0));
                assert_eq!(components[1], naga::Literal::F32(1.0));
            }
            other => panic!("expected vector, got {other:?}"),
        }
    }

    #[test]
    fn math_vector_size_mismatch_returns_none() {
        let a = ConstValue::Vector {
            components: vec![naga::Literal::F32(1.0), naga::Literal::F32(2.0)],
            size: naga::VectorSize::Bi,
            scalar: naga::Scalar::F32,
        };
        let b = ConstValue::Vector {
            components: vec![
                naga::Literal::F32(1.0),
                naga::Literal::F32(2.0),
                naga::Literal::F32(3.0),
            ],
            size: naga::VectorSize::Tri,
            scalar: naga::Scalar::F32,
        };
        assert_eq!(
            eval_const_math(naga::MathFunction::Min, a, Some(b), None),
            None,
            "mismatched vector sizes should not fold"
        );
    }

    #[test]
    fn folds_math_sqrt_in_local_arena() {
        let mut arena = naga::Arena::new();
        let four = arena.append(
            naga::Expression::Literal(naga::Literal::F32(4.0)),
            Default::default(),
        );
        let sqrt_expr = arena.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Sqrt,
                arg: four,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            !changed.is_empty(),
            "sqrt(4.0) should be folded to a literal"
        );
        assert_f32_literal(&arena, sqrt_expr, 2.0);
    }

    #[test]
    fn folds_math_max_in_local_arena() {
        let mut arena = naga::Arena::new();
        let a = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let b = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let max_expr = arena.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Max,
                arg: a,
                arg1: Some(b),
                arg2: None,
                arg3: None,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            !changed.is_empty(),
            "max(1.0, 2.0) should be folded to a literal"
        );
        assert_f32_literal(&arena, max_expr, 2.0);
    }

    #[test]
    fn folds_math_clamp_in_local_arena() {
        let mut arena = naga::Arena::new();
        let val = arena.append(
            naga::Expression::Literal(naga::Literal::F32(5.0)),
            Default::default(),
        );
        let lo = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        let hi = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let clamp_expr = arena.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Clamp,
                arg: val,
                arg1: Some(lo),
                arg2: Some(hi),
                arg3: None,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(!changed.is_empty(), "clamp(5.0, 0.0, 1.0) should be folded");
        assert_f32_literal(&arena, clamp_expr, 1.0);
    }

    #[test]
    fn does_not_fold_math_with_non_constant_arg() {
        let mut arena = naga::Arena::new();
        let param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let sqrt_expr = arena.append(
            naga::Expression::Math {
                fun: naga::MathFunction::Sqrt,
                arg: param,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Default::default(),
        );

        let (changed, _) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());
        assert!(
            changed.is_empty(),
            "sqrt(param) should not fold - arg is not constant"
        );
        assert!(
            matches!(arena[sqrt_expr], naga::Expression::Math { .. }),
            "expression should remain as Math"
        );
    }

    #[test]
    fn e2e_math_sqrt_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let x = sqrt(4.0);
    return vec4f(x, x, x, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        // After folding, sqrt(4.0) should become 2.0 (literal).
        // The output should NOT contain "sqrt".
        assert!(
            !out.source.contains("sqrt"),
            "sqrt(4.0) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_max_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let m = max(1.0, 2.0);
    return vec4f(m, m, m, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("max"),
            "max(1.0, 2.0) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_abs_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let a = abs(-3.0);
    return vec4f(a, a, a, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("abs"),
            "abs(-3.0) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_sin_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let s = sin(0.0);
    return vec4f(s, s, s, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("sin"),
            "sin(0.0) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_floor_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let f = floor(1.7);
    return vec4f(f, f, f, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("floor"),
            "floor(1.7) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_clamp_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let c = clamp(5.0, 0.0, 1.0);
    return vec4f(c, c, c, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("clamp"),
            "clamp(5.0, 0.0, 1.0) should be folded away: {}",
            out.source
        );
    }

    #[test]
    fn e2e_math_saturate_folds_through_pipeline() {
        let source = r#"
@fragment
fn main() -> @location(0) vec4f {
    let s = saturate(2.0);
    return vec4f(s, s, s, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        assert!(
            !out.source.contains("saturate"),
            "saturate(2.0) should be folded away: {}",
            out.source
        );
    }

    // Regression: absorbing rewrite must not corrupt vec <op> scalar types.

    /// `vec3<f32> * 0.0` must stay well-typed end-to-end.  Prior to the
    /// "both operands must be scalar Literal" gate, the Binary was rewritten
    /// to the scalar literal `0.0`, corrupting every downstream use.  The
    /// pipeline rolled back each sweep, wasting the sweep budget; with the
    /// gate, either the Binary is left alone or a later stage folds it to a
    /// correctly-typed vec3 zero.  Either outcome keeps the output valid.
    #[test]
    fn e2e_vec_times_scalar_zero_stays_valid() {
        let source = r#"
@fragment
fn main(@location(0) v: vec3<f32>) -> @location(0) vec4<f32> {
    let z = v * 0.0;
    return vec4<f32>(z, 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        // Output must parse back cleanly.
        crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
    }

    #[test]
    fn e2e_vec_and_zero_stays_valid() {
        let source = r#"
@fragment
fn main(@location(0) v: vec4<u32>) -> @location(0) vec4<u32> {
    return v & vec4<u32>(0u);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
    }

    #[test]
    fn e2e_bvec_or_true_stays_valid() {
        let source = r#"
@fragment
fn main(@location(0) v: vec3<f32>) -> @location(0) vec4<f32> {
    let b = vec3<bool>(v.x > 0.0, v.y > 0.0, v.z > 0.0);
    let t = b | vec3<bool>(true);
    return vec4<f32>(f32(t.x), f32(t.y), f32(t.z), 1.0);
}
"#;
        let out =
            crate::run(source, &crate::config::Config::default()).expect("source should compile");
        crate::io::validate_wgsl_text(&out.source).expect("output must reparse");
    }

    /// Unit-level regression: even when called with a vec operand, the
    /// simplify loop must NOT rewrite the Binary into a scalar literal.
    /// The type-safe gate requires both operands be scalar `Literal`; a
    /// `FunctionArgument` (typed as vec3) fails that gate.
    #[test]
    fn simplify_vec_times_zero_does_not_rewrite_to_scalar() {
        let mut arena = naga::Arena::new();
        let zero_f32 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(0.0)),
            Default::default(),
        );
        // FunctionArgument(0) is *typed* as a vector at the module level,
        // but within `fold_local_expressions` we have no type info for it.
        // The gate must therefore err on the side of not rewriting.
        let vec_param = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let mul = arena.append(
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: vec_param,
                right: zero_f32,
            },
            Default::default(),
        );

        let (_folded, simplified) =
            fold_local_expressions(&mut arena, &HashMap::new(), &naga::UniqueArena::new());

        // The Binary must NOT be replaced by the scalar literal.
        assert!(
            matches!(arena[mul], naga::Expression::Binary { .. }),
            "Binary(vec * scalar_0) must remain Binary, got {:?}",
            arena[mul]
        );
        assert_eq!(
            simplified, 0,
            "no simplification should fire for vec * scalar_0"
        );
    }

    // Regression: literal cache correctness (NaN / -0.0 / smallest-handle invariant).
    #[test]
    fn literal_key_distinguishes_negative_zero_from_positive_zero() {
        // `f32::to_bits` gives -0.0 and +0.0 different bit patterns, so the
        // cache must not collapse them.
        assert_ne!(
            literal_key(naga::Literal::F32(0.0)),
            literal_key(naga::Literal::F32(-0.0)),
            "-0.0 and +0.0 must have distinct cache keys"
        );
    }

    #[test]
    fn literal_key_distinguishes_distinct_nans() {
        // Two NaN bit patterns must be distinguishable; the cache must not
        // treat them as equal even though `f32::nan() != f32::nan()`.
        let nan_a = f32::from_bits(0x7FC00000); // quiet NaN
        let nan_b = f32::from_bits(0x7FC00001); // different payload
        assert_ne!(
            literal_key(naga::Literal::F32(nan_a)),
            literal_key(naga::Literal::F32(nan_b)),
            "distinct NaN payloads must have distinct cache keys"
        );
    }

    #[test]
    fn literal_cache_smallest_handle_wins() {
        // Two Literal(1.0f32) appended at different handles; the cache
        // must point to the one with the smaller index so materialize_vector
        // can satisfy its topological-order check more often.
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let h_early = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        // Insert an unrelated expression between the two literals.
        let _spacer = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        let _h_late = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );

        let cache = build_literal_cache(&arena);
        let got = cache
            .get(&literal_key(naga::Literal::F32(1.0)))
            .copied()
            .expect("cache must contain 1.0");
        assert_eq!(
            got.index(),
            h_early.index(),
            "cache must point to the smallest-index literal"
        );
    }

    #[test]
    fn materialize_vector_uses_cache_for_component_lookup() {
        // Verify the hash-cache path still produces a Compose with the right
        // component handles (integration test for the API change).
        let mut types: naga::UniqueArena<naga::Type> = naga::UniqueArena::new();
        let vec3f_ty = types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar::F32,
                },
            },
            Default::default(),
        );
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        let h1 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );
        let h2 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(2.0)),
            Default::default(),
        );
        let h3 = arena.append(
            naga::Expression::Literal(naga::Literal::F32(3.0)),
            Default::default(),
        );
        // `target` must be greater than all component handles.
        let target = arena.append(naga::Expression::FunctionArgument(0), Default::default());

        let lit_cache = build_literal_cache(&arena);
        let type_cache = build_vector_type_cache(&types);
        let out = materialize_vector(
            target,
            &[
                naga::Literal::F32(1.0),
                naga::Literal::F32(2.0),
                naga::Literal::F32(3.0),
            ],
            naga::VectorSize::Tri,
            naga::Scalar::F32,
            &lit_cache,
            &type_cache,
        )
        .expect("materialize_vector must succeed");

        match out {
            naga::Expression::Compose { ty, components } => {
                assert_eq!(ty, vec3f_ty);
                assert_eq!(components, vec![h1, h2, h3]);
            }
            other => panic!("expected Compose, got {other:?}"),
        }
    }

    #[test]
    fn materialize_vector_rejects_component_at_or_after_target() {
        // When the only matching literal is at or after `target`, the
        // topological-safety check must reject the materialization.
        let mut types: naga::UniqueArena<naga::Type> = naga::UniqueArena::new();
        types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar::F32,
                },
            },
            Default::default(),
        );
        let mut arena: naga::Arena<naga::Expression> = naga::Arena::new();
        // target first, then the literal.
        let target = arena.append(naga::Expression::FunctionArgument(0), Default::default());
        arena.append(
            naga::Expression::Literal(naga::Literal::F32(1.0)),
            Default::default(),
        );

        let lit_cache = build_literal_cache(&arena);
        let type_cache = build_vector_type_cache(&types);
        let out = materialize_vector(
            target,
            &[naga::Literal::F32(1.0), naga::Literal::F32(1.0)],
            naga::VectorSize::Bi,
            naga::Scalar::F32,
            &lit_cache,
            &type_cache,
        );
        assert!(
            out.is_none(),
            "component handle >= target must reject materialization"
        );
    }
}
