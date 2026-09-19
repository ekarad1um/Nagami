//! Constant folding: expressions over statically known operands become
//! `Literal` / `Compose` nodes, in `module.global_expressions` and in every
//! function arena.  A folded literal is declarative, so it must leave its
//! `Emit` range; each function fold returns that handle set and the body's
//! ranges are rebuilt around it.
//!
//! Exactly-rounded operations fold bit-exactly; accuracy-tolerant
//! transcendentals (sin/cos/exp/log/pow/...) and float divide / modulo fold
//! to a value inside WGSL's permitted error envelope, not a bit-identical
//! one.  Overflow-, NaN-, and signed-zero-sensitive cases decline.

use rustc_hash::FxHashMap;

use naga::Handle;

use crate::analysis::{Classes, ExprClass};
use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
use crate::ir::rewrite::rebuild_emit_ranges_after_removal;
use crate::passes::expr_util::{
    KeyToken, index_is_static_error, is_integer_zero_literal, is_library_module, lit_key,
    shift_amount_is_static_error,
};
use crate::pipeline::{Pass, PassContext};

/// Whether a clone of `expression` in another arena slot reproduces the same
/// value ([`ExprClass::PURE_TO_CLONE`]): declarative leaves and structural /
/// arithmetic wrappers; not memory reads, derivatives, or statement-attached
/// and cursor-dependent results, whose duplicate `Emit` would re-execute
/// against possibly-different shared state or land disconnected from its
/// producing statement.
fn is_pure_to_clone(expression: &naga::Expression) -> bool {
    ExprClass::node(expression).any(ExprClass::PURE_TO_CLONE)
}

/// Constant folding across globals, functions, and entry points.
#[derive(Debug, Default)]
pub struct ConstFoldPass;

impl Pass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut changed = 0usize;

        // A pure function of `module.types`, which no fold mutates.
        let vector_type_cache = build_vector_type_cache(&module.types);

        changed += fold_global_expressions(module, &vector_type_cache);
        // A mangled library module keeps every `const` under a one-letter
        // name, so folding a reference into its literal only lengthens the
        // text beside a declaration that stays (`dead_branch` resolves
        // `if FLAG` itself); unmangled, the name stays as written and the
        // literal wins.
        let const_literals = if is_library_module(module) && ctx.config.mangle() {
            Default::default()
        } else {
            constant_literals(module)
        };

        crate::ir::visit::for_each_function_taken(module, &mut |body, function, module| {
            // The identity gate keys on the pre-fold graph, not on mid-loop
            // partial rewrites; a fold replaces an expression with one of the
            // same type, so the index bounds sized here hold through the run.
            let census = reference_census(function);
            let emit_ranges = build_emit_range_map(&function.body, function.expressions.len());
            let zero_locals = zero_init_locals(function, &census);
            let access_lens = ctx.access_lens(body, function, module);
            let (folded, simplified) = fold_local_expressions(
                &mut function.expressions,
                &census.counts,
                &emit_ranges,
                &const_literals,
                &zero_locals,
                &access_lens,
                &module.types,
                &vector_type_cache,
            );
            changed += simplified;
            if !folded.is_empty() {
                changed += folded.len();
                rebuild_emit_ranges_after_removal(&mut function.body, &|h| folded.contains(h));
            }
        });

        Ok(changed > 0)
    }
}

// MARK: Global-expression folding

/// Fold `module.global_expressions` into the most compact emittable form.
fn fold_global_expressions(
    module: &mut naga::Module,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> usize {
    let const_inits = module
        .constants
        .iter()
        .map(|(h, c)| (h, c.init))
        .collect::<HandleMap<_, _>>();

    let handles = module
        .global_expressions
        .iter()
        .map(|(h, _)| h)
        .collect::<Vec<_>>();
    let mut changed = 0usize;

    let mut literal_cache = build_literal_cache(&module.global_expressions);

    let mut visiting = HandleSet::default();
    let mut memo: ConstValueMemo = vec![None; module.global_expressions.len()];
    let no_locals = HandleMap::default();
    for handle in handles {
        visiting.clear();
        let value = {
            let ctx = ConstFoldContext {
                arena: &module.global_expressions,
                types: &module.types,
                constants: ConstSource::Inits(&const_inits),
                zero_locals: &no_locals,
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        if let Some(ConstValue::Scalar(literal)) = value {
            if !matches!(module.global_expressions[handle], naga::Expression::Literal(existing) if existing == literal)
            {
                module.global_expressions[handle] = naga::Expression::Literal(literal);
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
            // One spelling for the zero vector here too, so a `const` the
            // source (or a previous minification) declared as `vec2f(0.0)`
            // anchors the sites const_hoist keys as `ZeroValue`.
            let new_expr = if is_zero_vector(components, scalar)
                && let Some(&ty) = vector_type_cache.get(&(size, scalar))
            {
                Some(naga::Expression::ZeroValue(ty))
            } else {
                materialize_vector(
                    handle,
                    components,
                    size,
                    scalar,
                    &literal_cache,
                    vector_type_cache,
                )
            };
            if let Some(new_expr) = new_expr
                && module.global_expressions[handle] != new_expr
            {
                module.global_expressions[handle] = new_expr;
                changed += 1;
            }
        }
    }

    changed
}

/// Named constants whose initializer is one concrete literal, the only form
/// a `Constant` resolves through outside the module-scope fold
/// ([`ConstSource::Literals`]).
pub(crate) type ConstantLiterals = HandleMap<naga::Constant, naga::Literal>;

/// [`ConstantLiterals`] for `module`; O(constants), so a caller asking about
/// several operands builds it once rather than per question.  A plain arena
/// read: naga's front-end evaluates every `const` initializer at parse, and
/// [`fold_global_expressions`] rewrites any initializer chain a pass leaves
/// (`Expression::Constant` links included) to its `Literal`, so "resolves
/// to a scalar" and "is a literal" are one predicate.  Abstract literals
/// are skipped: naga's validator rejects them in a function arena, and a
/// cached one would roll the whole pass back every sweep.
pub(crate) fn constant_literals(module: &naga::Module) -> ConstantLiterals {
    module
        .constants
        .iter()
        .filter_map(|(ch, c)| match module.global_expressions[c.init] {
            naga::Expression::Literal(
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_),
            ) => None,
            naga::Expression::Literal(lit) => Some((ch, lit)),
            _ => None,
        })
        .collect()
}

// MARK: Per-function folding

/// One walk serving both folding gates: the walk, not the tallying, is the
/// cost.
struct RefCensus {
    counts: Vec<u32>,
    /// Deferred rather than tallied: a mention's `counts` entry is only
    /// final once the walk is.
    local_var_exprs: Vec<(Handle<naga::Expression>, Handle<naga::LocalVariable>)>,
    /// `Load`s of a whole local, per local index.
    local_loads: Vec<u32>,
}

/// `counts` is the data-flow reference count per handle: expression-as-child
/// uses, statement operands and results, and local initializers.  `Emit`
/// ranges are NOT counted: they fix execution order, not consumption, and
/// an expression whose only "use" is its Emit entry is dead.  The identity gate
/// tests `== 1` for impure operands whose Emit entry can go after cloning,
/// so `saturating_add` is indistinguishable from exact.
///
/// Counting statement RESULTS is load-bearing: statement-attached expressions
/// (`CallResult`, `AtomicResult`, ...) are uncloneable, and their producer
/// bump pushes any consumed one to `>= 2` so the `== 1` escape never fires.
fn reference_census(function: &naga::Function) -> RefCensus {
    let mut counts = vec![0u32; function.expressions.len()];
    let mut local_var_exprs = Vec::with_capacity(function.local_variables.len());
    let mut local_loads = vec![0u32; function.local_variables.len()];

    fn bump(counts: &mut [u32], h: naga::Handle<naga::Expression>) {
        let i = h.index();
        if i < counts.len() {
            counts[i] = counts[i].saturating_add(1);
        }
    }

    for (handle, expr) in function.expressions.iter() {
        match expr {
            naga::Expression::LocalVariable(l) => local_var_exprs.push((handle, *l)),
            naga::Expression::Load { pointer } => {
                if let naga::Expression::LocalVariable(l) = function.expressions[*pointer] {
                    local_loads[l.index()] = local_loads[l.index()].saturating_add(1);
                }
            }
            _ => {}
        }
        crate::ir::visit::visit_expression_children(expr, |child| bump(&mut counts, child));
    }

    crate::ir::visit::visit_block_expression_handles(
        &function.body,
        /*include_emit_handles=*/ false,
        &mut |h| bump(&mut counts, h),
    );

    // naga restricts a local init to override-expressions (never impure), so
    // counting it only guards against a relaxation letting the gate drop an
    // impure init's Emit entry.
    for (_, lvar) in function.local_variables.iter() {
        if let Some(init) = lvar.init {
            bump(&mut counts, init);
        }
    }

    RefCensus {
        counts,
        local_var_exprs,
        local_loads,
    }
}

/// Locals still holding the zero WGSL initialised them with, mapped to the
/// type that names it: no initialiser, and every mention of the variable
/// being one of the `Load`s that read it, so no store, pointer argument or
/// element pointer reached it first.  A count can only exceed the loads (it
/// also covers named expressions and inits), and an over-count declines.
///
/// The language guarantees this, but no pass carries it: `load_dedup` seeds
/// forwards from an EXPLICIT init, and `dead_branch` deletes the `d = false`
/// stores that made the value knowable.  Unstated, a provably false `a && b`
/// survives as a runtime branch - dead code, and a const-expression demoted
/// to a runtime one, which loses the sign of zero under fast math.
fn zero_init_locals(
    function: &naga::Function,
    census: &RefCensus,
) -> HandleMap<naga::LocalVariable, Handle<naga::Type>> {
    let nlocals = function.local_variables.len();
    if nlocals == 0 {
        return Default::default();
    }
    let mut mentions = vec![0u32; nlocals];
    for &(handle, l) in &census.local_var_exprs {
        mentions[l.index()] = mentions[l.index()].saturating_add(census.counts[handle.index()]);
    }
    function
        .local_variables
        .iter()
        .filter(|(lh, lvar)| {
            let loads = census.local_loads[lh.index()];
            lvar.init.is_none() && loads > 0 && mentions[lh.index()] == loads
        })
        .map(|(lh, lvar)| (lh, lvar.ty))
        .collect()
}

const NO_EMIT: u32 = u32::MAX;

/// Per-handle id of the `Emit` range that materialises it (`NO_EMIT` when
/// none).  Two handles share an id IFF one `Emit` statement covers both, i.e.
/// no statement of any kind sits between them.  Every non-`Emit` statement is
/// a memory write, a barrier, or a control-flow edge, each of which makes
/// moving a read across it unsound, so "same `Emit` range" is exactly
/// "provably no intervening write": the store-aware guard for relocating an
/// impure operand (a `Load`) onto its consumer's slot, without re-walking
/// control flow per fold.  Nested blocks continue the counter, so ids never
/// collide across blocks; the map is read-only until the ranges are rebuilt
/// after the fold.
fn build_emit_range_map(body: &naga::Block, expression_count: usize) -> Vec<u32> {
    let mut map = vec![NO_EMIT; expression_count];
    let mut next_id = 0u32;
    crate::ir::visit::for_each_statement(body, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            let id = next_id;
            next_id += 1;
            for h in range.clone() {
                map[h.index()] = id;
            }
        }
    });
    map
}

/// Roles that make a folded literal a WGSL shader-creation error.
const ROLE_DIVISOR: u8 = 1;
const ROLE_SHIFT_AMOUNT: u8 = 2;
/// Anywhere inside a failable slot, however deep.  [`ROLE_DIVISOR`] and
/// [`ROLE_SHIFT_AMOUNT`] ask whether the literal written HERE is an error;
/// this one asks whether a fold here makes the whole slot const, the
/// question a `Load` poses: `5u / u32(b)` is legal until `b` folds to
/// `false`, and the error lands on the cast, which no fold touched.
const ROLE_IN_FAILABLE_SLOT: u8 = 4;
/// Anywhere inside an operand of an
/// [`is_sign_sensitive_op`](super::expr_util::is_sign_sensitive_op) node,
/// however deep: the value the GPU computes changes when an operand the
/// input read at run time turns const.
const ROLE_IN_SIGN_SENSITIVE_SLOT: u8 = 8;
/// The index of an `Access` whose base
/// [`access_static_lengths`](super::expr_util::access_static_lengths) sizes,
/// the bound being [`Role::index_len`].
const ROLE_INDEX: u8 = 16;
/// Anywhere inside such an index, however deep.  Not
/// [`ROLE_IN_FAILABLE_SLOT`]: the literal walk judges an index slot whole,
/// on its value ([`index_slot_declines`]), because `a[i + 1]` with a
/// zero-init `i` is the common shape and the interior taint rule would
/// decline every fold in it.  The bit exists for the narrowing arms, which
/// never evaluate the root.
const ROLE_IN_INDEX_SLOT: u8 = 32;
/// The index of an `Access` into an override-sized array
/// ([`IndexBound::pending`](super::expr_util::IndexBound)): the driver's
/// check skips it.
const ROLE_INDEX_PENDING: u8 = 64;
/// An unsigned index
/// ([`IndexBound::unsigned_index`](super::expr_util::IndexBound)): into a
/// runtime-sized base a value beyond the evaluator passes.
const ROLE_INDEX_UNSIGNED: u8 = 128;

/// Roles describing a whole subtree, so they survive the descent to children.
const PROPAGATING_ROLES: u8 =
    ROLE_IN_FAILABLE_SLOT | ROLE_IN_SIGN_SENSITIVE_SLOT | ROLE_IN_INDEX_SLOT;
/// Slots whose whole value the text evaluates for an error; a narrowing
/// inside one takes the crossing rule ([`clone_over`]).
const STATIC_ERROR_SLOT_ROLES: u8 = ROLE_IN_FAILABLE_SLOT | ROLE_IN_INDEX_SLOT;

/// What a slot does with a const-expression: the `ROLE_*` bits and, for an
/// index slot, the bound the index is judged against.  One value per
/// handle, so every consumer of "is this literal an error here" - the fold,
/// the narrowing clone, the driver's post-pass check, the inliner's and the
/// forwarders' per-site declines, the generator's const-hazard binding -
/// reads the same description.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Role {
    bits: u8,
    /// [`IndexBound::len`](super::expr_util::IndexBound) under
    /// [`ROLE_INDEX`].
    index_len: u32,
}

impl Role {
    const NONE: Role = Role {
        bits: 0,
        index_len: 0,
    };

    const fn bits(bits: u8) -> Role {
        Role { bits, index_len: 0 }
    }

    /// The index slot of an `Access` into a base
    /// [`access_static_lengths`](super::expr_util::access_static_lengths)
    /// bounds.
    pub(crate) const fn index(bound: super::expr_util::IndexBound) -> Role {
        Role {
            bits: ROLE_INDEX
                | ROLE_IN_INDEX_SLOT
                | if bound.pending { ROLE_INDEX_PENDING } else { 0 }
                | if bound.unsigned_index {
                    ROLE_INDEX_UNSIGNED
                } else {
                    0
                },
            index_len: bound.len,
        }
    }

    fn has(self, bit: u8) -> bool {
        self.bits & bit != 0
    }

    /// Every bit of `other` and at least as tight an index bound.
    fn covers(self, other: Role) -> bool {
        self.bits & other.bits == other.bits
            && (!other.has(ROLE_INDEX)
                || (self.index_len != 0 && self.index_len <= other.index_len)
                || other.index_len == 0)
    }

    /// Both slots' facts: an index shared by two bases is judged against
    /// the tighter bound (an unbounded base adds only the negative check,
    /// which every bound includes).
    fn merge(self, other: Role) -> Role {
        let index_len = match (self.index_len, other.index_len) {
            (0, n) | (n, 0) => n,
            (a, b) => a.min(b),
        };
        Role {
            bits: self.bits | other.bits,
            index_len,
        }
    }

    fn propagating(self) -> Role {
        Role::bits(self.bits & PROPAGATING_ROLES)
    }
}

/// Per-handle roles: the right operand of an integer `/` `%` or of a shift,
/// and the index of an `Access` whose base `access_lens` sizes, reached
/// through `Splat` / `Compose` since any offending lane condemns a
/// componentwise op.  Folding an offender into one of these cannot survive
/// post-pass validation, and the rollback discards every OTHER fold of the
/// same run - permanently, the pass being deterministic - so declining is
/// free.
///
/// [`PROPAGATING_ROLES`] ride the same walk but descend through every
/// operand: const-ness propagates up through any pure operation, not just
/// the two that carry a lane.  That is why a float `*` `/` `%` or unary `-`
/// seeds at all - its role marks the whole operand subtree, not the handle.
fn static_error_roles(
    arena: &naga::Arena<naga::Expression>,
    access_lens: &[Option<super::expr_util::IndexBound>],
) -> Vec<Role> {
    let mut stack: Vec<(Handle<naga::Expression>, Role)> = Vec::new();
    for (h, expr) in arena.iter() {
        match expr {
            naga::Expression::Binary { op, left, right } => match op {
                // Both operands of every sign-sensitive operator, untyped as
                // `is_sign_sensitive_op` is: the fold's float-literal test
                // keeps an integer slot inert.
                naga::BinaryOperator::Divide => {
                    stack.push((*right, Role::bits(ROLE_DIVISOR | ROLE_IN_FAILABLE_SLOT)));
                    stack.push((*left, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                    stack.push((*right, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                }
                naga::BinaryOperator::Modulo => {
                    stack.push((*right, Role::bits(ROLE_DIVISOR | ROLE_IN_FAILABLE_SLOT)));
                    stack.push((*left, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                    stack.push((*right, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                }
                naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight => {
                    stack.push((
                        *right,
                        Role::bits(ROLE_SHIFT_AMOUNT | ROLE_IN_FAILABLE_SLOT),
                    ));
                }
                naga::BinaryOperator::Multiply => {
                    stack.push((*left, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                    stack.push((*right, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT)));
                }
                _ => {}
            },
            naga::Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr,
            } => stack.push((*expr, Role::bits(ROLE_IN_SIGN_SENSITIVE_SLOT))),
            naga::Expression::Access { index, .. } => {
                if let Some(Some(bound)) = access_lens.get(h.index()) {
                    stack.push((*index, Role::index(*bound)));
                }
            }
            _ => {}
        }
    }
    // Nothing seeded: the empty vector costs no allocation and reads
    // through `role_of` as no role.
    if stack.is_empty() {
        return Vec::new();
    }
    let mut roles = vec![Role::NONE; arena.len()];
    while let Some((h, role)) = stack.pop() {
        // Every bit, not any: a push carries two roles, so being marked
        // INSIDE a slot must not swallow the push that makes it BE one.
        if roles[h.index()].covers(role) {
            continue;
        }
        roles[h.index()] = roles[h.index()].merge(role);
        match &arena[h] {
            naga::Expression::Splat { value, .. } => stack.push((*value, role)),
            naga::Expression::Compose { components, .. } => {
                stack.extend(components.iter().map(|&c| (c, role)));
            }
            _ => {
                crate::ir::visit::visit_expression_children(&arena[h], |child| {
                    stack.push((child, role.propagating()));
                });
            }
        }
    }
    roles
}

/// Handles with a [`zero_init_locals`] `Load` under them: the only
/// const-ness this pass adds that naga's front-end could not already see,
/// and so the only kind whose fold can make a slot newly const.  Children
/// precede their parent in a naga arena, so one forward pass fills it;
/// empty when no local qualifies, so read it through [`is_tainted`].
fn zero_local_taint(
    arena: &naga::Arena<naga::Expression>,
    zero_locals: &HandleMap<naga::LocalVariable, Handle<naga::Type>>,
) -> Vec<bool> {
    if zero_locals.is_empty() {
        return Vec::new();
    }
    let mut taint = vec![false; arena.len()];
    for (handle, expr) in arena.iter() {
        let mut tainted = is_zero_local_read(arena, expr, zero_locals);
        if !tainted {
            crate::ir::visit::visit_expression_children(expr, |child| {
                tainted |= taint[child.index()];
            });
        }
        taint[handle.index()] = tainted;
    }
    taint
}

/// Taint bit for `h`; an empty vector is "no zero-init local here".
fn is_tainted(taint: &[bool], h: Handle<naga::Expression>) -> bool {
    taint.get(h.index()).copied().unwrap_or(false)
}

/// Every handle inside an index slot this run must leave alone: the slot's
/// whole value, as the folds would leave it, is a static error (past the
/// base, below zero), or is beyond the evaluator while a zero-init local's
/// read ([`zero_local_taint`]) sits inside it and nothing else keeps it
/// runtime ([`const_after_zero_fold`]: `a[select(z + 4u, 0u, c)]` with `c`
/// runtime can never become a const-expression, whatever `z` folds to).
/// Judged at the root because the per-handle rule cannot see it: in
/// `a[i + 10]` with `i` reading zero, `0` at `i` is in bounds and `10` at
/// `+` declines, yet the text evaluates the sum once `i` alone has folded.
/// An in-bounds slot folds freely, whatever it holds.
fn index_slot_declines(
    ctx: &ConstFoldContext<'_>,
    access_lens: &[Option<super::expr_util::IndexBound>],
    zero_local_taint: &[bool],
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> HandleSet<naga::Expression> {
    let mut declined = HandleSet::default();
    let mut stack = Vec::new();
    let mut const_after_zero: Option<Vec<bool>> = None;
    for (h, expr) in ctx.arena.iter() {
        let naga::Expression::Access { index, .. } = expr else {
            continue;
        };
        let Some(Some(bound)) = access_lens.get(h.index()) else {
            continue;
        };
        visiting.clear();
        let bad = match resolve_const_value(*index, ctx, visiting, memo) {
            Some(ConstValue::Scalar(lit)) => index_is_static_error(bound.len, &lit),
            Some(ConstValue::Vector { .. }) => false,
            None => {
                is_tainted(zero_local_taint, *index)
                    && const_after_zero
                        .get_or_insert_with(|| const_after_zero_fold(ctx.arena, ctx.zero_locals))
                        [index.index()]
            }
        };
        if !bad {
            continue;
        }
        stack.push(*index);
        while let Some(h) = stack.pop() {
            if declined.insert(h) {
                crate::ir::visit::visit_expression_children(&ctx.arena[h], |c| stack.push(c));
            }
        }
    }
    declined
}

/// [`const_at_entry`] with a [`zero_init_locals`] read counted as the const
/// leaf the fold makes of it: what can turn const once every such read is
/// folded.
fn const_after_zero_fold(
    arena: &naga::Arena<naga::Expression>,
    zero_locals: &HandleMap<naga::LocalVariable, Handle<naga::Type>>,
) -> Vec<bool> {
    super::expr_util::const_cones(arena, |_, expr, _| {
        is_zero_local_read(arena, expr, zero_locals).then_some(true)
    })
}

/// A `Load` of a [`zero_init_locals`] local: the one runtime read this
/// pass folds.
fn is_zero_local_read(
    arena: &naga::Arena<naga::Expression>,
    expr: &naga::Expression,
    zero_locals: &HandleMap<naga::LocalVariable, Handle<naga::Type>>,
) -> bool {
    matches!(expr, naga::Expression::Load { pointer }
        if matches!(arena[*pointer], naga::Expression::LocalVariable(l)
            if zero_locals.get(l).is_some()))
}

/// Role of `h`; an empty vector - [`static_error_roles`] seeded nothing - is
/// no role.
fn role_of(roles: &[Role], h: Handle<naga::Expression>) -> Role {
    roles.get(h.index()).copied().unwrap_or(Role::NONE)
}

/// `literal` in a slot with `role` is a shader-creation error.
fn literal_is_static_error(role: Role, literal: naga::Literal) -> bool {
    (role.has(ROLE_DIVISOR) && is_integer_zero_literal(&literal))
        || (role.has(ROLE_SHIFT_AMOUNT) && shift_amount_is_static_error(&literal))
        || (role.has(ROLE_INDEX) && index_is_static_error(role.index_len, &literal))
}

/// Per-handle "already reads as a const-expression"
/// ([`ExprClass::CONST_CONE`]), over the arena as THIS RUN found it: a slot
/// true here is const whatever this run does, so folding inside it crosses
/// nothing.  Not the ORIGINAL module's const-ness - an earlier pass may have
/// supplied some, which this run then reads as given, so const-ness that
/// accretes across passes (each crossing nothing on its own) is the one gap;
/// closing it needs the original const-ness carried through the whole
/// pipeline to emission.
fn const_at_entry(arena: &naga::Arena<naga::Expression>) -> Classes {
    Classes::of(arena)
}

/// Role a failable operator puts on its right operand.
pub(crate) fn failable_op_role(op: naga::BinaryOperator) -> Option<Role> {
    match op {
        naga::BinaryOperator::Divide | naga::BinaryOperator::Modulo => {
            Some(Role::bits(ROLE_DIVISOR))
        }
        naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight => {
            Some(Role::bits(ROLE_SHIFT_AMOUNT))
        }
        _ => None,
    }
}

/// Every static-error slot of `arena` with its role: the right operand of a
/// `/` `%` `<<` `>>`, and the index of an `Access` whose base `access_lens`
/// sizes.
pub(crate) fn static_error_slots(
    arena: &naga::Arena<naga::Expression>,
    access_lens: &[Option<super::expr_util::IndexBound>],
) -> Vec<(Handle<naga::Expression>, Role)> {
    let mut slots = Vec::new();
    for (h, expr) in arena.iter() {
        match expr {
            naga::Expression::Binary { op, right, .. } => {
                if let Some(role) = failable_op_role(*op) {
                    slots.push((*right, role));
                }
            }
            naga::Expression::Access { index, .. } => {
                if let Some(Some(bound)) = access_lens.get(h.index()) {
                    slots.push((*index, Role::index(*bound)));
                }
            }
            _ => {}
        }
    }
    slots
}

/// The first static-error slot in `arena` that reads as a const-expression
/// evaluating to a shader-creation error - integer divide / modulo by zero,
/// a shift amount at or past the bit width, an index outside its base -
/// rendered for a diagnostic.
///
/// naga's validator checks the LITERAL spelling only (`5u / 0u`, `a[5]`), so
/// the const-EXPRESSION one (`5u / (1u - 1u)`, `a[4 + 1]`) validates yet has
/// no valid WGSL text at all - unreported, it would drop the run to lexical
/// compaction at emission.  A parsed module cannot carry the shape, so a
/// pass always manufactured it (inlining an argument that zeroes a divisor,
/// forwarding a stored literal into an index); reporting it through the
/// validator makes the driver's existing rollback undo that pass.
fn arena_static_error_slot(
    arena: &naga::Arena<naga::Expression>,
    access_lens: &[Option<super::expr_util::IndexBound>],
    types: &naga::UniqueArena<naga::Type>,
    const_literals: &HandleMap<naga::Constant, naga::Literal>,
) -> Option<String> {
    let slots = static_error_slots(arena, access_lens);
    // Slots are rare enough: an arena without one allocates nothing more.
    let mut memo: Option<ConstValueMemo> = None;
    let mut visiting = HandleSet::default();
    let no_zero_locals = HandleMap::default();
    for (operand, role) in slots {
        if role.has(ROLE_INDEX_PENDING) {
            continue;
        }
        let memo = memo.get_or_insert_with(|| vec![None; arena.len()]);
        visiting.clear();
        let ctx = ConstFoldContext {
            arena,
            types,
            constants: ConstSource::Literals(const_literals),
            // A never-written local is NOT a const-expression to the WGSL
            // front-end; only what it can see itself counts here.
            zero_locals: &no_zero_locals,
        };
        if operand_evaluates_to_static_error(operand, role, &ctx, &mut visiting, memo) {
            // Spelled out, not formatted: `{:?}` on a naga IR type links its
            // `Debug` impls, the single biggest size lever this crate has.
            return Some(
                if role.has(ROLE_DIVISOR) {
                    "divisor is a const-expression evaluating to zero, which \
                     WGSL rejects at shader creation"
                } else if role.has(ROLE_SHIFT_AMOUNT) {
                    "shift amount is a const-expression at or past the \
                     operand's bit width, which WGSL rejects at shader creation"
                } else {
                    "index is a const-expression outside its base's bounds, \
                     which WGSL rejects at shader creation"
                }
                .to_string(),
            );
        }
    }
    None
}

/// `right`, a static-error slot ([`static_error_slots`]) with `role`, reads
/// as a const-expression evaluating to a shader-creation error.
fn operand_evaluates_to_static_error(
    right: Handle<naga::Expression>,
    role: Role,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> bool {
    match resolve_const_value(right, ctx, visiting, memo) {
        Some(ConstValue::Scalar(lit)) => literal_is_static_error(role, lit),
        // One offending lane condemns the componentwise operation.
        Some(ConstValue::Vector { ref components, .. }) => components
            .iter()
            .any(|&lane| literal_is_static_error(role, lane)),
        None => false,
    }
}

/// [`arena_static_error_slot`]'s question for one operand the caller knows
/// reads as a const-expression: whether `operand` in `arena` evaluates to a
/// shader-creation error in a slot of `role`.  A pass asks it of a scratch
/// copy of the operand it is about to manufacture, which turns the driver's
/// whole-pass rollback into a per-site decline; the generator asks it of an
/// index whose lanes it cannot compute, to decide a binding.  A divisor or
/// shift amount the evaluator cannot model (`n << (reverseBits(v) & 31u)`)
/// passes, as it does after any rewrite: the generator's const-hazard
/// binding keeps it runtime - unless naga's front-end folds it
/// ([`folds_upstream_only`]).  An INDEX it cannot model reads as an error,
/// tint evaluating what neither this evaluator nor naga's does
/// (`unpack4xU8(..).w`), except an unsigned one into a runtime-sized base,
/// which has no error to reach.
pub(crate) fn operand_is_static_error(
    types: &naga::UniqueArena<naga::Type>,
    const_literals: &ConstantLiterals,
    arena: &naga::Arena<naga::Expression>,
    role: Role,
    operand: Handle<naga::Expression>,
) -> bool {
    let no_zero_locals = HandleMap::default();
    let ctx = ConstFoldContext {
        arena,
        types,
        constants: ConstSource::Literals(const_literals),
        zero_locals: &no_zero_locals,
    };
    let mut visiting = HandleSet::default();
    let mut memo = vec![None; arena.len()];
    match resolve_const_value(operand, &ctx, &mut visiting, &mut memo) {
        Some(ConstValue::Scalar(lit)) => literal_is_static_error(role, lit),
        Some(ConstValue::Vector { ref components, .. }) => components
            .iter()
            .any(|&lane| literal_is_static_error(role, lane)),
        None if role.has(ROLE_INDEX) => !(role.index_len == 0 && role.has(ROLE_INDEX_UNSIGNED)),
        None => folds_upstream_only(&ctx, operand, &mut visiting, &mut memo),
    }
}

/// Whether the const-expression at `root` holds a node naga's front-end
/// evaluates that [`resolve_const_value`] does not: the vector-reducing
/// builtins, `any` / `all`, a composite `const`, a vector conversion.  No
/// `let` keeps such a cone runtime - naga's lowerer folds the initializer
/// at the re-parse and its validator judges the literal - so a gate that
/// cannot evaluate it must decline.  The complement, where
/// the generator's binding holds, is the `unimplemented` arm of naga's
/// `ConstantEvaluator::math` (`modf`/`frexp`/`ldexp`, `mix`/`reflect`/
/// `refract`/`faceForward`/`smoothstep`, `transpose`/`determinant`/
/// `inverse`, `quantizeToF16`, `extractBits`/`insertBits`, `pack*`/`unpack*`)
/// plus `bitcast`; a naga upgrade that folds one of them ships a compacted
/// module (the self-check catches it) until it is listed here.
fn folds_upstream_only(
    ctx: &ConstFoldContext<'_>,
    root: Handle<naga::Expression>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> bool {
    use naga::MathFunction as M;
    let mut seen = HandleSet::default();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        if !seen.insert(h) {
            continue;
        }
        let expr = &ctx.arena[h];
        let upstream = match expr {
            naga::Expression::Math { fun, .. } => matches!(
                fun,
                M::Dot
                    | M::Dot4I8Packed
                    | M::Dot4U8Packed
                    | M::Cross
                    | M::Length
                    | M::Distance
                    | M::Normalize
            ),
            naga::Expression::Relational { .. } => true,
            naga::Expression::Constant(c) => match ctx.constants {
                ConstSource::Literals(literals) => !literals.contains_key(*c),
                ConstSource::Inits(_) => false,
            },
            // A scalar conversion is modelled; a vector one is not.
            naga::Expression::As { expr, .. } => {
                visiting.clear();
                matches!(
                    resolve_const_value(*expr, ctx, visiting, memo),
                    Some(ConstValue::Vector { .. })
                )
            }
            _ => false,
        };
        if upstream {
            return true;
        }
        crate::ir::visit::visit_expression_children(expr, |c| stack.push(c));
    }
    false
}

/// Whether the const-expression at `root` carries a float negative zero in
/// any lane: the one value a sign-sensitive slot changes by turning const
/// (Dawn on Metal flushes a `-0.0` LITERAL to `+0.0` and keeps the sign of a
/// runtime negation).  `None` when the evaluator cannot compute it - a gate
/// then declines, since tint evaluates every const-expression, this
/// evaluator's gaps included.
pub(crate) fn evaluates_to_negative_zero(
    types: &naga::UniqueArena<naga::Type>,
    const_literals: &ConstantLiterals,
    arena: &naga::Arena<naga::Expression>,
    root: Handle<naga::Expression>,
) -> Option<bool> {
    let no_zero_locals = HandleMap::default();
    let ctx = ConstFoldContext {
        arena,
        types,
        constants: ConstSource::Literals(const_literals),
        zero_locals: &no_zero_locals,
    };
    let mut visiting = HandleSet::default();
    let mut memo = vec![None; arena.len()];
    let value = resolve_const_value(root, &ctx, &mut visiting, &mut memo)?;
    Some(match value {
        ConstValue::Scalar(lit) => crate::passes::expr_util::is_negative_zero_literal(&lit),
        ConstValue::Vector { ref components, .. } => components
            .iter()
            .any(crate::passes::expr_util::is_negative_zero_literal),
    })
}

/// [`arena_static_error_slot`] over every arena in `module`.
pub(crate) fn module_static_error_slot(module: &naga::Module) -> Option<String> {
    let const_literals = constant_literals(module);
    let functions = module
        .functions
        .iter()
        .map(|(_, f)| (f.name.as_deref(), f))
        .chain(
            module
                .entry_points
                .iter()
                .map(|ep| (Some(ep.name.as_str()), &ep.function)),
        );
    for (name, function) in functions {
        let access_lens = super::expr_util::access_static_lengths(function, module);
        if let Some(detail) = arena_static_error_slot(
            &function.expressions,
            &access_lens,
            &module.types,
            &const_literals,
        ) {
            let where_ =
                name.map_or_else(|| "a function".to_string(), |n| format!("function `{n}`"));
            return Some(format!("{where_}: {detail}"));
        }
    }
    // Module scope holds const-expressions only, every one naga's front-end
    // evaluated at parse, so an `Access` there is never a manufactured slot.
    arena_static_error_slot(
        &module.global_expressions,
        &[],
        &module.types,
        &const_literals,
    )
    .map(|detail| format!("module scope: {detail}"))
}

/// Clone `source` over `target`, declining (and writing nothing) when that
/// would leave an offender where a runtime operation stood.  Every arm that
/// narrows an expression to one of its operands writes through here, so the
/// guard is structural rather than a rule each new arm must remember.
///
/// Narrowing is the other way a slot crosses runtime -> const, and no arm
/// evaluates the slot's root as the literal walk does, so both the
/// sign-sensitive and the static-error slots take the crossing rule here:
/// `a[select(x, x, c) + 1]` and `n << (select(x, x, c) + 1)` with `x` const
/// are runtime in the input and a const-expression the text evaluates once
/// `x` stands alone.  `select(x, x, c)` (dropping a runtime condition) and
/// `c && false` / `c || true` (dropping a runtime operand) are the arms that
/// cross; the others match a literal in the SIBLING slot, so a const
/// `source` had already made `target` const at entry.
fn clone_over(
    arena: &mut naga::Arena<naga::Expression>,
    roles: &[Role],
    const_at_entry: Option<&Classes>,
    target: Handle<naga::Expression>,
    source: Handle<naga::Expression>,
) -> bool {
    let role = role_of(roles, target);
    if matches!(arena[source], naga::Expression::Literal(lit)
        if literal_is_static_error(role, lit))
    {
        return false;
    }
    // Untyped like the role: an integer slot has no signed zero, but one
    // rule beats two that can drift.
    if role.has(ROLE_IN_SIGN_SENSITIVE_SLOT | STATIC_ERROR_SLOT_ROLES)
        && is_const_at_entry(const_at_entry, source)
        && !is_const_at_entry(const_at_entry, target)
    {
        return false;
    }
    arena[target] = arena[source].clone();
    true
}

/// Entry const-ness of `h`.  An absent entry - a handle this run appended,
/// or the `None` an arena with no sign-sensitive or static-error slot gets -
/// reads as runtime, which declines.
fn is_const_at_entry(const_at_entry: Option<&Classes>, h: Handle<naga::Expression>) -> bool {
    const_at_entry
        .and_then(|classes| classes.get(h))
        .is_some_and(|class| class.any(ExprClass::CONST_CONE))
}

/// Fold `arena` in place, returning the handles that must leave their `Emit`
/// ranges and the number of simplifications.  `refcounts` and `emit_ranges`
/// together gate cloning an impure operand in the identity / involution
/// arms: only when the folding expression is its sole consumer AND shares
/// its `Emit` range, so the operand dies (its Emit entry is dropped, no
/// double execution) and the relocated read crosses no statement.
#[allow(clippy::too_many_arguments)]
fn fold_local_expressions(
    arena: &mut naga::Arena<naga::Expression>,
    refcounts: &[u32],
    emit_ranges: &[u32],
    const_literals: &HandleMap<naga::Constant, naga::Literal>,
    zero_locals: &HandleMap<naga::LocalVariable, Handle<naga::Type>>,
    access_lens: &[Option<super::expr_util::IndexBound>],
    types: &naga::UniqueArena<naga::Type>,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> (HandleSet<naga::Expression>, usize) {
    let mut handles = Vec::with_capacity(arena.len());
    handles.extend(arena.iter().map(|(h, _)| h));
    // Sound to compute once: the loops below replace whole `arena[handle]`
    // values, never a `Binary`'s operand slots, so a role can retire but
    // never appear.
    let roles = static_error_roles(arena, access_lens);
    // Compute-once: a fold retires a taint but never creates one, and a stale
    // value only declines.  Each gates on its OWN role bits: `roles` is
    // non-empty once anything at all is seeded, a lone `*` included.
    let has_role = |bits: u8| roles.iter().any(|r| r.has(bits));
    let zero_local_taint = if has_role(ROLE_IN_FAILABLE_SLOT | ROLE_INDEX) {
        zero_local_taint(arena, zero_locals)
    } else {
        Vec::new()
    };
    let const_at_entry = has_role(ROLE_IN_SIGN_SENSITIVE_SLOT | STATIC_ERROR_SLOT_ROLES)
        .then(|| const_at_entry(arena));
    let const_at_entry = const_at_entry.as_ref();
    let mut folded = HandleSet::default();

    let mut literal_cache = build_literal_cache(arena);

    let mut simplify_count = 0usize;
    let mut visiting = HandleSet::default();
    let mut memo: ConstValueMemo = vec![None; arena.len()];
    let declined_slots = if has_role(ROLE_INDEX) {
        let ctx = ConstFoldContext {
            arena: &*arena,
            types,
            constants: ConstSource::Literals(const_literals),
            zero_locals,
        };
        index_slot_declines(
            &ctx,
            access_lens,
            &zero_local_taint,
            &mut visiting,
            &mut memo,
        )
    } else {
        HandleSet::default()
    };
    for handle in handles.iter().copied() {
        if declined_slots.contains(handle) {
            continue;
        }
        visiting.clear();
        let value = {
            let ctx = ConstFoldContext {
                arena: &*arena,
                types,
                constants: ConstSource::Literals(const_literals),
                zero_locals,
            };
            resolve_const_value(handle, &ctx, &mut visiting, &mut memo)
        };

        let role = role_of(&roles, handle);
        // Const-ness the front-end could see was already there when it
        // validated this slot, so folding inside one invents no error.  A
        // zero-local's was invisible to it, and declining only the `Load`
        // where it enters leaves the INTERIOR free to fold and make the slot
        // const anyway: `5u / (s + 1u - 1u)` would ship `5u / (1u - 1u)`, a
        // manufactured slot `arena_static_error_slot` rolls the whole pass
        // back for.  Checked before the match because a vector local
        // reaches the same cliff through the `Vector` arm.
        if role.has(ROLE_IN_FAILABLE_SLOT) && is_tainted(&zero_local_taint, handle) {
            continue;
        }
        match value {
            Some(ConstValue::Scalar(literal)) => {
                // An abstract result (both operands abstract) trips naga's
                // `WidthError::Abstract` in a function arena and would roll
                // the whole pass back.
                if matches!(
                    literal,
                    naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                ) {
                    continue;
                }
                // Same rollback, different cause; see `static_error_roles`.
                if literal_is_static_error(role, literal) {
                    continue;
                }
                // Manufacturing a `-0.0` where the shader computed one
                // changes what the GPU reads back.  A handle already holding
                // this literal is not rewritten below, so the input's own
                // survives.
                if crate::passes::expr_util::is_negative_zero_literal(&literal) {
                    continue;
                }
                // One level up: a float that only becomes const-foldable
                // here hands the enclosing `is_sign_sensitive_op` node a
                // const-expression the input did not have - any float, since
                // the zero can be the already-const SIBLING.
                if role.has(ROLE_IN_SIGN_SENSITIVE_SLOT)
                    && crate::passes::expr_util::is_float_literal(&literal)
                    && !is_const_at_entry(const_at_entry, handle)
                {
                    continue;
                }
                if !matches!(arena[handle], naga::Expression::Literal(existing) if existing == literal)
                {
                    arena[handle] = naga::Expression::Literal(literal);
                    note_literal_in_cache(&mut literal_cache, handle, literal);
                    folded.insert(handle);
                }
            }
            Some(ConstValue::Vector {
                ref components,
                size,
                scalar,
            }) => {
                // One offending lane condemns the whole componentwise op.
                if components
                    .iter()
                    .any(|&lane| literal_is_static_error(role, lane))
                {
                    continue;
                }
                // One manufactured `-0.0` lane is enough, as for a scalar.
                if components
                    .iter()
                    .any(crate::passes::expr_util::is_negative_zero_literal)
                {
                    continue;
                }
                if role.has(ROLE_IN_SIGN_SENSITIVE_SLOT)
                    && components
                        .iter()
                        .any(crate::passes::expr_util::is_float_literal)
                    && !is_const_at_entry(const_at_entry, handle)
                {
                    continue;
                }
                // A `Compose` needs an `Emit` range, so only an emittable
                // original materialises, and one that already spells these
                // literals is left alone: re-pointing its components at the
                // cache's canonical handles would declare a change that
                // moves nothing and leaves orphans for `compact` to cull, a
                // sweep spent on nothing.
                if !crate::passes::expr_util::expression_needs_emit(&arena[handle]) {
                    continue;
                }
                // The zero vector has one spelling, naga's `ZeroValue`:
                // built from nothing and declarative like a literal, so it
                // leaves its `Emit` range and needs no lane literal in the
                // arena (a zero-init local's read once folded only while its
                // deleted initializer's lanes were still lying around); it
                // is what `vec2f()` re-parses as, so a `Compose` or `Splat`
                // of zeros the source spelled is normalised to it as well,
                // and const_hoist keys the value by that one spelling.
                if is_zero_vector(components, scalar)
                    && let Some(&ty) = vector_type_cache.get(&(size, scalar))
                {
                    arena[handle] = naga::Expression::ZeroValue(ty);
                    folded.insert(handle);
                } else if let Some(new_expr) = materialize_vector(
                    handle,
                    components,
                    size,
                    scalar,
                    &literal_cache,
                    vector_type_cache,
                ) && !compose_spells(arena, handle, &new_expr)
                {
                    arena[handle] = new_expr;
                    simplify_count += 1;
                }
            }
            None => {}
        }
    }

    // Identity (`x * 1 -> x`), absorbing (`x * 0 -> 0`), involution
    // (`-(-x) -> x`), `!(a == b) -> a != b` and `select(x, x, c) -> x`.
    // A rewrite reports what it freed; one tail retires the handle when it
    // came out declarative, and the freed handles with it.
    let ownership = Ownership {
        refcounts,
        emit_ranges,
    };
    for handle in handles {
        if declined_slots.contains(handle) {
            continue;
        }
        let freed = match arena[handle] {
            naga::Expression::Binary { op, left, right } => 'rules: {
                // Cloning the literal over a Binary whose type follows naga
                // broadcasting (`vec3<f32> * 0.0` is a vec3) mis-types a
                // vector slot: an absorb is safe only for `&&` / `||`
                // (pinned to `bool x bool -> bool`) or when both operands
                // are literals (a scalar result); the latter admits what
                // `eval_binary` leaves unfolded (an overflow, a `%` its
                // lowering disputes), where the classes decline anything
                // sign-sensitive.  An identity is type-safe by construction:
                // the neutral element leaves `other` with the broadcast
                // result type.  The absorbing row goes first, and a declined
                // clone falls through to the identity row.
                let absorb_typed = matches!(
                    op,
                    naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr
                ) || (matches!(arena[left], naga::Expression::Literal(_))
                    && matches!(arena[right], naga::Expression::Literal(_)));
                for keep in [Keep::Literal, Keep::Other] {
                    let Some((literal, other)) = rule_match(op, keep, left, right, arena) else {
                        continue;
                    };
                    let (source, freed) = match keep {
                        Keep::Literal if absorb_typed => (literal, Vec::new()),
                        Keep::Literal => continue,
                        Keep::Other => match ownership.clonable(arena, other, handle, &[]) {
                            Some(freed) => (other, freed),
                            None => continue,
                        },
                    };
                    if clone_over(arena, &roles, const_at_entry, handle, source) {
                        break 'rules Some(freed);
                    }
                }
                None
            }
            naga::Expression::Unary { op, expr } => 'unary: {
                // Involution: the intermediate Unary at `expr` is the path
                // hoisted past; it leaves its `Emit` when solely owned, and
                // an impure `inner` additionally needs it solely owned,
                // since `expr`'s residual slot still references `inner`
                // until compact runs.
                if let naga::Expression::Unary {
                    op: inner_op,
                    expr: inner,
                } = arena[expr]
                    && op == inner_op
                    && let Some(freed) = ownership.clonable(arena, inner, handle, &[expr])
                    && clone_over(arena, &roles, const_at_entry, handle, inner)
                {
                    break 'unary Some(freed);
                }
                // `!(a == b)` -> `a != b`: equality negation is exact for
                // every type, NaN included, unlike ordered relations.  Done
                // in IR so the `Binary` emit path parenthesises correctly
                // (an emit-time fold mis-parenthesises a nested comparison),
                // and only when this `!` solely owns the comparison; a shared
                // one emits as `!name`.
                if op == naga::UnaryOperator::LogicalNot
                    && let naga::Expression::Binary {
                        op: cmp,
                        left,
                        right,
                    } = arena[expr]
                    && let Some(flipped) = flip_equality(cmp)
                    && ownership.sole(expr)
                {
                    arena[handle] = naga::Expression::Binary {
                        op: flipped,
                        left,
                        right,
                    };
                    break 'unary Some(vec![expr]);
                }
                None
            }
            // `x` has two consumers by construction, so only the pure escape
            // applies.
            naga::Expression::Select { accept, reject, .. }
                if accept == reject && is_pure_to_clone(&arena[accept]) =>
            {
                clone_over(arena, &roles, const_at_entry, handle, accept).then(Vec::new)
            }
            _ => None,
        };
        if let Some(freed) = freed {
            simplify_count += 1;
            if !crate::passes::expr_util::expression_needs_emit(&arena[handle]) {
                folded.insert(handle);
            }
            folded.extend(freed);
        }
    }

    (folded, simplify_count)
}

/// `==` <-> `!=`; ordered relations are excluded because their negation is
/// NaN-unsafe for floats and this pass does not resolve operand types.
fn flip_equality(op: naga::BinaryOperator) -> Option<naga::BinaryOperator> {
    match op {
        naga::BinaryOperator::Equal => Some(naga::BinaryOperator::NotEqual),
        naga::BinaryOperator::NotEqual => Some(naga::BinaryOperator::Equal),
        _ => None,
    }
}

// MARK: Constant value resolution

// Matrices are deliberately absent: rare in constant contexts, not worth the
// case analysis.
#[derive(Debug, Clone, PartialEq)]
enum ConstValue {
    Scalar(naga::Literal),
    Vector {
        components: Vec<naga::Literal>,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    },
}

impl ConstValue {
    fn as_scalar(&self) -> Option<naga::Literal> {
        match self {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }
}

/// Memo for [`resolve_const_value`], indexed by handle: outer `None` = not
/// computed, inner = the full result including "not a constant".  Failures
/// are cached too, since the dominant cost is re-walking long runtime chains
/// that fail at every ancestor (quadratic on the deep chains our own passes
/// build).  It stays valid across the folding loops' rewrites because those
/// only substitute a `Literal` / `Compose` of the value already resolved, and
/// it is safe even on cyclic IR: a cycle-guard hit returns `None` before any
/// store and no arm turns a child failure into `Some`, so a wrong `Some` is
/// never cached.
type ConstValueMemo = Vec<Option<Option<ConstValue>>>;

/// Where `Expression::Constant` resolves.  One enum rather than a trait with
/// two impls: the resolver below is a seven-function mutual recursion, and a
/// generic context monomorphised it twice for a single field's worth of
/// difference.
enum ConstSource<'a> {
    /// Module scope: a constant resolves through its initializer, which lives
    /// in the SAME arena, so the recursion continues and shares the memo.
    Inits(&'a HandleMap<naga::Constant, naga::Handle<naga::Expression>>),
    /// Function scope: constants were pre-resolved to literals by
    /// [`constant_literals`], so the lookup is terminal.
    Literals(&'a HandleMap<naga::Constant, naga::Literal>),
}

/// Everything [`resolve_const_value`] reads: the arena it walks, the type
/// arena `Compose` / `Splat` need for component type and vector size, and
/// where a constant handle resolves.
struct ConstFoldContext<'a> {
    arena: &'a naga::Arena<naga::Expression>,
    types: &'a naga::UniqueArena<naga::Type>,
    constants: ConstSource<'a>,
    /// See [`zero_init_locals`]; empty at module scope, which has no locals.
    zero_locals: &'a HandleMap<naga::LocalVariable, Handle<naga::Type>>,
}

impl ConstFoldContext<'_> {
    fn resolve_constant_value(
        &self,
        handle: naga::Handle<naga::Constant>,
        visiting: &mut HandleSet<naga::Expression>,
        memo: &mut ConstValueMemo,
    ) -> Option<ConstValue> {
        match self.constants {
            ConstSource::Inits(inits) => {
                resolve_const_value(*inits.get(handle)?, self, visiting, memo)
            }
            ConstSource::Literals(lits) => lits.get(handle).copied().map(ConstValue::Scalar),
        }
    }
}

// MARK: Resolver entry points

/// `u32(false)` and friends: dead weight while naga's frontend folded every
/// one, live now that [`zero_init_locals`] hands the resolver a `false` naga
/// never saw.  Bool is what join lowering produces, and without this a
/// never-written one ships as `u32(false)` - six of those cost more than the
/// variable they replaced.
fn cast_bool_to(src: bool, target: naga::Scalar) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    match (target.kind, target.width) {
        (K::Uint, 4) => Some(L::U32(src.into())),
        (K::Sint, 4) => Some(L::I32(src.into())),
        (K::Float, 4) => Some(L::F32(if src { 1.0 } else { 0.0 })),
        (K::Float, 2) => Some(L::F16(if src { half::f16::ONE } else { half::f16::ZERO })),
        (K::Bool, _) => Some(L::Bool(src)),
        // The 8-byte kinds decline, which only forfeits a fold.
        _ => None,
    }
}

/// Resolve `handle` to a [`ConstValue`], cycle-guarded by `visiting` and
/// memoised; composites resolve componentwise.
fn resolve_const_value(
    handle: Handle<naga::Expression>,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    if let Some(Some(cached)) = memo.get(handle.index()) {
        return cached.clone();
    }
    if !visiting.insert(handle) {
        return None;
    }
    let out = resolve_const_value_uncached(handle, ctx, visiting, memo);
    visiting.remove(handle);
    if let Some(slot) = memo.get_mut(handle.index()) {
        *slot = Some(out.clone());
    }
    out
}

/// `?` exits are safe here only because the wrapper owns `visiting.remove`
/// and the memo store on every return path.
fn resolve_const_value_uncached(
    handle: Handle<naga::Expression>,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let expr = &ctx.arena[handle];
    match expr {
        naga::Expression::Literal(lit) => Some(ConstValue::Scalar(*lit)),
        naga::Expression::Constant(ch) => ctx.resolve_constant_value(*ch, visiting, memo),

        naga::Expression::ZeroValue(ty) => resolve_zero_value(*ty, ctx),

        // A never-written local still holds its zero init; `zero_locals`
        // is the proof.
        naga::Expression::Load { pointer } => match ctx.arena[*pointer] {
            naga::Expression::LocalVariable(l) => resolve_zero_value(*ctx.zero_locals.get(l)?, ctx),
            _ => None,
        },

        naga::Expression::Splat { size, value } => {
            let inner = resolve_const_value(*value, ctx, visiting, memo)?;
            let lit = inner.as_scalar()?;
            Some(ConstValue::Vector {
                scalar: lit.scalar(),
                size: *size,
                components: vec![lit; *size as usize],
            })
        }

        naga::Expression::Compose { ty, components } => {
            resolve_compose(*ty, components, ctx, visiting, memo)
        }

        naga::Expression::AccessIndex { base, index } => {
            resolve_composite_element(*base, *index as usize, ctx, visiting, memo)
        }

        // A dynamic `Access` whose index folds is a static pick: naga
        // materialises a dynamically-indexed function-scope `const` array as
        // a full `Compose` at the use site and load_dedup forwards the index
        // literal, so without this the whole composite ships inline
        // (`array<u32,2310>(...)[0]`) and only the NEXT round collapses it
        // via naga's const-eval.  A pointer-typed base is a variable /
        // pointer chain that never resolves to a value, so it cannot fold.
        naga::Expression::Access { base, index } => {
            let idx = match resolve_const_value(*index, ctx, visiting, memo)? {
                ConstValue::Scalar(l) => literal_index(l)?,
                ConstValue::Vector { .. } => return None,
            };
            resolve_composite_element(*base, idx, ctx, visiting, memo)
        }

        naga::Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let vec_val = resolve_const_value(*vector, ctx, visiting, memo)?;
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

        naga::Expression::Unary { op, expr } => {
            let inner = resolve_const_value(*expr, ctx, visiting, memo)?;
            eval_const_unary(*op, inner)
        }

        naga::Expression::Binary { op, left, right } => {
            let l = resolve_const_value(*left, ctx, visiting, memo)?;
            let r = resolve_const_value(*right, ctx, visiting, memo)?;
            eval_const_binary(*op, l, r)
        }

        naga::Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3: _,
        } => {
            let a = resolve_const_value(*arg, ctx, visiting, memo)?;
            let b = match arg1 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting, memo)?),
                None => None,
            };
            let c = match arg2 {
                Some(h) => Some(resolve_const_value(*h, ctx, visiting, memo)?),
                None => None,
            };
            eval_const_math(*fun, a, b, c)
        }

        naga::Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond = resolve_const_value(*condition, ctx, visiting, memo)?;
            match cond {
                ConstValue::Scalar(naga::Literal::Bool(true)) => {
                    resolve_const_value(*accept, ctx, visiting, memo)
                }
                ConstValue::Scalar(naga::Literal::Bool(false)) => {
                    resolve_const_value(*reject, ctx, visiting, memo)
                }
                _ => None,
            }
        }

        // Converting casts fold scalars only: a VECTOR operand falls
        // through, its converted component literals do not exist in the
        // arena, and the generator's vector path handles it.
        naga::Expression::As {
            expr: operand,
            kind,
            convert,
        } => {
            let ConstValue::Scalar(lit) = resolve_const_value(*operand, ctx, visiting, memo)?
            else {
                return None;
            };
            match convert {
                Some(width) => {
                    let target = naga::Scalar {
                        kind: *kind,
                        width: *width,
                    };
                    match lit {
                        naga::Literal::F64(_) | naga::Literal::U64(_) | naga::Literal::I64(_) => {
                            super::expr_util::cast_width8_to(lit, target).map(ConstValue::Scalar)
                        }
                        naga::Literal::Bool(b) => cast_bool_to(b, target).map(ConstValue::Scalar),
                        _ => super::expr_util::cast_width4_to(lit, target).map(ConstValue::Scalar),
                    }
                }
                None => bitcast_literal(lit, *kind).map(ConstValue::Scalar),
            }
        }

        _ => None,
    }
}

/// Reinterpret a scalar literal's bits as `kind` at the same width, as
/// `bitcast` does at runtime; naga's evaluator declines bitcast, so without
/// this the tree above a literal bitcast survives every fold and tint
/// evaluates it at shader creation.  A float on either side must be NORMAL:
/// inf / NaN have no literal spelling, a subnormal may flush to zero on the
/// GPU, and a zero's sign is not kept by every platform (Dawn on Metal reads
/// `bitcast<u32>(-0f)` as 0), so those keep the runtime bitcast and the
/// platform decides, as it did for the input.  `f16` is declined too: its
/// partner width is naga's `i16` / `u16` alone, and its zero has the same
/// problem.
fn bitcast_literal(lit: naga::Literal, kind: naga::ScalarKind) -> Option<naga::Literal> {
    use naga::Literal as L;
    use naga::ScalarKind as K;
    let (bits, width) = match lit {
        L::F32(v) if v.is_normal() => (u64::from(v.to_bits()), 4),
        L::F64(v) if v.is_normal() => (v.to_bits(), 8),
        L::U32(v) => (u64::from(v), 4),
        L::I32(v) => (u64::from(v as u32), 4),
        L::U16(v) => (u64::from(v), 2),
        L::I16(v) => (u64::from(v as u16), 2),
        L::U64(v) => (v, 8),
        L::I64(v) => (v as u64, 8),
        _ => return None,
    };
    Some(match (kind, width) {
        (K::Uint, 4) => L::U32(bits as u32),
        (K::Sint, 4) => L::I32(bits as u32 as i32),
        (K::Float, 4) => {
            let f = f32::from_bits(bits as u32);
            if !f.is_normal() {
                return None;
            }
            L::F32(f)
        }
        (K::Uint, 2) => L::U16(bits as u16),
        (K::Sint, 2) => L::I16(bits as u16 as i16),
        (K::Uint, 8) => L::U64(bits),
        (K::Sint, 8) => L::I64(bits as i64),
        (K::Float, 8) => {
            let f = f64::from_bits(bits);
            if !f.is_normal() {
                return None;
            }
            L::F64(f)
        }
        _ => return None,
    })
}

/// Non-negative value of an integer index literal; anything else declines
/// the fold.
fn literal_index(lit: naga::Literal) -> Option<usize> {
    use naga::Literal as L;
    match lit {
        L::I32(v) => usize::try_from(v).ok(),
        L::U32(v) => Some(v as usize),
        L::I64(v) => usize::try_from(v).ok(),
        L::U64(v) => usize::try_from(v).ok(),
        L::AbstractInt(v) => usize::try_from(v).ok(),
        _ => None,
    }
}

/// One indexing step over a composite taken STRUCTURALLY, without a value:
/// a `Compose` yields the element's own expression, a `ZeroValue` the
/// element's type (or, for a matrix column / vector lane of one, its
/// shape), and an `AccessIndex` / const-indexed `Access` base is followed
/// through its own pick first, so `arr[1][2].m` and `array<vec2u,2>()[1].y`
/// reach their leaf where [`ConstValue`] (scalars and vectors only) cannot
/// carry the array / struct / matrix in between.
enum Picked {
    Expr(Handle<naga::Expression>),
    Zero(Handle<naga::Type>),
    ZeroVector {
        size: naga::VectorSize,
        scalar: naga::Scalar,
    },
    ZeroScalar(naga::Scalar),
}

/// Element `idx` of a zero value of type `ty`, by shape.
fn zero_element(
    ty: Handle<naga::Type>,
    idx: usize,
    types: &naga::UniqueArena<naga::Type>,
) -> Option<Picked> {
    match types[ty].inner {
        naga::TypeInner::Array {
            base,
            size: naga::ArraySize::Constant(n),
            ..
        } => (idx < n.get() as usize).then_some(Picked::Zero(base)),
        naga::TypeInner::Struct { ref members, .. } => {
            members.get(idx).map(|member| Picked::Zero(member.ty))
        }
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => (idx < columns as usize).then_some(Picked::ZeroVector { size: rows, scalar }),
        naga::TypeInner::Vector { size, scalar } => {
            (idx < size as usize).then_some(Picked::ZeroScalar(scalar))
        }
        _ => None,
    }
}

/// [`Picked`] element `idx` of `base`; `None` where the step needs a value
/// (a vector `Compose` flattens, so its lanes cannot be picked positionally)
/// or the base is not a composite the evaluator can see through.
fn pick_element(
    base: Picked,
    idx: usize,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<Picked> {
    let handle = match base {
        Picked::Expr(handle) => handle,
        Picked::Zero(ty) => return zero_element(ty, idx, ctx.types),
        Picked::ZeroVector { size, scalar } => {
            return (idx < size as usize).then_some(Picked::ZeroScalar(scalar));
        }
        Picked::ZeroScalar(_) => return None,
    };
    match &ctx.arena[handle] {
        naga::Expression::Compose { ty, components } => match ctx.types[*ty].inner {
            naga::TypeInner::Array { .. }
            | naga::TypeInner::Matrix { .. }
            | naga::TypeInner::Struct { .. } => components.get(idx).copied().map(Picked::Expr),
            _ => None,
        },
        naga::Expression::ZeroValue(ty) => zero_element(*ty, idx, ctx.types),
        naga::Expression::AccessIndex { base, index } => {
            let inner = pick_element(Picked::Expr(*base), *index as usize, ctx, visiting, memo)?;
            pick_element(inner, idx, ctx, visiting, memo)
        }
        naga::Expression::Access { base, index } => {
            let ConstValue::Scalar(lit) = resolve_const_value(*index, ctx, visiting, memo)? else {
                return None;
            };
            let inner = pick_element(
                Picked::Expr(*base),
                literal_index(lit)?,
                ctx,
                visiting,
                memo,
            )?;
            pick_element(inner, idx, ctx, visiting, memo)
        }
        _ => None,
    }
}

/// Element `idx` of the composite VALUE at `base`: the structural pick
/// resolved at its leaf, or the vector-component path when the pick needs
/// a value (a vector `Compose`, a vector-valued chain like `arr[1][2]`).
fn resolve_composite_element(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match pick_element(Picked::Expr(base), idx, ctx, visiting, memo) {
        Some(Picked::Expr(element)) => resolve_const_value(element, ctx, visiting, memo),
        Some(Picked::Zero(ty)) => resolve_zero_value(ty, ctx),
        Some(Picked::ZeroVector { size, scalar }) => {
            let zero = naga::Literal::zero(scalar)?;
            Some(ConstValue::Vector {
                components: vec![zero; size as usize],
                size,
                scalar,
            })
        }
        Some(Picked::ZeroScalar(scalar)) => naga::Literal::zero(scalar).map(ConstValue::Scalar),
        None => resolve_vector_component(base, idx, ctx, visiting, memo),
    }
}

/// Component `idx` of `base` once it resolves to a [`ConstValue::Vector`].
fn resolve_vector_component(
    base: Handle<naga::Expression>,
    idx: usize,
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    match resolve_const_value(base, ctx, visiting, memo)? {
        ConstValue::Vector { components, .. } => {
            components.get(idx).copied().map(ConstValue::Scalar)
        }
        ConstValue::Scalar(_) => None,
    }
}

/// `ZeroValue(ty)` for a scalar or vector type; matrices, structs, and
/// arrays return `None`.
fn resolve_zero_value(ty: Handle<naga::Type>, ctx: &ConstFoldContext<'_>) -> Option<ConstValue> {
    match ctx.types[ty].inner {
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

/// `Compose { ty, components }` as a [`ConstValue::Vector`] when every
/// component (scalar or flattened vector) resolves to `ty`'s scalar.  The
/// per-component scalar check keeps `materialize_vector` from building a
/// `Compose<vec4<f32>>` over `Literal::I32` handles, which naga's validator
/// rejects; naga's frontend concretises first, so it fires only on
/// hand-built IR.
fn resolve_compose(
    ty: Handle<naga::Type>,
    components: &[Handle<naga::Expression>],
    ctx: &ConstFoldContext<'_>,
    visiting: &mut HandleSet<naga::Expression>,
    memo: &mut ConstValueMemo,
) -> Option<ConstValue> {
    let inner = &ctx.types[ty].inner;
    match inner {
        naga::TypeInner::Vector { size, scalar } => {
            let expected = *size as usize;
            let target_scalar = *scalar;
            let mut out = Vec::with_capacity(expected);
            for &c in components {
                let val = resolve_const_value(c, ctx, visiting, memo)?;
                match val {
                    ConstValue::Scalar(l) => {
                        if l.scalar() != target_scalar {
                            return None;
                        }
                        out.push(l);
                    }
                    ConstValue::Vector {
                        components: v,
                        scalar: inner_scalar,
                        ..
                    } => {
                        if inner_scalar != target_scalar {
                            return None;
                        }
                        out.extend(v);
                    }
                }
            }
            if out.len() != expected {
                return None;
            }
            Some(ConstValue::Vector {
                components: out,
                size: *size,
                scalar: target_scalar,
            })
        }
        _ => None,
    }
}

/// [`eval_unary`] broadcast over vector components.
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

/// Operators whose result scalar is `bool` regardless of operand type.
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

/// Assemble a lane-wise fold's result: a relational operator narrows the
/// element type to `bool`, every other keeps the operand's - in one place, so
/// the element-wise and two broadcast arms cannot disagree about it.
fn vector_fold_result(
    op: naga::BinaryOperator,
    lanes: Option<Vec<naga::Literal>>,
    size: naga::VectorSize,
    scalar: naga::Scalar,
) -> Option<ConstValue> {
    Some(ConstValue::Vector {
        components: lanes?,
        size,
        scalar: if is_relational_op(op) {
            naga::Scalar::BOOL
        } else {
            scalar
        },
    })
}

/// [`eval_binary`] over scalar / same-size vector / broadcast scalar-vector
/// operand pairs.
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
        ) if ls == rs => vector_fold_result(
            op,
            lc.into_iter()
                .zip(rc)
                .map(|(l, r)| eval_binary(op, l, r))
                .collect(),
            ls,
            lscalar,
        ),
        // The two broadcast directions: `eval_binary` is not commutative
        // (`-`, `/`, `%`, the shifts), so each keeps the scalar on its side.
        (
            ConstValue::Scalar(l),
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
        ) => vector_fold_result(
            op,
            components
                .into_iter()
                .map(|r| eval_binary(op, l, r))
                .collect(),
            size,
            scalar,
        ),
        (
            ConstValue::Vector {
                components,
                size,
                scalar,
            },
            ConstValue::Scalar(r),
        ) => vector_fold_result(
            op,
            components
                .into_iter()
                .map(|l| eval_binary(op, l, r))
                .collect(),
            size,
            scalar,
        ),
        _ => None,
    }
}

// MARK: Materialisation helpers

/// Every lane the `+0` of `scalar`, bit for bit: `-0.0` is not zero here,
/// and a scalar without a zero literal (none, in practice) has no zero
/// vector.
fn is_zero_vector(lanes: &[naga::Literal], scalar: naga::Scalar) -> bool {
    naga::Literal::zero(scalar)
        .is_some_and(|zero| lanes.iter().all(|lane| lit_key(*lane) == lit_key(zero)))
}

/// A `Compose` for a folded vector built from `Literal` handles already in
/// the arena, or `None` unless every component's handle precedes `target`
/// (the Compose must stay topologically valid).  `literal_cache` maps each
/// literal to its SMALLEST handle, which maximises those hits.
fn materialize_vector(
    target: Handle<naga::Expression>,
    literals: &[naga::Literal],
    size: naga::VectorSize,
    scalar: naga::Scalar,
    literal_cache: &FxHashMap<KeyToken, Handle<naga::Expression>>,
    vector_type_cache: &FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>>,
) -> Option<naga::Expression> {
    let ty = *vector_type_cache.get(&(size, scalar))?;

    let mut handles = Vec::with_capacity(literals.len());
    for lit in literals {
        let h = *literal_cache.get(&lit_key(*lit))?;
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

/// `arena[handle]` is a `Compose` of the same type whose components are
/// literals bit-equal to `folded`'s (a `Compose` of cached literal handles),
/// so the two spell one value and rewriting one into the other moves
/// nothing.
fn compose_spells(
    arena: &naga::Arena<naga::Expression>,
    handle: Handle<naga::Expression>,
    folded: &naga::Expression,
) -> bool {
    let (
        naga::Expression::Compose {
            ty: have_ty,
            components: have,
        },
        naga::Expression::Compose {
            ty: want_ty,
            components: want,
        },
    ) = (&arena[handle], folded)
    else {
        return false;
    };
    have_ty == want_ty
        && have.len() == want.len()
        && have.iter().zip(want).all(|(&a, &b)| {
            matches!(
                (&arena[a], &arena[b]),
                (naga::Expression::Literal(x), naga::Expression::Literal(y))
                    if lit_key(*x) == lit_key(*y)
            )
        })
}

/// Each scalar `Literal` in `arena` -> the SMALLEST handle carrying it;
/// [`note_literal_in_cache`] keeps the invariant as the fold writes new
/// literals.
fn build_literal_cache(
    arena: &naga::Arena<naga::Expression>,
) -> FxHashMap<KeyToken, Handle<naga::Expression>> {
    let mut cache: FxHashMap<KeyToken, Handle<naga::Expression>> = Default::default();
    for (h, expr) in arena.iter() {
        if let naga::Expression::Literal(lit) = expr {
            cache
                .entry(lit_key(*lit))
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

/// `(size, scalar)` -> vector `Type` handle, so [`materialize_vector`] never
/// scans the type arena.
fn build_vector_type_cache(
    types: &naga::UniqueArena<naga::Type>,
) -> FxHashMap<(naga::VectorSize, naga::Scalar), naga::Handle<naga::Type>> {
    let mut cache = FxHashMap::default();
    for (h, t) in types.iter() {
        if let naga::TypeInner::Vector { size, scalar } = t.inner {
            cache.entry((size, scalar)).or_insert(h);
        }
    }
    cache
}

/// Record `handle` as a carrier of `literal`, keeping the smallest-handle
/// invariant.
fn note_literal_in_cache(
    cache: &mut FxHashMap<KeyToken, Handle<naga::Expression>>,
    handle: Handle<naga::Expression>,
    literal: naga::Literal,
) {
    cache
        .entry(lit_key(literal))
        .and_modify(|cur| {
            if handle.index() < cur.index() {
                *cur = handle;
            }
        })
        .or_insert(handle);
}

// MARK: Scalar evaluators

#[rustfmt::skip]
macro_rules! float_impl {
    // `f16` is the `f32` operation rounded once: correctly rounded for the
    // arithmetic (24 >= 2 * 11 + 2 bits, so the wide result rounds as the
    // exact one would; `half`'s own operators do the same), and for the
    // builtins a value in WGSL's `f16` envelope, as naga's evaluator does
    // through its own `f32` builtins (the `libm` crate's, where these are
    // the platform's: the two can differ by an `f32` ULP, so rarely by an
    // `f16` one).
    (f16 via f32; $($m:ident),*) => {
        impl WgslFloat for half::f16 {
            const ZERO: Self = half::f16::ZERO;
            const ONE: Self = half::f16::ONE;
            fn from_f64(v: f64) -> Self { half::f16::from_f64(v) }
            fn is_finite(self) -> bool { half::f16::is_finite(self) }
            fn is_nan(self) -> bool { half::f16::is_nan(self) }
            fn atan2(self, x: Self) -> Self { in_f32_2(self, x, f32::atan2) }
            fn powf(self, b: Self) -> Self { in_f32_2(self, b, f32::powf) }
            fn mul_add(self, b: Self, c: Self) -> Self { in_f32_3(self, b, c, f32::mul_add) }
            fn min(self, b: Self) -> Self { in_f32_2(self, b, f32::min) }
            fn max(self, b: Self) -> Self { in_f32_2(self, b, f32::max) }
            fn clamp(self, lo: Self, hi: Self) -> Self { in_f32_3(self, lo, hi, f32::clamp) }
            $(fn $m(self) -> Self { in_f32(self, f32::$m) })*
        }
    };
    ($t:ty; $($m:ident),*) => {
        impl WgslFloat for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn from_f64(v: f64) -> Self { v as $t }
            fn is_finite(self) -> bool { <$t>::is_finite(self) }
            fn is_nan(self) -> bool { <$t>::is_nan(self) }
            fn atan2(self, x: Self) -> Self { <$t>::atan2(self, x) }
            fn powf(self, b: Self) -> Self { <$t>::powf(self, b) }
            fn mul_add(self, b: Self, c: Self) -> Self { <$t>::mul_add(self, b, c) }
            fn min(self, b: Self) -> Self { <$t>::min(self, b) }
            fn max(self, b: Self) -> Self { <$t>::max(self, b) }
            fn clamp(self, lo: Self, hi: Self) -> Self { <$t>::clamp(self, lo, hi) }
            $(fn $m(self) -> Self { <$t>::$m(self) })*
        }
    };
}

/// An `f16` builtin as its `f32` builtin rounded once.  Out of line, or the
/// two conversions are inlined into each of the thirty methods.
#[inline(never)]
fn in_f32(v: half::f16, f: fn(f32) -> f32) -> half::f16 {
    half::f16::from_f32(f(v.to_f32()))
}

#[inline(never)]
fn in_f32_2(a: half::f16, b: half::f16, f: fn(f32, f32) -> f32) -> half::f16 {
    half::f16::from_f32(f(a.to_f32(), b.to_f32()))
}

#[inline(never)]
fn in_f32_3(a: half::f16, b: half::f16, c: half::f16, f: fn(f32, f32, f32) -> f32) -> half::f16 {
    half::f16::from_f32(f(a.to_f32(), b.to_f32(), c.to_f32()))
}

/// The float arithmetic a fold uses, so each fold is written once and
/// instantiated for `f16`, `f32` and `f64` (`AbstractFloat` is an `f64`
/// under its own [`FloatMode`]); the unary list is written once, at the
/// invocation.
#[rustfmt::skip]
macro_rules! wgsl_float {
    ($($m:ident),* $(,)?) => {
        trait WgslFloat:
            Copy + PartialOrd
            + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self>
            + std::ops::Mul<Output = Self> + std::ops::Div<Output = Self>
            + std::ops::Rem<Output = Self> + std::ops::Neg<Output = Self>
        {
            const ZERO: Self;
            const ONE: Self;
            fn from_f64(v: f64) -> Self;
            fn is_finite(self) -> bool;
            fn is_nan(self) -> bool;
            fn atan2(self, x: Self) -> Self;
            fn powf(self, b: Self) -> Self;
            fn mul_add(self, b: Self, c: Self) -> Self;
            fn min(self, b: Self) -> Self;
            fn max(self, b: Self) -> Self;
            fn clamp(self, lo: Self, hi: Self) -> Self;
            $(fn $m(self) -> Self;)*
        }
        float_impl!(f16 via f32; $($m),*);
        float_impl!(f32; $($m),*);
        float_impl!(f64; $($m),*);
    };
}

#[rustfmt::skip]
wgsl_float![
    abs, signum, floor, ceil, round_ties_even, trunc, sqrt, cos, sin, tan, cosh, sinh, tanh,
    acos, asin, atan, asinh, acosh, atanh, to_radians, to_degrees, exp, exp2, ln, log2,
];

/// The integer arithmetic a fold uses, instantiated for the four concrete
/// widths (`AbstractInt` is an `i64` under its own [`IntMode`]).  The
/// signedness lives in `int_impl!`, not in the folds: `add` / `sub` / `mul`
/// are checked for a signed type (an overflow declines rather than wraps:
/// the surviving expression still ships, both validators accept
/// `2147483647i+1i` in runtime position, and a const-context original never
/// reaches the passes) and wrapping for an unsigned one, and `neg` / `abs`
/// / `signum` answer for an unsigned type the way the type does: no
/// negation, its own absolute value, no sign.
trait WgslInt:
    Copy
    + Ord
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
{
    const SIGNED: bool;
    const BITS: u32;
    const ZERO: Self;
    const ALL_ONES: Self;
    fn from_count(n: u32) -> Self;
    fn add(self, b: Self) -> Option<Self>;
    fn sub(self, b: Self) -> Option<Self>;
    fn mul(self, b: Self) -> Option<Self>;
    fn neg(self) -> Option<Self>;
    fn abs(self) -> Option<Self>;
    fn signum(self) -> Option<Self>;
    fn checked_div(self, b: Self) -> Option<Self>;
    fn checked_rem(self, b: Self) -> Option<Self>;
    fn wrapping_shl(self, n: u32) -> Self;
    fn wrapping_shr(self, n: u32) -> Self;
    fn reverse_bits(self) -> Self;
    fn trailing_zeros(self) -> u32;
    fn leading_zeros(self) -> u32;
    fn leading_ones(self) -> u32;
    fn count_ones(self) -> u32;
}

#[rustfmt::skip]
macro_rules! int_impl {
    (signed $($t:ty),*) => {$(
        impl WgslInt for $t {
            const SIGNED: bool = true;
            fn add(self, b: Self) -> Option<Self> { self.checked_add(b) }
            fn sub(self, b: Self) -> Option<Self> { self.checked_sub(b) }
            fn mul(self, b: Self) -> Option<Self> { self.checked_mul(b) }
            fn neg(self) -> Option<Self> { self.checked_neg() }
            fn abs(self) -> Option<Self> { self.checked_abs() }
            fn signum(self) -> Option<Self> { Some(<$t>::signum(self)) }
            int_impl!(common $t);
        }
    )*};
    (unsigned $($t:ty),*) => {$(
        impl WgslInt for $t {
            const SIGNED: bool = false;
            fn add(self, b: Self) -> Option<Self> { Some(self.wrapping_add(b)) }
            fn sub(self, b: Self) -> Option<Self> { Some(self.wrapping_sub(b)) }
            fn mul(self, b: Self) -> Option<Self> { Some(self.wrapping_mul(b)) }
            fn neg(self) -> Option<Self> { None }
            fn abs(self) -> Option<Self> { Some(self) }
            fn signum(self) -> Option<Self> { None }
            int_impl!(common $t);
        }
    )*};
    (common $t:ty) => {
        const BITS: u32 = <$t>::BITS;
        const ZERO: Self = 0;
        const ALL_ONES: Self = !0;
        fn from_count(n: u32) -> Self { n as $t }
        fn checked_div(self, b: Self) -> Option<Self> { <$t>::checked_div(self, b) }
        fn checked_rem(self, b: Self) -> Option<Self> { <$t>::checked_rem(self, b) }
        fn wrapping_shl(self, n: u32) -> Self { <$t>::wrapping_shl(self, n) }
        fn wrapping_shr(self, n: u32) -> Self { <$t>::wrapping_shr(self, n) }
        fn reverse_bits(self) -> Self { <$t>::reverse_bits(self) }
        fn trailing_zeros(self) -> u32 { <$t>::trailing_zeros(self) }
        fn leading_zeros(self) -> u32 { <$t>::leading_zeros(self) }
        fn leading_ones(self) -> u32 { <$t>::leading_ones(self) }
        fn count_ones(self) -> u32 { <$t>::count_ones(self) }
    };
}

int_impl!(signed i32, i64);
int_impl!(unsigned u32, u64);

/// What tells the float variants of one Rust type apart.
#[derive(Clone, Copy)]
struct FloatMode {
    /// WGSL lowers float `a % b` to `a - b*trunc(a/b)` in OPERAND precision,
    /// which diverges from exact fmod by a FULL divisor whenever the rounded
    /// quotient crosses an integer the exact one does not (`33554432f % 3f`
    /// is fmod 2.0 but 0.0 on every round-to-nearest GPU).  A concrete float
    /// folds only when both agree, and only while `|a / b|` stays under
    /// this cap (the width's precision: every integer below it is exact),
    /// which keeps a quotient whose trunc is unrepresentable out of the
    /// comparison; an abstract float has no lowering to agree with.
    rem_quotient_cap: Option<f64>,
}

const F16_MODE: FloatMode = FloatMode {
    rem_quotient_cap: Some(2048.0),
};
const F32_MODE: FloatMode = FloatMode {
    rem_quotient_cap: Some(16_777_216.0),
};
/// WGSL has no runtime f64, so this sees only non-WGSL-frontend IR.
const F64_MODE: FloatMode = FloatMode {
    rem_quotient_cap: Some(9_007_199_254_740_992.0),
};
const ABSTRACT_FLOAT_MODE: FloatMode = FloatMode {
    rem_quotient_cap: None,
};

/// What tells `AbstractInt` apart from `I64`.
#[derive(Clone, Copy)]
struct IntMode {
    /// `MIN / -1` and `MIN % -1` MUST fold for a concrete type: the pair
    /// only arises from nagami's own literal substitution into runtime
    /// expressions, where WGSL defines the results as e1 and 0
    /// (<https://www.w3.org/TR/WGSL/#arithmetic-expr>); declined, it
    /// round-trips into naga's text const-eval, which rejects it and kills
    /// the emission.  An abstract pair is that const context, and declines.
    div_wraps: bool,
    /// The bit builtins take concrete integers only.
    bit_builtins: bool,
}

const CONCRETE_INT_MODE: IntMode = IntMode {
    div_wraps: true,
    bit_builtins: true,
};
const ABSTRACT_INT_MODE: IntMode = IntMode {
    div_wraps: false,
    bit_builtins: false,
};

/// The value of `$variant`'s payload, `None` for any other literal: a fold
/// takes its second and third operands from the SAME variant, `F64` and
/// `AbstractFloat` included, which share a Rust type and nothing else.
macro_rules! literal_of {
    ($variant:ident) => {
        |lit: naga::Literal| match lit {
            naga::Literal::$variant(v) => Some(v),
            _ => None,
        }
    };
}

/// Per-scalar unary evaluator; declines operator / type pairs whose fold
/// would change observable behaviour.
fn eval_unary(op: naga::UnaryOperator, rhs: naga::Literal) -> Option<naga::Literal> {
    use naga::Literal as L;
    match rhs {
        L::F32(v) => float_unary(op, v, L::F32),
        L::F64(v) => float_unary(op, v, L::F64),
        L::AbstractFloat(v) => float_unary(op, v, L::AbstractFloat),
        L::I32(v) => int_unary(op, v, L::I32),
        L::I64(v) => int_unary(op, v, L::I64),
        L::AbstractInt(v) => int_unary(op, v, L::AbstractInt),
        L::U32(v) => int_unary(op, v, L::U32),
        L::U64(v) => int_unary(op, v, L::U64),
        L::F16(v) => float_unary(op, v, L::F16),
        L::Bool(v) => (op == naga::UnaryOperator::LogicalNot).then_some(L::Bool(!v)),
        L::U16(_) | L::I16(_) => None,
    }
}

/// naga's validator rejects non-finite `F32` / `F64` literals
/// (`LiteralError::NaN` / `Infinity`); `is_finite()` rather than `!is_nan()`
/// keeps a negated infinity out of the IR whatever upstream injects.
fn float_unary<T: WgslFloat>(
    op: naga::UnaryOperator,
    v: T,
    lit: fn(T) -> naga::Literal,
) -> Option<naga::Literal> {
    (op == naga::UnaryOperator::Negate && (-v).is_finite()).then(|| lit(-v))
}

fn int_unary<T: WgslInt>(
    op: naga::UnaryOperator,
    v: T,
    lit: fn(T) -> naga::Literal,
) -> Option<naga::Literal> {
    match op {
        naga::UnaryOperator::Negate => v.neg().map(lit),
        naga::UnaryOperator::BitwiseNot => Some(lit(!v)),
        naga::UnaryOperator::LogicalNot => None,
    }
}

/// Per-scalar binary evaluator; NaN, overflow, and divide-by-zero cases
/// decline so folding never changes observable output.  Equality is the
/// derived `PartialEq` over ANY pair, mixed variants included (`-0.0 ==
/// 0.0`, `1i != 1u`); every other operator takes one variant on both sides,
/// except the shifts, whose amount naga's WGSL frontend concretises to `u32`
/// whatever the left operand's width (a `U64` amount could never match),
/// abstract pairs aside.
fn eval_binary(
    op: naga::BinaryOperator,
    lhs: naga::Literal,
    rhs: naga::Literal,
) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    use naga::Literal as L;
    match op {
        B::Equal => return Some(L::Bool(lhs == rhs)),
        B::NotEqual => return Some(L::Bool(lhs != rhs)),
        B::ShiftLeft | B::ShiftRight => {
            return match (lhs, rhs) {
                (L::U32(a), L::U32(b)) => int_shift(op, a, b, L::U32),
                (L::U64(a), L::U32(b)) => int_shift(op, a, b, L::U64),
                (L::I32(a), L::U32(b)) => int_shift(op, a, b, L::I32),
                (L::I64(a), L::U32(b)) => int_shift(op, a, b, L::I64),
                (L::AbstractInt(a), L::AbstractInt(b)) => abstract_shift(op, a, b),
                _ => None,
            };
        }
        _ => {}
    }
    match (lhs, rhs) {
        (L::F16(a), L::F16(b)) => float_binary(op, a, b, F16_MODE, L::F16),
        (L::F32(a), L::F32(b)) => float_binary(op, a, b, F32_MODE, L::F32),
        (L::F64(a), L::F64(b)) => float_binary(op, a, b, F64_MODE, L::F64),
        (L::AbstractFloat(a), L::AbstractFloat(b)) => {
            float_binary(op, a, b, ABSTRACT_FLOAT_MODE, L::AbstractFloat)
        }
        (L::I32(a), L::I32(b)) => int_binary(op, a, b, CONCRETE_INT_MODE, L::I32),
        (L::I64(a), L::I64(b)) => int_binary(op, a, b, CONCRETE_INT_MODE, L::I64),
        (L::U32(a), L::U32(b)) => int_binary(op, a, b, CONCRETE_INT_MODE, L::U32),
        (L::U64(a), L::U64(b)) => int_binary(op, a, b, CONCRETE_INT_MODE, L::U64),
        (L::AbstractInt(a), L::AbstractInt(b)) => {
            int_binary(op, a, b, ABSTRACT_INT_MODE, L::AbstractInt)
        }
        (L::Bool(a), L::Bool(b)) => bool_binary(op, a, b),
        _ => None,
    }
}

fn float_binary<T: WgslFloat>(
    op: naga::BinaryOperator,
    a: T,
    b: T,
    mode: FloatMode,
    lit: fn(T) -> naga::Literal,
) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    let r = match op {
        B::Add => a + b,
        B::Subtract => a - b,
        B::Multiply => a * b,
        B::Divide if b != T::ZERO => a / b,
        B::Modulo if b != T::ZERO => {
            let r = a % b;
            if let Some(cap) = mode.rem_quotient_cap {
                let q = a / b;
                if !(q.abs() < T::from_f64(cap) && r == a - b * q.trunc()) {
                    return None;
                }
            }
            r
        }
        B::Less => return Some(naga::Literal::Bool(a < b)),
        B::LessEqual => return Some(naga::Literal::Bool(a <= b)),
        B::Greater => return Some(naga::Literal::Bool(a > b)),
        B::GreaterEqual => return Some(naga::Literal::Bool(a >= b)),
        _ => return None,
    };
    r.is_finite().then(|| lit(r))
}

fn int_binary<T: WgslInt>(
    op: naga::BinaryOperator,
    a: T,
    b: T,
    mode: IntMode,
    lit: fn(T) -> naga::Literal,
) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    let r = match op {
        B::Add => a.add(b)?,
        B::Subtract => a.sub(b)?,
        B::Multiply => a.mul(b)?,
        B::Divide if b != T::ZERO => {
            let q = a.checked_div(b);
            if mode.div_wraps { q.unwrap_or(a) } else { q? }
        }
        B::Modulo if b != T::ZERO => {
            let r = a.checked_rem(b);
            if mode.div_wraps {
                r.unwrap_or(T::ZERO)
            } else {
                r?
            }
        }
        B::And => a & b,
        B::ExclusiveOr => a ^ b,
        B::InclusiveOr => a | b,
        B::Less => return Some(naga::Literal::Bool(a < b)),
        B::LessEqual => return Some(naga::Literal::Bool(a <= b)),
        B::Greater => return Some(naga::Literal::Bool(a > b)),
        B::GreaterEqual => return Some(naga::Literal::Bool(a >= b)),
        _ => return None,
    };
    Some(lit(r))
}

/// A concrete shift by a `u32` amount under the type's width.  A
/// sign-changing `e1 << e2` is a shader-creation error only in CONST
/// contexts, which naga rejected at ingest; a literal pair here is a runtime
/// expression manufactured by nagami's own transforms, where WGSL defines
/// the plain bit-pattern result (<https://www.w3.org/TR/WGSL/#bit-expr>).
/// Declined, the pair fails naga's text const-eval on re-parse and kills the
/// emission.
fn int_shift<T: WgslInt>(
    op: naga::BinaryOperator,
    a: T,
    b: u32,
    lit: fn(T) -> naga::Literal,
) -> Option<naga::Literal> {
    (b < T::BITS).then(|| {
        lit(if op == naga::BinaryOperator::ShiftLeft {
            a.wrapping_shl(b)
        } else {
            a.wrapping_shr(b)
        })
    })
}

/// An abstract shift: the amount is abstract too, and a left shift that
/// leaves the 64-bit range declines.
fn abstract_shift(op: naga::BinaryOperator, a: i64, b: i64) -> Option<naga::Literal> {
    if !(0..64).contains(&b) {
        return None;
    }
    if op == naga::BinaryOperator::ShiftLeft {
        let wide = (a as i128).wrapping_shl(b as u32);
        let narrowed = wide as i64;
        (narrowed as i128 == wide).then_some(naga::Literal::AbstractInt(narrowed))
    } else {
        Some(naga::Literal::AbstractInt(a.wrapping_shr(b as u32)))
    }
}

fn bool_binary(op: naga::BinaryOperator, a: bool, b: bool) -> Option<naga::Literal> {
    use naga::BinaryOperator as B;
    Some(naga::Literal::Bool(match op {
        B::LogicalAnd => a && b,
        B::LogicalOr => a || b,
        B::And => a & b,
        B::InclusiveOr => a | b,
        B::ExclusiveOr => a ^ b,
        _ => return None,
    }))
}

/// Fold a math built-in over scalar literals.  Comparison, decomposition,
/// and integer-bit functions are bit-exact; the trigonometric / exponential
/// family has a WGSL-defined error envelope, so those folds substitute a
/// conformant value, not a bit-identical one.  Unsupported functions, type
/// mismatches, NaN-sensitive cases, and domain errors decline.
fn eval_math_scalar(
    fun: naga::MathFunction,
    arg: naga::Literal,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
) -> Option<naga::Literal> {
    use naga::Literal as L;
    match arg {
        L::F16(a) => float_math(fun, a, arg1, arg2, L::F16, literal_of!(F16)),
        L::F32(a) => float_math(fun, a, arg1, arg2, L::F32, literal_of!(F32)),
        L::F64(a) => float_math(fun, a, arg1, arg2, L::F64, literal_of!(F64)),
        L::AbstractFloat(a) => float_math(
            fun,
            a,
            arg1,
            arg2,
            L::AbstractFloat,
            literal_of!(AbstractFloat),
        ),
        L::I32(a) => int_math(
            fun,
            a,
            arg1,
            arg2,
            CONCRETE_INT_MODE,
            L::I32,
            literal_of!(I32),
        ),
        L::I64(a) => int_math(
            fun,
            a,
            arg1,
            arg2,
            CONCRETE_INT_MODE,
            L::I64,
            literal_of!(I64),
        ),
        L::U32(a) => int_math(
            fun,
            a,
            arg1,
            arg2,
            CONCRETE_INT_MODE,
            L::U32,
            literal_of!(U32),
        ),
        L::U64(a) => int_math(
            fun,
            a,
            arg1,
            arg2,
            CONCRETE_INT_MODE,
            L::U64,
            literal_of!(U64),
        ),
        L::AbstractInt(a) => int_math(
            fun,
            a,
            arg1,
            arg2,
            ABSTRACT_INT_MODE,
            L::AbstractInt,
            literal_of!(AbstractInt),
        ),
        L::Bool(_) | L::U16(_) | L::I16(_) => None,
    }
}

/// The float built-ins.  Every result routes through the finiteness check
/// (naga's validator rejects a non-finite literal) except the selections -
/// `min` / `max` / `clamp` / `step` / `sign` - which return an operand, or a
/// constant, of their own.
fn float_math<T: WgslFloat>(
    fun: naga::MathFunction,
    a: T,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
    lit: fn(T) -> naga::Literal,
    of: fn(naga::Literal) -> Option<T>,
) -> Option<naga::Literal> {
    use naga::MathFunction as M;
    let b = || arg1.and_then(of);
    let c = || arg2.and_then(of);
    let r = match fun {
        M::Abs => a.abs(),
        // WGSL propagates NaN through min / max / clamp; Rust's return the
        // other operand (and `clamp` panics on NaN bounds).
        M::Min => {
            let b = b()?;
            return (!a.is_nan() && !b.is_nan()).then(|| lit(a.min(b)));
        }
        M::Max => {
            let b = b()?;
            return (!a.is_nan() && !b.is_nan()).then(|| lit(a.max(b)));
        }
        M::Clamp => {
            let (lo, hi) = (b()?, c()?);
            return (lo <= hi && !a.is_nan() && !lo.is_nan() && !hi.is_nan())
                .then(|| lit(a.clamp(lo, hi)));
        }
        M::Saturate => a.clamp(T::ZERO, T::ONE),
        // Rust's `signum(0.0)` is 1.0; WGSL `sign(0)` is 0.
        M::Sign => {
            return (!a.is_nan()).then(|| lit(if a == T::ZERO { T::ZERO } else { a.signum() }));
        }
        M::Floor => a.floor(),
        M::Ceil => a.ceil(),
        // WGSL `round` is ties-to-even.
        M::Round => a.round_ties_even(),
        M::Trunc => a.trunc(),
        // WGSL `fract(e)` is `e - floor(e)`, not Rust's `e - trunc(e)`.
        M::Fract => a - a.floor(),
        // A NaN operand compares false and would fold to a wrong 0.0; WGSL
        // propagates it.
        M::Step => {
            let x = b()?;
            return (!a.is_nan() && !x.is_nan())
                .then(|| lit(if a <= x { T::ONE } else { T::ZERO }));
        }
        M::Sqrt if a >= T::ZERO => a.sqrt(),
        M::InverseSqrt if a > T::ZERO => T::ONE / a.sqrt(),
        M::Fma => {
            let (b, c) = (b()?, c()?);
            a.mul_add(b, c)
        }
        M::Cos => a.cos(),
        M::Sin => a.sin(),
        M::Tan => a.tan(),
        M::Cosh => a.cosh(),
        M::Sinh => a.sinh(),
        M::Tanh => a.tanh(),
        M::Acos if a.abs() <= T::ONE => a.acos(),
        M::Asin if a.abs() <= T::ONE => a.asin(),
        M::Atan => a.atan(),
        // WGSL leaves `atan2(0, 0)` implementation-defined: a GPU may return
        // any of {0, +/-pi/2, pi}.
        M::Atan2 => {
            let x = b()?;
            if a == T::ZERO && x == T::ZERO {
                return None;
            }
            a.atan2(x)
        }
        M::Asinh => a.asinh(),
        M::Acosh if a >= T::ONE => a.acosh(),
        M::Atanh if a.abs() < T::ONE => a.atanh(),
        M::Radians => a.to_radians(),
        M::Degrees => a.to_degrees(),
        M::Exp => a.exp(),
        M::Exp2 => a.exp2(),
        M::Log if a > T::ZERO => a.ln(),
        M::Log2 if a > T::ZERO => a.log2(),
        // WGSL requires `e1 >= 0`, and `pow(0, b)` with `b <= 0` is
        // implementation-defined (Rust says 1.0 for 0^0; a GPU may say NaN
        // or 0).
        M::Pow => {
            let b = b()?;
            if !(a > T::ZERO || (a == T::ZERO && b > T::ZERO)) {
                return None;
            }
            a.powf(b)
        }
        _ => return None,
    };
    r.is_finite().then(|| lit(r))
}

/// The integer built-ins: the selections, then the bit builtins, which
/// `mode` withholds from an abstract operand.
fn int_math<T: WgslInt>(
    fun: naga::MathFunction,
    a: T,
    arg1: Option<naga::Literal>,
    arg2: Option<naga::Literal>,
    mode: IntMode,
    lit: fn(T) -> naga::Literal,
    of: fn(naga::Literal) -> Option<T>,
) -> Option<naga::Literal> {
    use naga::MathFunction as M;
    let b = || arg1.and_then(of);
    let c = || arg2.and_then(of);
    let r = match fun {
        M::Abs => a.abs()?,
        M::Min => a.min(b()?),
        M::Max => a.max(b()?),
        M::Clamp => {
            let (lo, hi) = (b()?, c()?);
            if lo > hi {
                return None;
            }
            a.clamp(lo, hi)
        }
        M::Sign => a.signum()?,
        _ if !mode.bit_builtins => return None,
        M::CountTrailingZeros => T::from_count(a.trailing_zeros()),
        M::CountLeadingZeros => T::from_count(a.leading_zeros()),
        M::CountOneBits => T::from_count(a.count_ones()),
        M::ReverseBits => a.reverse_bits(),
        M::FirstTrailingBit => {
            if a == T::ZERO {
                T::ALL_ONES
            } else {
                T::from_count(a.trailing_zeros())
            }
        }
        M::FirstLeadingBit => {
            if a == T::ZERO || (T::SIGNED && a == T::ALL_ONES) {
                T::ALL_ONES
            } else if !T::SIGNED || a > T::ZERO {
                T::from_count(T::BITS - 1 - a.leading_zeros())
            } else {
                // Negative: the highest bit differing from the sign bit.
                T::from_count(T::BITS - 1 - a.leading_ones())
            }
        }
        _ => return None,
    };
    Some(lit(r))
}

/// [`eval_math_scalar`] broadcast over vector arguments; sizes must match.
fn eval_const_math(
    fun: naga::MathFunction,
    arg: ConstValue,
    arg1: Option<ConstValue>,
    arg2: Option<ConstValue>,
) -> Option<ConstValue> {
    fn as_scalar(v: &ConstValue) -> Option<naga::Literal> {
        match v {
            ConstValue::Scalar(l) => Some(*l),
            _ => None,
        }
    }

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

    if let Some((comps, size, scalar)) = as_vector(&arg) {
        let n = comps.len();

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

// MARK: Identity and absorbing rules

/// The literal classes the rules match.  `IntZero` stands apart from `Zero`
/// because a float zero has a sign: under IEEE 754 round-to-nearest
/// `(-0.0) + (+0.0)` and `(-0.0) - (-0.0)` are both `+0.0`, so dropping a
/// float zero can flip the sign of `x = -0.0` (and the safe sign differs
/// per operator), and a float product carries the sign of BOTH operands
/// (`-2.0h * 0.0h` is `-0.0h`) and is NaN for a non-finite `x`.  The
/// additive identity and the multiplicative absorber take integer zeros
/// only, uniformly, at the cost of a few unfolded `float + 0.0`; a bare
/// float product is left for naga to re-parse to the correctly-signed zero.
#[derive(Clone, Copy)]
enum Class {
    IntZero,
    Zero,
    One,
    AllOnes,
    True,
    False,
}

impl Class {
    fn matches(self, lit: naga::Literal) -> bool {
        use naga::Literal as L;
        match self {
            Class::IntZero => matches!(
                lit,
                L::I32(0) | L::U32(0) | L::I64(0) | L::U64(0) | L::AbstractInt(0)
            ),
            // `to_bits() & 0x7FFF == 0` is either binary16 zero.
            Class::Zero => {
                Class::IntZero.matches(lit)
                    || matches!(lit, L::F32(v) if v == 0.0)
                    || matches!(lit, L::F64(v) if v == 0.0)
                    || matches!(lit, L::AbstractFloat(v) if v == 0.0)
                    || matches!(lit, L::F16(v) if v.to_bits() & 0x7FFF == 0)
            }
            // `0x3C00` is binary16 `+1.0`.
            Class::One => {
                matches!(
                    lit,
                    L::I32(1) | L::U32(1) | L::I64(1) | L::U64(1) | L::AbstractInt(1)
                ) || matches!(lit, L::F32(v) if v == 1.0)
                    || matches!(lit, L::F64(v) if v == 1.0)
                    || matches!(lit, L::AbstractFloat(v) if v == 1.0)
                    || matches!(lit, L::F16(v) if v.to_bits() == 0x3C00)
            }
            Class::AllOnes => matches!(
                lit,
                L::U32(u32::MAX) | L::I32(-1) | L::U64(u64::MAX) | L::I64(-1) | L::AbstractInt(-1)
            ),
            Class::True => matches!(lit, L::Bool(true)),
            Class::False => matches!(lit, L::Bool(false)),
        }
    }
}

/// Which operand an `x <op> literal` match leaves standing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// The other operand: the literal was the neutral element.
    Other,
    /// The literal: it absorbed the other operand.
    Literal,
}

/// One rewrite of `x <op> literal` - and of `literal <op> x` where `either`
/// - to the operand `keep` names.
struct Rule {
    op: naga::BinaryOperator,
    literal: Class,
    either_side: bool,
    keep: Keep,
}

macro_rules! rules {
    ($($op:ident: $lit:ident $side:ident => $keep:ident),* $(,)?) => {
        [$(Rule {
            op: naga::BinaryOperator::$op,
            literal: Class::$lit,
            either_side: rules!(@side $side),
            keep: Keep::$keep,
        }),*]
    };
    (@side either) => { true };
    (@side right) => { false };
}

/// The absorbing rules, then the identities:
///
/// ```text
/// x * 0 = 0          x & 0 = 0          x | all_ones = all_ones
/// x && false = false x || true = true
///
/// x + 0 = x          x - 0 = x          x * 1 = x          x / 1 = x
/// x | 0 = x          x ^ 0 = x          x & all_ones = x
/// x && true = x      x || false = x
/// ```
///
/// Every row is exact for the operands its class admits: the two zero rows
/// that would not be for a float take `IntZero`.  `x - (+0.0)` would be
/// exact for a float too (no shader seen spells it as a scalar; the vector
/// form needs the operand's type), and `x + (-0.0)`, exact in IEEE 754, is not
/// on Dawn / Metal, which reads a `-0.0` literal as `+0.0`.
const RULES: [Rule; 14] = rules![
    Multiply: IntZero either => Literal,
    And: Zero either => Literal,
    InclusiveOr: AllOnes either => Literal,
    LogicalAnd: False either => Literal,
    LogicalOr: True either => Literal,
    Add: IntZero either => Other,
    Subtract: IntZero right => Other,
    Multiply: One either => Other,
    Divide: One right => Other,
    InclusiveOr: Zero either => Other,
    ExclusiveOr: Zero either => Other,
    And: AllOnes either => Other,
    LogicalAnd: True either => Other,
    LogicalOr: False either => Other,
];

/// The rule of `keep`'s kind matching `left <op> right`, as the pair
/// `(the literal operand, the other one)`; the left operand is tried first.
fn rule_match(
    op: naga::BinaryOperator,
    keep: Keep,
    left: naga::Handle<naga::Expression>,
    right: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<(
    naga::Handle<naga::Expression>,
    naga::Handle<naga::Expression>,
)> {
    let rule = RULES
        .iter()
        .find(|rule| rule.op == op && rule.keep == keep)?;
    let is = |h: naga::Handle<naga::Expression>| matches!(arena[h], naga::Expression::Literal(lit) if rule.literal.matches(lit));
    if rule.either_side && is(left) {
        Some((left, right))
    } else if is(right) {
        Some((right, left))
    } else {
        None
    }
}

/// Whether `source` may be cloned over `target`, which references it
/// through `path` (the handles between the two, both excluded): a pure
/// expression clones freely; an impure one only when `target` is the sole
/// consumer of it and of every handle on the path AND the two share an
/// `Emit` range, so the operand dies (its `Emit` entry is dropped, no double
/// execution) and the relocated read crosses no statement.  The handles
/// that go dead with the clone: the path's solely-owned ones, and `source`
/// itself on the impure escape.
struct Ownership<'a> {
    refcounts: &'a [u32],
    emit_ranges: &'a [u32],
}

impl Ownership<'_> {
    fn sole(&self, h: naga::Handle<naga::Expression>) -> bool {
        self.refcounts.get(h.index()).copied() == Some(1)
    }

    /// Absent ids (`NO_EMIT`, or a handle past the map) read as not
    /// co-located, which only suppresses a relocation.
    fn same_emit_range(
        &self,
        a: naga::Handle<naga::Expression>,
        b: naga::Handle<naga::Expression>,
    ) -> bool {
        match (
            self.emit_ranges.get(a.index()),
            self.emit_ranges.get(b.index()),
        ) {
            (Some(&x), Some(&y)) => x != NO_EMIT && x == y,
            _ => false,
        }
    }

    fn clonable(
        &self,
        arena: &naga::Arena<naga::Expression>,
        source: naga::Handle<naga::Expression>,
        target: naga::Handle<naga::Expression>,
        path: &[naga::Handle<naga::Expression>],
    ) -> Option<Vec<naga::Handle<naga::Expression>>> {
        let pure = is_pure_to_clone(&arena[source]);
        if !pure
            && !(self.sole(source)
                && path.iter().all(|&h| self.sole(h))
                && self.same_emit_range(source, target))
        {
            return None;
        }
        let mut freed: Vec<_> = path.iter().copied().filter(|&h| self.sole(h)).collect();
        if !pure {
            freed.push(source);
        }
        Some(freed)
    }
}

// MARK: Tests

#[cfg(test)]
#[path = "const_fold_tests.rs"]
mod tests;
