//! Expression ref-count analysis: how many live consumers each
//! expression handle has, and which handles appear in `Emit` ranges.
//!
//! A count is a stand-in for how many times the emitter will render the
//! expression, and several emission-time collapses render one operand where
//! the IR holds several references to it.  Every such collapse needs a
//! discount here or the pricing binds a `let` the text then uses once - a
//! second pass, whose input already carries the collapsed spelling, drops it.

use super::twins::Twins;
use crate::handle_set::HandleSet;
use crate::passes::expr_util::RefCount;

use crate::generator::core::FunctionExprInfo;
use crate::generator::expr_emit::{
    child_needs_parens, compose_is_all_zero, compose_is_splat, matrix_flatten_scalars,
    swizzle_component, unary_child_needs_parens,
};

use crate::analysis::{Classes, ExprClass};

/// [`crate::passes::expr_util::live_expression_ref_counts`] in the shape the
/// generator caches it, with the parenthesised uses beside the counts.
pub(super) fn compute_expression_ref_counts(func: &naga::Function) -> FunctionExprInfo {
    let (ref_counts, live) = crate::passes::expr_util::live_expression_ref_counts(func);
    let paren_uses = paren_uses_in(&func.expressions, &live);
    FunctionExprInfo {
        ref_counts,
        live,
        paren_uses,
    }
}

/// Per expression, how many of its live consumers wrap its text in
/// parentheses when it renders inline there - a binary or unary operand
/// under an operator that binds tighter or a grammar level that takes no
/// bare operand ([`child_needs_parens`]), any operator under a postfix
/// access.  The byte rule of the binder charges an inlined use these two
/// bytes; the name a `let` gives the value never takes them.
fn paren_uses_in(expressions: &naga::Arena<naga::Expression>, live: &[bool]) -> Vec<u16> {
    let mut uses = vec![0u16; expressions.len()];
    let mut wrap = |h: naga::Handle<naga::Expression>| {
        uses[h.index()] = uses[h.index()].saturating_add(1);
    };
    for (h, expr) in expressions.iter() {
        if !live[h.index()] {
            continue;
        }
        match expr {
            naga::Expression::Binary { op, left, right } => {
                if child_needs_parens(*left, expressions, *op, false, false) {
                    wrap(*left);
                }
                if child_needs_parens(*right, expressions, *op, true, false) {
                    wrap(*right);
                }
            }
            naga::Expression::Unary { expr, .. } => {
                if unary_child_needs_parens(*expr, expressions, false) {
                    wrap(*expr);
                }
            }
            naga::Expression::AccessIndex { base, .. }
            | naga::Expression::Access { base, .. }
            | naga::Expression::Swizzle { vector: base, .. } => {
                if matches!(
                    expressions[*base],
                    naga::Expression::Binary { .. }
                        | naga::Expression::Unary { .. }
                        | naga::Expression::Select { .. }
                ) {
                    wrap(*base);
                }
            }
            _ => {}
        }
    }
    uses
}

/// Discount references made from initializer trees that no body `let` can
/// serve: a hoisted `var x = <init>;` renders its tree at the top of the body,
/// before any `Emit` range is priced, and a deferred or dead local never
/// renders it; only a `for`-header local renders its init after the `Emit`,
/// where a name does serve it.  Without the discount a sub-expression the tree
/// shares (`array(l(1,0), .., l(1,0))` with one `l(1,0)`) is bound on the tree's
/// account alone, dead text.  Each child occurrence counts once, so a subtree
/// two initializers share is discounted once.
pub(super) fn discount_initializer_refs(
    func: &naga::Function,
    for_loop_vars: &[Option<naga::Handle<naga::Expression>>],
    counts: &mut [RefCount],
) {
    let mut visited = vec![false; func.expressions.len()];
    let mut stack: Vec<_> = func
        .local_variables
        .iter()
        .filter(|(lh, _)| for_loop_vars[lh.index()].is_none())
        .filter_map(|(_, local)| local.init)
        .collect();
    while let Some(h) = stack.pop() {
        if std::mem::replace(&mut visited[h.index()], true) {
            continue;
        }
        crate::ir::visit::visit_expression_children(&func.expressions[h], |child| {
            counts[child.index()] = counts[child.index()].saturating_sub(1);
            stack.push(child);
        });
    }
}

/// Base of a `Compose` component the emitter folds into a swizzle, through
/// [`swizzle_component`]; types resolve against `finfo` because the census
/// runs before any `FunctionCtx` exists.
fn swizzle_component_base<'t>(
    handle: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    finfo: &'t naga::valid::FunctionInfo,
    types: &'t naga::UniqueArena<naga::Type>,
    twins: &Twins,
) -> Option<naga::Handle<naga::Expression>> {
    swizzle_component(handle, expressions, types, &|b| {
        finfo[b].ty.inner_with(types)
    })
    .map(|(base, _)| twins.first_of(base).unwrap_or(base))
}

/// Whether `h`'s operand cone holds a statement result or an expensive
/// operation, both of which make an over-discount worse than a byte loss: a
/// stashed call re-rendered at a second use runs twice
/// (`find_inlineable_calls` reads these counts after this walk and gates on
/// `== 1`), and `loop_sunk_work` priced its bound-or-inlined model on the
/// counts before it, so a fetch it believes bound could sink into a loop
/// unpinned.  The classes are computed once, on first demand.
fn cone_has_call_or_image(
    h: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    classes: &mut Option<Classes>,
) -> bool {
    classes.get_or_insert_with(|| Classes::of(expressions))[h]
        .any(ExprClass::STMT_RESULT_IN_CONE | ExprClass::EXPENSIVE_IN_CONE)
}

/// Discount the component references a `Compose` will not render: the
/// census counts slots, and three of the emitter's folds render fewer
/// operands than the arena holds slots.  Mirrors the chain of `emit_expr`'s
/// `Compose` arm, in its order:
///
/// 1. all-zero -> `vecNf()` / `matNxMf()`: no component renders at all;
/// 2. same-base swizzle over every component -> `v.xyz`, or a run of them ->
///    `vec4(a.xy,0,1)`: the base renders once per run;
/// 3. splat -> `vec4f(d)`: only slot 0 renders.
///
/// Only 2 and 3 discount.  1 is exact and would be one line, but its
/// components are the zero leaves a `let` never binds anyway, so a discount
/// there changes nothing; it is recognised only to keep 2 and 3 off a
/// `Compose` the emitter collapses before trying them.
///
/// 2 compares bases as values (a lane read off a twin is a lane of the
/// first spelling, as the emitter folds it) and declines a component the
/// emitter has bound, which is not known yet, so a run forms only where the
/// emitter is certain to inline every component (a count of 1 never binds
/// by bytes, and `must_bind` covers the force-bind).  3 reads the slots
/// alone and is withheld from a column the matrix arm may flatten.  Both
/// decline a cone holding a call or an image operation
/// ([`cone_has_call_or_image`]); the over-discount paths that remain (a
/// component the render-depth cap or a const-hazard binding names anyway, a
/// run in a column the matrix arm does flatten) cost bytes, not meaning: the
/// operand inlines at every use instead of binding once.
pub(super) fn discount_compose_folds(
    func: &naga::Function,
    finfo: &naga::valid::FunctionInfo,
    types: &naga::UniqueArena<naga::Type>,
    must_bind: &HandleSet<naga::Expression>,
    live: &[bool],
    twins: &Twins,
    counts: &mut [RefCount],
) {
    let expressions = &func.expressions;
    // Only a live `Compose` (`live`: in an `Emit` range, the census's own
    // bitmap) had its components counted, so only a live one is owed a
    // discount.  The splat discount is withheld from a column of a matrix
    // the emitter may flatten (`matrix_flatten_scalars`): the flat form
    // spells a splat column's operand once per row, and which form wins
    // turns on rendered lengths unknown here.
    let mut matrix_column = vec![false; expressions.len()];
    for (mh, expr) in expressions.iter() {
        if !live[mh.index()] {
            continue;
        }
        let naga::Expression::Compose { ty, components } = expr else {
            continue;
        };
        if matrix_flatten_scalars(*ty, components, types, expressions).is_some() {
            for &c in components.iter() {
                matrix_column[c.index()] = true;
            }
        }
    }

    // Decisions read the counts as the census left them and land together, so
    // one run's discount cannot change whether a later run forms.
    let mut deltas = vec![0 as RefCount; expressions.len()];
    let mut cone_memo = None;
    let inlined = |c: naga::Handle<naga::Expression>, counts: &[RefCount]| {
        counts[c.index()] == 1 && !must_bind.contains(c)
    };

    for (ch, expr) in expressions.iter() {
        if !live[ch.index()] {
            continue;
        }
        let naga::Expression::Compose { ty, components } = expr else {
            continue;
        };
        // The all-zero fold takes matrices too; the swizzle and splat folds
        // are vector-only.
        let vector_size = match types[*ty].inner {
            naga::TypeInner::Vector { size, .. } => Some(size as usize),
            naga::TypeInner::Matrix { .. } => None,
            _ => continue,
        };
        if components.len() < 2 {
            continue;
        }
        // 1. All-zero: recognised, not discounted.
        if counts[ch.index()] <= 1
            && components
                .iter()
                .all(|&c| compose_is_all_zero(c, expressions, &|_| false))
        {
            continue;
        }
        let Some(vector_size) = vector_size.filter(|_| components.len() <= 4) else {
            continue;
        };
        // 3. Splat: slot 0 alone renders.
        if components.len() == vector_size && compose_is_splat(components, expressions, &|_| false)
        {
            if !matrix_column[ch.index()] {
                for &c in components[1..].iter() {
                    if !cone_has_call_or_image(c, expressions, &mut cone_memo) {
                        deltas[c.index()] += 1;
                    }
                }
            }
            continue;
        }
        // 2. Runs of same-base swizzle components.
        let mut i = 0;
        while i < components.len() {
            let Some(base) =
                swizzle_component_base(components[i], expressions, finfo, types, twins)
                    .filter(|_| inlined(components[i], counts))
            else {
                i += 1;
                continue;
            };
            let mut j = i + 1;
            while j < components.len()
                && inlined(components[j], counts)
                && swizzle_component_base(components[j], expressions, finfo, types, twins)
                    == Some(base)
            {
                j += 1;
            }
            if j - i >= 2 && !cone_has_call_or_image(base, expressions, &mut cone_memo) {
                deltas[base.index()] += (j - i - 1) as RefCount;
            }
            i = j;
        }
    }

    for (i, d) in deltas.into_iter().enumerate() {
        if d != 0 {
            counts[i] = counts[i].saturating_sub(d);
        }
    }
}
