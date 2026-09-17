//! Whole-arena facts the passes and the generator share: one classification
//! of every expression variant ([`expr_class`]) so the effect, const-ness and
//! emission questions are asked of one table, and one of every statement's
//! memory effects with the summary a call exposes ([`effects`]).

pub(crate) mod effects;
pub(crate) mod expr_class;

pub(crate) use effects::{
    Effect, FnEffects, PointerRoot, compute_fn_effects, resolve_pointer_root, statement_effects,
};
pub(crate) use expr_class::{Classes, ExprClass};
