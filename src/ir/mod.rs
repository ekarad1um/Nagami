//! The IR traversal and rewriting layer beneath the passes and the
//! generator: [`visit`] walks, [`rewrite`] edits.

pub(crate) mod rewrite;
pub(crate) mod visit;
