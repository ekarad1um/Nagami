//! Byte-cost pricing.  The savings formulas live with their decisions
//! (each has context-specific terms); this module holds the one shared
//! constant and the inventory of every pricing site:
//!
//! * [`decl_boilerplate`] - `core`'s type-alias savings and
//!   `literal_extract`'s extraction pricing.
//! * `passes::const_hoist` - prices un-rendered IR (the sites above
//!   measure rendered text): `est_lit_len` estimates literal width,
//!   with a 2-character bound-name assumption and the compact-form `8`
//!   embedded in its formula - style-blind by design, passes price for
//!   the compact target.
//! * `expr_emit::emit_zero_init_tail` - every zero-init `var` routes
//!   through it; picks the shorter of `var N:T` / `var N=0i` by direct
//!   comparison, no estimate.
//!
//! A future unified rendered-text model should start from this inventory.

/// Fixed overhead of one `<keyword> <name>=<body>;` module-scope
/// declaration, `const` and `alias` both being 5 characters: compact
/// `const N=D;` = 6+1+1 = 8; beautify `const N = D;\n` = 6+3+2 = 11.
/// Callers must pass their actual output style - pricing compact under
/// beautify accepts borderline extractions that net-cost two bytes per
/// use.
pub(super) fn decl_boilerplate(beautify: bool) -> usize {
    if beautify { 11 } else { 8 }
}
