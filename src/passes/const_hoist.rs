//! Repeated-vector-constant hoisting.
//!
//! A WGSL shader that constructs the same all-literal vector constant in many
//! places (`vec4f(0, 2, 0, 0)` appearing six times, etc.) pays the full
//! constructor text at every site.  This pass finds such constants, creates one
//! shared module `const`, and rewrites each occurrence to a reference to it, so
//! the later [`crate::passes::rename`] pass can give the (now frequent) constant
//! a short name by the same frequency model it uses for everything else.
//!
//! Doing this in the IR - rather than as a post-rename text substitution -
//! is what makes it IDEMPOTENT: the constant participates in renaming on the
//! first pass exactly as it would on any re-minification, so re-minifying the
//! output reproduces the same names.  It is also SAFE by construction: the
//! relocated `Compose` is bit-identical, so sharing it changes no value; the
//! per-pass re-validation is only a structural backstop (it rejects malformed
//! IR, not wrong values).
//!
//! Scope is deliberately narrow: only `Compose` expressions that are a full
//! vector built entirely from plain `Literal` components are hoisted (no
//! nesting, no `Splat`, no non-literal operands), and only when the estimated
//! byte saving is positive at a conservative 2-character bound name.  Global
//! expressions are never scanned (the hoisted constant's own initializer lives
//! there) so the pass reaches a fixed point after one application.

use std::collections::HashMap;

use crate::error::Error;
use crate::pipeline::{Pass, PassContext};

/// A hashable, value-exact key for one literal (floats keyed by bit pattern so
/// `-0.0` and `+0.0` - and NaNs - never collide).
type LitKey = (u8, u64);

fn lit_key(l: naga::Literal) -> LitKey {
    use naga::Literal as L;
    match l {
        L::F32(v) => (0, v.to_bits() as u64),
        L::F64(v) => (1, v.to_bits()),
        L::F16(v) => (2, v.to_bits() as u64),
        L::I32(v) => (3, v as u32 as u64),
        L::U32(v) => (4, v as u64),
        L::I64(v) => (5, v as u64),
        L::U64(v) => (6, v),
        L::Bool(v) => (7, v as u64),
        L::AbstractInt(v) => (8, v as u64),
        L::AbstractFloat(v) => (9, v.to_bits()),
        L::I16(v) => (10, v as u16 as u64),
        L::U16(v) => (11, v as u64),
    }
}

/// Rough minified length of a literal's bare token.  It slightly
/// OVER-estimates the float forms (the emitter may render `.5` for `0.5`),
/// and since the savings model scales the per-use term by `count` while the
/// declaration pays once, an over-estimate biases toward MORE hoists - a
/// marginal hoist can grow output by the (small, float-form-bounded) delta.
/// Pass-local by design: it prices un-rendered IR, unlike the rendered-text
/// pricing sites inventoried in `crate::generator::cost`.
fn est_lit_len(l: naga::Literal) -> usize {
    use naga::Literal as L;
    let s = match l {
        L::Bool(b) => return if b { 4 } else { 5 },
        L::I32(v) => v.to_string(),
        L::U32(v) => v.to_string(),
        L::I64(v) => v.to_string(),
        L::U64(v) => v.to_string(),
        L::AbstractInt(v) => v.to_string(),
        L::F32(v) => (v as f64).to_string(),
        L::F64(v) => v.to_string(),
        L::AbstractFloat(v) => v.to_string(),
        L::F16(v) => v.to_f32().to_string(),
        // i16/u16 always emit via the constructor form `i16(N)` / `u16(N)`
        // (there is no bare int16 literal in WGSL), so price that width.
        L::I16(v) => format!("i16({v})"),
        L::U16(v) => format!("u16({v})"),
    };
    s.len().max(1)
}

/// Identifies where a hoistable `Compose` lives so the rewrite phase can reach
/// it after the immutable collection phase.
#[derive(Clone, Copy)]
enum FuncRef {
    Function(naga::Handle<naga::Function>),
    EntryPoint(usize),
}

/// One hoistable vector `Compose`: where it lives (`loc`), its expression
/// handle (Phase 4 overwrites that slot in place and records its Emit-range
/// removal), its vector type, and its component literals - which form both the
/// grouping key and the hoisted initializer.
struct Candidate {
    loc: FuncRef,
    handle: naga::Handle<naga::Expression>,
    ty: naga::Handle<naga::Type>,
    lits: Vec<naga::Literal>,
}

/// If `expr` is a full-width vector `Compose` worth hoisting - every component a
/// plain `Literal`, a standard concrete element type, and NOT a splat - return
/// the literal values; else `None`.
///
/// Restrictions, each closing a measured corpus regression:
/// - element type limited to `f32` / `i32` / `u32` (the standard WGSL vector
///   scalars).  A `vec4<f64>` is valid only inside naga's internal expression
///   space; emitting it as a standalone `const`/alias is tint-rejected
///   ("unresolved type 'f64'").  `f16` / 64-bit / bool are excluded
///   conservatively (their decls/suffixes break the simple cost model).
/// - splat composes (all lanes equal) are excluded: they emit in the short
///   `vecNf(x)` splat form, which the generic-length cost model over-prices, so
///   hoisting a `vec2i(1)` used twice would GROW the output.
fn full_literal_vector(
    expr: &naga::Expression,
    types: &naga::UniqueArena<naga::Type>,
    arena: &naga::Arena<naga::Expression>,
) -> Option<(naga::Handle<naga::Type>, Vec<naga::Literal>)> {
    let naga::Expression::Compose { ty, components } = expr else {
        return None;
    };
    let naga::TypeInner::Vector { size, scalar } = types[*ty].inner else {
        return None;
    };
    use naga::ScalarKind::{Float, Sint, Uint};
    let standard = matches!(
        (scalar.kind, scalar.width),
        (Float, 4) | (Sint, 4) | (Uint, 4)
    );
    if !standard || components.len() != size as usize {
        return None;
    }
    let mut lits = Vec::with_capacity(components.len());
    for &c in components {
        match &arena[c] {
            naga::Expression::Literal(l) => lits.push(*l),
            _ => return None,
        }
    }
    // Exclude splats (all lanes equal by bit pattern).
    if lits[1..].iter().all(|&l| lit_key(l) == lit_key(lits[0])) {
        return None;
    }
    Some((*ty, lits))
}

/// Collect every expression handle that appears in an `Emit` range anywhere in
/// `block` (recursing into nested control flow).  A vector `Compose` is live -
/// and so worth hoisting - exactly when it is emitted, i.e. present in some
/// Emit range; composes left dead in the arena (e.g. the initializer of a
/// DCE'd `var` in an empty function) are NOT, and hoisting them would emit a
/// `const` no statement references, growing the output.  Each handle is emitted
/// at most once, so a plain `Vec` collects the live set without deduplication
/// and lets the caller scan only those handles rather than the whole arena.
fn collect_emitted(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Expression>>) {
    for stmt in block.iter() {
        if let naga::Statement::Emit(range) = stmt {
            out.extend(range.clone());
        }
        for nested in super::expr_util::nested_blocks(stmt) {
            collect_emitted(nested, out);
        }
    }
}

/// Pass entry point; the algorithm and its scope limits are the
/// module-level docs.
pub struct ConstHoistPass;

impl Pass for ConstHoistPass {
    fn name(&self) -> &'static str {
        "const-hoist"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        // Phase 1: collect every hoistable vector-literal Compose in every
        // function and entry-point body (global expressions are skipped).
        let mut candidates: Vec<Candidate> = Vec::new();
        let collect = |loc: FuncRef,
                       func: &naga::Function,
                       types: &naga::UniqueArena<naga::Type>,
                       out: &mut Vec<Candidate>| {
            let mut emitted = Vec::new();
            collect_emitted(&func.body, &mut emitted);
            for h in emitted {
                if let Some((ty, lits)) =
                    full_literal_vector(&func.expressions[h], types, &func.expressions)
                {
                    out.push(Candidate {
                        loc,
                        handle: h,
                        ty,
                        lits,
                    });
                }
            }
        };
        for (fh, func) in module.functions.iter() {
            collect(FuncRef::Function(fh), func, &module.types, &mut candidates);
        }
        for (i, ep) in module.entry_points.iter().enumerate() {
            collect(
                FuncRef::EntryPoint(i),
                &ep.function,
                &module.types,
                &mut candidates,
            );
        }
        if candidates.is_empty() {
            return Ok(false);
        }

        // Phase 2: group by (type, literal bits) and keep the profitable groups.
        // Profit is modelled at a conservative 2-char bound name: a hoist that
        // is not clearly worthwhile is left inline (the rename pass may give the
        // constant a 1-char name, so this only ever UNDER-hoists - never grows).
        // Key = (vector type index, per-lane literal bits).
        type GroupKey = (usize, Vec<LitKey>);
        let mut groups: HashMap<GroupKey, Vec<usize>> = HashMap::new();
        for (idx, c) in candidates.iter().enumerate() {
            let key = (c.ty.index(), c.lits.iter().map(|&l| lit_key(l)).collect());
            groups.entry(key).or_default().push(idx);
        }

        // Deterministic order: sort group keys so const creation (and thus the
        // names rename later assigns) does not depend on HashMap iteration order.
        let mut group_list: Vec<(GroupKey, Vec<usize>)> = groups.into_iter().collect();
        group_list.sort_by(|a, b| a.0.cmp(&b.0));

        // Names the placeholder must avoid.  The load-bearing case is a
        // preserve-listed (preamble) name: rename keeps such a name verbatim, and
        // if it matches a preamble symbol the generator suppresses the hoisted
        // declaration as preamble-owned, silently rebinding every use to the
        // preamble's (different) value.  The raw `_hoist{constants.len()}` scheme
        // can hit this because preamble consts count in `constants.len()`.  The
        // remaining module-level names are belt-and-suspenders (rename mangles a
        // non-preserved placeholder to a fresh unique name anyway).  Seed the
        // avoid-set once and record each minted name so repeated hoists differ.
        let mut reserved_names: std::collections::HashSet<String> =
            ctx.config.preserve_symbols.iter().cloned().collect();
        for (_, c) in module.constants.iter() {
            if let Some(n) = c.name.as_deref() {
                reserved_names.insert(n.to_string());
            }
        }
        for (_, g) in module.global_variables.iter() {
            if let Some(n) = g.name.as_deref() {
                reserved_names.insert(n.to_string());
            }
        }
        for (_, ov) in module.overrides.iter() {
            if let Some(n) = ov.name.as_deref() {
                reserved_names.insert(n.to_string());
            }
        }
        for (_, f) in module.functions.iter() {
            if let Some(n) = f.name.as_deref() {
                reserved_names.insert(n.to_string());
            }
        }
        for ep in module.entry_points.iter() {
            reserved_names.insert(ep.name.clone());
        }
        for (_, ty) in module.types.iter() {
            if let Some(n) = ty.name.as_deref() {
                reserved_names.insert(n.to_string());
            }
        }

        const NAME_LEN: usize = 2;
        let mut changed = false;
        // Per-function set of handles converted Compose -> Constant, so their
        // Emit ranges can be rebuilt afterwards (a `Constant` is not emittable).
        type ExprSet = std::collections::HashSet<naga::Handle<naga::Expression>>;
        let mut hoisted_fn: HashMap<naga::Handle<naga::Function>, ExprSet> = HashMap::new();
        let mut hoisted_ep: HashMap<usize, ExprSet> = HashMap::new();

        for (_key, members) in group_list {
            let count = members.len();
            // Require >=3 IR occurrences: a group of exactly 2 can be cut to a
            // single emitted use by downstream CSE / copy-prop, leaving the
            // hoisted `const` referenced once - pure overhead that grows the
            // output.
            if count < 3 {
                continue;
            }
            let rep = &candidates[members[0]];
            // Estimated inline length: aliased type name (~2) + parens + literal
            // tokens + separators.  vecN constructor with all-literal args never
            // collapses to a swizzle, only to a splat/zero form; those short
            // forms are filtered out by the savings check below anyway.
            let lit_len: usize = rep.lits.iter().map(|&l| est_lit_len(l)).sum();
            let inline_len = 2 + 2 + lit_len + rep.lits.len().saturating_sub(1);
            // savings = count*(inline - name) - (decl boilerplate + name + decl body)
            let savings = (count as isize) * (inline_len as isize - NAME_LEN as isize)
                - (8 + NAME_LEN as isize + inline_len as isize);
            if savings <= 0 {
                continue;
            }

            // Phase 3: materialise the shared constant.  Re-create the literal
            // components and the Compose in `global_expressions` (const-exprs
            // must live there), then a named `Constant` over them.  `name` is a
            // throwaway placeholder; the rename pass replaces it.
            let ty = rep.ty;
            let comp_handles: Vec<naga::Handle<naga::Expression>> = rep
                .lits
                .iter()
                .map(|&l| {
                    module
                        .global_expressions
                        .append(naga::Expression::Literal(l), naga::Span::UNDEFINED)
                })
                .collect();
            let init = module.global_expressions.append(
                naga::Expression::Compose {
                    ty,
                    components: comp_handles,
                },
                naga::Span::UNDEFINED,
            );
            // Mint a collision-free placeholder: start at the arena length and
            // walk forward until the name is fresh in `reserved_names` (which
            // also records it, keeping later hoists in this loop distinct).
            let mut suffix = module.constants.len();
            let hoist_name = loop {
                let cand = format!("_hoist{}", suffix);
                if reserved_names.insert(cand.clone()) {
                    break cand;
                }
                suffix += 1;
            };
            let const_handle = module.constants.append(
                naga::Constant {
                    name: Some(hoist_name),
                    ty,
                    init,
                },
                naga::Span::UNDEFINED,
            );

            // Phase 4: rewrite every occurrence's Compose to reference the const.
            // The orphaned literal components in each function arena become dead
            // and are removed by the later compaction/DCE pass.
            for &m in &members {
                let cand = &candidates[m];
                let expr_slot = match cand.loc {
                    FuncRef::Function(fh) => {
                        hoisted_fn.entry(fh).or_default().insert(cand.handle);
                        &mut module.functions[fh].expressions[cand.handle]
                    }
                    FuncRef::EntryPoint(i) => {
                        hoisted_ep.entry(i).or_default().insert(cand.handle);
                        &mut module.entry_points[i].function.expressions[cand.handle]
                    }
                };
                *expr_slot = naga::Expression::Constant(const_handle);
            }
            changed = true;
        }

        // A `Constant` is not emittable, so every converted handle must be
        // dropped from its `Emit` range (the expression itself stays in the
        // arena at the same index, preserving topological order; only the Emit
        // bookkeeping changes).
        for (fh, removed) in &hoisted_fn {
            crate::passes::expr_util::rebuild_emit_ranges_after_removal(
                &mut module.functions[*fh].body,
                removed,
            );
        }
        for (i, removed) in &hoisted_ep {
            crate::passes::expr_util::rebuild_emit_ranges_after_removal(
                &mut module.entry_points[*i].function.body,
                removed,
            );
        }

        Ok(changed)
    }
}
