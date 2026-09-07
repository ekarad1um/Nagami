//! Repeated-vector-constant hoisting: an all-literal vector constant built
//! at many sites (`vec4f(0, 2, 0, 0)` six times) becomes one shared module
//! `const`, so the rename pass can give the now-frequent constant a short
//! name by its usual frequency model.
//!
//! Done in the IR rather than as a post-rename text substitution so the
//! result is idempotent: the constant takes part in renaming exactly as it
//! would on any re-minification.  Safe by construction: the relocated
//! `Compose` is bit-identical, so sharing it changes no value (per-pass
//! re-validation only rejects malformed IR, not wrong values).
//!
//! Scope is deliberately narrow: only full vectors built entirely from
//! plain `Literal` components (no nesting, no `Splat`), and only when the
//! estimated saving is positive at a conservative 2-character bound name.
//! Global expressions are never scanned (the hoisted initializer lives
//! there), so the pass reaches a fixed point after one application.

use rustc_hash::FxHashMap;

use crate::error::Error;
use crate::handle_set::{HandleMap, HandleSet};
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

/// Rough minified length of a literal's bare token.  Over-estimates the
/// float forms (the emitter may render `.5` for `0.5`); since the per-use
/// term scales by `count` while the declaration pays once, that biases
/// toward more hoists, and a marginal hoist can grow output by the small
/// float-form-bounded delta.  Prices un-rendered IR, unlike the
/// rendered-text pricing in `crate::generator::cost`.
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

#[derive(Clone, Copy)]
enum FuncRef {
    Function(naga::Handle<naga::Function>),
    EntryPoint(usize),
}

/// One hoistable vector `Compose`; the rewrite overwrites `handle`'s slot in
/// place, and `lits` is both the grouping key and the hoisted initializer.
struct Candidate {
    loc: FuncRef,
    handle: naga::Handle<naga::Expression>,
    ty: naga::Handle<naga::Type>,
    lits: Vec<naga::Literal>,
}

/// The literal lanes of a full-width vector `Compose` worth hoisting.
/// Restrictions, each closing a measured corpus regression: element type
/// limited to `f32` / `i32` / `u32` (a standalone `const` of `vec4<f64>` is
/// tint-rejected, "unresolved type 'f64'"; `f16` / 64-bit / bool
/// declarations and suffixes break the simple cost model); splats (all
/// lanes equal) excluded, since their short `vecNf(x)` form is over-priced
/// by the generic-length model and hoisting a `vec2i(1)` used twice would
/// grow the output.
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
    if lits[1..].iter().all(|&l| lit_key(l) == lit_key(lits[0])) {
        return None;
    }
    Some((*ty, lits))
}

/// Every handle in an `Emit` range under `block`: a `Compose` is live, and
/// worth hoisting, exactly when emitted; composes left dead in the arena
/// (the initializer of a DCE'd `var`) would hoist a `const` no statement
/// references, growing the output.  Each handle is emitted at most once, so
/// a plain `Vec` needs no deduplication.
fn collect_emitted(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Expression>>) {
    super::expr_util::for_each_statement(block, &mut |stmt| {
        if let naga::Statement::Emit(range) = stmt {
            out.extend(range.clone());
        }
    });
}

/// Hoists repeated all-literal vector constants into shared module constants.
pub struct ConstHoistPass;

impl Pass for ConstHoistPass {
    fn name(&self) -> &'static str {
        "const-hoist"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
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

        // Profit at a conservative 2-char bound name: rename may hand out a
        // 1-char name, so this only ever under-hoists.
        type GroupKey = (usize, Vec<LitKey>);
        let mut groups: FxHashMap<GroupKey, Vec<usize>> = Default::default();
        for (idx, c) in candidates.iter().enumerate() {
            let key = (c.ty.index(), c.lits.iter().map(|&l| lit_key(l)).collect());
            groups.entry(key).or_default().push(idx);
        }

        // Sorted so const creation, hence the names rename assigns, does not
        // depend on hash order.
        let mut group_list: Vec<(GroupKey, Vec<usize>)> = groups.into_iter().collect();
        group_list.sort_by(|a, b| a.0.cmp(&b.0));

        // The placeholder must avoid preserve-listed (preamble) names: rename
        // keeps those verbatim, and a match makes the generator suppress the
        // hoisted declaration as preamble-owned, silently rebinding every use
        // to the preamble's value; preamble consts count in `constants.len()`,
        // so the raw `_hoist{n}` scheme can hit one.  Other module names are
        // belt-and-suspenders (rename mangles any non-preserved placeholder).
        let mut reserved_names: std::collections::HashSet<String> =
            ctx.config.preserve_symbols.iter().cloned().collect();
        reserved_names.extend(
            crate::name_gen::module_scope_names(module)
                .chain(crate::name_gen::type_names(module))
                .map(str::to_owned),
        );

        const NAME_LEN: usize = 2;
        let mut changed = false;
        // Per-function set of handles converted Compose -> Constant, so their
        // Emit ranges can be rebuilt afterwards (a `Constant` is not emittable).
        type ExprSet = HandleSet<naga::Expression>;
        let mut hoisted_fn: HandleMap<naga::Function, ExprSet> = Default::default();
        let mut hoisted_ep: FxHashMap<usize, ExprSet> = Default::default();

        for (_key, members) in group_list {
            let count = members.len();
            // A group of exactly 2 can be cut to one emitted use by downstream
            // CSE / copy-prop, leaving a `const` referenced once: pure overhead.
            if count < 3 {
                continue;
            }
            let rep = &candidates[members[0]];
            // Inline length: aliased type name (~2) + parens + literal tokens +
            // separators; splat / zero forms are excluded upstream, so the full
            // token list is the right price.
            let lit_len: usize = rep.lits.iter().map(|&l| est_lit_len(l)).sum();
            let inline_len = 2 + 2 + lit_len + rep.lits.len().saturating_sub(1);
            // savings = count*(inline - name) - (decl boilerplate + name + decl body)
            let savings = (count as isize) * (inline_len as isize - NAME_LEN as isize)
                - (8 + NAME_LEN as isize + inline_len as isize);
            if savings <= 0 {
                continue;
            }

            // Const-exprs must live in `global_expressions`; the name is a
            // placeholder rename replaces.
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
            // `reserved_names` records each minted name, keeping later hoists
            // distinct.
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

            // The orphaned literal components become dead for the next
            // compaction.
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

        // The expression keeps its arena slot (topological order preserved);
        // only the Emit bookkeeping changes.
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
