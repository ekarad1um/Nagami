//! Repeated-constant hoisting: a vector or matrix constant built at several
//! sites (`vec4f(0, 2, 0, 0)` in three functions), or the zero of such a
//! type (`vec4f()`, which the emitter never `let`-binds: a `ZeroValue` has
//! no `Emit` to bind at), becomes one shared module `const`, so the rename
//! pass can give the now-frequent constant a short name by its usual
//! frequency model.
//!
//! Done in the IR rather than as a post-rename text substitution so the
//! result is idempotent: the constant takes part in renaming exactly as it
//! would on any re-minification.  Safe by construction: the relocated
//! constructor is bit-identical, so sharing it changes no value (per-pass
//! re-validation only rejects malformed IR, not wrong values).
//!
//! Priced by the generator itself.  A per-site model (`Pricer`: each
//! site's rendered text, whether the emitter would `let`-bind it, the `let`
//! name it would take and the name rename will give the constant) shortlists
//! the groups that can pay; each is then confirmed by emitting the renamed
//! module with the hoist applied and comparing lengths.  The per-site model
//! alone over-hoists: a new module-scope name pushes the lightest holder of
//! a single letter - often a type alias with fifty uses - to two letters,
//! and the vector's lanes stop paying for a literal's extraction; modelling
//! those means re-running the alias planner and the literal census on the
//! hypothetical module, which is what rendering it does.  Global
//! expressions are never scanned (the hoisted initializer lives there), so
//! the pass reaches a fixed point after one application.

use rustc_hash::FxHashMap;

use crate::error::Error;
use crate::generator::price::Pricer;
use crate::handle_set::{HandleMap, HandleSet};
use crate::passes::expr_util::{KeyToken, lit_key};
use crate::pipeline::{Pass, PassContext};

/// Append the value key of the constant cone under `h`: constructors by
/// type and arity, lanes by literal bits or zero type, a constant by its
/// initializer's key.  `false` on any other node - an `Override` is not a
/// value, and a runtime lane makes the cone a runtime value.  Keyed by
/// value, not spelling, so a re-minification finds the constant the first
/// pass declared instead of minting a twin: the emitter inlines a scalar
/// constant as its literal, and prints a vector of one repeated literal as
/// the splat of it, which re-parse as a `Literal` lane and a `Splat`.
fn const_cone_key(
    h: naga::Handle<naga::Expression>,
    arena: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    key: &mut Vec<KeyToken>,
) -> bool {
    let literal = |h: naga::Handle<naga::Expression>| match arena[h] {
        naga::Expression::Literal(l) => Some(l),
        naga::Expression::Constant(c) => {
            match module.global_expressions[module.constants[c].init] {
                naga::Expression::Literal(l) => Some(l),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(l) = literal(h) {
        key.push(lit_key(l));
        return true;
    }
    match &arena[h] {
        naga::Expression::Constant(c) => {
            let init = module.constants[*c].init;
            return const_cone_key(init, &module.global_expressions, module, key);
        }
        naga::Expression::ZeroValue(ty) => key.push((17, ty.index() as u64)),
        naga::Expression::Splat { size, value } => {
            key.push((18, *size as u64));
            return const_cone_key(*value, arena, module, key);
        }
        naga::Expression::Compose { ty, components } => {
            if let naga::TypeInner::Vector { size, .. } = module.types[*ty].inner
                && let Some(first) = literal(components[0])
                && components.len() == size as usize
                && components
                    .iter()
                    .all(|&c| literal(c).is_some_and(|l| lit_key(l) == lit_key(first)))
            {
                key.push((18, size as u64));
                key.push(lit_key(first));
                return true;
            }
            key.push((19, ty.index() as u64));
            key.push((20, components.len() as u64));
            return components
                .iter()
                .all(|&c| const_cone_key(c, arena, module, key));
        }
        _ => return false,
    }
    true
}

/// Whether a constant cone rooted at a constructor of `ty` may become a
/// module `const`: vectors of `f32` / `i32` / `u32` and `f32` matrices.  A
/// standalone `const` of `vec4<f64>` is tint-rejected ("unresolved type
/// 'f64'"), `f16` needs its enable, and a bool or 64-bit vector is too
/// rare to earn a rule.
fn hoistable_type(ty: naga::Handle<naga::Type>, types: &naga::UniqueArena<naga::Type>) -> bool {
    use naga::ScalarKind::{Float, Sint, Uint};
    match types[ty].inner {
        naga::TypeInner::Vector { scalar, .. } => {
            matches!(
                (scalar.kind, scalar.width),
                (Float, 4) | (Sint, 4) | (Uint, 4)
            )
        }
        naga::TypeInner::Matrix { scalar, .. } => matches!((scalar.kind, scalar.width), (Float, 4)),
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FuncRef {
    Function(naga::Handle<naga::Function>),
    EntryPoint(usize),
}

/// One hoistable constructor; the rewrite overwrites `handle`'s slot in
/// place, and `key` groups the sites that build the same value.  `zero`
/// is the type of a `ZeroValue` site, whose bare spelling the shortlist
/// prices.
struct Candidate {
    loc: FuncRef,
    handle: naga::Handle<naga::Expression>,
    key: Vec<KeyToken>,
    zero: Option<naga::Handle<naga::Type>>,
    /// Live consumers: how often the site renders unless the emitter
    /// binds it.
    uses: u32,
}

/// What one site costs today and what its `const` reference would: the
/// bytes it spells today - `let`-bound by the emitter, the `let` and its
/// name at every use, else its text at every use - its text as a
/// declaration initializer (type spelled), and its renderings.  A
/// `ZeroValue` site that is `B()` under a type alias may be what pays for
/// the alias, so its today is priced with the type spelled in full: an
/// upper bound that admits the group to the render, which knows whether
/// the alias survives the hoist.  A constructor's bare use is its use.
struct SitePrice {
    today: usize,
    decl_len: usize,
    uses: usize,
}

/// The constructors of `func` worth grouping: emitted (in an `Emit` range)
/// AND consumed - a constructor left dead in the arena (the initializer of
/// a DCE'd `var`) or emitted but consumed by nothing (`const_fold`
/// materialises a vector at `arr[1]` and then folds the `.z` that read it,
/// leaving it in its range until `compact` culls it) would hoist a `const`
/// no statement references, growing the output - a hoistable constant cone,
/// and not a lane of an emitted vector or matrix constructor: the emitter
/// renders a nested constructor through such a parent (a matrix flattens
/// its columns to scalars), so the parent is the candidate and its lanes
/// are priced as part of it.  An array or struct constructor spells its
/// lanes as they are, so those stay candidates.  A `ZeroValue` of a
/// hoistable type is a candidate on the same terms less the range: it is
/// pre-emit, so consumers are its only liveness, and the counts come from
/// live consumers alone.
fn collect(loc: FuncRef, func: &naga::Function, module: &naga::Module, out: &mut Vec<Candidate>) {
    let types = &module.types;
    let arena = &func.expressions;
    let (counts, live) = super::expr_util::live_expression_ref_counts(func);
    let mut lane = vec![false; arena.len()];
    for (h, expr) in arena.iter() {
        if live[h.index()]
            && let naga::Expression::Compose { ty, components } = expr
            && matches!(
                types[*ty].inner,
                naga::TypeInner::Vector { .. } | naga::TypeInner::Matrix { .. }
            )
        {
            for &c in components {
                lane[c.index()] = true;
            }
        }
    }
    for (h, expr) in arena.iter() {
        if counts[h.index()] == 0 || lane[h.index()] {
            continue;
        }
        let (ty, zero) = match *expr {
            naga::Expression::Compose { ty, .. } if live[h.index()] => (ty, None),
            naga::Expression::ZeroValue(ty) => (ty, Some(ty)),
            _ => continue,
        };
        if !hoistable_type(ty, types) {
            continue;
        }
        let mut key = Vec::new();
        if const_cone_key(h, arena, module, &mut key) {
            out.push(Candidate {
                loc,
                handle: h,
                key,
                zero,
                uses: counts[h.index()],
            });
        }
    }
}

/// The named constants whose initializer is a hoistable constant cone, by
/// value: a site building that value references the constant instead of
/// minting a twin - the source's own `const`, or one the previous
/// minification hoisted, whose value a second pass builds afresh where it
/// folds arithmetic over that constant.
fn anchors(module: &naga::Module) -> Vec<(Vec<KeyToken>, naga::Handle<naga::Constant>)> {
    let mut out = Vec::new();
    for (ch, c) in module.constants.iter() {
        if c.name.is_none() || !hoistable_type(c.ty, &module.types) {
            continue;
        }
        let mut key = Vec::new();
        if const_cone_key(c.init, &module.global_expressions, module, &mut key) {
            out.push((key, ch));
        }
    }
    out
}

/// The constant already declaring a value, if any, and the candidate sites
/// building it.
type Group = (Option<naga::Handle<naga::Constant>>, Vec<usize>);

/// Repeated-constant hoisting; the module doc has the model.
pub struct ConstHoistPass;

impl Pass for ConstHoistPass {
    fn name(&self) -> &'static str {
        "const-hoist"
    }

    fn run(&mut self, module: &mut naga::Module, ctx: &PassContext<'_>) -> Result<bool, Error> {
        let mut candidates: Vec<Candidate> = Vec::new();
        for (fh, func) in module.functions.iter() {
            collect(FuncRef::Function(fh), func, module, &mut candidates);
        }
        for (i, ep) in module.entry_points.iter().enumerate() {
            collect(
                FuncRef::EntryPoint(i),
                &ep.function,
                module,
                &mut candidates,
            );
        }

        // A group is the sites of one value, plus the constant that already
        // declares it, if any; without one it takes two renderings to share
        // a declaration - two sites, or one site the emitter spells at every
        // use (a `ZeroValue` is never `let`-bound; a constructor with that
        // many uses is, and the model declines it without a render).  Groups
        // are numbered in order of first site, so const creation, hence the
        // names rename assigns, does not depend on hash order (which differs
        // between the native and the wasm build).
        let anchors = anchors(module);
        let mut by_key: FxHashMap<&[KeyToken], usize> = Default::default();
        let mut groups: Vec<Group> = Vec::new();
        for (idx, c) in candidates.iter().enumerate() {
            let g = *by_key.entry(c.key.as_slice()).or_insert_with(|| {
                groups.push((None, Vec::new()));
                groups.len() - 1
            });
            groups[g].1.push(idx);
        }
        for (key, ch) in &anchors {
            if let Some(&g) = by_key.get(key.as_slice()) {
                groups[g].0 = Some(*ch);
            }
        }
        groups.retain(|(anchor, members)| {
            anchor.is_some()
                || members
                    .iter()
                    .map(|&m| candidates[m].uses as usize)
                    .sum::<usize>()
                    >= 2
        });
        if groups.is_empty() {
            return Ok(false);
        }
        let mut grouped = vec![false; candidates.len()];
        for (_, members) in &groups {
            for &m in members {
                grouped[m] = true;
            }
        }

        // The pricer renders each site as the generator will, over the
        // names rename will assign; one visit per function, whose
        // candidates `collect` left contiguous.
        let mut preserve = std::collections::HashSet::new();
        preserve.extend(ctx.config.preserve_symbols.iter().cloned());
        let info = ctx.info(module)?;
        let plan = super::rename::plan_names(module, &preserve, ctx.config.mangle());
        let renamed = plan.applied(module);
        let mut pricer = Pricer::new(&renamed, &info, ctx.config, &plan);
        let mut prices: Vec<Option<SitePrice>> = (0..candidates.len()).map(|_| None).collect();
        let mut start = 0;
        while start < candidates.len() {
            let loc = candidates[start].loc;
            let end = start
                + candidates[start..]
                    .iter()
                    .take_while(|c| c.loc == loc)
                    .count();
            if grouped[start..end].iter().any(|&g| g) {
                let mut fp = match loc {
                    FuncRef::Function(fh) => pricer.function(fh),
                    FuncRef::EntryPoint(i) => pricer.entry_point(i),
                };
                for idx in (start..end).filter(|&idx| grouped[idx]) {
                    let h = candidates[idx].handle;
                    let (Some(use_len), Some(decl_len)) = (fp.expr_len(h), fp.expr_len_pinned(h))
                    else {
                        continue;
                    };
                    let let_len = fp.let_name_len();
                    let bare_use_len = candidates[idx]
                        .zero
                        .map_or(use_len, |ty| fp.bare_zero_value_len(ty));
                    let bound = fp.binds(h);
                    let uses = fp.uses(h);
                    let today = if bound {
                        fp.let_cost(uses, let_len, use_len)
                    } else {
                        uses * bare_use_len
                    };
                    prices[idx] = Some(SitePrice {
                        today,
                        decl_len,
                        uses,
                    });
                }
            }
            start = end;
        }

        // Shortlist by the model (an upper bound on today for zero sites, see
        // `SitePrice`); confirm by the output.  Decisions are sequential: an
        // accepted hoist is in the module the next trial renders, so its
        // name-pool and extraction effects are in the base.
        let mut base_len = None;
        let mut changed = false;
        for (anchor, members) in groups {
            let Some(site_prices) = members
                .iter()
                .map(|&m| prices[m].as_ref())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            // The constant's weight is its declaration plus every rendering
            // of its name; the `let`s the sites paid for disappear with them.
            // An anchored group pays no declaration.
            let uses: usize = site_prices.iter().map(|p| p.uses).sum();
            let name_len = pricer.name_len_at_weight(1 + uses);
            let today: usize = site_prices.iter().map(|p| p.today).sum();
            let hoisted = match anchor {
                Some(_) => uses * name_len,
                None => {
                    pricer.decl_cost(uses, name_len, site_prices[0].decl_len)
                        + pricer.name_insertion_cost(1 + uses)
                }
            };
            if hoisted >= today {
                continue;
            }
            let sites: Vec<(FuncRef, naga::Handle<naga::Expression>)> = members
                .iter()
                .map(|&m| (candidates[m].loc, candidates[m].handle))
                .collect();
            let base = match base_len {
                Some(len) => len,
                None => match super::shipped_len(module.clone(), ctx) {
                    Some(len) => *base_len.insert(len),
                    None => return Ok(changed),
                },
            };
            // What the tail ships with the hoist: the orphaned lanes culled
            // (the alias planner counts every constructor in the arena), the
            // names.  A trial naga rejects or the emitter declines is no win.
            let mut trial = module.clone();
            hoist_group(&mut trial, &sites, anchor, &preserve);
            if let Some(len) = super::shipped_len(trial, ctx)
                && len < base
            {
                hoist_group(module, &sites, anchor, &preserve);
                base_len = Some(len);
                changed = true;
            }
        }

        Ok(changed)
    }
}

/// Point every site at one shared `const` (`anchor`, or a new constant
/// initialised with the first site's cone cloned into `global_expressions`)
/// and rebuild the sites' `Emit` ranges: a `Constant` is not emittable, and
/// the orphaned lanes become dead for the next compaction (a `ZeroValue`
/// site was in no range and leaves no lanes).  The expressions keep their
/// arena slots, so topological order is preserved and the same call on a
/// copy of the module yields the same module.
fn hoist_group(
    module: &mut naga::Module,
    sites: &[(FuncRef, naga::Handle<naga::Expression>)],
    anchor: Option<naga::Handle<naga::Constant>>,
    preserve: &std::collections::HashSet<String>,
) {
    let const_handle = anchor.unwrap_or_else(|| declare_hoisted(module, sites[0], preserve));

    // The sites replaced per body, free functions first, then entry points.
    let functions = module.functions.len();
    let mut hoisted: Vec<HandleSet<naga::Expression>> =
        vec![HandleSet::default(); functions + module.entry_points.len()];
    for &(loc, handle) in sites {
        let (body, function) = match loc {
            FuncRef::Function(fh) => (fh.index(), &mut module.functions[fh]),
            FuncRef::EntryPoint(i) => (functions + i, &mut module.entry_points[i].function),
        };
        hoisted[body].insert(handle);
        function.expressions[handle] = naga::Expression::Constant(const_handle);
    }
    let bodies = module
        .functions
        .iter_mut()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter_mut().map(|ep| &mut ep.function));
    for (function, removed) in bodies.zip(&hoisted) {
        if !removed.is_empty() {
            crate::ir::rewrite::rebuild_emit_ranges_after_removal(&mut function.body, &|h| {
                removed.contains(h)
            });
        }
    }
}

fn declare_hoisted(
    module: &mut naga::Module,
    site: (FuncRef, naga::Handle<naga::Expression>),
    preserve: &std::collections::HashSet<String>,
) -> naga::Handle<naga::Constant> {
    let (rep_loc, rep_handle) = site;
    let rep_func = match rep_loc {
        FuncRef::Function(fh) => &module.functions[fh],
        FuncRef::EntryPoint(i) => &module.entry_points[i].function,
    };
    let (naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty)) =
        rep_func.expressions[rep_handle]
    else {
        unreachable!("every candidate is a constructor or a zero value");
    };
    // A const-expression lives in `global_expressions`.
    let mut cloned = HandleMap::default();
    let init = crate::ir::rewrite::clone_expression_handle(
        rep_handle,
        &rep_func.expressions,
        &mut module.global_expressions,
        &mut cloned,
    );

    // The placeholder name rename replaces must avoid preserve-listed
    // (preamble) names: rename keeps those verbatim, and a match makes the
    // generator suppress the hoisted declaration as preamble-owned,
    // silently rebinding every use to the preamble's value; preamble consts
    // count in `constants.len()`, so the raw `_hoist{n}` scheme can hit one.
    // Other module names are belt-and-suspenders (rename mangles any
    // non-preserved placeholder).
    let mut suffix = module.constants.len();
    let hoist_name = loop {
        let cand = format!("_hoist{}", suffix);
        if !preserve.contains(&cand)
            && !crate::name_gen::module_scope_names(module)
                .chain(crate::name_gen::type_names(module))
                .any(|n| n == cand)
        {
            break cand;
        }
        suffix += 1;
    };
    module.constants.append(
        naga::Constant {
            name: Some(hoist_name),
            ty,
            init,
        },
        naga::Span::UNDEFINED,
    )
}
