//! Repeated-literal extraction.
//!
//! Scans every function body for textual literals that appear enough
//! times across the module to justify replacing them with a shared
//! `const NAME = ...;` declaration.  The scan has to mirror every
//! emission-time literal-collapsing rule exactly, otherwise the
//! extraction either overcounts (and emits unused constants) or
//! undercounts (and misses profitable shares).
//!
//! The counting logic in [`Generator::scan_and_extract_literals`]
//! documents each emission bypass path in detail; keep those notes in
//! sync with `expr_emit` if new collapse rules are added there.

use std::collections::{HashMap, HashSet};

use crate::name_gen::next_name_unique;

use super::core::Generator;
use super::expr_emit::{
    ConcretizedAbstract, compose_is_splat, concretize_abstract_literal_via_inner,
};
use super::syntax::{LiteralExtractKey, literal_extract_key};

impl<'a> Generator<'a> {
    /// Scan every function's expression arena for repeated literal
    /// values.  When the combined textual emission count of a literal
    /// exceeds the break-even threshold, extract it into a shared
    /// module-scope `const` so each use site shrinks to the bound name.
    pub(super) fn scan_and_extract_literals(&mut self) {
        let max_precision = self.options.max_precision;
        let module = &self.module;

        // Helper: count textual literal emissions in a single function.
        //
        // Two IR shapes produce literal text in a function body:
        //   1. `Expression::Literal(lit)`               - direct literal use.
        //   2. `Expression::Constant(h)` where the constant is *unnamed*
        //      and its init in `global_expressions` is `Literal(lit)` -
        //      naga inlines the literal text at every reference.
        //
        // Counting strategy: start from `ref_counts` (which already restricts
        // to live handles), then correct for emission-time collapsing.
        //
        // - naga never places `Literal` or `Constant` expressions in
        //   `Statement::Emit` ranges, so they are *always* inlined at every
        //   reference rather than being let-bound.  The textual emission
        //   count per literal-like handle therefore starts at `ref_counts[h]`,
        //   not 1.  Counting one-per-handle (the previous behaviour) would
        //   undercount any literal shared across N inline use sites and
        //   suppress profitable extractions.
        //
        // - However, vector `Compose` constructors with all components equal
        //   are emitted in splat form (`vec3f(x)` instead of `vec3f(x,x,x)`)
        //   - see `compose_is_splat` and the splat path in
        //   `emit_expr_uncached::Compose`.  In that case the parent Compose
        //   bumped the literal handle's ref count once *per slot* (N times)
        //   but emission produces the literal text only **once**.  Worse,
        //   value-equivalent-but-distinct literal handles in the same
        //   splat-collapsable Compose all map to the same key, but only
        //   `components[0]` is emitted; the rest contribute zero emissions.
        //
        //   For each live splat-collapsable Compose we therefore subtract
        //   the over-count that `count_expr_children` introduced for each
        //   literal-like component handle:
        //     - `components[0]`: parent-contributed bumps reduce to 1.
        //     - other slots:     parent-contributed bumps reduce to 0.
        let count_literals =
            |func: &naga::Function,
             func_info: &naga::valid::FunctionInfo,
             ref_counts: &[usize],
             live: &[bool],
             literal_counts: &mut HashMap<LiteralExtractKey, usize>| {
                // Predicate: is this handle a literal-like that we would tally
                // (a `Literal`, or an unnamed `Constant` whose init is a literal)?
                let literal_lit = |h: naga::Handle<naga::Expression>| -> Option<naga::Literal> {
                    match func.expressions[h] {
                        naga::Expression::Literal(lit) => Some(lit),
                        naga::Expression::Constant(ch) => {
                            let c = &module.constants[ch];
                            if c.name.is_some() {
                                return None;
                            }
                            match module.global_expressions[c.init] {
                                naga::Expression::Literal(lit) => Some(lit),
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                };

                // Per-handle adjustment: how many bumps to *subtract* from
                // `ref_counts[h]` to obtain the true textual emission count.
                //
                // Initially zero; populated by walking live arena expressions
                // below.  The two expression-level bypasses are fused into a
                // single iteration because their predicates are disjoint
                // (`Compose` vs. `Select` vs. `Derivative`) and the per-iteration
                // overhead (`live[]` index, dispatch on `expr` kind) dominates
                // the inner work for typical shader sizes.
                //
                // Bypass paths (kept in sync with the emission code):
                //   * `emit_expr::Compose` splat-collapse - vector composes whose
                //     components are all equal (per `compose_is_splat`) emit only
                //     `components[0]`; literal-like components in slots `>= 1`
                //     contribute zero textual emissions despite each bumping
                //     `ref_counts` once via `count_expr_children`.
                //   * `emit_expr::Select` - direct `Expression::Literal` operands
                //     (NOT unnamed `Constant` operands) in the `reject`/`accept`
                //     slots are forced to typed form so both branches share a
                //     single concrete type.
                //   * `emit_expr::Derivative` - a direct `Expression::Literal`
                //     `expr` arg is forced to typed form to avoid i32 inference
                //     for derivatives (which require float).
                //   * `stmt_emit::emit_expr_for_atomic` - INTEGER literal/unnamed
                //     constant args of atomic statements are forced to the
                //     atomic's scalar type (handled in the second walk below).
                //
                // Note for Select/Derivative: unnamed-Constant operands take
                // `emit_expr`'s normal path and substitute correctly, so they
                // remain eligible (no adjustment).
                let mut adjust: Vec<usize> = vec![0; func.expressions.len()];

                for (ch, expr) in func.expressions.iter() {
                    if !live[ch.index()] {
                        continue;
                    }
                    match expr {
                        naga::Expression::Compose { ty, components } => {
                            if components.len() < 2 {
                                continue;
                            }
                            // Splat-collapse only applies to vector composes;
                            // matrix / array / struct composes always emit each
                            // slot.
                            if !matches!(module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
                                continue;
                            }
                            if !compose_is_splat(components, &func.expressions) {
                                continue;
                            }
                            // Only the first slot is emitted; subtract over-counts
                            // for every literal-like component slot.  We walk
                            // slot-by-slot (rather than de-duplicating component
                            // handles first) so repeated handles get one
                            // subtraction per slot, exactly cancelling
                            // `count_expr_children`'s per-slot bumps.
                            for (i, &comp) in components.iter().enumerate() {
                                if i == 0 {
                                    // First slot is emitted once; no adjustment.
                                    continue;
                                }
                                if literal_lit(comp).is_some() {
                                    adjust[comp.index()] += 1;
                                }
                            }
                        }
                        naga::Expression::Select { accept, reject, .. } => {
                            if matches!(func.expressions[*reject], naga::Expression::Literal(_)) {
                                adjust[reject.index()] += 1;
                            }
                            if matches!(func.expressions[*accept], naga::Expression::Literal(_)) {
                                adjust[accept.index()] += 1;
                            }
                        }
                        naga::Expression::Derivative { expr: e, .. } => {
                            if matches!(func.expressions[*e], naga::Expression::Literal(_)) {
                                adjust[e.index()] += 1;
                            }
                        }
                        _ => {}
                    }
                }
                // Statement-level bypass: walk all atomic statements.
                //
                // Only INTEGER literals get type-pinned by `emit_expr_for_atomic`;
                // float / bool literals fall through to `emit_expr`, which
                // already consults `extracted_literals`.  We must therefore only
                // subtract over-counts for integer-literal arguments, otherwise
                // we wrongly suppress profitable float-literal extractions
                // (e.g. `atomicAdd(&a, 1.5)` repeated N times).
                fn is_int_lit(lit: naga::Literal) -> bool {
                    matches!(
                        lit,
                        naga::Literal::U32(_)
                            | naga::Literal::I32(_)
                            | naga::Literal::U64(_)
                            | naga::Literal::I64(_)
                            | naga::Literal::AbstractInt(_)
                    )
                }
                fn walk_block_for_atomic_lits<F: FnMut(naga::Handle<naga::Expression>)>(
                    block: &naga::Block,
                    visit: &mut F,
                ) {
                    for stmt in block {
                        match stmt {
                            naga::Statement::Atomic { fun, value, .. } => {
                                if let naga::AtomicFunction::Exchange { compare: Some(c) } = fun {
                                    visit(*c);
                                }
                                visit(*value);
                            }
                            naga::Statement::Block(b) => walk_block_for_atomic_lits(b, visit),
                            naga::Statement::If { accept, reject, .. } => {
                                walk_block_for_atomic_lits(accept, visit);
                                walk_block_for_atomic_lits(reject, visit);
                            }
                            naga::Statement::Switch { cases, .. } => {
                                for case in cases {
                                    walk_block_for_atomic_lits(&case.body, visit);
                                }
                            }
                            naga::Statement::Loop {
                                body, continuing, ..
                            } => {
                                walk_block_for_atomic_lits(body, visit);
                                walk_block_for_atomic_lits(continuing, visit);
                            }
                            _ => {}
                        }
                    }
                }
                walk_block_for_atomic_lits(&func.body, &mut |h| {
                    if let Some(lit) = literal_lit(h)
                        && is_int_lit(lit)
                    {
                        adjust[h.index()] += 1;
                    }
                });

                for (h, _expr) in func.expressions.iter() {
                    let refs = ref_counts[h.index()];
                    if refs == 0 {
                        continue;
                    }
                    let Some(lit) = literal_lit(h) else { continue };
                    // Subtract over-count from splat-collapsed Composes.
                    // `adjust[h]` cannot exceed `refs[h]` (it counts a strict
                    // subset of the bumps that produced `refs[h]`), but we
                    // saturate defensively.
                    let emissions = refs.saturating_sub(adjust[h.index()]);
                    if emissions == 0 {
                        continue;
                    }
                    // Project abstract literals to their concrete form via
                    // the same helper the emission path uses
                    // (`expr_emit::concretize_abstract_literal_via_inner`).
                    // Without this, an abstract literal counted under its
                    // bare-form key would never match the
                    // typed-form-via-`literal_extract_key(concrete)` lookup
                    // performed in
                    // `Generator::concretize_abstract_literal_for_expr`,
                    // and the extracted `const` would sit unreferenced.
                    //
                    // Three cases:
                    //   * `Some(Lit(c))`  - count under the concrete key.
                    //   * `Some(Text(_))` - emission goes through a wrapper
                    //                       (`f16(...)`, `i32(<huge>)`)
                    //                       that bypasses
                    //                       `extracted_literals`; skip.
                    //   * `None`          - non-abstract literal; key on
                    //                       the original `lit` directly.
                    let key_lit = match concretize_abstract_literal_via_inner(
                        lit,
                        func_info[h].ty.inner_with(&module.types),
                    ) {
                        Some(ConcretizedAbstract::Lit(c)) => c,
                        Some(ConcretizedAbstract::Text(_)) => continue,
                        None => lit,
                    };
                    let emitted = literal_extract_key(key_lit, max_precision);
                    *literal_counts.entry(emitted).or_insert(0) += emissions;
                }
            };

        // Gather the forbidden-name set contributed by one function.
        // Function-scope names shadow module-scope names, so the
        // extracted constant must avoid colliding with any argument
        // or local of any function or entry point.
        let collect_func_names = |func: &naga::Function, forbidden: &mut HashSet<String>| {
            for arg in &func.arguments {
                if let Some(n) = &arg.name {
                    forbidden.insert(n.clone());
                }
            }
            for (_, local) in func.local_variables.iter() {
                if let Some(n) = &local.name {
                    forbidden.insert(n.clone());
                }
            }
        };

        // 1. Count literal strings exactly as they are emitted in general
        //    expression contexts.  Walk regular functions then entry
        //    points; the `cache_idx` counter mirrors the order in which
        //    `compute_expression_ref_counts` populated `ref_count_cache`.
        //    Splitting the chain (vs. the previous `chain().enumerate()`
        //    pattern) lets each branch index `self.info` naturally
        //    (`self.info[handle]` vs. `self.info.get_entry_point(idx)`)
        //    without an O(N) `.nth(cache_idx)` walk.
        let mut literal_counts: HashMap<LiteralExtractKey, usize> = HashMap::new();
        let mut cache_idx: usize = 0;
        for (handle, func) in self.module.functions.iter() {
            let live = std::mem::take(&mut self.ref_count_cache[cache_idx].live);
            count_literals(
                func,
                &self.info[handle],
                &self.ref_count_cache[cache_idx].ref_counts,
                &live,
                &mut literal_counts,
            );
            cache_idx += 1;
        }
        for (ep_idx, ep) in self.module.entry_points.iter().enumerate() {
            let live = std::mem::take(&mut self.ref_count_cache[cache_idx].live);
            count_literals(
                &ep.function,
                self.info.get_entry_point(ep_idx),
                &self.ref_count_cache[cache_idx].ref_counts,
                &live,
                &mut literal_counts,
            );
            cache_idx += 1;
        }
        debug_assert_eq!(cache_idx, self.ref_count_cache.len());

        // 2. Build forbidden name set: module-scope + all function-scope names.
        //    This prevents the extracted const name from being shadowed by
        //    any argument or local variable in any function.
        let mut forbidden = HashSet::new();
        for name in self.type_names.values() {
            forbidden.insert(name.clone());
        }
        for name in self.constant_names.iter() {
            forbidden.insert(name.clone());
        }
        for name in self.override_names.iter() {
            forbidden.insert(name.clone());
        }
        for name in self.global_names.iter() {
            forbidden.insert(name.clone());
        }
        for name in self.function_names.iter() {
            forbidden.insert(name.clone());
        }
        for ep in self.module.entry_points.iter() {
            forbidden.insert(ep.name.clone());
        }
        for func in self
            .module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(self.module.entry_points.iter().map(|ep| &ep.function))
        {
            collect_func_names(func, &mut forbidden);
        }

        // 3. Collect profitable candidates sorted by estimated savings.
        //    `savings = K * (L - N) - (8 + N + D)` where `K` is the use
        //    count, `L` the literal text length, `N` the bound name
        //    length (estimated as 1 for the initial filter), and `D`
        //    the declaration text length.  The `8 + N + D` term covers
        //    `const N=D;` boilerplate.
        let mut candidates: Vec<(isize, LiteralExtractKey, usize)> = literal_counts
            .into_iter()
            .filter_map(|(key, count)| {
                let expr_len = key.expr_text.len() as isize;
                let decl_len = key.decl_text.len() as isize;
                let k = count as isize;
                let est = k * (expr_len - 1) - (8 + 1 + decl_len);
                if est > 0 {
                    Some((est, key, count))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by(|a, b| b.0.cmp(&a.0));

        // 4. Greedily assign names, re-computing savings with the true
        //    name length once a concrete name has been picked; extracts
        //    are kept only when the corrected savings stay positive.
        let mut counter = 0usize;
        for (_, key, count) in candidates {
            let name = next_name_unique(&mut counter, &forbidden);
            let n = name.len() as isize;
            let expr_len = key.expr_text.len() as isize;
            let decl_len = key.decl_text.len() as isize;
            let k = count as isize;
            let savings = k * (expr_len - n) - (8 + n + decl_len);
            if savings > 0 {
                forbidden.insert(name.clone());
                self.extracted_literals.insert(key, name);
            }
        }
    }
}
