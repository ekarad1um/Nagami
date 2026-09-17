//! Repeated-literal extraction: literals whose textual emission count across
//! the module pays for a shared `const NAME = ...;` are bound to one.  The
//! count must mirror every emission-time literal-collapsing rule exactly:
//! overcounting emits unused constants, undercounting misses profitable shares.

use rustc_hash::FxHashMap;

use crate::passes::expr_util::{KeyToken, RefCount, lit_key};

use super::core::Generator;
use super::expr_emit::{
    ConcretizedAbstract, as_operand_keeps_suffix, compose_is_splat,
    concretize_abstract_literal_via_inner, literal_bare_form_changes_type, literal_is_width8,
    literal_needs_typed_form_outside_constructor,
};
use super::syntax::{LiteralExtractKey, literal_extract_key};
use crate::handle_set::HandleSet;

impl<'a> Generator<'a> {
    /// How often the text spells each literal over every body, and whether
    /// ever bare (`Analyses::literal_counts`): a function of the arenas and
    /// the float precision, counted over the prepared census.
    fn literal_census(&self) -> FxHashMap<LiteralExtractKey, (usize, bool)> {
        let precision = &self.options.float_precision;
        let module = &self.module;

        // Textual literal emissions in one function.  Literal-like handles are
        // `Literal`s and unnamed `Constant`s with a `Literal` init (naga inlines
        // the text at every reference).  Neither ever sits in an `Emit` range,
        // so each is inlined at every use and its emission count starts at
        // `ref_counts[h]` (live handles only), then drops by the emission-time
        // bypasses tallied in `adjust`.
        let count_literals =
            |func: &naga::Function,
             func_info: &naga::valid::FunctionInfo,
             ref_counts: &[RefCount],
             live: &[bool],
             deferrable: &[bool],
             rendered: &mut FxHashMap<KeyToken, LiteralExtractKey>,
             literal_counts: &mut FxHashMap<LiteralExtractKey, (usize, bool)>| {
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

                // Bumps to subtract from `ref_counts[h]` for uses the emitter
                // renders without consulting `extracted_literals`: splat-collapsed
                // `Compose` slots, width-8 narrowing folds, and operands forced to
                // typed form (Select / Derivative branches, bitcast, shift and
                // bit-op operands, atomic arguments, a deferred local's first
                // store, a switch selector).  Select / Derivative force only a
                // direct `Literal`; an unnamed `Constant` there takes the normal,
                // extraction-aware path and needs no adjustment.
                let mut adjust: Vec<RefCount> = vec![0; func.expressions.len()];

                // `bare_handle[h]`: `h` is emitted somewhere in a bare
                // constructor slot (`Compose` component or `Splat` value).  A
                // needs-typed literal (F16/F64/I64/U64) used only standalone emits
                // the longer suffixed form and is priced there; one that also
                // appears bare is priced at the bare length so extraction never
                // nets larger.
                let mut bare_handle: Vec<bool> = vec![false; func.expressions.len()];

                // Operands of the const width-8 vector-narrowing fold: an
                // `As { convert: Some }` over a single-use inlined vector
                // `Compose` / `Splat` of width-8 literals emits each component's
                // CONVERTED text and never consults `extracted_literals`, so
                // those literals contribute no substitutable emission under their
                // suffixed key; counting them would extract a `const` no use site
                // references.  `ref_counts == 1` mirrors the emitter's
                // not-`let`-bound gate; over-matching only forgoes an extraction.
                let is_width8_lit = |h: naga::Handle<naga::Expression>| matches!(func.expressions[h], naga::Expression::Literal(l) if literal_is_width8(l));
                let mut narrow_folded: HandleSet<naga::Expression> = Default::default();
                for (ch, expr) in func.expressions.iter() {
                    if !live[ch.index()] {
                        continue;
                    }
                    let naga::Expression::As {
                        expr: src,
                        convert: Some(_),
                        ..
                    } = expr
                    else {
                        continue;
                    };
                    if ref_counts.get(src.index()).copied() != Some(1) {
                        continue;
                    }
                    let folds = match &func.expressions[*src] {
                        naga::Expression::Compose { ty, components } => {
                            matches!(module.types[*ty].inner, naga::TypeInner::Vector { .. })
                                && components.iter().all(|&c| is_width8_lit(c))
                        }
                        naga::Expression::Splat { value, .. } => is_width8_lit(*value),
                        _ => false,
                    };
                    if folds {
                        narrow_folded.insert(*src);
                    }
                }

                for (ch, expr) in func.expressions.iter() {
                    if !live[ch.index()] {
                        continue;
                    }
                    match expr {
                        naga::Expression::Compose { ty, components } => {
                            // Every slot renders as converted text under another
                            // key: drop all per-slot bumps, skip the splat
                            // accounting.
                            if narrow_folded.contains(ch) {
                                for &comp in components.iter() {
                                    adjust[comp.index()] += 1;
                                }
                                continue;
                            }
                            // Every slot is emitted bare; mark before the splat
                            // early-outs.
                            for &comp in components.iter() {
                                if literal_lit(comp).is_some() {
                                    bare_handle[comp.index()] = true;
                                }
                            }
                            if components.len() < 2 {
                                continue;
                            }
                            // Only vector composes splat-collapse.
                            if !matches!(module.types[*ty].inner, naga::TypeInner::Vector { .. }) {
                                continue;
                            }
                            if !compose_is_splat(components, &func.expressions, &|_| false) {
                                continue;
                            }
                            // Only slot 0 is emitted; one subtraction per later
                            // slot, repeated handles included, exactly cancels
                            // the census's per-slot bumps.
                            for (i, &comp) in components.iter().enumerate() {
                                if i == 0 {
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
                        // Forced typed by the emitter (no sibling pins them).
                        naga::Expression::Binary {
                            op: naga::BinaryOperator::ShiftLeft | naga::BinaryOperator::ShiftRight,
                            left,
                            ..
                        } => {
                            if literal_lit(*left).is_some_and(literal_bare_form_changes_type) {
                                adjust[left.index()] += 1;
                            }
                        }
                        naga::Expression::Math {
                            fun: naga::MathFunction::ExtractBits,
                            arg,
                            ..
                        } => {
                            if literal_lit(*arg).is_some_and(literal_bare_form_changes_type) {
                                adjust[arg.index()] += 1;
                            }
                        }
                        naga::Expression::Math {
                            fun: naga::MathFunction::InsertBits,
                            arg,
                            arg1,
                            ..
                        } => {
                            for a in [Some(*arg), *arg1].into_iter().flatten() {
                                if literal_lit(a).is_some_and(literal_bare_form_changes_type) {
                                    adjust[a.index()] += 1;
                                }
                            }
                        }
                        // A float builtin whose every argument is a bare literal
                        // spells its first one typed (the emitter's pin against
                        // an abstract const-expression), so that occurrence
                        // never renders an extracted name.
                        naga::Expression::Math {
                            arg,
                            arg1,
                            arg2,
                            arg3,
                            ..
                        } => {
                            let float_scalar = matches!(
                                func_info[*arg].ty.inner_with(&module.types),
                                naga::TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Float
                            );
                            if float_scalar
                                && [Some(*arg), *arg1, *arg2, *arg3].into_iter().flatten().all(
                                    |a| {
                                        literal_lit(a).is_some_and(|l| {
                                            !literal_needs_typed_form_outside_constructor(l)
                                        })
                                    },
                                )
                            {
                                adjust[arg.index()] += 1;
                            }
                        }
                        // Kept-suffix operands bypass `extracted_literals`.
                        naga::Expression::As {
                            expr: src,
                            kind,
                            convert,
                        } => {
                            if literal_lit(*src)
                                .is_some_and(|l| as_operand_keeps_suffix(l, *kind, *convert))
                            {
                                adjust[src.index()] += 1;
                            }
                        }
                        // The value is emitted bare inside the vector constructor.
                        naga::Expression::Splat { value, .. } if literal_lit(*value).is_some() => {
                            // Width-8 narrowing fold: converted text, never under
                            // its key.
                            if narrow_folded.contains(ch) {
                                adjust[value.index()] += 1;
                            } else {
                                bare_handle[value.index()] = true;
                            }
                        }
                        _ => {}
                    }
                }
                // Atomic statement arguments: `emit_expr_for_atomic` pins only
                // INTEGER literals to the atomic's scalar type; float / bool
                // literals reach `emit_expr` and stay extraction-aware.
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
                // A `Store` through an `atomic<T>` pointer lowers to `atomicStore`
                // and pins its integer literal the same way.
                fn pointer_is_atomic(
                    pointer: naga::Handle<naga::Expression>,
                    func_info: &naga::valid::FunctionInfo,
                    types: &naga::UniqueArena<naga::Type>,
                ) -> bool {
                    match func_info[pointer].ty.inner_with(types) {
                        naga::TypeInner::Atomic(_) => true,
                        naga::TypeInner::Pointer { base, .. } => {
                            matches!(types[*base].inner, naga::TypeInner::Atomic(_))
                        }
                        _ => false,
                    }
                }
                fn walk_block_for_atomic_lits<F: FnMut(naga::Handle<naga::Expression>)>(
                    block: &naga::Block,
                    func_info: &naga::valid::FunctionInfo,
                    types: &naga::UniqueArena<naga::Type>,
                    visit: &mut F,
                ) {
                    crate::ir::visit::for_each_statement(block, &mut |stmt| match stmt {
                        naga::Statement::Atomic { fun, value, .. } => {
                            crate::ir::visit::visit_atomic_function_handles(fun, visit);
                            visit(*value);
                        }
                        naga::Statement::Store { pointer, value }
                            if pointer_is_atomic(*pointer, func_info, types) =>
                        {
                            visit(*value);
                        }
                        naga::Statement::ImageAtomic { fun, value, .. } => {
                            crate::ir::visit::visit_atomic_function_handles(fun, visit);
                            visit(*value);
                        }
                        _ => {}
                    });
                }
                walk_block_for_atomic_lits(&func.body, func_info, &module.types, &mut |h| {
                    if let Some(lit) = literal_lit(h)
                        && is_int_lit(lit)
                    {
                        adjust[h.index()] += 1;
                    }
                });

                // Two statement paths render a direct `Literal` through
                // `literal_to_wgsl`, bypassing substitution: the FIRST store to a
                // deferred local (`var X = <lit>;`, for-init form included; later
                // stores are extraction-aware, hence the `consumed` gate) and a
                // literal switch selector, forced to match the typed case labels.
                // Both force every literal kind, and only a direct `Literal` (an
                // unnamed `Constant` takes the extraction-aware path), so this
                // must not use `literal_lit`.
                fn walk_typed_form_lits<F: FnMut(naga::Handle<naga::Expression>)>(
                    block: &naga::Block,
                    expressions: &naga::Arena<naga::Expression>,
                    deferrable: &[bool],
                    consumed: &mut [bool],
                    visit: &mut F,
                ) {
                    crate::ir::visit::for_each_statement(block, &mut |stmt| match stmt {
                        naga::Statement::Store { pointer, value } => {
                            if let naga::Expression::LocalVariable(lh) = expressions[*pointer]
                                && deferrable[lh.index()]
                                && !consumed[lh.index()]
                            {
                                consumed[lh.index()] = true;
                                visit(*value);
                            }
                        }
                        naga::Statement::Switch { selector, .. } => visit(*selector),
                        _ => {}
                    });
                }
                let mut deferred_consumed = vec![false; func.local_variables.len()];
                walk_typed_form_lits(
                    &func.body,
                    &func.expressions,
                    deferrable,
                    &mut deferred_consumed,
                    &mut |h| {
                        if matches!(func.expressions[h], naga::Expression::Literal(_)) {
                            adjust[h.index()] += 1;
                        }
                    },
                );

                for (h, _expr) in func.expressions.iter() {
                    let refs = ref_counts[h.index()];
                    if refs == 0 {
                        continue;
                    }
                    let Some(lit) = literal_lit(h) else { continue };
                    // `adjust[h]` counts a strict subset of the bumps behind
                    // `refs`; saturate anyway.
                    let emissions = refs.saturating_sub(adjust[h.index()]);
                    if emissions == 0 {
                        continue;
                    }
                    // Key abstract literals by their concrete form, the key the
                    // emitter looks up; a bare-form key would never match and the
                    // `const` would sit unreferenced.  A `Text` concretization
                    // (`f16(...)`, `i32(<huge>)`) bypasses `extracted_literals`.
                    let key_lit = match concretize_abstract_literal_via_inner(
                        lit,
                        func_info[h].ty.inner_with(&module.types),
                    ) {
                        Some(ConcretizedAbstract::Lit(c)) => c,
                        Some(ConcretizedAbstract::Text(_)) => continue,
                        None => lit,
                    };
                    // Rendered once per value: a module spells its few
                    // literals many times over, and a float's shortest
                    // spelling is three candidates and a parse back.
                    let emitted = rendered
                        .entry(lit_key(key_lit))
                        .or_insert_with(|| literal_extract_key(key_lit, precision));
                    let bare = bare_handle[h.index()];
                    let emissions = emissions as usize;
                    if let Some(entry) = literal_counts.get_mut(emitted) {
                        entry.0 += emissions;
                        entry.1 |= bare;
                    } else {
                        literal_counts.insert(emitted.clone(), (emissions, bare));
                    }
                }
            };

        // `all_functions` is also the order `compute_expression_ref_counts`
        // filled `ref_count_cache` in; `ModuleInfo` has no iterator spanning
        // both arenas, so the two are zipped rather than indexed apart.
        let (module, info) = (self.module, self.info);
        debug_assert_eq!(
            self.ref_count_cache.len(),
            module.functions.len() + module.entry_points.len(),
            "a cache built over a different function set would zip short"
        );
        let infos = module
            .functions
            .iter()
            .map(|(handle, _)| &info[handle])
            .chain((0..module.entry_points.len()).map(|i| info.get_entry_point(i)));
        let mut rendered: FxHashMap<KeyToken, LiteralExtractKey> = Default::default();
        let mut literal_counts: FxHashMap<LiteralExtractKey, (usize, bool)> = Default::default();
        for (cache_idx, (func, fn_info)) in crate::ir::visit::all_functions(module)
            .zip(infos)
            .enumerate()
        {
            count_literals(
                func,
                fn_info,
                &self.ref_count_cache[cache_idx].ref_counts,
                &self.ref_count_cache[cache_idx].live,
                &self.analyses.defer[cache_idx].0,
                &mut rendered,
                &mut literal_counts,
            );
        }
        literal_counts
    }

    /// Bind module-wide repeated literals to shared `const`s where the
    /// emission count beats the break-even threshold.
    pub(super) fn scan_and_extract_literals(&mut self) {
        if self.analyses.literal_counts.is_none() {
            self.analyses.literal_counts = Some(self.literal_census());
        }
        let literal_counts = self
            .analyses
            .literal_counts
            .as_ref()
            .expect("counted above");

        // Names the extracted `const` must avoid: every module-scope name
        // (the preserve list among them: a pruned preamble binding is in no
        // arena yet stands in the consumer's spliced document) and, since
        // function-scope names shadow them, every argument and local.
        let mut forbidden: crate::name_gen::NameScope =
            self.emitted_module_names().map(str::to_owned).collect();
        forbidden.extend(
            crate::ir::visit::all_functions(self.module)
                .flat_map(crate::name_gen::function_local_names)
                .map(str::to_owned),
        );

        // Estimated savings `K * (L - N) - (BOILERPLATE + N + D)`: `K` uses, `L`
        // per-use length, `N` the name length (1 for this filter), `D` the
        // declaration text.  A needs-typed literal (F16/F64/I64/U64, the only
        // kinds whose `decl_text` carries a suffix `expr_text` lacks) emits the
        // typed form at every standalone use and is priced there, unless it also
        // appears bare (`has_bare`), where the shorter `expr_text` keeps the
        // estimate conservative; for every other kind the two texts are equal.
        let boilerplate = super::syntax::decl_boilerplate(self.options.beautify) as isize;
        let mut candidates: Vec<(isize, LiteralExtractKey, usize, bool)> = literal_counts
            .iter()
            .filter_map(|(key, &(count, has_bare))| {
                let expr_len = key.expr_text.len() as isize;
                let decl_len = key.decl_text.len() as isize;
                let typed_only = !has_bare && key.decl_text != key.expr_text;
                let use_len = if typed_only { decl_len } else { expr_len };
                let k = count as isize;
                let est = k * (use_len - 1) - (boilerplate + 1 + decl_len);
                if est > 0 {
                    Some((est, key.clone(), count, has_bare))
                } else {
                    None
                }
            })
            .collect();
        // Ties break on the full key for a total order independent of map
        // iteration order.
        candidates.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.expr_text.cmp(&b.1.expr_text))
                .then_with(|| a.1.decl_text.cmp(&b.1.decl_text))
        });

        // Re-price with the real name; a rejected candidate leaves its name
        // to the next.
        for (_, key, count, has_bare) in candidates {
            let name = crate::name_gen::shortest_free_name(&forbidden, &[], &|_| false);
            let n = name.len() as isize;
            let expr_len = key.expr_text.len() as isize;
            let decl_len = key.decl_text.len() as isize;
            let typed_only = !has_bare && key.decl_text != key.expr_text;
            let use_len = if typed_only { decl_len } else { expr_len };
            let k = count as isize;
            let savings = k * (use_len - n) - (boilerplate + n + decl_len);
            if savings > 0 {
                forbidden.insert(name.clone());
                self.extracted_literals.insert(key, name);
            }
        }
    }
}
