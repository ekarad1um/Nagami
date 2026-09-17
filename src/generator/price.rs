//! The renderer as the cost model.  A pass that decides a rewrite by its
//! bytes asks the generator for the text it would emit instead of modelling
//! it: every collapse the emitter performs (constructor elision, splat and
//! swizzle folds, literal extraction, alias names) is priced by the code that
//! performs it, so a decision cannot drift from the output.  The pricer runs
//! a real [`Generator`] over a copy of the module carrying the names rename
//! will assign, so a `let`, an alias or a constant's name is the shipped one.
//! A price is `None` where the emitter declines the expression, which a pass
//! reads as "do not rewrite".

use std::collections::HashSet;

use super::core::{FunctionCtx, GenerateOptions, Generator};
use crate::config::Config;
use crate::passes::rename::NamePlan;

pub(crate) struct Pricer<'m> {
    generator: Generator<'m>,
    module_used_names: HashSet<String>,
    plan: &'m NamePlan,
}

impl<'m> Pricer<'m> {
    /// `module` is [`NamePlan::applied`] of the module the pass rewrites,
    /// so handles agree, and `info` its validation (naming touches no
    /// arena, so the pass's own `info` serves).
    pub(crate) fn new(
        module: &'m naga::Module,
        info: &'m naga::valid::ModuleInfo,
        config: &Config,
        plan: &'m NamePlan,
    ) -> Self {
        let mut generator = Generator::new(module, info, GenerateOptions::from_config(config));
        generator.prepare();
        let module_used_names = generator.module_used_names();
        Self {
            generator,
            module_used_names,
            plan,
        }
    }

    /// The pricer of one function body.  The generator's per-function
    /// caches are taken when the context is built, exactly as emission
    /// takes them, so each function is priced in one visit.
    pub(crate) fn function(
        &mut self,
        handle: naga::Handle<naga::Function>,
    ) -> FunctionPricer<'_, 'm> {
        let module = self.generator.module;
        let info = self.generator.info;
        let name = self.generator.function_names[handle.index()].clone();
        self.function_pricer(
            &name,
            &module.functions[handle],
            &info[handle],
            handle.index(),
        )
    }

    /// [`Self::function`] for `module.entry_points[index]`.
    pub(crate) fn entry_point(&mut self, index: usize) -> FunctionPricer<'_, 'm> {
        let module = self.generator.module;
        let info = self.generator.info;
        let entry = &module.entry_points[index];
        self.function_pricer(
            &entry.name,
            &entry.function,
            info.get_entry_point(index),
            module.functions.len() + index,
        )
    }

    /// `name` is the one the declaration prints.
    fn function_pricer(
        &mut self,
        name: &str,
        func: &'m naga::Function,
        finfo: &'m naga::valid::FunctionInfo,
        cache_idx: usize,
    ) -> FunctionPricer<'_, 'm> {
        let ctx =
            self.generator
                .function_ctx(name, func, finfo, &self.module_used_names, cache_idx);
        FunctionPricer {
            generator: &mut self.generator,
            ctx,
        }
    }

    /// Fixed cost of one module-scope `<keyword> <name>=<body>;`.
    pub(crate) fn decl_boilerplate(&self) -> usize {
        super::syntax::decl_boilerplate(self.generator.options.beautify)
    }

    /// Length of the name rename would give a new module-scope identifier
    /// of `weight` occurrences.
    pub(crate) fn name_len_at_weight(&self, weight: usize) -> usize {
        self.plan.name_len_at_weight(weight)
    }

    /// Bytes the other identifiers pay for that new name
    /// ([`NamePlan::insertion_cost`]).
    pub(crate) fn name_insertion_cost(&self, weight: usize) -> usize {
        self.plan.insertion_cost(weight)
    }
}

/// Prices inside one function, in the state its first statement renders in:
/// no `let` bound yet, every argument and local named - until [`Self::emit`]
/// advances it.
pub(crate) struct FunctionPricer<'p, 'm> {
    generator: &'p mut Generator<'m>,
    ctx: FunctionCtx<'m, 'p>,
}

impl FunctionPricer<'_, '_> {
    /// Bytes of the whole declaration, `fn N(..)->R{..}`, as emitted; the
    /// pricer's own state is untouched (a copy renders).
    pub(crate) fn definition_len(&mut self) -> Option<usize> {
        let mut ctx = self.ctx.clone();
        let name = ctx.display_name.clone();
        let start = self.generator.out.len();
        let rendered = self
            .generator
            .generate_function_in(&name, ctx.func, false, &mut ctx)
            .is_ok();
        let len = self.generator.out.len() - start;
        self.generator.out.truncate(start);
        rendered.then_some(len)
    }

    /// Length of the name the declaration prints.
    pub(crate) fn name_len(&self) -> usize {
        self.ctx.display_name.len()
    }

    /// Length of the name parameter `index` reads as.
    pub(crate) fn argument_name_len(&self, index: usize) -> usize {
        self.ctx.argument_names[index].len()
    }

    /// Price the `Emit` of `h` and advance past it as the emitter does: the
    /// bytes of the `let N=E;` it renders, `h` bound to `N` from here on,
    /// or 0 where `h` renders at its uses.  `None` where the emitter
    /// declines the expression.
    pub(crate) fn emit(&mut self, h: naga::Handle<naga::Expression>) -> Option<usize> {
        let Some(text) = self.generator.binding_decision(h, &mut self.ctx).ok()? else {
            return Some(0);
        };
        let name = self.ctx.next_expr_name();
        let cost = text.len() + name.len() + self.let_boilerplate();
        self.ctx.expr_names.insert(h, name);
        self.ctx.name_twins(h);
        Some(cost)
    }

    /// Whether [`Self::emit`] bound `h`.
    pub(crate) fn bound(&self, h: naga::Handle<naga::Expression>) -> bool {
        self.ctx.expr_names.contains_key(h)
    }

    /// Bytes `h` renders at a use site.
    pub(crate) fn expr_len(&mut self, h: naga::Handle<naga::Expression>) -> Option<usize> {
        self.generator
            .emit_expr_uncached(h, &mut self.ctx)
            .ok()
            .map(|text| text.len())
    }

    /// [`Self::expr_len`] with `h` spelling its own type, as the
    /// initializer of a declaration that prints no `: T` does.
    pub(crate) fn expr_len_pinned(&mut self, h: naga::Handle<naga::Expression>) -> Option<usize> {
        let outer = self.ctx.pinned_root.replace(h);
        let len = self.expr_len(h);
        self.ctx.pinned_root = outer;
        len
    }

    /// Whether the emitter binds `h` to a `let` at its `Emit`.
    pub(crate) fn binds(&mut self, h: naga::Handle<naga::Expression>) -> bool {
        matches!(
            self.generator.binding_decision(h, &mut self.ctx),
            Ok(Some(_))
        )
    }

    /// Length of the name the function's next `let` takes.
    pub(crate) fn let_name_len(&self) -> usize {
        self.ctx.peek_expr_name_len()
    }

    /// Fixed cost of one body `let <name>=<value>;` around its name and
    /// value.
    pub(crate) fn let_boilerplate(&self) -> usize {
        super::syntax::let_boilerplate(self.generator.options.beautify)
    }

    /// Whether the call producing `result` renders at the result's one use
    /// (its text stashed) rather than as a `let`.
    pub(crate) fn stashed(&self, result: naga::Handle<naga::Expression>) -> bool {
        self.ctx.inlineable_calls.contains(result)
    }

    /// How many times the emitter renders `h` (or its `let` name).
    pub(crate) fn uses(&self, h: naga::Handle<naga::Expression>) -> usize {
        self.ctx.ref_counts[h.index()]
    }
}
