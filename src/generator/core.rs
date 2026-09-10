//! Generator state and per-function context: [`Generator`] owns the output
//! buffer, the cached module-wide analyses (liveness, alias plan, per-function
//! ref counts, layouts) and the precomputed format tokens; [`FunctionCtx`] is
//! the per-function bundle threaded through the statement and expression
//! emitters so bindings, deferred-variable flags and inline decisions stay
//! coherent within one function.

use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::config::FloatPrecision;

use super::syntax::LiteralExtractKey;
use crate::handle_set::{HandleMap, HandleSet};
use crate::passes::expr_util::all_functions;

// MARK: Options

/// Caller-facing generator knobs, each resolved from a [`crate::config::Config`]
/// entry by [`crate::run`]; changing a default is a public-surface change.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Emit human-readable output with indentation and newlines.
    pub beautify: bool,
    /// Spaces per indentation level; honoured only when `beautify` is set.
    pub indent: u8,
    /// Rename struct types and struct members to short identifiers.
    pub mangle: bool,
    /// Per-type precision caps applied to float literals; any non-`Full` mode
    /// is lossy.
    pub float_precision: FloatPrecision,
    /// Output buffer reservation, to amortise reallocation across emission.
    pub initial_capacity: usize,
    /// Struct type / member names exempt from mangling.
    pub preserve_symbols: HashSet<String>,
    /// Struct member names kept verbatim (a preamble struct's, accessed by
    /// the body).  Members are their own namespace, so unlike
    /// `preserve_symbols` this never freezes a same-named function or local.
    pub preserve_members: HashSet<String>,
    /// Names of preamble (external) declarations to exclude from output.
    pub preamble_names: HashSet<String>,
    /// Emit `alias` declarations for repeated types when that shortens output.
    pub type_alias: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            beautify: false,
            indent: 2,
            mangle: false,
            float_precision: FloatPrecision::default(),
            initial_capacity: 16 * 1024,
            preserve_symbols: HashSet::new(),
            preserve_members: HashSet::new(),
            preamble_names: HashSet::new(),
            type_alias: false,
        }
    }
}

// MARK: Cached analyses

/// Per-function expression analysis, computed once and in lock-step by
/// `compute_expression_ref_counts`.  `live` is `mem::take`n by literal
/// extraction and `ref_counts` by `generate_function`, so neither is readable
/// after its consumer has run.
pub(super) struct FunctionExprInfo {
    pub(super) ref_counts: Vec<usize>,
    pub(super) live: Vec<bool>,
}

// MARK: Generator state

/// Emission state for one module; created per [`super::generate_wgsl`] call and
/// never reused.
pub(super) struct Generator<'a> {
    pub(super) module: &'a naga::Module,
    pub(super) info: &'a naga::valid::ModuleInfo,
    pub(super) options: GenerateOptions,
    pub(super) out: String,
    pub(super) indent_depth: u32,
    pub(super) type_names: HandleMap<naga::Type, String>,
    pub(super) member_names: FxHashMap<(naga::Handle<naga::Type>, u32), String>,
    pub(super) constant_names: Vec<String>,
    pub(super) override_names: Vec<String>,
    pub(super) global_names: Vec<String>,
    pub(super) function_names: Vec<String>,
    /// See [`super::syntax::ShadowedAliases`].
    pub(super) shadowed_type_aliases: super::syntax::ShadowedAliases,
    pub(super) extracted_literals: FxHashMap<LiteralExtractKey, String>,
    /// `(alias_name, type_string)` alias declarations awaiting emission.
    pub(super) type_alias_decls: Vec<(String, String)>,
    /// Global-expression handle -> named constant whose `init` it is, filled
    /// as constants are emitted so later ones reference earlier ones by name
    /// instead of re-inlining the tree.
    pub(super) expr_to_const: HandleMap<naga::Expression, naga::Handle<naga::Constant>>,
    /// Constants reachable from live code; dead ones are not emitted.
    pub(super) live_constants: HandleSet<naga::Constant>,
    /// Types reachable from live code; dead struct declarations are not emitted.
    pub(super) live_types: HandleSet<naga::Type>,
    /// Struct types a host can address: live, not naga-predeclared,
    /// preamble-owned included (declared in the consumer's preamble text).
    pub(super) map_visible_structs: HandleSet<naga::Type>,
    /// Type layouts for reconstructing `@size` / `@align` on struct members.
    pub(super) layouter: naga::proc::Layouter,
    /// `true` when `layouter` holds an entry for every type; indexing a missing
    /// one panics, so layout-dependent emission must refuse on `false`.
    pub(super) layouter_complete: bool,
    /// Per-function analyses indexed `[0..N)` for functions and `[N..N+E)` for
    /// entry points.
    pub(super) ref_count_cache: Vec<FunctionExprInfo>,
    /// Per-function `(deferrable, dead)` local bitmaps, indexed like
    /// `ref_count_cache`; computed once so the live-type census, alias cost
    /// model, literal extraction and emission cannot disagree.
    pub(super) defer_cache: Vec<(Vec<bool>, Vec<bool>)>,
    /// Per-`module.functions` purity bitmap (`true` = no side effect beyond the
    /// return value); keeps impure single-use calls bound rather than inlined
    /// past a read of what they write.
    pub(super) pure_functions: Vec<bool>,
    /// Format tokens fixed at construction from `options.beautify`, so the
    /// hot path never branches per character.
    tok: &'static Tokens,
    indent_unit: &'static str,
}

/// The two spellings of every token `beautify` alone decides, so adding one
/// is a field and two entries.  `indent_unit` is not here: its width comes
/// from `options.indent`, which would force these tables to be built at run
/// time.
struct Tokens {
    separator: &'static str,
    assign: &'static str,
    colon: &'static str,
    arrow: &'static str,
    open_brace: &'static str,
    newline: &'static str,
    bin_op_sep: &'static str,
    binding_sep: &'static str,
    attr_end: &'static str,
    angle_end: &'static str,
    else_kw: &'static str,
    for_open: &'static str,
    for_sep: &'static str,
}

static COMPACT_TOKENS: Tokens = Tokens {
    separator: ",",
    assign: "=",
    colon: ":",
    arrow: "->",
    open_brace: "{",
    newline: "",
    bin_op_sep: "",
    binding_sep: ")@binding(",
    attr_end: ")",
    angle_end: ">",
    else_kw: "else",
    for_open: "for(",
    for_sep: ";",
};

static PRETTY_TOKENS: Tokens = Tokens {
    separator: ", ",
    assign: " = ",
    colon: ": ",
    arrow: " -> ",
    open_brace: " {\n",
    newline: "\n",
    bin_op_sep: " ",
    binding_sep: ") @binding(",
    attr_end: ") ",
    angle_end: "> ",
    else_kw: " else",
    for_open: "for (",
    for_sep: "; ",
};

// MARK: Function context

/// Per-function emission context, threaded through every statement and
/// expression emitter.
pub(super) struct FunctionCtx<'a, 'm> {
    /// The enclosing function, or an empty one at module scope.
    pub(super) func: &'a naga::Function,
    /// Arena every handle here indexes: the function's expressions, or
    /// `module.global_expressions` at module scope.
    pub(super) exprs: &'a naga::Arena<naga::Expression>,
    pub(super) types: ExprTypes<'a>,
    /// `array<T,N>(..)` may drop to `array(..)` only in a function body: naga's
    /// front-end rejects the elided form against a module-scope declaration
    /// whose type resolves through an alias.
    pub(super) elide_array_ctor: bool,
    /// Root of a declaration that prints no `: T`, so its own text must spell
    /// the concrete type: an elided `vec2(42,43)` would leave the constant
    /// abstract, a different type that naga drops from the arena entirely.
    pub(super) pinned_root: Option<naga::Handle<naga::Expression>>,
    pub(super) argument_names: Vec<String>,
    pub(super) local_names: HandleMap<naga::LocalVariable, String>,
    pub(super) expr_names: HandleMap<naga::Expression, String>,
    pub(super) ref_counts: Vec<usize>,
    pub(super) deferred_vars: Vec<bool>,
    pub(super) dead_vars: Vec<bool>,
    /// Matrix / array locals passed by address to a call while their type
    /// renders under a minted `alias`: their `var` declares `: ALIAS`.  naga
    /// types an inferred matrix / array as the unnamed twin of the alias and
    /// compares matrix / array pointer bases by HANDLE (scalar / vector ones
    /// canonicalise, structs are always named), so `f(&b)` against
    /// `ptr<function, ALIAS>` fails the re-parse.
    pub(super) typed_pointer_arg_locals: Vec<bool>,
    /// Locals whose references stay inside one `Loop`, absorbable into a
    /// `for (var x = init; ...)` header.
    pub(super) for_loop_vars: Vec<bool>,
    pub(super) expr_name_counter: usize,
    /// Module-scope names shared across functions; not cloned per call.
    pub(super) module_names: &'m std::collections::HashSet<String>,
    /// Names claimed in this function: arguments, locals, expression bindings.
    pub(super) local_used_names: std::collections::HashSet<String>,
    /// Call results inlinable at their use site: `ref_count == 1` and no
    /// side-effecting statement between the `Call` and the use.
    pub(super) inlineable_calls: HandleSet<naga::Expression>,
    /// `Load`s that must be `let`-bound: their place is written between the
    /// `Load`'s `Emit` and a use, so inlining would relocate the read past the
    /// write and yield the post-write value (the classic swap `let t=x;x=y;y=t`).
    pub(super) must_bind_loads: HandleSet<naga::Expression>,
    /// Memo for the rendered-depth cap (`0` = not computed).  Depths are
    /// queried child-first in arena order, so a child bound after its entry was
    /// taken can only make an ancestor's stored depth an overestimate, the safe
    /// direction (at worst an extra `let`).
    pub(super) render_depth_memo: Vec<u16>,
    /// True rendered depth of each stashed single-use call text, keyed by
    /// `CallResult`: the result has no expression children (the arguments hang
    /// off the `Call` statement), so without it a chain of stashed calls prices
    /// as nested leaves and escapes the depth cap.
    pub(super) stashed_call_depth: HandleMap<naga::Expression, u16>,
    /// Operands `let`-bound by the `const_hazard` guard, in emission order:
    /// pre-emitted expressions usable from any block, so each name must leave
    /// `expr_names` when its block closes.
    pub(super) const_hazard_bindings: Vec<naga::Handle<naga::Expression>>,
    /// Function name for diagnostics.
    pub(super) display_name: String,
}

/// Where an expression's resolved type comes from: naga keeps function
/// expressions in `FunctionInfo` and module-scope ones in `ModuleInfo`.
pub(super) enum ExprTypes<'a> {
    Function(&'a naga::valid::FunctionInfo),
    Module(&'a naga::valid::ModuleInfo),
}

static EMPTY_FUNCTION: std::sync::LazyLock<naga::Function> =
    std::sync::LazyLock::new(naga::Function::default);
static NO_NAMES: std::sync::LazyLock<HashSet<String>> = std::sync::LazyLock::new(HashSet::new);

impl<'a, 'm> FunctionCtx<'a, 'm> {
    /// `typed_pointer_arg_locals` entry whose initialiser is not a
    /// constructor / zero value of its own type (those spell the alias).
    pub(super) fn needs_declared_type(
        &self,
        local: naga::Handle<naga::LocalVariable>,
        init: naga::Handle<naga::Expression>,
    ) -> bool {
        self.typed_pointer_arg_locals
            .get(local.index())
            .is_some_and(|&typed| typed)
            && !matches!(
                self.exprs[init],
                naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty)
                    if ty == self.func.local_variables[local].ty
            )
    }

    pub(super) fn ty(
        &self,
        expr: naga::Handle<naga::Expression>,
    ) -> &'a naga::proc::TypeResolution {
        match self.types {
            ExprTypes::Function(info) => &info[expr].ty,
            ExprTypes::Module(info) => &info[expr],
        }
    }

    pub(super) fn next_expr_name(&mut self) -> String {
        loop {
            let name = crate::name_gen::next_name(&mut self.expr_name_counter);
            if !self.module_names.contains(&name) && !self.local_used_names.contains(&name) {
                self.local_used_names.insert(name.clone());
                return name;
            }
        }
    }
}

/// Every module-scope identifier as the output SPELLS it - renamed
/// declarations, entry points, and the preserve / preamble list, whose pruned
/// bindings are in no arena yet stand in the consumer's spliced text.  Not
/// `name_gen::module_scope_names`, which censuses the IR's source names.
///
/// The three name decisions - which alias spellings are taken, and what a
/// minted `alias` or `let` must dodge - all start here and chain on what
/// each alone adds, so none can drift from the others.
fn emitted_module_names<'a>(
    module: &'a naga::Module,
    options: &'a GenerateOptions,
    type_names: &'a HandleMap<naga::Type, String>,
    constant_names: &'a [String],
    override_names: &'a [String],
    global_names: &'a [String],
    function_names: &'a [String],
) -> impl Iterator<Item = &'a str> + 'a {
    type_names
        .values()
        .chain(constant_names)
        .chain(override_names)
        .chain(global_names)
        .chain(function_names)
        .chain(module.entry_points.iter().map(|ep| &ep.name))
        .map(String::as_str)
        .chain(options.preserve_symbols.iter().map(String::as_str))
}

/// [`emitted_module_names`] plus the function-locals that shadow them.
fn emitted_names<'a>(
    module: &'a naga::Module,
    options: &'a GenerateOptions,
    type_names: &'a HandleMap<naga::Type, String>,
    constant_names: &'a [String],
    override_names: &'a [String],
    global_names: &'a [String],
    function_names: &'a [String],
) -> impl Iterator<Item = &'a str> + 'a {
    emitted_module_names(
        module,
        options,
        type_names,
        constant_names,
        override_names,
        global_names,
        function_names,
    )
    .chain(all_functions(module).flat_map(crate::name_gen::function_local_names))
}

impl<'a> Generator<'a> {
    /// [`emitted_module_names`] over a constructed generator's own tables.
    pub(super) fn emitted_module_names(&self) -> impl Iterator<Item = &str> + '_ {
        emitted_module_names(
            self.module,
            &self.options,
            &self.type_names,
            &self.constant_names,
            &self.override_names,
            &self.global_names,
            &self.function_names,
        )
    }
}

impl<'a> Generator<'a> {
    /// Context for module-scope expressions: no locals, and every earlier named
    /// constant's initializer bound to that constant's name, so a shared
    /// initializer re-emits as the name.
    pub(super) fn module_ctx(&self) -> FunctionCtx<'a, 'static> {
        let n = self.module.global_expressions.len();
        FunctionCtx {
            func: &EMPTY_FUNCTION,
            exprs: &self.module.global_expressions,
            types: ExprTypes::Module(self.info),
            elide_array_ctor: false,
            pinned_root: None,
            argument_names: Vec::new(),
            local_names: HandleMap::default(),
            expr_names: self
                .expr_to_const
                .iter()
                .map(|(h, c)| (*h, self.constant_names[c.index()].clone()))
                .collect(),
            ref_counts: vec![0; n],
            deferred_vars: Vec::new(),
            typed_pointer_arg_locals: Vec::new(),
            dead_vars: Vec::new(),
            for_loop_vars: Vec::new(),
            expr_name_counter: 0,
            module_names: &NO_NAMES,
            local_used_names: HashSet::new(),
            inlineable_calls: HandleSet::default(),
            must_bind_loads: HandleSet::default(),
            render_depth_memo: vec![0; n],
            stashed_call_depth: HandleMap::default(),
            const_hazard_bindings: Vec::new(),
            display_name: String::from("<module>"),
        }
    }
}

// MARK: Type alias planning

/// References to each `Handle<Type>` from live declarations and expressions
/// that flow through `type_ref()`; dead constants and types are excluded so
/// the alias-cost estimate matches the emitted output.
fn count_type_handle_refs(
    module: &naga::Module,
    live_constants: &HandleSet<naga::Constant>,
    live_types: &HandleSet<naga::Type>,
    defer_cache: &[(Vec<bool>, Vec<bool>)],
) -> HandleMap<naga::Type, usize> {
    let mut counts: HandleMap<naga::Type, usize> = Default::default();
    let mut inc = |h: naga::Handle<naga::Type>| {
        *counts.entry(h).or_default() += 1;
    };

    for (h, ty) in module.types.iter() {
        if !live_types.contains(h) {
            continue;
        }
        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for member in members {
                inc(member.ty);
            }
        }
    }

    // A live constant's `: <type>` is emitted only when the init text does not
    // already spell it; counting it otherwise double-counts (here and in the
    // Compose / ZeroValue walk) and can introduce a net-larger alias.  Two safe
    // imprecisions only ever forgo a borderline alias: a `Splat`-init const
    // under-counts (the global-expr walk skips `Splat`), and an unnamed
    // constant that `generate_constants` skips is still counted.
    for (h, c) in module.constants.iter() {
        if !live_constants.contains(h) {
            continue;
        }
        if !super::module_emit::const_init_has_explicit_type(&module.global_expressions[c.init]) {
            inc(c.ty);
        }
    }

    for (_, ov) in module.overrides.iter() {
        inc(ov.ty);
    }

    for (_, g) in module.global_variables.iter() {
        inc(g.ty);
    }

    for (_, expr) in module.global_expressions.iter() {
        match expr {
            naga::Expression::Compose { ty, .. } if live_types.contains(ty) => inc(*ty),
            naga::Expression::ZeroValue(ty) if live_types.contains(ty) => inc(*ty),
            _ => {}
        }
    }

    // A dead local is never printed, so counting its type would credit an
    // `alias X=T;` used nowhere.  Argument / result types always print;
    // `Compose` / `ZeroValue` entries are counted wholesale because
    // `collect_emitted_handles` under-marks some expressions the emitter does
    // print, which makes a live-only gate unsound here.  The alias planner is
    // greedy, so any count perturbation can flip a marginal choice; all outputs
    // stay valid.
    for (func, (_, dead_locals)) in all_functions(module).zip(defer_cache) {
        for arg in &func.arguments {
            inc(arg.ty);
        }
        if let Some(result) = &func.result {
            inc(result.ty);
        }
        for (h, local) in func.local_variables.iter() {
            if !dead_locals[h.index()] {
                inc(local.ty);
            }
        }
        for (_, expr) in func.expressions.iter() {
            match expr {
                naga::Expression::Compose { ty, .. } => inc(*ty),
                naga::Expression::ZeroValue(ty) => inc(*ty),
                _ => {}
            }
        }
    }

    counts
}

// MARK: Liveness analyses

/// Constants transitively reachable from function / entry-point expressions,
/// global-variable and override initialisers, and preserved names.  A library
/// module (no entry points) keeps every constant, matching the `Compact`
/// pass's `KeepUnused::Yes`.
fn compute_live_constants(
    module: &naga::Module,
    preserve_names: &HashSet<String>,
) -> HandleSet<naga::Constant> {
    let mut live: HandleSet<naga::Constant> = Default::default();

    if module.entry_points.is_empty() {
        return module.constants.iter().map(|(h, _)| h).collect();
    }

    if !preserve_names.is_empty() {
        for (h, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref()
                && preserve_names.contains(name)
            {
                live.insert(h);
            }
        }
    }

    for func in all_functions(module) {
        for (_, expr) in func.expressions.iter() {
            if let naga::Expression::Constant(h) = expr {
                live.insert(*h);
            }
        }
    }

    for (_, g) in module.global_variables.iter() {
        if let Some(init) = g.init {
            collect_const_refs_in_global_expr(init, module, &mut live);
        }
    }

    for (_, ov) in module.overrides.iter() {
        if let Some(init) = ov.init {
            collect_const_refs_in_global_expr(init, module, &mut live);
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        let snapshot: Vec<_> = live.iter().copied().collect();
        for ch in snapshot {
            let before = live.len();
            collect_const_refs_in_global_expr(module.constants[ch].init, module, &mut live);
            if live.len() > before {
                changed = true;
            }
        }
    }

    live
}

/// Collect `Constant` references in a global-expression tree, following into
/// each constant's own init.
fn collect_const_refs_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Constant>,
) {
    use naga::Expression as E;
    match &module.global_expressions[expr_h] {
        E::Constant(h) if live.insert(*h) => {
            collect_const_refs_in_global_expr(module.constants[*h].init, module, live);
        }
        E::Compose { components, .. } => {
            for c in components {
                collect_const_refs_in_global_expr(*c, module, live);
            }
        }
        E::Binary { left, right, .. } => {
            collect_const_refs_in_global_expr(*left, module, live);
            collect_const_refs_in_global_expr(*right, module, live);
        }
        E::Unary { expr, .. } | E::As { expr, .. } => {
            collect_const_refs_in_global_expr(*expr, module, live);
        }
        E::Splat { value, .. } => {
            collect_const_refs_in_global_expr(*value, module, live);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            collect_const_refs_in_global_expr(*condition, module, live);
            collect_const_refs_in_global_expr(*accept, module, live);
            collect_const_refs_in_global_expr(*reject, module, live);
        }
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            collect_const_refs_in_global_expr(*arg, module, live);
            if let Some(a) = arg1 {
                collect_const_refs_in_global_expr(*a, module, live);
            }
            if let Some(a) = arg2 {
                collect_const_refs_in_global_expr(*a, module, live);
            }
            if let Some(a) = arg3 {
                collect_const_refs_in_global_expr(*a, module, live);
            }
        }
        E::Access { base, index } => {
            collect_const_refs_in_global_expr(*base, module, live);
            collect_const_refs_in_global_expr(*index, module, live);
        }
        E::AccessIndex { base, .. }
        | E::Swizzle { vector: base, .. }
        | E::Relational { argument: base, .. } => {
            collect_const_refs_in_global_expr(*base, module, live);
        }
        _ => {}
    }
}

/// Types transitively reachable from live code, so dead struct declarations
/// can be skipped; a library module keeps every type.
fn compute_live_types(
    module: &naga::Module,
    live_constants: &HandleSet<naga::Constant>,
    defer_cache: &[(Vec<bool>, Vec<bool>)],
) -> HandleSet<naga::Type> {
    let mut live: HandleSet<naga::Type> = Default::default();

    if module.entry_points.is_empty() {
        return module.types.iter().map(|(h, _)| h).collect();
    }

    let mut mark = |h: naga::Handle<naga::Type>| {
        live.insert(h);
    };

    for (h, c) in module.constants.iter() {
        if live_constants.contains(h) {
            mark(c.ty);
        }
    }

    for (_, ov) in module.overrides.iter() {
        mark(ov.ty);
    }

    for (_, g) in module.global_variables.iter() {
        mark(g.ty);
    }

    for (func, (_, dead_locals)) in all_functions(module).zip(defer_cache) {
        for arg in &func.arguments {
            mark(arg.ty);
        }
        if let Some(result) = &func.result {
            mark(result.ty);
        }
        // A dead local is never declared, so its type is not live: naga's
        // compactor roots ALL locals, and a DCE'd `var res: __frexp_result_f16;`
        // would otherwise pin its special type and, through the member scalars,
        // a spurious `enable f16;`.  `defer_cache` is the emitter's own verdict.
        for (h, local) in func.local_variables.iter() {
            if !dead_locals[h.index()] {
                mark(local.ty);
            }
        }
        for (_, expr) in func.expressions.iter() {
            match expr {
                naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => {
                    mark(*ty);
                }
                _ => {}
            }
        }
    }

    // Init-tree `Compose` / `ZeroValue` types should already be reachable
    // through the constant's `.ty`; walking them is defense in depth.
    for (h, c) in module.constants.iter() {
        if live_constants.contains(h) {
            collect_types_in_global_expr(c.init, module, &mut live);
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        let snapshot: Vec<_> = live.iter().copied().collect();
        for th in snapshot {
            let before = live.len();
            collect_inner_types(th, module, &mut live);
            if live.len() > before {
                changed = true;
            }
        }
    }

    live
}

/// Insert the types nested directly in `ty_h` (struct members, array element,
/// pointer base) so liveness propagates through composites.
fn collect_inner_types(
    ty_h: naga::Handle<naga::Type>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Type>,
) {
    match &module.types[ty_h].inner {
        naga::TypeInner::Struct { members, .. } => {
            for m in members {
                live.insert(m.ty);
            }
        }
        naga::TypeInner::Array { base, .. } | naga::TypeInner::BindingArray { base, .. } => {
            live.insert(*base);
        }
        naga::TypeInner::Pointer { base, .. } => {
            live.insert(*base);
        }
        _ => {}
    }
}

/// Insert every `Compose` / `ZeroValue` type in a global-expression tree,
/// following into constants' inits.
fn collect_types_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Type>,
) {
    use naga::Expression as E;
    match &module.global_expressions[expr_h] {
        E::Compose { ty, components } => {
            live.insert(*ty);
            for c in components {
                collect_types_in_global_expr(*c, module, live);
            }
        }
        E::ZeroValue(ty) => {
            live.insert(*ty);
        }
        E::Constant(h) => {
            collect_types_in_global_expr(module.constants[*h].init, module, live);
        }
        E::Binary { left, right, .. }
        | E::Access {
            base: left,
            index: right,
        } => {
            collect_types_in_global_expr(*left, module, live);
            collect_types_in_global_expr(*right, module, live);
        }
        E::Unary { expr, .. }
        | E::As { expr, .. }
        | E::Splat { value: expr, .. }
        | E::AccessIndex { base: expr, .. }
        | E::Swizzle { vector: expr, .. }
        | E::Relational { argument: expr, .. } => {
            collect_types_in_global_expr(*expr, module, live);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            collect_types_in_global_expr(*condition, module, live);
            collect_types_in_global_expr(*accept, module, live);
            collect_types_in_global_expr(*reject, module, live);
        }
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            collect_types_in_global_expr(*arg, module, live);
            if let Some(a) = arg1 {
                collect_types_in_global_expr(*a, module, live);
            }
            if let Some(a) = arg2 {
                collect_types_in_global_expr(*a, module, live);
            }
            if let Some(a) = arg3 {
                collect_types_in_global_expr(*a, module, live);
            }
        }
        _ => {}
    }
}

// MARK: Construction and output

impl<'a> Generator<'a> {
    /// Pre-computes liveness, layout and alias tables so emission stays
    /// allocation-free on the hot path.
    pub(super) fn new(
        module: &'a naga::Module,
        info: &'a naga::valid::ModuleInfo,
        options: GenerateOptions,
    ) -> Self {
        let mangle = options.mangle;

        // Minted names (mangled struct / member names, the anonymous
        // override's) must dodge every name in scope where they are
        // referenced: all module-scope names, every argument and local
        // (function scope shadows type names) and the preserve list.  Built
        // only when something draws on it.
        let mut used_names = HashSet::new();
        if mangle || module.overrides.iter().any(|(_, o)| o.name.is_none()) {
            used_names.extend(
                crate::name_gen::module_scope_names(module)
                    .chain(all_functions(module).flat_map(crate::name_gen::function_local_names))
                    .map(str::to_owned),
            );
            used_names.extend(options.preserve_symbols.iter().cloned());
        }
        let mut mangle_counter = 0usize;

        let preserve = &options.preserve_symbols;
        let mut type_names = HandleMap::default();
        let mut member_names = FxHashMap::default();

        // naga predeclared / special struct types are never renamed: their
        // members are accessed through canonical names (`.old_value`,
        // `.fract`, `.kind`, ...) and no declaration is emitted for them, so a
        // mangled accessor would be invalid WGSL.
        let predeclared_type_handles = super::module_emit::special_struct_handles(module);

        for (h, ty) in module.types.iter() {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                // Name-based fallback for IRs where `special_types.ray_desc` is
                // not populated.
                let is_ray_descriptor = ty.name.as_deref() == Some("RayDesc");

                let is_predeclared = predeclared_type_handles.contains(h);

                if mangle {
                    if is_predeclared || is_ray_descriptor {
                        type_names.insert(
                            h,
                            ty.name.clone().unwrap_or_else(|| format!("T{}", h.index())),
                        );
                        for (idx, member) in members.iter().enumerate() {
                            member_names.insert(
                                (h, idx as u32),
                                member.name.clone().unwrap_or_else(|| format!("m{}", idx)),
                            );
                        }
                        continue;
                    }

                    if let Some(name) = ty.name.as_deref() {
                        if preserve.contains(name) {
                            type_names.insert(h, name.to_string());
                        } else {
                            type_names.insert(
                                h,
                                crate::name_gen::next_name_unique(&mut mangle_counter, &used_names),
                            );
                        }
                    } else {
                        type_names.insert(
                            h,
                            crate::name_gen::next_name_unique(&mut mangle_counter, &used_names),
                        );
                    }
                    for (idx, member) in members.iter().enumerate() {
                        if let Some(name) = member.name.as_deref() {
                            if preserve.contains(name) || options.preserve_members.contains(name) {
                                member_names.insert((h, idx as u32), name.to_string());
                            } else {
                                member_names.insert(
                                    (h, idx as u32),
                                    crate::name_gen::next_name_unique(
                                        &mut mangle_counter,
                                        &used_names,
                                    ),
                                );
                            }
                        } else {
                            member_names.insert(
                                (h, idx as u32),
                                crate::name_gen::next_name_unique(&mut mangle_counter, &used_names),
                            );
                        }
                    }
                } else {
                    type_names.insert(
                        h,
                        ty.name.clone().unwrap_or_else(|| format!("T{}", h.index())),
                    );
                }
            }
        }

        let mut constant_names = Vec::with_capacity(module.constants.len());
        for (h, c) in module.constants.iter() {
            debug_assert_eq!(h.index(), constant_names.len());
            constant_names.push(c.name.clone().unwrap_or_else(|| format!("C{}", h.index())));
        }

        // naga's anonymous override (an override-expression array size) is
        // declared under a minted name: the positional `O<n>` collided with
        // user, preserved and preamble names.  Named here rather than in the
        // IR so a dead one still compacts away instead of being rooted as a
        // host-visible override.
        let mut override_names = Vec::with_capacity(module.overrides.len());
        let mut anonymous_counter = 0usize;
        for (h, ov) in module.overrides.iter() {
            debug_assert_eq!(h.index(), override_names.len());
            override_names.push(ov.name.clone().unwrap_or_else(|| {
                crate::name_gen::next_name_insert(&mut anonymous_counter, &mut used_names)
            }));
        }

        let mut global_names = Vec::with_capacity(module.global_variables.len());
        for (h, g) in module.global_variables.iter() {
            debug_assert_eq!(h.index(), global_names.len());
            global_names.push(g.name.clone().unwrap_or_else(|| format!("G{}", h.index())));
        }

        let mut function_names = Vec::with_capacity(module.functions.len());
        for (h, f) in module.functions.iter() {
            debug_assert_eq!(h.index(), function_names.len());
            function_names.push(f.name.clone().unwrap_or_else(|| format!("f{}", h.index())));
        }

        let tok = if options.beautify {
            &PRETTY_TOKENS
        } else {
            &COMPACT_TOKENS
        };
        let indent_unit = if options.beautify {
            // `options.indent` is a `u8`, so the 256-byte buffer always yields
            // an in-bounds all-ASCII slice; a `&'static str` keeps `indent_unit`
            // borrow-free.
            static SPACES: [u8; 256] = [b' '; 256];
            std::str::from_utf8(&SPACES[..options.indent as usize])
                .expect("ASCII spaces are valid UTF-8")
        } else {
            ""
        };

        let mut layouter = naga::proc::Layouter::default();
        // `Layouter::update` stops at the first un-layoutable type (overflowing
        // arrays, type-arena cycles) and leaves later slots unpopulated, which
        // panic on indexing; record success so layout-dependent emission can
        // bail.
        let layouter_complete = layouter.update(module.to_ctx()).is_ok();

        let defer_cache: Vec<(Vec<bool>, Vec<bool>)> = all_functions(module)
            .map(super::module_emit::find_deferrable_vars)
            .collect();
        let live_constants = compute_live_constants(module, &options.preserve_symbols);
        let live_types = compute_live_types(module, &live_constants, &defer_cache);

        let in_scope = || {
            emitted_names(
                module,
                &options,
                &type_names,
                &constant_names,
                &override_names,
                &global_names,
                &function_names,
            )
        };

        let shadowed_type_aliases: super::syntax::ShadowedAliases = in_scope()
            .filter(|name| super::syntax::is_predeclared_type_alias(name))
            .map(str::to_owned)
            .collect();

        let type_alias_decls = if options.type_alias {
            let ref_counts =
                count_type_handle_refs(module, &live_constants, &live_types, &defer_cache);

            let mut alias_used: HashSet<String> = in_scope().map(str::to_owned).collect();

            let mut alias_counter = 0usize;
            let mut decls: Vec<(String, String)> = Vec::new();

            let fixed_overhead = super::syntax::decl_boilerplate(options.beautify);

            // `UniqueArena<Type>` deduplicates by the full `Type` (name
            // included), so one `TypeInner` can sit under several handles when
            // the source mixes bare types with named aliases.  Group by inner
            // once: `canonical[h]` is the arena-first handle of h's group and
            // `group_ref_count[head]` the group's summed `ref_counts`.
            let mut canonical: HandleMap<naga::Type, naga::Handle<naga::Type>> = Default::default();
            let mut inner_to_first: FxHashMap<&naga::TypeInner, naga::Handle<naga::Type>> =
                Default::default();
            for (h, ty) in module.types.iter() {
                let first = *inner_to_first.entry(&ty.inner).or_insert(h);
                canonical.insert(h, first);
            }
            let mut group_ref_count: HandleMap<naga::Type, usize> = Default::default();
            for (h, _) in module.types.iter() {
                let first = canonical[&h];
                *group_ref_count.entry(first).or_insert(0) +=
                    ref_counts.get(h).copied().unwrap_or(0);
            }

            for (h, ty) in module.types.iter() {
                // Structs already have short names.
                if type_names.contains_key(h) {
                    continue;
                }

                let group_head = canonical[&h];
                if h != group_head
                    && let Some(existing) = type_names.get(group_head).cloned()
                {
                    type_names.insert(h, existing);
                    continue;
                }

                let count = group_ref_count.get(group_head).copied().unwrap_or(0);
                if count == 0 {
                    continue;
                }
                // Rendered with the aliases minted so far, so aliases nest.
                let type_str = match super::syntax::type_inner_name(
                    &ty.inner,
                    module,
                    &type_names,
                    &override_names,
                    &shadowed_type_aliases,
                ) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let alias_name = crate::name_gen::next_name_unique(&mut alias_counter, &alias_used);
                let alias_len = alias_name.len();
                let type_len = type_str.len();

                let decl_cost = fixed_overhead + alias_len + type_len;
                let savings_per_use = type_len.saturating_sub(alias_len);
                let total_savings = count * savings_per_use;

                if total_savings > decl_cost {
                    alias_used.insert(alias_name.clone());
                    decls.push((alias_name.clone(), type_str));
                    type_names.insert(h, alias_name);
                }
            }
            decls
        } else {
            Vec::new()
        };

        let initial_capacity = options.initial_capacity;
        Self {
            module,
            info,
            options,
            out: String::with_capacity(initial_capacity),
            indent_depth: 0,
            type_names,
            member_names,
            constant_names,
            override_names,
            global_names,
            function_names,
            shadowed_type_aliases,
            extracted_literals: Default::default(),
            type_alias_decls,
            expr_to_const: Default::default(),
            live_constants,
            live_types,
            map_visible_structs: Default::default(),
            layouter,
            layouter_complete,
            ref_count_cache: Vec::new(),
            defer_cache,
            pure_functions: Vec::new(),
            tok,
            indent_unit,
        }
    }

    pub(super) fn into_output(self) -> String {
        self.out
    }

    // MARK: Output helpers

    #[inline]
    pub(super) fn push_indent(&mut self) {
        for _ in 0..self.indent_depth {
            self.out.push_str(self.indent_unit);
        }
    }

    #[inline]
    pub(super) fn open_brace(&mut self) {
        self.out.push_str(self.tok.open_brace);
        self.indent_depth += 1;
    }

    #[inline]
    pub(super) fn close_brace(&mut self) {
        self.indent_depth = self.indent_depth.saturating_sub(1);
        self.push_indent();
        self.out.push('}');
    }

    #[inline]
    pub(super) fn push_separator(&mut self) {
        self.out.push_str(self.tok.separator);
    }

    #[inline]
    pub(super) fn push_assign(&mut self) {
        self.out.push_str(self.tok.assign);
    }

    #[inline]
    pub(super) fn push_colon(&mut self) {
        self.out.push_str(self.tok.colon);
    }

    #[inline]
    pub(super) fn push_arrow(&mut self) {
        self.out.push_str(self.tok.arrow);
    }

    #[inline]
    pub(super) fn push_newline(&mut self) {
        self.out.push_str(self.tok.newline);
    }

    #[inline]
    pub(super) fn comma_sep(&self) -> &'static str {
        self.tok.separator
    }

    #[inline]
    pub(super) fn bin_op_sep(&self) -> &'static str {
        self.tok.bin_op_sep
    }

    #[inline]
    pub(super) fn assign_sep(&self) -> &'static str {
        self.tok.assign
    }

    #[inline]
    pub(super) fn push_binding_sep(&mut self) {
        self.out.push_str(self.tok.binding_sep);
    }

    #[inline]
    pub(super) fn push_attr_end(&mut self) {
        self.out.push_str(self.tok.attr_end);
    }

    #[inline]
    pub(super) fn push_angle_end(&mut self) {
        self.out.push_str(self.tok.angle_end);
    }

    #[inline]
    pub(super) fn push_else(&mut self) {
        self.out.push_str(self.tok.else_kw);
    }

    #[inline]
    pub(super) fn push_for_open(&mut self) {
        self.out.push_str(self.tok.for_open);
    }

    #[inline]
    pub(super) fn push_for_sep(&mut self) {
        self.out.push_str(self.tok.for_sep);
    }
}
