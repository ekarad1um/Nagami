//! Generator state and per-function context.
//!
//! [`Generator`] owns the output buffer, cached module-wide analyses
//! (reachable constants and types, type-alias plan, per-function
//! expression ref-counts, layout table), and the precomputed format
//! tokens that beautify-mode and minify-mode both read from.
//!
//! [`FunctionCtx`] is the per-function bundle; the statement and
//! expression emitters thread it through their recursive walks so
//! name bindings, deferred-variable flags, and inline-eligibility
//! decisions stay coherent inside a single function.

use std::collections::{HashMap, HashSet};

use super::syntax::LiteralExtractKey;

// MARK: Options

/// Caller-facing generator knobs.  Every field is tied directly to a
/// [`crate::config::Config`] entry resolved by [`crate::run`].  Changing
/// a default here is a public-surface change and requires updating
/// both that call site and the tests.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Emit human-readable output with indentation and newlines.
    pub beautify: bool,
    /// Spaces per indentation level; honoured only when `beautify` is set.
    pub indent: u8,
    /// Rename struct types and struct members to short identifiers.
    pub mangle: bool,
    /// Maximum decimal places for float literals; `None` preserves
    /// full precision and any `Some(n)` is lossy.
    pub max_precision: Option<u8>,
    /// Reserve this many bytes up front in the output buffer to amortise
    /// reallocation across the emission pass.
    pub initial_capacity: usize,
    /// Symbol names to preserve from mangling (struct types and
    /// members).  Consulted only when `mangle` is true.
    pub preserve_symbols: HashSet<String>,
    /// Names of preamble (external) declarations to exclude from output.
    pub preamble_names: HashSet<String>,
    /// Emit `alias` declarations for frequently-referenced types when
    /// the alias shortens total output.
    pub type_alias: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            beautify: false,
            indent: 2,
            mangle: false,
            max_precision: None,
            initial_capacity: 16 * 1024,
            preserve_symbols: HashSet::new(),
            preamble_names: HashSet::new(),
            type_alias: false,
        }
    }
}

// MARK: Cached analyses

/// Per-function expression analysis cached on [`Generator`] so both
/// `ref_counts` and `live` are computed exactly once per function
/// instead of once per consumer.  `compute_expression_ref_counts`
/// produces the pair in lock-step; keeping them together eliminates
/// the duplicate body walk that `literal_extract::count_literals`
/// previously needed to rebuild `live`.
///
/// The two fields have different lifecycles:
///
/// - `live` is consumed once by
///   [`super::literal_extract::scan_and_extract_literals`], which
///   `mem::take`s the `Vec`.  Do not read `live` after that call.
/// - `ref_counts` is consumed per function by
///   `module_emit::generate_function`, which `mem::take`s the slot
///   as each function is emitted.
pub(super) struct FunctionExprInfo {
    pub(super) ref_counts: Vec<usize>,
    pub(super) live: Vec<bool>,
}

// MARK: Generator state

/// Owns the output buffer, cached module-wide analyses, and the
/// identifier allocation tables the sub-emitters consult.  A fresh
/// instance is created per [`super::generate_wgsl`] call; the state
/// is never reused across modules.
pub(super) struct Generator<'a> {
    pub(super) module: &'a naga::Module,
    pub(super) info: &'a naga::valid::ModuleInfo,
    pub(super) options: GenerateOptions,
    pub(super) out: String,
    pub(super) indent_depth: u32,
    pub(super) type_names: HashMap<naga::Handle<naga::Type>, String>,
    pub(super) member_names: HashMap<(naga::Handle<naga::Type>, u32), String>,
    pub(super) constant_names: Vec<String>,
    pub(super) override_names: Vec<String>,
    pub(super) global_names: Vec<String>,
    pub(super) function_names: Vec<String>,
    pub(super) extracted_literals: HashMap<LiteralExtractKey, String>,
    /// Alias declarations awaiting emission, stored as
    /// `(alias_name, type_string)`.
    pub(super) type_alias_decls: Vec<(String, String)>,
    /// Map from global-expression handle to a named constant whose
    /// `init` is that handle.  Populated incrementally during constant
    /// emission so later constants can reference earlier ones by name
    /// instead of re-inlining the entire expression tree.
    pub(super) expr_to_const: HashMap<naga::Handle<naga::Expression>, naga::Handle<naga::Constant>>,
    /// Constants reachable from live code (functions, entry points,
    /// global-variable initialisers).  Dead constants are skipped
    /// during emission.
    pub(super) live_constants: HashSet<naga::Handle<naga::Constant>>,
    /// Types reachable from live code.  Dead struct declarations are
    /// skipped during emission.
    pub(super) live_types: HashSet<naga::Handle<naga::Type>>,
    /// Pre-computed type layouts used when reconstructing `@size` and
    /// `@align` attributes on struct members.
    pub(super) layouter: naga::proc::Layouter,
    /// Cached per-function analyses: ref counts plus the live
    /// (in-`Emit`-range) bitmap, indexed `[0..N)` for regular
    /// functions and `[N..N+E)` for entry points.  Both halves are
    /// produced together by `compute_expression_ref_counts` so
    /// consumers that need the live mask (e.g. `literal_extract`) do
    /// not re-walk the body.
    pub(super) ref_count_cache: Vec<FunctionExprInfo>,
    // Pre-computed format tokens chosen at construction time from
    // `options.beautify` so the hot path never branches per character.
    tok_separator: &'static str,
    tok_assign: &'static str,
    tok_colon: &'static str,
    tok_arrow: &'static str,
    tok_open_brace: &'static str,
    tok_newline: &'static str,
    tok_bin_op_sep: &'static str,
    tok_binding_sep: &'static str,
    tok_attr_end: &'static str,
    tok_angle_end: &'static str,
    tok_else: &'static str,
    tok_for_open: &'static str,
    tok_for_sep: &'static str,
    indent_unit: &'static str,
}

// MARK: Function context

/// Per-function emission context.  Created once per function by
/// `module_emit::generate_function` and threaded through every
/// statement and expression emitter so name bindings, deferred
/// variable flags, and inline-eligibility decisions stay coherent
/// inside a single function body.
pub(super) struct FunctionCtx<'a, 'm> {
    pub(super) func: &'a naga::Function,
    pub(super) info: &'a naga::valid::FunctionInfo,
    pub(super) argument_names: Vec<String>,
    pub(super) local_names: HashMap<naga::Handle<naga::LocalVariable>, String>,
    pub(super) expr_names: HashMap<naga::Handle<naga::Expression>, String>,
    pub(super) ref_counts: Vec<usize>,
    pub(super) deferred_vars: Vec<bool>,
    pub(super) dead_vars: Vec<bool>,
    /// Locals whose references stay inside a single `Loop` and can
    /// therefore be absorbed into a `for (var x = init; ...)` header.
    pub(super) for_loop_vars: Vec<bool>,
    pub(super) expr_name_counter: usize,
    /// Module-scope names shared across functions; not cloned per call.
    pub(super) module_names: &'m std::collections::HashSet<String>,
    /// Names already claimed inside this function (arguments, locals,
    /// and expression bindings).
    pub(super) local_used_names: std::collections::HashSet<String>,
    /// Call results that can safely be inlined at their use site:
    /// `ref_count == 1`, no side-effecting statement between the
    /// `Call` and its use.
    pub(super) inlineable_calls: std::collections::HashSet<naga::Handle<naga::Expression>>,
    /// Display name for the current function, used to decorate
    /// diagnostic messages.
    pub(super) display_name: String,
}

impl<'a, 'm> FunctionCtx<'a, 'm> {
    /// Allocate the next short, collision-free expression-binding
    /// name, claiming it in `local_used_names` so siblings cannot
    /// reuse it later in the same function.
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

// MARK: Type alias planning

/// Count how many times each `Handle<Type>` is referenced by **live**
/// declarations and expressions that flow through `type_ref()`.  Dead
/// constants and dead types are excluded so the alias-cost estimate
/// reflects what will actually appear in the emitted output.
fn count_type_handle_refs(
    module: &naga::Module,
    live_constants: &HashSet<naga::Handle<naga::Constant>>,
    live_types: &HashSet<naga::Handle<naga::Type>>,
) -> HashMap<naga::Handle<naga::Type>, usize> {
    let mut counts: HashMap<naga::Handle<naga::Type>, usize> = HashMap::new();
    let mut inc = |h: naga::Handle<naga::Type>| {
        *counts.entry(h).or_default() += 1;
    };

    // Struct member types (only live structs).
    for (h, ty) in module.types.iter() {
        if !live_types.contains(&h) {
            continue;
        }
        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for member in members {
                inc(member.ty);
            }
        }
    }

    // Constants (only live ones).
    for (h, c) in module.constants.iter() {
        if live_constants.contains(&h) {
            inc(c.ty);
        }
    }

    // Overrides.
    for (_, ov) in module.overrides.iter() {
        inc(ov.ty);
    }

    // Global variables.
    for (_, g) in module.global_variables.iter() {
        inc(g.ty);
    }

    // Global expressions (Compose / ZeroValue) - only count types that are live.
    for (_, expr) in module.global_expressions.iter() {
        match expr {
            naga::Expression::Compose { ty, .. } if live_types.contains(ty) => inc(*ty),
            naga::Expression::ZeroValue(ty) if live_types.contains(ty) => inc(*ty),
            _ => {}
        }
    }

    // Functions and entry points.
    let all_funcs = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function));
    for func in all_funcs {
        for arg in &func.arguments {
            inc(arg.ty);
        }
        if let Some(result) = &func.result {
            inc(result.ty);
        }
        for (_, local) in func.local_variables.iter() {
            inc(local.ty);
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

/// Transitively gather every `Constant` reachable from live code.
/// Seeds are constants referenced by function and entry-point
/// expressions, by global-variable initialisers, by override
/// initialisers, and by any name the caller asked to preserve.
/// Library modules (no entry points) keep every constant to match
/// the `Compact` pass's `KeepUnused::Yes` behaviour.
fn compute_live_constants(
    module: &naga::Module,
    preserve_names: &HashSet<String>,
) -> HashSet<naga::Handle<naga::Constant>> {
    let mut live: HashSet<naga::Handle<naga::Constant>> = HashSet::new();

    // Library module (no entry points): keep everything, just like the compact
    // pass does with KeepUnused::Yes.
    if module.entry_points.is_empty() {
        return module.constants.iter().map(|(h, _)| h).collect();
    }

    // Constants whose names the user asked to preserve are always live.
    if !preserve_names.is_empty() {
        for (h, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref() {
                if preserve_names.contains(name) {
                    live.insert(h);
                }
            }
        }
    }

    // Seed: constants referenced directly in function / entry-point expressions.
    let all_funcs = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function));
    for func in all_funcs {
        for (_, expr) in func.expressions.iter() {
            if let naga::Expression::Constant(h) = expr {
                live.insert(*h);
            }
        }
    }

    // Seed: constants referenced in global variable initialisers.
    for (_, g) in module.global_variables.iter() {
        if let Some(init) = g.init {
            collect_const_refs_in_global_expr(init, module, &mut live);
        }
    }

    // Seed: constants referenced in override initialisers.
    for (_, ov) in module.overrides.iter() {
        if let Some(init) = ov.init {
            collect_const_refs_in_global_expr(init, module, &mut live);
        }
    }

    // Transitive closure: follow each live constant's init expression for
    // further constant references.
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

/// Recursively collect `Constant` references inside a global expression tree.
///
/// Covers all expression variants that can appear in `module.global_expressions`
/// and may contain child expression handles.
/// Walk a global-expression sub-tree, pushing every referenced
/// constant into `out`.  Recursive to mirror the composite structure
/// of `Compose`, `Splat`, `Swizzle`, and their relatives.
fn collect_const_refs_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HashSet<naga::Handle<naga::Constant>>,
) {
    use naga::Expression as E;
    match &module.global_expressions[expr_h] {
        E::Constant(h) => {
            if live.insert(*h) {
                collect_const_refs_in_global_expr(module.constants[*h].init, module, live);
            }
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
        // Leaf expressions: Literal, ZeroValue, Override - no child handles.
        _ => {}
    }
}

/// Transitively collect every `Type` reachable from live code so
/// dead struct declarations can be skipped during emission.  Mirrors
/// [`compute_live_constants`] with a different set of seed roots.
fn compute_live_types(
    module: &naga::Module,
    live_constants: &HashSet<naga::Handle<naga::Constant>>,
) -> HashSet<naga::Handle<naga::Type>> {
    let mut live: HashSet<naga::Handle<naga::Type>> = HashSet::new();

    // Library module: keep everything.
    if module.entry_points.is_empty() {
        return module.types.iter().map(|(h, _)| h).collect();
    }

    let mut mark = |h: naga::Handle<naga::Type>| {
        live.insert(h);
    };

    // From live constants.
    for (h, c) in module.constants.iter() {
        if live_constants.contains(&h) {
            mark(c.ty);
        }
    }

    // From overrides.
    for (_, ov) in module.overrides.iter() {
        mark(ov.ty);
    }

    // From global variables.
    for (_, g) in module.global_variables.iter() {
        mark(g.ty);
    }

    // From functions and entry points.
    let all_funcs = module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function));
    for func in all_funcs {
        for arg in &func.arguments {
            mark(arg.ty);
        }
        if let Some(result) = &func.result {
            mark(result.ty);
        }
        for (_, local) in func.local_variables.iter() {
            mark(local.ty);
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

    // From global expressions of live constants (defense-in-depth: types
    // referenced by Compose / ZeroValue in init trees should already be
    // reachable through the constant's `.ty` member chain, but walking
    // them explicitly guards against edge cases).
    for (h, c) in module.constants.iter() {
        if live_constants.contains(&h) {
            collect_types_in_global_expr(c.init, module, &mut live);
        }
    }

    // Transitive closure: struct member types and inner types.
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

/// Collect types referenced inside the given type (struct members, array
/// element, pointer base, etc.).
/// Push every nested `Handle<Type>` referenced by a
/// [`naga::TypeInner`] into `out` (array element types, pointer
/// pointee types, struct member types, and so on) so liveness
/// propagates through composite declarations.
fn collect_inner_types(
    ty_h: naga::Handle<naga::Type>,
    module: &naga::Module,
    live: &mut HashSet<naga::Handle<naga::Type>>,
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

/// Recursively collect `Compose` / `ZeroValue` type handles inside a global
/// expression tree.
/// Walk a global expression and push every directly-referenced
/// `Handle<Type>` into `out` (for example the target type of a
/// `Compose` or `ZeroValue`).  Used during live-type discovery.
fn collect_types_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HashSet<naga::Handle<naga::Type>>,
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
            // Trace into the constant's own init.
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
    /// Construct a generator for `module` with `options`.  Pre-computes
    /// liveness, layout, and alias tables up front so downstream
    /// emission stays allocation-free along the hot path.
    pub(super) fn new(
        module: &'a naga::Module,
        info: &'a naga::valid::ModuleInfo,
        options: GenerateOptions,
    ) -> Self {
        use std::collections::HashSet;

        let mangle = options.mangle;

        // When mangling, collect all names already used by constants,
        // globals, overrides, functions, entry points, and function-local
        // identifiers (arguments, locals) so that struct type/member
        // mangled names don't collide.  In WGSL, function-scope names
        // shadow module-scope type names, so a struct type named `B`
        // becomes unusable inside any function that has a parameter `B`.
        let mut used_names = HashSet::new();
        if mangle {
            for (_, c) in module.constants.iter() {
                if let Some(name) = c.name.as_deref() {
                    used_names.insert(name.to_string());
                }
            }
            for (_, ov) in module.overrides.iter() {
                if let Some(name) = ov.name.as_deref() {
                    used_names.insert(name.to_string());
                }
            }
            for (_, g) in module.global_variables.iter() {
                if let Some(name) = g.name.as_deref() {
                    used_names.insert(name.to_string());
                }
            }
            for (_, f) in module.functions.iter() {
                if let Some(name) = f.name.as_deref() {
                    used_names.insert(name.to_string());
                }
                for arg in &f.arguments {
                    if let Some(name) = arg.name.as_deref() {
                        used_names.insert(name.to_string());
                    }
                }
                for (_, local) in f.local_variables.iter() {
                    if let Some(name) = local.name.as_deref() {
                        used_names.insert(name.to_string());
                    }
                }
            }
            for ep in module.entry_points.iter() {
                used_names.insert(ep.name.clone());
                for arg in &ep.function.arguments {
                    if let Some(name) = arg.name.as_deref() {
                        used_names.insert(name.to_string());
                    }
                }
                for (_, local) in ep.function.local_variables.iter() {
                    if let Some(name) = local.name.as_deref() {
                        used_names.insert(name.to_string());
                    }
                }
            }
            // Reserve all preserved symbol names upfront so that the
            // mangle counter never generates a name that collides with
            // a preserved struct type or member - regardless of the
            // order types appear in the arena.
            used_names.extend(options.preserve_symbols.iter().cloned());
        }
        let mut mangle_counter = 0usize;

        let preserve = &options.preserve_symbols;
        let mut type_names = HashMap::new();
        let mut member_names = HashMap::new();

        // Pre-compute the set of naga predeclared-type handles
        // (AtomicCompareExchangeWeakResult, ModfResult, FrexpResult).
        // Their struct names and member names must never be mangled: WGSL
        // code accesses the members by canonical names (`.old_value`,
        // `.exchanged`, `.fract`, `.whole`, `.exp`) and there is no struct
        // declaration emitted for them, so a mangled accessor would produce
        // invalid WGSL (e.g. "invalid field accessor 'B'").
        let predeclared_type_handles: std::collections::HashSet<naga::Handle<naga::Type>> = module
            .special_types
            .predeclared_types
            .values()
            .copied()
            .collect();

        for (h, ty) in module.types.iter() {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                // `RayDesc` is a WGSL predeclared type for the ray-tracing extension.
                // Its member names must be preserved so that the constructor
                // `RayDesc(flags, cull_mask, ...)` remains valid, and no struct
                // declaration is emitted for it.
                let is_ray_descriptor = ty.name.as_deref() == Some("RayDesc");

                // All naga-internal synthetic result structs (AtomicCmpExch,
                // Modf, Frexp) live in `predeclared_types`.
                let is_predeclared = predeclared_type_handles.contains(&h);

                if mangle {
                    // Keep predeclared struct type/member names stable.
                    // Their field accessors must match the canonical WGSL names:
                    // `.old_value`/`.exchanged` for atomicCompareExchangeWeak,
                    // `.fract`/`.whole` for modf, `.fract`/`.exp` for frexp.
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

                    // Preserve the struct type name if it appears in the preserve set.
                    // All preserved names were added to `used_names` upfront,
                    // so `next_name_unique` will never generate a collision.
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
                            if preserve.contains(name) {
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

        let mut override_names = Vec::with_capacity(module.overrides.len());
        for (h, ov) in module.overrides.iter() {
            debug_assert_eq!(h.index(), override_names.len());
            override_names.push(ov.name.clone().unwrap_or_else(|| format!("O{}", h.index())));
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

        let (
            tok_separator,
            tok_assign,
            tok_colon,
            tok_arrow,
            tok_open_brace,
            tok_newline,
            tok_bin_op_sep,
            tok_binding_sep,
            tok_attr_end,
            tok_angle_end,
            tok_else,
            tok_for_open,
            tok_for_sep,
            indent_unit,
        ) = if options.beautify {
            // Pre-compute single indent unit from a large static buffer, 16 spaces.
            const SPACES: &str = "                ";
            let unit = &SPACES[..(options.indent as usize).min(SPACES.len())];
            (
                ", ",
                " = ",
                ": ",
                " -> ",
                " {\n",
                "\n",
                " ",
                ") @binding(",
                ") ",
                "> ",
                " else",
                "for (",
                "; ",
                unit,
            )
        } else {
            (
                ",",
                "=",
                ":",
                "->",
                "{",
                "",
                "",
                ")@binding(",
                ")",
                ">",
                "else",
                "for(",
                ";",
                "",
            )
        };

        let mut layouter = naga::proc::Layouter::default();
        // Ignore layout errors here; the module passed validation so
        // all types are well-formed.  If a truly exotic type sneaks
        // through, generate_struct will simply treat its natural size
        // as 0 and always emit the explicit @size attribute.
        let _ = layouter.update(module.to_ctx());

        let live_constants = compute_live_constants(module, &options.preserve_symbols);
        let live_types = compute_live_types(module, &live_constants);

        // Type aliasing: introduce `alias` declarations when they shrink output
        let type_alias_decls = if options.type_alias {
            let ref_counts = count_type_handle_refs(module, &live_constants, &live_types);

            // Collect every name already in use so alias names don't collide.
            // This includes function-local names (arguments, locals) because a
            // local with the same name would shadow the type alias inside its
            // function body, breaking type references.
            let mut alias_used: HashSet<String> = HashSet::new();
            alias_used.extend(type_names.values().cloned());
            alias_used.extend(constant_names.iter().cloned());
            alias_used.extend(override_names.iter().cloned());
            alias_used.extend(global_names.iter().cloned());
            alias_used.extend(function_names.iter().cloned());
            alias_used.extend(module.entry_points.iter().map(|ep| ep.name.clone()));
            let all_funcs = module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|ep| &ep.function));
            for func in all_funcs {
                for arg in &func.arguments {
                    if let Some(name) = &arg.name {
                        alias_used.insert(name.clone());
                    }
                }
                for (_, local) in func.local_variables.iter() {
                    if let Some(name) = &local.name {
                        alias_used.insert(name.clone());
                    }
                }
            }

            let mut alias_counter = 0usize;
            let mut decls: Vec<(String, String)> = Vec::new();

            // Fixed overhead per alias declaration:
            //   compact:  "alias <name>=<type>;"   -> 6 + 1 + 1 = 8 fixed chars
            //   beautify: "alias <name> = <type>;\n" -> 6 + 3 + 2 = 11 fixed chars
            let fixed_overhead: usize = if options.beautify { 11 } else { 8 };

            for (h, ty) in module.types.iter() {
                // Structs already have short names in type_names.
                if type_names.contains_key(&h) {
                    continue;
                }

                // If another handle with the same TypeInner already received
                // an alias, reuse it (naga's UniqueArena may hold duplicate
                // handles when the source mixes bare types with named aliases).
                if let Some(existing) = module.types.iter().find_map(|(h2, ty2)| {
                    (h2 != h && ty2.inner == ty.inner)
                        .then(|| type_names.get(&h2).cloned())
                        .flatten()
                }) {
                    type_names.insert(h, existing);
                    continue;
                }

                // Sum ref counts across ALL handles with the same TypeInner.
                // naga may create multiple handles for identical inners when
                // the source mixes bare types with named aliases.
                let count: usize = module
                    .types
                    .iter()
                    .filter(|(_, ty2)| ty2.inner == ty.inner)
                    .map(|(h2, _)| ref_counts.get(&h2).copied().unwrap_or(0))
                    .sum();
                if count == 0 {
                    continue;
                }
                // Compute the type string using current type_names (which may
                // include previously-created aliases for base types).
                let type_str = match super::syntax::type_inner_name(
                    &ty.inner,
                    module,
                    &type_names,
                    &override_names,
                ) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                // Generate candidate alias name.
                let alias_name = crate::name_gen::next_name_unique(&mut alias_counter, &alias_used);
                let alias_len = alias_name.len();
                let type_len = type_str.len();

                // Net savings = (count * chars_saved_per_use) - declaration_cost
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
            extracted_literals: HashMap::new(),
            type_alias_decls,
            expr_to_const: HashMap::new(),
            live_constants,
            live_types,
            layouter,
            ref_count_cache: Vec::new(),
            tok_separator,
            tok_assign,
            tok_colon,
            tok_arrow,
            tok_open_brace,
            tok_newline,
            tok_bin_op_sep,
            tok_binding_sep,
            tok_attr_end,
            tok_angle_end,
            tok_else,
            tok_for_open,
            tok_for_sep,
            indent_unit,
        }
    }

    /// Consume the generator and return the final output string.
    pub(super) fn into_output(self) -> String {
        self.out
    }

    // MARK: Output helpers

    // All of the helpers below read their text from the precomputed
    // `tok_*` fields so the hot path does not branch on `beautify`
    // for each pushed separator.

    /// Emit indentation for the current depth.  No-op when
    /// `beautify` is off because `indent_unit` is an empty string.
    #[inline]
    pub(super) fn push_indent(&mut self) {
        for _ in 0..self.indent_depth {
            self.out.push_str(self.indent_unit);
        }
    }

    /// Push ` {\n` (beautify) or `{` (compact), then increment depth.
    #[inline]
    pub(super) fn open_brace(&mut self) {
        self.out.push_str(self.tok_open_brace);
        self.indent_depth += 1;
    }

    /// Decrement depth, then push indent + `}`.
    #[inline]
    pub(super) fn close_brace(&mut self) {
        self.indent_depth = self.indent_depth.saturating_sub(1);
        self.push_indent();
        self.out.push('}');
    }

    /// Separator after a comma in output written to `self.out`.
    #[inline]
    pub(super) fn push_separator(&mut self) {
        self.out.push_str(self.tok_separator);
    }

    /// Assignment `=` with optional surrounding spaces.
    #[inline]
    pub(super) fn push_assign(&mut self) {
        self.out.push_str(self.tok_assign);
    }

    /// Colon `:` with optional trailing space (for type annotations).
    #[inline]
    pub(super) fn push_colon(&mut self) {
        self.out.push_str(self.tok_colon);
    }

    /// Return-type arrow `->` with optional surrounding spaces.
    #[inline]
    pub(super) fn push_arrow(&mut self) {
        self.out.push_str(self.tok_arrow);
    }

    /// Push a newline when beautifying, no-op in compact mode.
    #[inline]
    pub(super) fn push_newline(&mut self) {
        self.out.push_str(self.tok_newline);
    }

    /// Return the comma separator string for building expression strings.
    #[inline]
    pub(super) fn comma_sep(&self) -> &'static str {
        self.tok_separator
    }

    /// Return space-around binary operator format for expression strings.
    #[inline]
    pub(super) fn bin_op_sep(&self) -> &'static str {
        self.tok_bin_op_sep
    }

    /// Return the assignment token (` = ` or `=`) for string building.
    #[inline]
    pub(super) fn assign_sep(&self) -> &'static str {
        self.tok_assign
    }

    /// Push `) @binding(` (beautify) or `)@binding(` (compact).
    #[inline]
    pub(super) fn push_binding_sep(&mut self) {
        self.out.push_str(self.tok_binding_sep);
    }

    /// Push `) ` (beautify) or `)` (compact) - attribute closing.
    #[inline]
    pub(super) fn push_attr_end(&mut self) {
        self.out.push_str(self.tok_attr_end);
    }

    /// Push `> ` (beautify) or `>` (compact) - generic close.
    #[inline]
    pub(super) fn push_angle_end(&mut self) {
        self.out.push_str(self.tok_angle_end);
    }

    /// Push ` else` (beautify) or `else` (compact).
    #[inline]
    pub(super) fn push_else(&mut self) {
        self.out.push_str(self.tok_else);
    }

    /// Push `for (` (beautify) or `for(` (compact).
    #[inline]
    pub(super) fn push_for_open(&mut self) {
        self.out.push_str(self.tok_for_open);
    }

    /// Push `; ` (beautify) or `;` (compact) - for-loop clause separator.
    #[inline]
    pub(super) fn push_for_sep(&mut self) {
        self.out.push_str(self.tok_for_sep);
    }
}
