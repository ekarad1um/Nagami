//! Generator state and per-function context: [`Generator`] owns the output
//! buffer, the cached module-wide analyses (liveness, alias plan, per-function
//! ref counts, layouts) and the precomputed format tokens; [`FunctionCtx`] is
//! the per-function bundle threaded through the statement and expression
//! emitters so bindings, deferred-variable flags and inline decisions stay
//! coherent within one function.

use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::config::FloatPrecision;

use super::expr_emit::subsplat_runs;
use super::module_emit::Twins;
use super::syntax::LiteralExtractKey;
use crate::handle_set::{HandleMap, HandleSet};
use crate::ir::visit::{all_functions, visit_expression_children};
use crate::passes::expr_util::{RefCount, is_library_module};

// MARK: Options

/// Caller-facing generator knobs, each resolved from a [`crate::config::Config`]
/// entry by [`crate::run`]; changing a default is a public-surface change.
#[derive(Debug, Clone, PartialEq)]
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
    /// The type spellings a render of this module produced, which the
    /// alias planner prices instead of its IR census: the census counts
    /// constructors the text elides or spells bare (`vec3(`) and preamble-
    /// owned declarations, and misses conversions (`vec3f(v)`), so it mints
    /// aliases that lose and skips ones that pay.  The tail's rename renders
    /// the module once already; its emission carries these.
    pub type_uses: Option<TypeUses>,
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
            type_uses: None,
        }
    }
}

/// The type spellings a render produced, per type handle: how many, and
/// the bytes they take without an alias of their own (a constructor that
/// may drop its suffix - `vec3(`, `array(` - counts as the shorter form,
/// a scalar `var x=0i` tail as its literal), so an alias of `n` bytes
/// saves `bytes - sites * n` at them.  A type spelled inside another's
/// (`array<T,N>`, `ptr<_,T>`) is not a site of `T`: those spellings
/// collapse into one declaration when the outer type takes an alias, so
/// counting them would credit `T` for sites that may not survive; the
/// under-count only forgoes an alias, never mints a losing one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeUses {
    sites: Vec<u32>,
    bytes: Vec<u32>,
}

impl TypeUses {
    pub(super) fn sized(types: usize) -> Self {
        Self {
            sites: vec![0; types],
            bytes: vec![0; types],
        }
    }

    pub(super) fn note(&mut self, ty: naga::Handle<naga::Type>, len: u32) {
        self.sites[ty.index()] += 1;
        self.bytes[ty.index()] += len;
    }

    fn unnote(&mut self, index: usize, len: u32) {
        self.sites[index] -= 1;
        self.bytes[index] -= len;
    }

    /// A function's counts, once its render settled.
    pub(super) fn absorb(&mut self, other: &TypeUses) {
        for (mine, theirs) in self.sites.iter_mut().zip(&other.sites) {
            *mine += theirs;
        }
        for (mine, theirs) in self.bytes.iter_mut().zip(&other.bytes) {
            *mine += theirs;
        }
    }

    /// `(sites, bytes)` of `ty`; none for a type the counted render never
    /// had (the counts are sized by the module they came from).
    fn at(&self, ty: naga::Handle<naga::Type>) -> (usize, usize) {
        let at = |v: &[u32]| v.get(ty.index()).copied().unwrap_or(0) as usize;
        (at(&self.sites), at(&self.bytes))
    }
}

impl GenerateOptions {
    /// The options every emission of a compacted module shares; the
    /// preamble-derived sets are the caller's.  One resolution of the
    /// config, so the pricer's generator renders the text the shipped one
    /// will.
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            beautify: config.beautify,
            indent: config.indent,
            mangle: config.mangle(),
            float_precision: config.float_precision,
            preserve_symbols: config.preserve_symbols.iter().cloned().collect(),
            type_alias: true,
            ..Default::default()
        }
    }
}

/// What the type spellings decide in [`Generator::new`]: the alias
/// declarations, the type names ranked by use and the member names minted
/// after them.  Nothing else there reads the spellings, so two generators
/// of one module under one set of options that agree here render the same
/// text (`super::generate_reusing`).
#[derive(Debug, Default)]
pub(crate) struct TypePlan {
    type_names: HandleMap<naga::Type, String>,
    member_names: HandleMap<naga::Type, Vec<String>>,
    type_alias_decls: Vec<(String, String)>,
}

impl Generator<'_> {
    pub(super) fn take_type_plan(&mut self) -> TypePlan {
        TypePlan {
            type_names: std::mem::take(&mut self.type_names),
            member_names: std::mem::take(&mut self.member_names),
            type_alias_decls: std::mem::take(&mut self.type_alias_decls),
        }
    }

    /// Whether this generator decided `plan`.
    pub(super) fn plans_types_as(&self, plan: &TypePlan) -> bool {
        fn same<T, V: PartialEq>(a: &HandleMap<T, V>, b: &HandleMap<T, V>) -> bool {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|((ha, va), (hb, vb))| ha == hb && va == vb)
        }
        same(&self.type_names, &plan.type_names)
            && same(&self.member_names, &plan.member_names)
            && self.type_alias_decls == plan.type_alias_decls
    }
}

/// What a render computes from the arenas alone: per body, its local
/// bitmaps, the type groups it may spell and the analyses its context is
/// built from ([`FunctionAnalyses`], kept once the body rendered), and the
/// literal census.  The rename sets names and nothing else, so the render
/// that ships takes the tail's rename render's instead of computing them
/// again (`super::generate_after`).
#[derive(Default)]
pub(crate) struct Analyses {
    /// Per-function `(deferrable, dead)` local bitmaps, indexed like
    /// `ref_count_cache`; computed once so the live-type census, alias cost
    /// model, literal extraction and emission cannot disagree.
    pub(super) defer: Vec<(Vec<bool>, Vec<bool>)>,
    /// Per function, the type groups its body may spell
    /// (`spelled_type_groups`).
    spelled_groups: Vec<HandleSet<naga::Type>>,
    /// How often the text spells each literal, and whether ever bare
    /// (`Generator::literal_census`); `None` until counted.
    pub(super) literal_counts: Option<FxHashMap<LiteralExtractKey, (usize, bool)>>,
    /// Indexed like `ref_count_cache`; `None` until the body rendered.
    functions: Vec<Option<FunctionAnalyses>>,
}

impl Analyses {
    /// What a render computes before the census, for one computing its
    /// own.
    fn fresh(
        module: &naga::Module,
        info: &naga::valid::ModuleInfo,
        canonical: &HandleMap<naga::Type, naga::Handle<naga::Type>>,
        inner_to_first: &FxHashMap<&naga::TypeInner, naga::Handle<naga::Type>>,
    ) -> Self {
        let bodies = module.functions.len() + module.entry_points.len();
        Self {
            defer: all_functions(module)
                .map(super::module_emit::find_deferrable_vars)
                .collect(),
            spelled_groups: spelled_type_groups(module, info, canonical, inner_to_first),
            literal_counts: None,
            functions: (0..bodies).map(|_| None).collect(),
        }
    }

    #[cfg(test)]
    pub(crate) fn bodies_built(&self) -> usize {
        self.functions.iter().flatten().count()
    }
}

/// Handles and counts, nothing a report reads: the size stands in.
impl std::fmt::Debug for Analyses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analyses")
            .field("functions", &self.functions.len())
            .finish()
    }
}

impl Generator<'_> {
    pub(super) fn take_analyses(&mut self) -> Analyses {
        std::mem::take(&mut self.analyses)
    }

    /// The analyses `body`'s context is built from, if a render of these
    /// arenas left them; taken, so each body is built once.
    pub(super) fn take_function_analyses(&mut self, body: usize) -> Option<FunctionAnalyses> {
        self.analyses.functions.get_mut(body).and_then(Option::take)
    }

    /// Keeps what `ctx` was built from, for a later render of these
    /// arenas.
    pub(super) fn keep_function_analyses(&mut self, body: usize, ctx: FunctionCtx<'_, '_>) {
        if let Some(slot) = self.analyses.functions.get_mut(body) {
            *slot = Some(FunctionAnalyses::of(ctx));
        }
    }
}

// MARK: Cached analyses

/// Per-function expression analysis, computed once and in lock-step by
/// `compute_expression_ref_counts`: the census the literal extraction and
/// [`FunctionAnalyses`] are built from, taken by the latter.
#[derive(Default)]
pub(super) struct FunctionExprInfo {
    pub(super) ref_counts: Vec<RefCount>,
    pub(super) live: Vec<bool>,
    /// How many live consumers would wrap the expression's own text in
    /// parentheses (`paren_uses_in`).
    pub(super) paren_uses: Vec<u16>,
}

/// The analyses one body's context is built from before any text
/// (`Generator::function_analyses`): the [`FunctionCtx`] fields of these
/// names, as built.  A function of the arenas, the types and the callees'
/// effects, so a render of the same arenas under other names is built
/// from the same.
pub(super) struct FunctionAnalyses {
    pub(super) ref_counts: Vec<RefCount>,
    pub(super) paren_uses: Vec<u16>,
    pub(super) twins: Twins,
    pub(super) must_bind: HandleSet<naga::Expression>,
    pub(super) for_loop_vars: Vec<Option<naga::Handle<naga::Expression>>>,
    pub(super) inlineable_calls: HandleSet<naga::Expression>,
}

impl FunctionAnalyses {
    /// `ctx` as built, before any text; the render hands it back as passed.
    fn of(ctx: FunctionCtx<'_, '_>) -> Self {
        Self {
            ref_counts: ctx.ref_counts,
            paren_uses: ctx.paren_uses,
            twins: ctx.twins,
            must_bind: ctx.must_bind,
            for_loop_vars: ctx.for_loop_vars,
            inlineable_calls: ctx.inlineable_calls,
        }
    }
}

// MARK: Generator state

/// Emission state for one module; created per [`super::generate`] call and
/// never reused.
pub(super) struct Generator<'a> {
    pub(super) module: &'a naga::Module,
    pub(super) info: &'a naga::valid::ModuleInfo,
    pub(super) options: GenerateOptions,
    pub(super) out: String,
    pub(super) indent_depth: u32,
    pub(super) type_names: HandleMap<naga::Type, String>,
    /// Per struct type, a name per member; absent for a struct spelled as
    /// the source has it.
    pub(super) member_names: HandleMap<naga::Type, Vec<String>>,
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
    /// What this render computed from the arenas, or took over from a
    /// render of the same arenas.
    pub(super) analyses: Analyses,
    /// Per-`module.functions` effect summary: keeps impure single-use calls
    /// bound rather than inlined past a read of what they write, and tells
    /// the forced-binding analysis which loads a call stales.
    pub(super) fn_effects: Vec<crate::analysis::FnEffects>,
    /// How often the text spells each renameable name, gathered as the
    /// bodies render (`FunctionCtx::render_counts`), for the rename that
    /// ranks by it.
    pub(super) name_weights: crate::passes::rename::Weights,
    /// The type spellings this render produced (`FunctionCtx::type_uses`
    /// absorbed as each body settles, the declarations counted here), for
    /// the alias plan of the render that ships.
    pub(super) type_uses: TypeUses,
    /// Bytes of each type's spelling without an alias of its own (nested
    /// aliases as planned), what a site would cost were its alias dropped.
    pub(super) type_spelled_len: Vec<u32>,
    /// The arena-first handle of each `TypeInner`, for a spelling that has
    /// the inner alone (a conversion's target, a splat's vector).
    pub(super) inner_to_first: FxHashMap<&'a naga::TypeInner, naga::Handle<naga::Type>>,
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

/// One `let`-or-inline decision the byte rule made: the value, its text as
/// priced, the name length priced against, and the verdict.
#[derive(Clone, Copy)]
pub(super) struct ByteDecision {
    pub(super) handle: naga::Handle<naga::Expression>,
    pub(super) len: usize,
    pub(super) name: usize,
    pub(super) bound: bool,
}

/// Per-function emission context, threaded through every statement and
/// expression emitter.  `Clone` so a pricer can render from a copy and keep
/// its own state pristine.
#[derive(Clone)]
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
    /// A root whose own text must spell its concrete type: the initializer
    /// of a declaration that prints no `: T` (an elided `vec2(42,43)` would
    /// leave the constant abstract, a different type that naga drops from
    /// the arena entirely), or the all-literal `Compose` operand of a
    /// conversion that would convert its lanes differently as abstract
    /// values (`compose_lane_keeps_suffix`).
    pub(super) pinned_root: Option<naga::Handle<naga::Expression>>,
    pub(super) argument_names: Vec<String>,
    pub(super) local_names: HandleMap<naga::LocalVariable, String>,
    pub(super) expr_names: HandleMap<naga::Expression, String>,
    /// Each later spelling of a value in scope, to its first
    /// (`structural_twins`): it renders as the first's name once that has
    /// one, and `ref_counts` / `paren_uses` of the first hold its uses.
    pub(super) twins: Twins,
    pub(super) ref_counts: Vec<RefCount>,
    /// Of `ref_counts`, the uses that parenthesise an inlined text
    /// (`FunctionExprInfo::paren_uses`).
    pub(super) paren_uses: Vec<u16>,
    pub(super) deferred_vars: Vec<bool>,
    pub(super) dead_vars: Vec<bool>,
    /// Matrix / array locals passed by address to a call while their type
    /// renders under a minted `alias`: their `var` declares `: ALIAS`.  naga
    /// types an inferred matrix / array as the unnamed twin of the alias and
    /// compares matrix / array pointer bases by HANDLE (scalar / vector ones
    /// canonicalise, structs are always named), so `f(&b)` against
    /// `ptr<function, ALIAS>` fails the re-parse.
    pub(super) typed_pointer_arg_locals: Vec<bool>,
    /// Per local, the guard of the `Loop` whose `for(var x=init;...)` header
    /// declares it (`find_for_loop_vars`); cleared once rendered.
    pub(super) for_loop_vars: Vec<Option<naga::Handle<naga::Expression>>>,
    pub(super) expr_name_counter: usize,
    /// The name the next `let` takes and the counter past it, drawn once
    /// per state of the names in use: a decision prices it before the
    /// binding claims it, and a declined one prices the same name again.
    pub(super) drawn_expr_name: Option<(String, usize)>,
    /// Module-scope names shared across functions; not cloned per call.
    pub(super) module_names: &'m std::collections::HashSet<String>,
    /// Names claimed in this function: arguments, locals, expression bindings.
    pub(super) local_used_names: std::collections::HashSet<String>,
    /// Call results inlinable at their use site: `ref_count == 1` and no
    /// side-effecting statement between the `Call` and the use.
    pub(super) inlineable_calls: HandleSet<naga::Expression>,
    /// Expressions `let`-bound whatever the byte cost (`compute_must_bind`).
    pub(super) must_bind: HandleSet<naga::Expression>,
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
    /// How many times each expression has rendered at a use so far (its
    /// text or its name) and how many of those a consumer would wrap in
    /// parentheses: exactly what the text holds, since a rendering the text
    /// drops (a pricing render of a value that inlines, the longer of two
    /// constructor spellings) is undone through `count_journal` and a
    /// `let`'s own rendering is never counted.
    pub(super) render_counts: Vec<i32>,
    pub(super) paren_counts: Vec<i32>,
    /// Every count taken, in order, so a range of them can be undone;
    /// a type spelling (`note_type`) is a negative entry.
    pub(super) count_journal: Vec<i32>,
    /// The type spellings this render produced, journaled like the counts.
    pub(super) type_uses: TypeUses,
    /// The counts a previous render of this function measured, which the
    /// byte rule prices by instead of the census; `None` on the first.
    pub(super) measured: Option<(Vec<i32>, Vec<i32>)>,
    /// Every byte decision this render made, for the check against the
    /// counts it produced.
    pub(super) decisions: Vec<ByteDecision>,
    /// Operands the `const_hazard` guard named, in emission order: a `let` it
    /// emitted, or an `override` read routed through an open binding's name.
    /// Pre-emitted, so usable from any block, and each name must leave
    /// `expr_names` when its block closes.
    pub(super) const_hazard_bindings: Vec<naga::Handle<naga::Expression>>,
    /// The name the declaration prints, and the diagnostics' name.
    pub(super) display_name: String,
}

/// Where an expression's resolved type comes from: naga keeps function
/// expressions in `FunctionInfo` and module-scope ones in `ModuleInfo`.
#[derive(Clone, Copy)]
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

    /// The next free `let` name and the counter past it; nothing claimed.
    /// Valid until [`Self::next_expr_name`] claims it: the names in use
    /// change nowhere else once the context is built.
    fn draw_expr_name(&mut self) -> &(String, usize) {
        if self.drawn_expr_name.is_none() {
            let mut counter = self.expr_name_counter;
            let drawn = loop {
                let name = crate::name_gen::next_name(&mut counter);
                if !self.module_names.contains(&name) && !self.local_used_names.contains(&name) {
                    break (name, counter);
                }
            };
            self.drawn_expr_name = Some(drawn);
        }
        self.drawn_expr_name.as_ref().expect("drawn above")
    }

    pub(super) fn next_expr_name(&mut self) -> String {
        self.draw_expr_name();
        let (name, counter) = self.drawn_expr_name.take().expect("drawn above");
        self.expr_name_counter = counter;
        self.local_used_names.insert(name.clone());
        name
    }

    /// Length of the name [`Self::next_expr_name`] would return, for
    /// pricing a `let` before deciding on it.
    pub(super) fn peek_expr_name_len(&mut self) -> usize {
        self.draw_expr_name().0.len()
    }

    /// The twins of `h` take the name it was just given.
    pub(super) fn name_twins(&mut self, h: naga::Handle<naga::Expression>) {
        if self.twins.is_empty() {
            return;
        }
        let Some(name) = self.expr_names.get(h).cloned() else {
            return;
        };
        let twins: Vec<_> = self
            .twins
            .iter()
            .filter(|&(_, first)| first == h)
            .map(|(twin, _)| twin)
            .collect();
        for twin in twins {
            self.expr_names.insert(twin, name.clone());
        }
    }

    /// The counts with each twin's folded into its first spelling's: what
    /// the byte rule of that value is judged by.
    pub(super) fn counts_by_value(&self) -> (Vec<i32>, Vec<i32>) {
        let mut renders = self.render_counts.clone();
        let mut parens = self.paren_counts.clone();
        for (twin, first) in self.twins.iter() {
            renders[first.index()] += renders[twin.index()];
            parens[first.index()] += parens[twin.index()];
        }
        (renders, parens)
    }

    /// `h` rendered once more, its text or its name.
    pub(super) fn rendered(&mut self, h: naga::Handle<naga::Expression>) {
        self.render_counts[h.index()] += 1;
        self.count_journal.push(Self::journal_entry(h, false));
    }

    /// A consumer rendering `h` would wrap its text in parentheses (it may
    /// be rendering the name instead).
    pub(super) fn wrapped(&mut self, h: naga::Handle<naga::Expression>) {
        self.paren_counts[h.index()] += 1;
        self.count_journal.push(Self::journal_entry(h, true));
    }

    /// The text spelled `ty`, `len` bytes without an alias of its own.
    pub(super) fn note_type(&mut self, ty: naga::Handle<naga::Type>, len: usize) {
        let len = u32::try_from(len).map_or(TYPE_LEN_CAP, |l| l.min(TYPE_LEN_CAP));
        self.type_uses.note(ty, len);
        let entry =
            i32::try_from(ty.index() as u32 * (TYPE_LEN_CAP + 1) + len).expect("a type index");
        self.count_journal.push(-1 - entry);
    }

    /// The journal's spelling of one count: the handle and which count,
    /// never zero (a zero is an entry already undone) and never negative
    /// (a type spelling).
    fn journal_entry(h: naga::Handle<naga::Expression>, paren: bool) -> i32 {
        let idx = i32::try_from(h.index()).expect("an arena index");
        idx * 2 + i32::from(paren) + 1
    }

    /// Where the journal stands, for a rendering that may be dropped.
    pub(super) fn mark(&self) -> usize {
        self.count_journal.len()
    }

    /// Undo the counts taken since `mark`: the rendering they came from is
    /// not in the text.
    pub(super) fn discard_since(&mut self, mark: usize) {
        self.discard_range(mark, self.count_journal.len());
        self.count_journal.truncate(mark);
    }

    /// [`Self::discard_since`] for the counts in `start..end`, later ones
    /// kept (an earlier spelling lost to a later one).
    pub(super) fn discard_range(&mut self, start: usize, end: usize) {
        for i in start..end {
            let entry = std::mem::replace(&mut self.count_journal[i], 0);
            if entry == 0 {
                continue;
            }
            if entry < 0 {
                let packed = (-1 - entry) as u32;
                let (idx, len) = (packed / (TYPE_LEN_CAP + 1), packed % (TYPE_LEN_CAP + 1));
                self.type_uses.unnote(idx as usize, len);
                continue;
            }
            let idx = ((entry - 1) / 2) as usize;
            if (entry - 1) % 2 == 0 {
                self.render_counts[idx] -= 1;
            } else {
                self.paren_counts[idx] -= 1;
            }
        }
    }

    /// [`Self::wrapped`] for the two operands of a binary rendering.
    pub(super) fn wrapped_operands(
        &mut self,
        left: naga::Handle<naga::Expression>,
        wrap_l: bool,
        right: naga::Handle<naga::Expression>,
        wrap_r: bool,
    ) {
        if wrap_l {
            self.wrapped(left);
        }
        if wrap_r {
            self.wrapped(right);
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
    /// Context for module-scope expressions: no locals, and
    /// [`Generator::expr_to_const`]'s constants pre-bound to their names.
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
            twins: Twins::default(),
            ref_counts: vec![0; n],
            paren_uses: Vec::new(),
            deferred_vars: Vec::new(),
            typed_pointer_arg_locals: Vec::new(),
            dead_vars: Vec::new(),
            for_loop_vars: Vec::new(),
            expr_name_counter: 0,
            drawn_expr_name: None,
            module_names: &NO_NAMES,
            local_used_names: HashSet::new(),
            inlineable_calls: HandleSet::default(),
            must_bind: HandleSet::default(),
            render_depth_memo: vec![0; n],
            stashed_call_depth: HandleMap::default(),
            render_counts: vec![0; n],
            paren_counts: vec![0; n],
            count_journal: Vec::new(),
            type_uses: TypeUses::sized(self.module.types.len()),
            measured: None,
            decisions: Vec::new(),
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
    // constant that `emit_constant_decls` skips is still counted.
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
    // `Compose` / `ZeroValue` entries are counted whether or not their
    // expression prints, since no per-expression liveness is known here and
    // an under-count forgoes a paying alias.  The alias planner is greedy,
    // so any count perturbation can flip a marginal choice; all outputs stay
    // valid.
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

struct StructNames {
    type_names: HandleMap<naga::Type, String>,
    member_names: HandleMap<naga::Type, Vec<String>>,
    /// Struct names assigned under mangling, preserved ones included and
    /// predeclared ones excluded; [`crate::name_gen::module_scope_names`]
    /// lists no type, so the anonymous override's minted name dodges these
    /// explicitly.
    minted: HashSet<String>,
    /// Names the source or the language fixes (preserved, predeclared), which
    /// [`reassign_type_names_by_use`] must not hand to anything else.
    fixed: HashSet<String>,
}

/// Name every struct type; without mangling the source names are kept
/// verbatim.  A minted struct name lives at module scope and dodges what an
/// alias dodges ([`plan_type_aliases`]) plus the other minted type names.
/// Members are named by [`mint_member_names`] once every type name is
/// final; only the predeclared structs get theirs here, because those are
/// canonical.
fn mint_struct_names(
    module: &naga::Module,
    options: &GenerateOptions,
    canonical: &HandleMap<naga::Type, naga::Handle<naga::Type>>,
    spelling: &SpellingScopes<'_>,
) -> StructNames {
    let mangle = options.mangle;
    let preserve = &options.preserve_symbols;
    let mut type_names = HandleMap::default();
    let mut member_names = HandleMap::default();
    let mut minted: HashSet<String> = HashSet::new();
    let mut fixed: HashSet<String> = HashSet::new();

    // naga predeclared / special struct types are never renamed: their members
    // are accessed through canonical names (`.old_value`, `.fract`, `.kind`,
    // ...) and no declaration is emitted for them, so a mangled accessor would
    // be invalid WGSL.
    let predeclared_type_handles = super::module_emit::special_struct_handles(module);

    let mut module_used = crate::name_gen::NameScope::default();
    if mangle {
        module_used.extend(crate::name_gen::module_scope_names(module).map(str::to_owned));
        module_used.extend(preserve.iter().cloned());
        module_used.extend(options.preserve_members.iter().cloned());
    }

    for (h, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            continue;
        };
        if !mangle {
            type_names.insert(
                h,
                ty.name.clone().unwrap_or_else(|| format!("T{}", h.index())),
            );
            continue;
        }
        // Name-based fallback for IRs where `special_types.ray_desc` is not
        // populated.
        let is_ray_descriptor = ty.name.as_deref() == Some("RayDesc");
        if predeclared_type_handles.contains(h) || is_ray_descriptor {
            let name = ty.name.clone().unwrap_or_else(|| format!("T{}", h.index()));
            fixed.insert(name.clone());
            type_names.insert(h, name);
            member_names.insert(
                h,
                members
                    .iter()
                    .enumerate()
                    .map(|(idx, member)| member.name.clone().unwrap_or_else(|| format!("m{}", idx)))
                    .collect(),
            );
            continue;
        }

        let name = match ty.name.as_deref().filter(|n| preserve.contains(*n)) {
            Some(kept) => {
                fixed.insert(kept.to_string());
                kept.to_string()
            }
            None => spelling.shortest_free_name(&module_used, canonical[&h]),
        };
        minted.insert(name.clone());
        module_used.insert(name.clone());
        type_names.insert(h, name);
    }

    StructNames {
        type_names,
        member_names,
        minted,
        fixed,
    }
}

/// Name the members of every mangled struct.  A member lives in its
/// struct's own namespace, so it dodges only that struct's kept members
/// and every struct counts from the first letter - except for the names of
/// the types the struct body spells.  WGSL keeps members and types apart;
/// the C++ that naga's MSL writer emits does not, and there a member named
/// like a type hides that type for the members declared after it, so the
/// shader compiles under Dawn (which renames every symbol first) and fails
/// under wgpu on Metal.  Direct member types are the whole
/// list: naga spells an array through a module-scope wrapper, so an element
/// type's name never appears in the containing body, and pointers cannot be
/// members.  Runs after the type names are final, and the predeclared
/// structs keep the canonical names [`mint_struct_names`] gave them.
fn mint_member_names(
    module: &naga::Module,
    options: &GenerateOptions,
    type_names: &HandleMap<naga::Type, String>,
    member_names: &mut HandleMap<naga::Type, Vec<String>>,
) {
    if !options.mangle {
        return;
    }
    let predeclared = super::module_emit::special_struct_handles(module);
    for (h, ty) in module.types.iter() {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            continue;
        };
        if predeclared.contains(h) || ty.name.as_deref() == Some("RayDesc") {
            continue;
        }
        let kept: HashSet<String> = members
            .iter()
            .filter_map(|m| m.name.as_deref())
            .filter(|n| {
                options.preserve_symbols.contains(*n) || options.preserve_members.contains(*n)
            })
            .map(str::to_owned)
            .collect();
        let mut forbidden = kept.clone();
        forbidden.extend(members.iter().filter_map(|m| type_names.get(m.ty).cloned()));
        let mut counter = 0usize;
        let names = members
            .iter()
            .map(|member| match member.name.as_deref() {
                Some(name) if kept.contains(name) => name.to_string(),
                _ => crate::name_gen::next_name_unique(&mut counter, &forbidden),
            })
            .collect();
        member_names.insert(h, names);
    }
}

/// The longest type spelling a journal entry records, beyond which the
/// bytes are capped (an alias for such a type is under-credited, never
/// over): the entry packs the type index above it.
const TYPE_LEN_CAP: u32 = 1023;

/// The type-group and shadowing state [`plan_type_aliases`] and
/// [`rank_type_names_by_use`] share.
struct AliasScope<'s> {
    /// Arena-first handle of each type's `TypeInner` group.
    canonical: &'s HandleMap<naga::Type, naga::Handle<naga::Type>>,
    /// Summed reference count per group head.
    group_ref_count: &'s HandleMap<naga::Type, usize>,
    /// Summed bytes the sites spell without an alias, per group head, from
    /// a render (`TypeUses`); `None` prices every site at the full spelling.
    group_bytes: Option<&'s HandleMap<naga::Type, usize>>,
    /// Per function, the groups it spells and the locals a minted name
    /// must then dodge.
    spelling: &'s SpellingScopes<'s>,
    /// Predeclared aliases a module-scope name already shadows.
    shadowed_type_aliases: &'s super::syntax::ShadowedAliases,
}

/// Mint an alias for each non-struct type whose uses pay for the declaration,
/// naming it in `type_names`; the declarations and the handles they name,
/// parallel.  An alias dodges every module-scope name and the locals of
/// the functions that spell its type ([`SpellingScopes`]).
#[allow(clippy::too_many_arguments)]
fn plan_type_aliases(
    module: &naga::Module,
    options: &GenerateOptions,
    type_names: &mut HandleMap<naga::Type, String>,
    scope: &AliasScope<'_>,
    constant_names: &[String],
    override_names: &[String],
    global_names: &[String],
    function_names: &[String],
) -> (Vec<(String, String)>, Vec<naga::Handle<naga::Type>>) {
    let mut module_names: crate::name_gen::NameScope = emitted_module_names(
        module,
        options,
        type_names,
        constant_names,
        override_names,
        global_names,
        function_names,
    )
    .map(str::to_owned)
    .collect();
    module_names.extend(options.preserve_members.iter().cloned());
    let mut minted_aliases = module_names.clone();
    let mut decls: Vec<(String, String)> = Vec::new();
    let mut decl_handles: Vec<naga::Handle<naga::Type>> = Vec::new();
    let fixed_overhead = super::syntax::decl_boilerplate(options.beautify);

    for (h, ty) in module.types.iter() {
        // Structs are already named by `mint_struct_names`.
        if type_names.contains_key(h) {
            continue;
        }

        let group_head = scope.canonical[&h];
        if h != group_head
            && let Some(existing) = type_names.get(group_head).cloned()
        {
            type_names.insert(h, existing);
            continue;
        }

        let count = scope.group_ref_count.get(group_head).copied().unwrap_or(0);
        if count == 0 {
            continue;
        }
        // Rendered with the aliases minted so far, so aliases nest.
        let Ok(type_str) = super::syntax::type_inner_name(
            &ty.inner,
            module,
            type_names,
            override_names,
            scope.shadowed_type_aliases,
        ) else {
            continue;
        };
        // Priced at the shortest name the ranking below can hand it (the
        // other aliases' letters are reassigned there by use); minted
        // provisionally past them.
        let name_len = scope
            .spelling
            .shortest_free_name(&module_names, group_head)
            .len();
        let alias_name = scope
            .spelling
            .shortest_free_name(&minted_aliases, group_head);

        let decl_cost = fixed_overhead + name_len + type_str.len();
        let spelled = scope.group_bytes.map_or(count * type_str.len(), |bytes| {
            bytes.get(group_head).copied().unwrap_or(0)
        });
        let total_savings = spelled.saturating_sub(count * name_len);
        if total_savings > decl_cost {
            minted_aliases.insert(alias_name.clone());
            decls.push((alias_name.clone(), type_str));
            decl_handles.push(h);
            type_names.insert(h, alias_name);
        }
    }

    (decls, decl_handles)
}

/// Re-letter the minted type names by use count
/// ([`reassign_type_names_by_use`]) and re-render the alias declarations
/// against the result, last, so a nested alias spells the final name.
#[allow(clippy::too_many_arguments)]
fn rank_type_names_by_use(
    module: &naga::Module,
    type_names: &mut HandleMap<naga::Type, String>,
    type_alias_decls: &mut [(String, String)],
    decl_handles: &[naga::Handle<naga::Type>],
    scope: &AliasScope<'_>,
    ref_counts: &HandleMap<naga::Type, usize>,
    fixed_type_names: &HashSet<String>,
    // Spelled inside an `array<_, N>` alias, so the re-render needs them.
    override_names: &[String],
    options: &GenerateOptions,
    // Constant, override, global and function names.
    taken: [&[String]; 4],
) {
    // A kept member name is dodged too: the type letter would otherwise be
    // hidden inside the very struct that spells it (see `mint_member_names`).
    let mut assigned: crate::name_gen::NameScope = taken
        .iter()
        .flat_map(|names| names.iter())
        .chain(fixed_type_names)
        .chain(&options.preserve_symbols)
        .chain(&options.preserve_members)
        .cloned()
        .collect();
    assigned.extend(module.entry_points.iter().map(|entry| entry.name.clone()));

    reassign_type_names_by_use(
        type_names,
        assigned,
        scope.canonical,
        |h| {
            if matches!(module.types[h].inner, naga::TypeInner::Struct { .. }) {
                (ref_counts.get(h).copied().unwrap_or(0), true)
            } else {
                (
                    scope
                        .group_ref_count
                        .get(scope.canonical[&h])
                        .copied()
                        .unwrap_or(0),
                    false,
                )
            }
        },
        scope.spelling,
    );

    for (decl, &h) in type_alias_decls.iter_mut().zip(decl_handles) {
        if let (Some(name), Ok(type_str)) = (
            type_names.get(h).cloned(),
            super::syntax::type_inner_name(
                &module.types[h].inner,
                module,
                type_names,
                override_names,
                scope.shadowed_type_aliases,
            ),
        ) {
            *decl = (name, type_str);
        }
    }
}

/// Letters by use count.  The arena-order minting hands the first free
/// letter to whichever type comes first, so a struct spelled once can hold
/// the letter an alias spelled a hundred times wanted, and a re-minify,
/// registering the types in another order, derives other letters (text
/// drift on the second pass).  Every minted name in `type_names` - one not
/// in `assigned`, which holds the module-scope names and the fixed type
/// names - is reassigned in descending `count`, each dodging `assigned`,
/// the names given so far and the locals of the functions that spell its
/// group.  Ties go to aliases before structs, then handle order: a
/// re-minify registers types in the order the output declares them,
/// aliases first, so that is the one order both passes agree on.  Only
/// letters move; no minting decision does.  Plain loops throughout: a sort
/// or a map instantiation here is kilobytes of binary for a dozen names.
fn reassign_type_names_by_use(
    type_names: &mut HandleMap<naga::Type, String>,
    mut assigned: crate::name_gen::NameScope,
    canonical: &HandleMap<naga::Type, naga::Handle<naga::Type>>,
    count: impl Fn(naga::Handle<naga::Type>) -> (usize, bool),
    spelling: &SpellingScopes<'_>,
) {
    // (old name, (count, is struct), group head).
    let mut entities: Vec<(String, (usize, bool), naga::Handle<naga::Type>)> = Vec::new();
    for (&h, name) in type_names.iter() {
        if assigned.contains(name) || entities.iter().any(|(n, _, _)| n == name) {
            continue;
        }
        entities.push((name.clone(), count(h), canonical[&h]));
    }
    let rank = |i: usize| {
        let (_, (n, is_struct), head) = &entities[i];
        (usize::MAX - *n, *is_struct, head.index())
    };
    let mut renamed: Vec<(String, String)> = Vec::with_capacity(entities.len());
    let mut done = vec![false; entities.len()];
    for _ in 0..entities.len() {
        let mut pick = usize::MAX;
        for (i, &taken) in done.iter().enumerate() {
            if !taken && (pick == usize::MAX || rank(i) < rank(pick)) {
                pick = i;
            }
        }
        done[pick] = true;
        let (old, _, head) = &entities[pick];
        let new = spelling.shortest_free_name(&assigned, *head);
        assigned.insert(new.clone());
        renamed.push((old.clone(), new));
    }
    let handles: Vec<_> = type_names.keys().copied().collect();
    for h in handles {
        let new = type_names
            .get(h)
            .and_then(|old| renamed.iter().find(|(o, _)| o == old))
            .map(|(_, new)| new.clone());
        if let Some(new) = new {
            type_names.insert(h, new);
        }
    }
}

/// `UniqueArena<Type>` deduplicates by the full `Type` (name included), so
/// one `TypeInner` can sit under several handles when the source mixes bare
/// types with named aliases.  `canonical[h]` is the arena-first handle of
/// h's group, `inner_to_first` the group head by inner.
#[allow(clippy::type_complexity)]
fn type_groups(
    module: &naga::Module,
) -> (
    HandleMap<naga::Type, naga::Handle<naga::Type>>,
    FxHashMap<&naga::TypeInner, naga::Handle<naga::Type>>,
) {
    let mut canonical: HandleMap<naga::Type, naga::Handle<naga::Type>> = Default::default();
    let mut inner_to_first: FxHashMap<&naga::TypeInner, naga::Handle<naga::Type>> =
        Default::default();
    for (h, ty) in module.types.iter() {
        let first = *inner_to_first.entry(&ty.inner).or_insert(h);
        canonical.insert(h, first);
    }
    (canonical, inner_to_first)
}

/// Per function (functions, then entry points), the type groups its body
/// may spell: argument, result and local types, the resolved type of every
/// expression (constructors, splats, casts, zero values), and the bases those
/// nest (`array<T,N>`, `ptr<_,T>` print `T` inline).  Over-approximate on
/// purpose: a group listed here that never prints only widens a minted type
/// name's dodge set, a missed one would let a local shadow the name.
fn spelled_type_groups(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    canonical: &HandleMap<naga::Type, naga::Handle<naga::Type>>,
    inner_to_first: &FxHashMap<&naga::TypeInner, naga::Handle<naga::Type>>,
) -> Vec<HandleSet<naga::Type>> {
    let infos = module
        .functions
        .iter()
        .map(|(h, _)| &info[h])
        .chain((0..module.entry_points.len()).map(|i| info.get_entry_point(i)));
    all_functions(module)
        .zip(infos)
        .map(|(func, finfo)| {
            let mut groups: HandleSet<naga::Type> = Default::default();
            let mut stack: Vec<naga::Handle<naga::Type>> = Vec::new();
            let mut add = |h: naga::Handle<naga::Type>, groups: &mut HandleSet<naga::Type>| {
                stack.push(h);
                while let Some(t) = stack.pop() {
                    if !groups.insert(canonical[&t]) {
                        continue;
                    }
                    match &module.types[t].inner {
                        naga::TypeInner::Array { base, .. }
                        | naga::TypeInner::BindingArray { base, .. }
                        | naga::TypeInner::Pointer { base, .. } => stack.push(*base),
                        _ => {}
                    }
                }
            };
            for arg in &func.arguments {
                add(arg.ty, &mut groups);
            }
            if let Some(result) = &func.result {
                add(result.ty, &mut groups);
            }
            for (_, local) in func.local_variables.iter() {
                add(local.ty, &mut groups);
            }
            for (h, expr) in func.expressions.iter() {
                match &finfo[h].ty {
                    naga::proc::TypeResolution::Handle(t) => add(*t, &mut groups),
                    naga::proc::TypeResolution::Value(inner) => {
                        if let Some(&t) = inner_to_first.get(inner) {
                            add(t, &mut groups);
                        }
                        if let naga::TypeInner::Pointer { base, .. } = inner {
                            add(*base, &mut groups);
                        }
                    }
                }
                // A vector `Compose` with a run of equal lanes may print it
                // as a SUB-vector splat (`vec4f(vec3f(2),x)`), a type no
                // expression of the function resolves to.  Only such a
                // constructor counts: listing the sub-vectors of every
                // compose widened the dodge set of their aliases module-wide
                // (a two-letter alias name).
                if let naga::Expression::Compose { ty, components } = expr
                    && let naga::TypeInner::Vector { size, scalar } = module.types[*ty].inner
                    && let Some(runs) = subsplat_runs(
                        components,
                        &func.expressions,
                        size as usize,
                        |c| {
                            matches!(
                                finfo[c].ty.inner_with(&module.types),
                                naga::TypeInner::Scalar(_)
                            )
                        },
                        &|_| false,
                    )
                {
                    for (_, k) in runs {
                        let sub = match k {
                            2 => naga::VectorSize::Bi,
                            3 => naga::VectorSize::Tri,
                            _ => continue,
                        };
                        if let Some(&t) =
                            inner_to_first.get(&naga::TypeInner::Vector { size: sub, scalar })
                        {
                            add(t, &mut groups);
                        }
                    }
                }
            }
            groups
        })
        .collect()
}

/// Per function (functions, then entry points), the type groups its body
/// may spell ([`spelled_type_groups`]) and its argument and local names: a
/// minted type name must dodge the locals of every function that spells it
/// (a same-named local there would shadow it) and only those - reserving
/// every local module-wide would push a frequent type's alias past the
/// single characters, one byte at every use (the alias cliff), and make a
/// re-minify derive them against a different taken set.
struct SpellingScopes<'m> {
    groups: &'m [HandleSet<naga::Type>],
    locals: Vec<crate::name_gen::LocalNames<'m>>,
}

impl<'m> SpellingScopes<'m> {
    fn new(module: &'m naga::Module, groups: &'m [HandleSet<naga::Type>]) -> Self {
        Self {
            groups,
            locals: all_functions(module)
                .map(crate::name_gen::LocalNames::of)
                .collect(),
        }
    }

    /// The shortest name free at module scope and in every function that
    /// spells the group `head` leads.
    fn shortest_free_name(
        &self,
        scope: &crate::name_gen::NameScope,
        head: naga::Handle<naga::Type>,
    ) -> String {
        crate::name_gen::shortest_free_name(scope, &self.locals, &|i| self.groups[i].contains(head))
    }
}

// MARK: Liveness analyses

/// Constants transitively reachable from function / entry-point expressions,
/// global-variable and override initialisers, and preserved names; a library
/// module ([`is_library_module`]) keeps every constant.
fn compute_live_constants(
    module: &naga::Module,
    preserve_names: &HashSet<String>,
) -> HandleSet<naga::Constant> {
    let mut live: HandleSet<naga::Constant> = Default::default();

    if is_library_module(module) {
        return module.constants.iter().map(|(h, _)| h).collect();
    }

    if !preserve_names.is_empty() {
        for (h, c) in module.constants.iter() {
            if let Some(name) = c.name.as_deref()
                && preserve_names.contains(name)
            {
                mark_const_live(h, module, &mut live);
            }
        }
    }

    for func in all_functions(module) {
        for (_, expr) in func.expressions.iter() {
            if let naga::Expression::Constant(h) = expr {
                mark_const_live(*h, module, &mut live);
            }
        }
    }

    for init in module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.init)
        .chain(module.overrides.iter().filter_map(|(_, o)| o.init))
    {
        collect_const_refs_in_global_expr(init, module, &mut live);
    }

    live
}

/// Marks `h` live and, the first time, the constants its init reads: every
/// insertion walks, so the set needs no fixpoint loop over itself.
fn mark_const_live(
    h: naga::Handle<naga::Constant>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Constant>,
) {
    if live.insert(h) {
        collect_const_refs_in_global_expr(module.constants[h].init, module, live);
    }
}

fn collect_const_refs_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Constant>,
) {
    let expr = &module.global_expressions[expr_h];
    if let naga::Expression::Constant(h) = expr {
        mark_const_live(*h, module, live);
    }
    visit_expression_children(expr, |c| collect_const_refs_in_global_expr(c, module, live));
}

/// Types transitively reachable from live code, so dead struct declarations
/// can be skipped; a library module keeps every type.
fn compute_live_types(
    module: &naga::Module,
    live_constants: &HandleSet<naga::Constant>,
    defer_cache: &[(Vec<bool>, Vec<bool>)],
) -> HandleSet<naga::Type> {
    let mut live: HandleSet<naga::Type> = Default::default();

    if is_library_module(module) {
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

    // A live composite carries its members, element and pointee along.
    let mut work: Vec<_> = live.iter().copied().collect();
    while let Some(th) = work.pop() {
        let mut nest = |t: naga::Handle<naga::Type>| {
            if live.insert(t) {
                work.push(t);
            }
        };
        match &module.types[th].inner {
            naga::TypeInner::Struct { members, .. } => members.iter().for_each(|m| nest(m.ty)),
            naga::TypeInner::Array { base, .. }
            | naga::TypeInner::BindingArray { base, .. }
            | naga::TypeInner::Pointer { base, .. } => nest(*base),
            _ => {}
        }
    }

    live
}

fn collect_types_in_global_expr(
    expr_h: naga::Handle<naga::Expression>,
    module: &naga::Module,
    live: &mut HandleSet<naga::Type>,
) {
    let expr = &module.global_expressions[expr_h];
    match expr {
        naga::Expression::Compose { ty, .. } | naga::Expression::ZeroValue(ty) => {
            live.insert(*ty);
        }
        naga::Expression::Constant(h) => {
            collect_types_in_global_expr(module.constants[*h].init, module, live);
        }
        _ => {}
    }
    visit_expression_children(expr, |c| collect_types_in_global_expr(c, module, live));
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
        Self::with_analyses(module, info, options, None)
    }

    /// [`Self::new`] over `prior`, the [`Analyses`] of a render of these
    /// arenas under these options, whatever its names.
    pub(super) fn with_analyses(
        module: &'a naga::Module,
        info: &'a naga::valid::ModuleInfo,
        options: GenerateOptions,
        prior: Option<Analyses>,
    ) -> Self {
        let mangle = options.mangle;
        let (canonical, inner_to_first) = type_groups(module);
        let analyses =
            prior.unwrap_or_else(|| Analyses::fresh(module, info, &canonical, &inner_to_first));
        let spelling = SpellingScopes::new(module, &analyses.spelled_groups);
        let StructNames {
            mut type_names,
            mut member_names,
            minted: minted_types,
            fixed: fixed_type_names,
        } = mint_struct_names(module, &options, &canonical, &spelling);

        // The anonymous override's minted name is referenced from module
        // scope and function bodies alike, so it dodges every name in scope
        // plus `StructNames::minted`.
        let mut used_names = HashSet::new();
        if module.overrides.iter().any(|(_, o)| o.name.is_none()) {
            used_names.extend(
                crate::name_gen::module_scope_names(module)
                    .chain(all_functions(module).flat_map(crate::name_gen::function_local_names))
                    .map(str::to_owned),
            );
            used_names.extend(options.preserve_symbols.iter().cloned());
            used_names.extend(minted_types.iter().cloned());
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

        let live_constants = compute_live_constants(module, &options.preserve_symbols);
        let live_types = compute_live_types(module, &live_constants, &analyses.defer);

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

        let (ref_counts, ref_bytes) = match &options.type_uses {
            Some(uses) => {
                let mut sites = HandleMap::default();
                let mut bytes = HandleMap::default();
                for (h, _) in module.types.iter() {
                    let (n, b) = uses.at(h);
                    if n > 0 {
                        sites.insert(h, n);
                        bytes.insert(h, b);
                    }
                }
                (sites, Some(bytes))
            }
            None => (
                count_type_handle_refs(module, &live_constants, &live_types, &analyses.defer),
                None,
            ),
        };
        let mut group_ref_count: HandleMap<naga::Type, usize> = Default::default();
        let mut group_bytes: HandleMap<naga::Type, usize> = Default::default();
        for (h, _) in module.types.iter() {
            let first = canonical[&h];
            *group_ref_count.entry(first).or_insert(0) += ref_counts.get(h).copied().unwrap_or(0);
            if let Some(bytes) = &ref_bytes {
                *group_bytes.entry(first).or_insert(0) += bytes.get(h).copied().unwrap_or(0);
            }
        }
        let alias_scope = AliasScope {
            canonical: &canonical,
            group_ref_count: &group_ref_count,
            group_bytes: ref_bytes.is_some().then_some(&group_bytes),
            spelling: &spelling,
            shadowed_type_aliases: &shadowed_type_aliases,
        };
        let (type_alias_decls, decl_handles) = if options.type_alias {
            plan_type_aliases(
                module,
                &options,
                &mut type_names,
                &alias_scope,
                &constant_names,
                &override_names,
                &global_names,
                &function_names,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let mut type_alias_decls = type_alias_decls;
        if mangle {
            rank_type_names_by_use(
                module,
                &mut type_names,
                &mut type_alias_decls,
                &decl_handles,
                &alias_scope,
                &ref_counts,
                &fixed_type_names,
                &override_names,
                &options,
                [
                    constant_names.as_slice(),
                    override_names.as_slice(),
                    global_names.as_slice(),
                    function_names.as_slice(),
                ],
            );
        }
        mint_member_names(module, &options, &type_names, &mut member_names);
        // A struct spells its name; anything else its structure, through
        // the aliases nested types got.
        let type_spelled_len = module
            .types
            .iter()
            .map(|(h, ty)| {
                let spelled = match &ty.inner {
                    naga::TypeInner::Struct { .. } => type_names.get(h).map_or(0, String::len),
                    inner => super::syntax::type_inner_name(
                        inner,
                        module,
                        &type_names,
                        &override_names,
                        &shadowed_type_aliases,
                    )
                    .map_or(0, |s| s.len()),
                };
                u32::try_from(spelled).unwrap_or(u32::MAX)
            })
            .collect();

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
            analyses,
            fn_effects: Vec::new(),
            name_weights: Default::default(),
            type_uses: TypeUses::sized(module.types.len()),
            type_spelled_len,
            inner_to_first,
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
