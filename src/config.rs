//! Configuration for [`crate::run`] / [`crate::run_module`]: [`Profile`]
//! selects the pass bundle, [`TraceConfig`] gates diagnostics,
//! [`FloatPrecision`] controls lossy float-literal trimming.

use std::path::PathBuf;

/// Float-literal precision trim applied at emission; every variant but
/// [`Self::Full`] is lossy and opted into per type via [`FloatPrecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Preserve the full IR value; emit the shortest round-trip text.
    #[default]
    Full,
    /// Round to at most `N` digits after the decimal point (`0` yields an
    /// integer-valued float), e.g. `0.123456 -> 0.12`.
    DecimalPlaces(u8),
    /// Round to at most `N` significant figures (`0` is treated as `1`),
    /// e.g. `1234567.89 -> 1230000`, `0.0012345 -> 0.0012`.
    SignificantFigures(u8),
}

/// Per-type precision caps for float literals; each IR float kind has its
/// own [`PrecisionMode`] so a lossy `f32` budget can pair with a
/// full-precision `f64`.  Defaults to `Full` everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FloatPrecision {
    /// Precision applied to `f16` literals (rounded via their exact `f32`
    /// widening).
    pub f16: PrecisionMode,
    /// Precision applied to `f32` literals.
    pub f32: PrecisionMode,
    /// Precision applied to `f64` literals.
    pub f64: PrecisionMode,
    /// Precision applied to literals still `AbstractFloat` at emission
    /// (extracted `const` decls, module-scope abstract literals); naga
    /// holds these as f64.
    pub abstract_float: PrecisionMode,
}

impl FloatPrecision {
    /// Apply `mode` to every float kind.
    pub fn all(mode: PrecisionMode) -> Self {
        Self {
            f16: mode,
            f32: mode,
            f64: mode,
            abstract_float: mode,
        }
    }
}

/// Optimization aggressiveness level, selecting which pass bundle
/// [`crate::passes::build_ir_passes`] constructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// Minimal DCE-driven pipeline: compact, const fold, dead-branch,
    /// dead-param, emit merge, rename.  No inlining or load dedup.
    Baseline,
    /// Full pipeline (inlining, load dedup, coalescing) without mangling
    /// unless [`Config::mangle`] requests it.
    Aggressive,
    /// [`Profile::Aggressive`] plus higher multi-site inlining budgets,
    /// identifier mangling on by default and, with mangling on,
    /// vector-constant hoisting.
    #[default]
    Max,
}

/// Per-pass diagnostic tracing, opt-in and off the hot path: with `enabled`
/// false the pipeline emits no intermediate text, validates only after
/// declared changes and allocates no trace directory.
#[derive(Debug, Clone, Default)]
pub struct TraceConfig {
    /// Master switch for per-pass before/after dumps to disk.
    pub enabled: bool,
    /// Base directory for trace output; defaults to `./trace` when `None`.
    pub dump_dir: Option<PathBuf>,
    /// Re-validate the WGSL text after every pass and escalate failures to
    /// hard errors instead of rolling back; meant for CI, not daily use.
    pub validate_each_pass: bool,
    /// Stop accepting IR pass changes after this many, then emit: the
    /// output carries exactly the first N accepted runs, so a wrong output
    /// is bisected to one pass run in log2(runs) executions (LLVM's
    /// `-opt-bisect-limit`).  `None` runs to convergence.
    pub opt_bisect_limit: Option<u64>,
}

/// Top-level minification configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Optimization profile.
    pub profile: Profile,
    /// Symbol names to preserve from renaming and mangling.  Applies
    /// uniformly to globals, functions, constants, overrides, arguments,
    /// locals, struct type names, and struct member names.
    pub preserve_symbols: Vec<String>,
    /// Keep every name a host may address by string: resource-bound
    /// globals, overrides, and the struct types those globals reach with
    /// their members; entry points are never renamed and `@location`
    /// arguments carry their contract in the attribute.  For hosts that
    /// re-reflect the shipped text by name; [`crate::name_map::NameMap`] is
    /// the alternative.  Computed into the preserve set at run start.
    pub preserve_interface: bool,
    /// Explicit mangle override; `None` defers to the profile default
    /// (only [`Profile::Max`] enables mangling implicitly).
    pub mangle: Option<bool>,
    /// Emit human-readable output with indentation and newlines.
    pub beautify: bool,
    /// Spaces per indentation level; honoured only when `beautify` is set.
    pub indent: u8,
    /// Per-type float-literal precision caps; any non-`Full` mode is lossy.
    pub float_precision: FloatPrecision,
    /// Expression-node ceiling for cloning a helper into each of several
    /// call sites; `None` selects the profile default.  A function called
    /// once is spliced into its caller whatever its size.
    pub max_inline_node_count: Option<usize>,
    /// Call-site ceiling for cloning a helper into its call sites; `None`
    /// selects the profile default.
    pub max_inline_call_sites: Option<usize>,
    /// Per-pass tracing and diagnostic settings.
    pub trace: TraceConfig,
    /// WGSL preamble of external declarations (e.g. a playground's
    /// bindings): prepended for parsing and optimization, its names
    /// auto-preserved, its declarations stripped from the output; leading
    /// directives of both texts are hoisted so the combination stays
    /// spec-compliant.
    pub preamble: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            profile: Profile::Max,
            preserve_symbols: Vec::new(),
            preserve_interface: false,
            mangle: None,
            beautify: false,
            indent: 2,
            float_precision: FloatPrecision::default(),
            max_inline_node_count: None,
            max_inline_call_sites: None,
            trace: TraceConfig::default(),
            preamble: None,
        }
    }
}

impl Config {
    /// Effective mangle setting: the explicit override, else on only for
    /// [`Profile::Max`].
    pub fn mangle(&self) -> bool {
        self.mangle.unwrap_or(self.profile == Profile::Max)
    }
}
