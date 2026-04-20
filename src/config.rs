//! Public configuration surface for the minification pipeline.
//!
//! The three types below form the knobs callers tune before invoking
//! [`crate::run`] or [`crate::run_module`]: [`Profile`] selects the pass
//! bundle, [`TraceConfig`] gates diagnostic instrumentation, and
//! [`Config`] composes both with the user-visible output options.

use std::path::PathBuf;

/// Optimization aggressiveness level, selecting which pass bundle
/// [`crate::passes::build_ir_passes`] constructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// Minimal DCE-driven pipeline: compact, const fold, dead-branch,
    /// dead-param, emit merge, rename.  No inlining, CSE, or load dedup.
    Baseline,
    /// Full pipeline including inlining, load dedup, and coalescing,
    /// but without mangling unless explicitly requested via
    /// [`Config::mangle`].
    Aggressive,
    /// [`Profile::Aggressive`] plus CSE, higher inlining budgets, and
    /// identifier mangling on by default.
    #[default]
    Max,
}

/// Format used when dumping per-pass trace output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraceDumpFormat {
    /// Emit each per-pass dump as a `.wgsl` source file.
    #[default]
    WGSL,
}

/// Configuration for per-pass diagnostic tracing.
///
/// Tracing is opt-in and off the hot path: when `enabled` is `false` the
/// pipeline never emits intermediate text, validates only once per run,
/// and skips trace directory allocation.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Master switch for per-pass before/after dumps to disk.
    pub enabled: bool,
    /// Base directory for trace output; defaults to `./trace` when `None`.
    pub dump_dir: Option<PathBuf>,
    /// Output format for trace dumps.
    pub dump_format: TraceDumpFormat,
    /// Re-validate the WGSL text after every pass and escalate any
    /// failure to a hard error instead of silently rolling back.  Intended
    /// for CI regressions, not day-to-day minification.
    pub validate_each_pass: bool,
    /// Emit a `before.wgsl` alongside each step's `after.wgsl`.  Disable
    /// to roughly halve trace volume when only the final state matters.
    pub dump_before_after: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dump_dir: None,
            dump_format: TraceDumpFormat::WGSL,
            validate_each_pass: false,
            dump_before_after: true,
        }
    }
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
    /// Explicit mangle override; `None` defers to the profile default
    /// (only [`Profile::Max`] enables mangling implicitly).
    pub mangle: Option<bool>,
    /// Emit human-readable output with indentation and newlines.
    pub beautify: bool,
    /// Spaces per indentation level; honoured only when `beautify` is set.
    pub indent: u8,
    /// Maximum decimal places for float literals.  `None` preserves full
    /// precision; any `Some(n)` is lossy and must be opted into by the caller.
    pub max_precision: Option<u8>,
    /// Override the per-function expression-node ceiling used to gate
    /// inlining.  `None` selects the profile default.
    pub max_inline_node_count: Option<usize>,
    /// Override the call-site ceiling used to gate inlining.  `None`
    /// selects the profile default.
    pub max_inline_call_sites: Option<usize>,
    /// Per-pass tracing and diagnostic settings.
    pub trace: TraceConfig,
    /// Optional WGSL preamble providing external declarations (e.g. uniform
    /// bindings from a shader playground).  The preamble is
    /// prepended for parsing and optimization, its symbol names are added
    /// to `preserve_symbols` automatically, and its declarations are
    /// stripped from the final output.  Leading directives in both the
    /// preamble and the user source are hoisted so the combined text
    /// remains spec-compliant.
    pub preamble: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            profile: Profile::Max,
            preserve_symbols: Vec::new(),
            mangle: None,
            beautify: false,
            indent: 2,
            max_precision: None,
            max_inline_node_count: None,
            max_inline_call_sites: None,
            trace: TraceConfig::default(),
            preamble: None,
        }
    }
}

impl Config {
    /// Resolve the effective mangle setting: an explicit `Some(v)` wins
    /// over the profile default, otherwise [`Profile::Max`] enables it
    /// and every other profile leaves it off.
    pub fn mangle(&self) -> bool {
        self.mangle.unwrap_or(self.profile == Profile::Max)
    }
}
