//! Structured diagnostics returned alongside every minification run.
//!
//! [`Report`] is the aggregate result; [`PassReport`] is its per-pass
//! leaf.  Both are plain-data and meant for downstream consumption
//! (CLI summaries, wasm callers, test assertions) so every field is
//! public and populated even when tracing is off.

/// Diagnostics for a single optimization pass execution.
#[derive(Debug, Clone)]
pub struct PassReport {
    /// Stable pass identifier (e.g. `constant_folding`, `generator_emit`).
    pub pass_name: String,
    /// Emitted WGSL byte size before the pass ran.  `None` when tracing
    /// is off (no text is emitted on the hot path).
    pub before_bytes: Option<usize>,
    /// Emitted WGSL byte size after the pass ran.  `None` under the
    /// same condition as [`before_bytes`](PassReport::before_bytes).
    pub after_bytes: Option<usize>,
    /// `true` when the pass modified the module (either declared a
    /// change or produced different output text).
    pub changed: bool,
    /// Wall-clock time spent in the pass, in microseconds.  Zero on
    /// the wasm target where no high-resolution clock is available.
    pub duration_us: u64,
    /// Whether the IR passed naga validation immediately after the pass.
    pub validation_ok: bool,
    /// Whether the emitted WGSL text round-trip validated.  `None`
    /// when `validate_each_pass` is off.
    pub text_validation_ok: Option<bool>,
    /// `true` when the pipeline reverted the pass after a validation
    /// failure (only possible when `validate_each_pass` is off).
    pub rolled_back: bool,
}

/// Aggregate report for an entire compaction run.
#[derive(Debug, Clone)]
pub struct Report {
    /// Size of the user-supplied input in bytes.  Excludes any preamble.
    pub input_bytes: usize,
    /// Size of the final output in bytes.
    pub output_bytes: usize,
    /// Per-pass diagnostics in execution order.
    pub pass_reports: Vec<PassReport>,
    /// `true` when the pipeline reached a fixed point before hitting
    /// the sweep cap; `false` when the cap forced an early exit.
    pub converged: bool,
    /// Number of full pass sweeps executed.
    pub sweeps: usize,
}

impl Report {
    /// Initialise a report with `input_bytes` as both the recorded input
    /// size and the initial output size.  `output_bytes` is overwritten
    /// by the driver after emission; seeding it with the input length
    /// keeps the invariant `output_bytes <= input_bytes` true even if
    /// the pipeline short-circuits before emitting anything.
    pub fn new(input_bytes: usize) -> Self {
        Self {
            input_bytes,
            output_bytes: input_bytes,
            pass_reports: Vec::new(),
            converged: true,
            sweeps: 0,
        }
    }
}
