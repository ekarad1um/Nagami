//! Structured diagnostics returned alongside every minification run:
//! [`Report`] aggregates per-pass [`PassReport`]s.  Both are plain data
//! for downstream consumers, so every field is public and populated even
//! when tracing is off.

/// Diagnostics for a single optimization pass execution.
#[derive(Debug, Clone)]
pub struct PassReport {
    /// Stable pass identifier (e.g. `constant_folding`, `generator_emit`).
    pub pass_name: String,
    /// Emitted WGSL byte size before the pass ran.  `None` when tracing
    /// is off (no text is emitted on the hot path).
    pub before_bytes: Option<usize>,
    /// Emitted WGSL byte size after the pass ran; `None` when tracing is off.
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
    /// naga's rendered error, prefixed with the stage that gave up, when
    /// the input shipped compacted instead of the pipeline's output;
    /// `None` on every normal path.
    pub bailout: Option<String>,
    /// Why naga's emitter printed the module instead of nagami's generator:
    /// still IR-minified and renamed, without the generator's spelling
    /// (aliases, elisions), so larger.  The CLI warns on stderr, which the
    /// web build cannot show.
    pub fallback: Option<String>,
}

impl Report {
    /// Seeds `output_bytes` with `input_bytes` so a pipeline that
    /// short-circuits before emitting still reports a size; the driver
    /// overwrites it with the true emitted size.
    ///
    /// `output_bytes <= input_bytes` is not an invariant: [`crate::run`]
    /// records the raw source length while the output is the generator's
    /// complete emission, which can exceed it for already-minified input,
    /// so callers must guard `input_bytes - output_bytes` against underflow.
    pub fn new(input_bytes: usize) -> Self {
        Self {
            input_bytes,
            output_bytes: input_bytes,
            pass_reports: Vec::new(),
            converged: true,
            sweeps: 0,
            bailout: None,
            fallback: None,
        }
    }
}
