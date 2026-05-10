//! Streaming inference engine: PCM ring -> preproc -> backbone -> head ->
//! `InferenceFrame` broadcast.
//!
//! ## Threading model
//!
//! `InferenceEngine` is `Send + !Sync` (the underlying `rknn_runtime::Session`
//! is `!Sync`).  It is constructed in the daemon's tokio runtime, then moved
//! into a single
//! `tokio::task::spawn_blocking(move || engine.run_blocking(...))`
//! worker that runs forever.  Inside [`InferenceEngine::run_blocking`]
//! everything is sync: no async state, no `&mut self`
//! from outside, no tokio Reactor.  All scratch buffers
//! are heap-allocated once pre-loop, then reused.
//!
//! ## Cancellation
//!
//! The engine polls `shutdown.is_cancelled()` once per outer-loop
//! iteration.  The token is created in `daemon::main`; cancelling it
//! lets the engine drain its current frame and return `Ok(())`.  The
//! daemon's supervisor `await`s the spawn_blocking JoinHandle.
//!
//! ## Backpressure
//!
//! On `ReadStatus::Wait` we sleep `min(hop_dur / 4, 50ms)` -- short
//! enough that we wake up well before the next hop is due, long enough
//! that we don't burn a CPU on a tight poll loop.  On `Lagged` we
//! `seek_latest(WaveformLen::USIZE)` and skip one frame: the audible glitch
//! has already happened, no point compounding it by chasing the
//! backlog.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use bytes::Bytes;
use thiserror::Error;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use crate::audio_buffer::{ReadStatus, Reader};
use crate::common::dims::{BackboneFeatureDim, NBins, NFrames, WaveformLen};
use crate::preproc::Preproc;
use crate::proto::{InferenceFrame, TopK};

use crate::inference::backbone::{Backbone, BackboneError};
use crate::inference::head::HotHead;
use crate::inference::kernel::{head_forward, softmax_into, top_k_indices_into};

/// Buffer-channel sample rate.  The mic arbitrator
/// always normalizes to 44.1 kHz; if we ever support a
/// configurable rate, this becomes a runtime field.
pub const SAMPLE_RATE_HZ: u64 = 44_100;

/// Duration of one full inference window in nanoseconds.  Used to
/// compute `window_start_ns` from `ts_ns` without per-sample timing
/// info from the buffer.
pub const WAVEFORM_DURATION_NS: u64 = WaveformLen::VALUE as u64 * 1_000_000_000 / SAMPLE_RATE_HZ;

/// Initial reservation for top-k / logits / probs vectors.  Heads with
/// up to 32 classes never realloc.  Larger heads do exactly one realloc
/// on first frame (cold path).
const INITIAL_CLASSES_HINT: usize = 32;

/// Wall-clock budget the engine spends absorbing back-to-back
/// backbone failures before surfacing them.  Long enough to ride
/// out a transient NPU-driver hiccup, short enough that a wedged
/// backbone doesn't keep the engine spinning silently.
const BACKBONE_FAILURE_BUDGET_SECS: u64 = 5;

/// Sleep duration (in milliseconds) inserted after each failed
/// backbone call before retrying.  Without this a fast-failing
/// backbone (e.g. "device not ready" returning microseconds)
/// tight-loops the engine.  100 ms paces retries to ~10 Hz which
/// is above the typical 4 Hz inference cadence so we don't
/// compound throughput loss in steady state.  Held in
/// milliseconds (not `Duration`) so the failure-count derivation
/// below is a single integer division.
const BACKBONE_FAILURE_BACKOFF_MS: u64 = 100;

/// Sleep duration inserted after each failed backbone call.
/// Derived from [`BACKBONE_FAILURE_BACKOFF_MS`].
const BACKBONE_FAILURE_BACKOFF: Duration = Duration::from_millis(BACKBONE_FAILURE_BACKOFF_MS);

/// After this many consecutive backbone failures, the engine
/// gives up and returns `EngineError::Backbone` so the daemon's
/// supervisor surfaces the failure.  Derived from the wall-clock
/// budget and the per-failure backoff so the two numbers stay in
/// sync: editing one of the inputs automatically retunes the cap.
const MAX_CONSECUTIVE_BACKBONE_FAILURES: u32 =
    ((BACKBONE_FAILURE_BUDGET_SECS * 1_000) / BACKBONE_FAILURE_BACKOFF_MS) as u32;

/// Failure shapes from the streaming inference engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("backbone: {0}")]
    Backbone(#[from] BackboneError),
    #[error("encode InferenceFrame: {0}")]
    Encode(#[from] prost::EncodeError),
}

/// Tunable inference cadence.  Held in the engine via
/// `Arc<ArcSwap<InferenceCfg>>` so the API can change it mid-stream
/// without touching the engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceCfg {
    /// Stride between successive windows, in samples.  The first
    /// inference fires once `WaveformLen::USIZE` samples are buffered;
    /// subsequent fires advance the reader by this many samples.
    /// Typical values: 11_025 (250 ms hop, 4 Hz cadence) for
    /// real-time UI; 22_050 (500 ms hop) when CPU is constrained.
    /// Smaller hops cost more CPU but reduce per-frame latency.
    pub hop_samples: usize,

    /// Number of (idx, label, prob) entries in each emitted
    /// `InferenceFrame.top_k`.
    pub top_k: usize,
}

impl Default for InferenceCfg {
    fn default() -> Self {
        Self {
            hop_samples: 11_025, // 250 ms @ 44.1 kHz
            top_k: 3,
        }
    }
}

impl InferenceCfg {
    /// Largest accepted `hop_samples`, capped so successive
    /// inference windows overlap by at least 25%.  Equivalently
    /// the operator-tunable overlap ratio `(WaveformLen - hop) /
    /// WaveformLen` is constrained to `[0.25, 1.0)` (`hop = 0`
    /// is rejected by the lower bound below).  At the upper
    /// bound (`hop = MAX_HOP_SAMPLES`) overlap is exactly 25%;
    /// at the lower bound (`hop = 1`) overlap is ~100%.
    pub const MAX_HOP_SAMPLES: usize = WaveformLen::USIZE * 3 / 4;

    /// Hard cap on `top_k`.  The runtime allocates a small fixed
    /// buffer; keeping the cap modest avoids surprising memory
    /// behaviour and matches the protobuf message footprint.
    pub const MAX_TOP_K: usize = 64;

    /// Reject values that would make the inference loop misbehave.
    /// The engine hot loop additionally clamps `hop_samples >= 1` as
    /// a defense-in-depth measure for any code path that bypasses
    /// this validator (e.g. a `Default::default()` then mutate).
    /// `validate` is the explicit-reject hook for the API + config
    /// loaders that take operator input.
    pub fn validate(&self) -> Result<(), String> {
        if self.hop_samples == 0 || self.hop_samples > Self::MAX_HOP_SAMPLES {
            return Err(format!(
                "hop_samples must be 1..={}; got {}",
                Self::MAX_HOP_SAMPLES,
                self.hop_samples
            ));
        }
        if self.top_k == 0 || self.top_k > Self::MAX_TOP_K {
            return Err(format!(
                "top_k must be 1..={}; got {}",
                Self::MAX_TOP_K,
                self.top_k
            ));
        }
        Ok(())
    }
}

/// Internal: running totals tracked across iterations of the engine
/// loop.  Bundled into one `Copy` struct so the heartbeat-emit helpers
/// take a single reference instead of 4 positional `u64`s -- the
/// previous flat-args signature was a future-bug surface (silent
/// argument reordering would compile cleanly).  `Heartbeat` mirrors
/// these fields for the public watch channel.
#[derive(Debug, Clone, Copy, Default)]
struct EngineCounters {
    last_seq: u64,
    frames_emitted: u64,
    frames_dropped_nan: u64,
    frames_dropped_lag: u64,
}

/// Per-iteration liveness signal.  Emitted via `watch::Sender<Heartbeat>`
/// so the status crate can show "engine alive at HH:MM:SS,
/// last frame seq=N" without polling the broadcast channel.
///
/// `Default` is safe to construct before the engine starts, so the
/// daemon can build the watch channel up-front.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    pub at: Instant,
    pub state: EngineState,
    /// Most recent emitted seq, or 0 if no frame yet.
    pub last_seq: u64,
    pub frames_emitted: u64,
    /// Frames dropped because preproc returned NaN/Inf
    /// (silent input is intentionally silenced rather than
    /// classified).
    pub frames_dropped_nan: u64,
    /// Frames skipped because the buffer reader observed a `Lagged`.
    pub frames_dropped_lag: u64,
}

impl Default for Heartbeat {
    fn default() -> Self {
        Self {
            at: Instant::now(),
            state: EngineState::Starting,
            last_seq: 0,
            frames_emitted: 0,
            frames_dropped_nan: 0,
            frames_dropped_lag: 0,
        }
    }
}

/// Coarse health state of the inference engine, surfaced
/// to operators via the [`Heartbeat`] feed.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EngineState {
    /// Pre-loop init.
    Starting,
    /// Running normally; last iteration emitted a frame.
    Running,
    /// Last iteration found buffer underrun (`ReadStatus::Wait`).
    Waiting,
    /// Last iteration found buffer lap (`ReadStatus::Lagged`).
    Lagged,
    /// Loop exited cleanly via cancellation.
    Stopped,
    /// Loop exited via error (returned to caller).
    Failed,
}

/// Owning engine.  Constructed once per daemon lifetime; moved into
/// `spawn_blocking`; never cloned.
///
/// `backbone` is a trait object so tests substitute mocks
/// without going through `BackbonePipeline`'s cfg-gated arms.  The
/// daemon constructs the load-time enum then converts via
/// `BackbonePipeline::into_boxed`.
pub struct InferenceEngine {
    preproc: Preproc,
    backbone: Box<dyn Backbone>,
    head: HotHead,
    cfg: Arc<ArcSwap<InferenceCfg>>,
    monitor: watch::Sender<Heartbeat>,
    /// Optional producer-side
    /// [`crate::common::time::BufferTimingAnchor`] cell.  When
    /// `Some`, `InferenceFrame.t_us_capture_monotonic` is
    /// projected from the window's FIRST 44.1 kHz sample
    /// (window-start convention).  When `None`, the engine
    /// falls back to a publish-time `CaptureTime::now()`
    /// stamp -- correct clock domain, wrong instant.  Tests
    /// pass `None`; production wiring always supplies one.
    timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("backbone", &self.backbone)
            .field("head", &self.head)
            .field("cfg", &self.cfg.load())
            .finish_non_exhaustive()
    }
}

impl InferenceEngine {
    /// Initialize the engine with a pre-constructed backbone (RKNN
    /// or Burn -- see [`crate::inference::backbone::BackbonePipeline`]).  The caller is responsible
    /// for choosing + loading the backbone; this lets the daemon
    /// implement its RKNN-first / Burn-fallback policy in one place
    /// without baking the choice into the engine's constructor.
    ///
    /// Caller-provided `head`, `cfg`, and `monitor` are typically
    /// `Clone`d from `Arc<...>`/watch handles owned by the daemon's
    /// `async_main`; the engine takes ownership of the pipeline +
    /// `Preproc` (both are `!Sync`).  Returns infallibly today --
    /// validation is the backbone constructor's job.
    pub fn new(
        backbone: Box<dyn Backbone>,
        head: HotHead,
        cfg: Arc<ArcSwap<InferenceCfg>>,
        monitor: watch::Sender<Heartbeat>,
        timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
    ) -> Self {
        Self {
            preproc: Preproc::new(),
            backbone,
            head,
            cfg,
            monitor,
            timing_anchor,
        }
    }

    /// Run forever.  Consumes self (Session owns runtime resources that
    /// shouldn't outlive this call).
    ///
    /// Behavior:
    /// * `shutdown.is_cancelled()` polled once per outer-loop iter.
    /// * `Ok(())` on clean shutdown.
    /// * `Err(EngineError)` on backbone failure mid-stream -- the daemon
    ///   supervisor restarts the task.
    pub fn run_blocking(
        mut self,
        mut reader: Reader,
        out: broadcast::Sender<Bytes>,
        shutdown: CancellationToken,
    ) -> Result<(), EngineError> {
        let result = self.run_blocking_inner(&mut reader, &out, &shutdown);
        // Final heartbeat reflecting the loop's terminal state.  `Ok(())`
        // can only be reached from a `shutdown.is_cancelled()` early return
        // in `run_blocking_inner`; the cancelled/non-cancelled split was
        // redundant.
        let final_state = if result.is_err() {
            EngineState::Failed
        } else {
            EngineState::Stopped
        };
        self.send_heartbeat(|hb| {
            hb.at = Instant::now();
            hb.state = final_state;
        });
        result
    }

    fn run_blocking_inner(
        &mut self,
        reader: &mut Reader,
        out: &broadcast::Sender<Bytes>,
        shutdown: &CancellationToken,
    ) -> Result<(), EngineError> {
        // Pre-loop scratch (allocated once)
        let mut pcm = Box::new([0.0f32; WaveformLen::USIZE]);
        // Spectrogram scratch reused across every frame via
        // `Preproc::spectrogram_into`.  The in-place
        // variant (verified bit-identical against
        // `Preproc::spectrogram` by `tests/preproc_parity.rs`)
        // overwrites every cell, so initial state is
        // immaterial.
        let mut spec_buf: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
            Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        // Features buffer matches the [`Backbone::infer`]
        // in-place signature; both backends overwrite every
        // cell.
        let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
        let mut logits: Vec<f32> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        let mut probs: Vec<f32> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        let mut top_idx: Vec<usize> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        // Encode scratch reused across
        // every emitted Envelope via `wrap_inference_into`.
        // `BytesMut::split().freeze()` returns the head as an
        // Arc-backed `Bytes` (zero-copy fan-out across the
        // broadcast channel's clones) and re-uses the residual
        // capacity for the next frame.  After a few frames the
        // underlying allocation is reused indefinitely; the
        // per-frame `wrap_inference` Vec alloc that this replaces
        // was ~3-4 KB at typical top_k=5, costing ~16 KB/sec at
        // 4 Hz cadence.  The 4 KiB initial capacity sizes for one
        // typical envelope without the first frame triggering
        // a grow.
        let mut encode_buf: bytes::BytesMut = bytes::BytesMut::with_capacity(4096);

        // Counters live in self.monitor's last-sent heartbeat; we mirror
        // them locally to avoid load-then-store on every frame.  The
        // watch::Sender::send-modify protocol takes a closure; we use it
        // for the top-of-loop liveness ping.
        let mut counters = EngineCounters::default();

        // Bounded backbone-failure retry.  `consecutive_failures`
        // tracks the in-progress streak so log volume is throttled
        // (power-of-two thinning) and the streak triggers a hard exit
        // at `MAX_CONSECUTIVE_BACKBONE_FAILURES`.
        let mut consecutive_failures: u32 = 0;

        // Fire one Starting -> Running heartbeat before we enter the loop
        // so a fast `await` on the receiver immediately observes
        // liveness.
        self.send_heartbeat_state(EngineState::Running, &counters);

        loop {
            if shutdown.is_cancelled() {
                return Ok(());
            }

            let cfg_snap = self.cfg.load_full();
            let hop_samples = cfg_snap.hop_samples.max(1);
            // hop_dur = hop_samples / 44_100 s.  Compute in nanoseconds
            // to avoid f64 noise on small hops.
            let hop_ns = hop_samples as u64 * 1_000_000_000 / SAMPLE_RATE_HZ;
            let wait_ns = (hop_ns / 4).min(50_000_000);
            let wait_sleep = Duration::from_nanos(wait_ns.max(1_000_000));

            match reader.peek_into(&mut pcm[..]) {
                ReadStatus::Wait => {
                    self.send_heartbeat_state(EngineState::Waiting, &counters);
                    std::thread::sleep(wait_sleep);
                    continue;
                }
                ReadStatus::Lagged { by } => {
                    tracing::warn!(
                        target: "inference",
                        by_samples = by,
                        "audio reader lagged; resyncing to latest window",
                    );
                    reader.seek_latest(WaveformLen::USIZE);
                    counters.frames_dropped_lag = counters.frames_dropped_lag.saturating_add(1);
                    self.send_heartbeat_state(EngineState::Lagged, &counters);
                    continue;
                }
                ReadStatus::Ready => {}
            }
            // Snapshot the reader's tail BEFORE advance so the
            // anchor-derived capture timestamp on emitted
            // inference frames refers to the FIRST 44.1 kHz
            // sample of the window (window-start convention:
            // "this prediction is about audio that started at
            // t").  After advance the tail points at the next
            // hop's start, which is the WRONG anchor point for
            // this frame's stamp.
            let window_start_tail = reader.tail();
            // Ready: advance tail by hop_samples (NOT WaveformLen --
            // successive windows overlap by WaveformLen - hop_samples).
            reader.advance(hop_samples);

            // Per-frame timestamps
            // Clock-domain disambiguation.  The proto
            // surface carries two distinct stamps (per
            // `crate::common::time::CaptureTime` / `WallTime`):
            //   * `t_us_capture_monotonic` -- the capture
            //     monotonic time of the FIRST 44.1 kHz sample of
            //     the inference window (window-start convention),
            //     projected through the producer's
            //     [`crate::common::time::BufferTimingAnchor`] via
            //     `capture_us_for(...)`.  Accurate to within one
            //     mic-arbitrator push period (~5-23 ms) plus
            //     integer-division rounding.  Tests and
            //     adversarial harnesses that construct the engine
            //     without an anchor (`None`) fall back to a
            //     publish-time `CaptureTime::now()` stamp; the
            //     daemon's wiring always supplies an anchor.
            //   * `t_us_publish_unix` -- engine emit time on the
            //     wall clock, microseconds since Unix epoch.
            //     Always publish-time (cross-process correlation).
            let t_us_capture_monotonic = match self.timing_anchor.as_ref() {
                Some(cell) => {
                    let anchor = **cell.load();
                    crate::common::time::capture_us_for(anchor, window_start_tail)
                }
                None => crate::common::time::CaptureTime::now().as_micros(),
            };
            // `WallTime::now` returns `None` only if the system
            // clock is set before the Unix epoch -- pathological;
            // emit `None` on the wire in that case so consumers
            // see "no wall-clock stamp" rather than a misleading
            // zero.
            let t_us_publish_unix = crate::common::time::WallTime::now().map(|w| w.as_micros());

            // Preproc
            // Zero-alloc per frame: `spectrogram_into`
            // overwrites the pre-allocated `spec_buf`.
            self.preproc.spectrogram_into(&pcm, &mut spec_buf);
            // Flat-slice scan auto-vectorizes; an
            // `iter().flatten()` formulation goes through
            // two iterator levels and inhibits the SIMD
            // lowering of `is_finite`.
            if spec_buf
                .as_slice()
                .as_flattened()
                .iter()
                .any(|v| !v.is_finite())
            {
                tracing::warn!(
                    target: "inference",
                    seq = counters.last_seq + 1,
                    "frame dropped: NaN/Inf in spec (silence on log)",
                );
                counters.frames_dropped_nan = counters.frames_dropped_nan.saturating_add(1);
                self.send_heartbeat_state(EngineState::Running, &counters);
                continue;
            }

            // Backbone (NPU or CPU/Burn)
            // The pipeline encapsulates the layout transformation
            // (bin-major flatten for RKNN, frame-major
            // NCHW for Burn) so the engine doesn't need to
            // know which is in use.  Failures (typically
            // transient NPU driver hiccups) are logged on
            // power-of-two streak boundaries so persistent
            // failure produces at most `ceil(log2(N))`
            // lines, and the engine bails with
            // [`EngineError::Backbone`] once the streak
            // hits [`MAX_CONSECUTIVE_BACKBONE_FAILURES`].
            match self.backbone.infer(&spec_buf, &mut features) {
                Ok(()) => {
                    if consecutive_failures > 0 {
                        tracing::info!(
                            target: "inference",
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            "backbone recovered after failure streak",
                        );
                        consecutive_failures = 0;
                    }
                }
                Err(e) => {
                    consecutive_failures = consecutive_failures.saturating_add(1);
                    if consecutive_failures == 1 {
                        tracing::error!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            "backbone failure; engine will retry",
                        );
                    } else if consecutive_failures.is_power_of_two() {
                        tracing::warn!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            "backbone failure (continued)",
                        );
                    }
                    // Silent for the inter-power frames:
                    // the heartbeat counters keep ticking
                    // so operators still have a status
                    // signal when log lines are throttled.
                    if consecutive_failures >= MAX_CONSECUTIVE_BACKBONE_FAILURES {
                        tracing::error!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            "backbone failure streak exceeded threshold; engine giving up",
                        );
                        return Err(e.into());
                    }
                    std::thread::sleep(BACKBONE_FAILURE_BACKOFF);
                    continue;
                }
            }

            // Defence: a misbehaving backbone (RKNN driver
            // hiccup, fp16 underflow on a corner-case
            // input) could emit NaN / Inf in the feature
            // vector.  Without this check, [`head_forward`]'s
            // matmul propagates NaN into logits and
            // [`softmax_into`] then emits a uniform `1/n`
            // distribution, so the API client sees a
            // valid-looking but meaningless top-k.  Dropping
            // the frame here gives the same observable
            // signal as a spec-NaN drop (counted in
            // `frames_dropped_nan`).
            if features.iter().any(|v| !v.is_finite()) {
                tracing::warn!(
                    target: "inference",
                    seq = counters.last_seq + 1,
                    backbone = self.backbone.description(),
                    "frame dropped: NaN/Inf in backbone output features",
                );
                counters.frames_dropped_nan = counters.frames_dropped_nan.saturating_add(1);
                self.send_heartbeat_state(EngineState::Running, &counters);
                continue;
            }

            // Head (CPU)
            // Atomic `(snapshot, version)`: stamping the
            // version separately from the weight snapshot
            // would race a swap landing between the two
            // reads, producing a frame whose logits came
            // from version N but whose stamped version
            // reads N+1.
            let (snap, head_version) = self.head.snapshot_with_version();
            let n = snap.n_classes;
            // resize(n, 0.0) is a no-op when n == old len; otherwise it
            // grows or shrinks (no realloc until we exceed initial
            // capacity, which is rare for typical heads). head_forward
            // overwrites every entry, so the fill value is immaterial.
            logits.resize(n, 0.0);
            probs.resize(n, 0.0);
            head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);
            softmax_into(&logits, &mut probs);
            top_k_indices_into(&probs, cfg_snap.top_k, &mut top_idx);

            // Build + encode the proto frame
            // Per-frame heap traffic (top-k Vec, label
            // clones, head_id String, prost varint scratch)
            // is unavoidable without bypassing
            // [`prost::Message::encode`].
            let next_seq = counters.last_seq.wrapping_add(1);
            let frame = InferenceFrame {
                seq: next_seq,
                t_us_capture_monotonic: Some(t_us_capture_monotonic),
                t_us_publish_unix,
                head_id: Some(snap.head_id.to_string()),
                head_version: Some(head_version.get()),
                top_k: top_idx
                    .iter()
                    .map(|&i| TopK {
                        class_idx: i as u32,
                        label: snap.labels[i].clone(),
                        prob: probs[i],
                    })
                    .collect(),
            };

            // Wrap into an Envelope; [`wrap_inference_into`]
            // reuses `encode_buf` so the steady-state
            // envelope-encode allocation rate is zero.
            // Ignore `SendError`: no subscribers is steady
            // state when no UI is connected.
            let payload = crate::proto::framing::wrap_inference_into(&mut encode_buf, frame);
            let _ = out.send(payload);

            // `last_seq` is bumped only on successful
            // publish so the heartbeat counts emitted
            // frames.
            counters.last_seq = next_seq;
            counters.frames_emitted = counters.frames_emitted.saturating_add(1);
            self.send_heartbeat_state(EngineState::Running, &counters);
        }
    }

    fn send_heartbeat<F: FnOnce(&mut Heartbeat)>(&self, edit: F) {
        self.monitor.send_modify(edit);
    }

    fn send_heartbeat_state(&self, state: EngineState, counters: &EngineCounters) {
        // Snapshot by-value: [`EngineCounters`] is `Copy`,
        // so the closure avoids borrow conflicts with
        // `self.monitor`.
        let snap = *counters;
        self.send_heartbeat(|hb| {
            hb.at = Instant::now();
            hb.state = state;
            hb.last_seq = snap.last_seq;
            hb.frames_emitted = snap.frames_emitted;
            hb.frames_dropped_nan = snap.frames_dropped_nan;
            hb.frames_dropped_lag = snap.frames_dropped_lag;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `WAVEFORM_DURATION_NS` is the integer ratio.  Sanity-check.
    #[test]
    fn waveform_duration_ns_is_998_458_049() {
        assert_eq!(WAVEFORM_DURATION_NS, 998_458_049);
    }

    /// The default cfg matches the canonical cadence: 250 ms hop,
    /// top-3.
    #[test]
    fn default_cfg_is_250ms_hop() {
        let c = InferenceCfg::default();
        assert_eq!(c.hop_samples, 11_025);
        assert_eq!(c.top_k, 3);
    }

    /// Heartbeat default has at-now and Starting state.
    #[test]
    fn heartbeat_default_state_is_starting() {
        let hb = Heartbeat::default();
        assert_eq!(hb.state, EngineState::Starting);
        assert_eq!(hb.last_seq, 0);
        assert_eq!(hb.frames_emitted, 0);
    }

    /// `WallTime::now` returns a post-epoch value (the
    /// test process's wall clock is almost certainly post-2001).
    /// This asserted on `wall_clock_unix_ns`; the
    /// engine now uses `crate::common::time::WallTime` directly.
    #[test]
    fn wall_clock_now_post_epoch() {
        let t = crate::common::time::WallTime::now()
            .expect("wall clock post-epoch")
            .as_micros();
        // After 2001-09-09 (Y2001 us ~= 10^15) -> t > 10^15 us.
        assert!(t > 1_000_000_000_000_000, "wall clock < 2001? t={t}");
    }

    /// The engine's bounded-retry loop logs on power-of-two
    /// boundaries, so a failure streak of N produces ceil(log2(N+1))
    /// lines instead of N.  Verify the formula directly (the
    /// throttling logic is straight-line code in the engine; this
    /// test pins the algebra so a future refactor can't silently
    /// change the cadence).
    #[test]
    fn backbone_failure_log_thinning_is_logarithmic() {
        let mut log_lines: Vec<u32> = Vec::new();
        for streak in 1..=MAX_CONSECUTIVE_BACKBONE_FAILURES {
            if streak == 1 || streak.is_power_of_two() {
                log_lines.push(streak);
            }
        }
        // For MAX = 50 the sequence is 1, 2, 4, 8, 16, 32 -> 6 lines.
        // Plus the final hard-exit error line at the threshold itself,
        // which the engine emits unconditionally; that's accounted
        // for at the call site, not in this throttling formula.
        assert_eq!(
            log_lines,
            vec![1, 2, 4, 8, 16, 32],
            "log thinning shifted; expected 6 boundaries within 50",
        );
    }
}
