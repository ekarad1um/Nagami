//! Mic arbitrator: developer-curated mic whitelist + intra-mic
//! channel arbitration.
//!
//! ## Conceptual model
//!
//! Production audio capture is *intra-mic* RMS-arbitrated channel
//! switching, optionally with mic-level failover across a developer-
//! curated whitelist. **There is no cross-mic RMS comparison** --
//! adding two USB mics doesn't make the daemon switch between them
//! based on signal level.  Mic selection is operator-set or
//! first-available; channel selection within the active mic is the
//! signal-level knob.
//!
//! Two-level policy:
//!
//! ```text
//!   MicPolicy {
//!       mic:     MicSelection { FirstAvailable | Fixed(MicId) },
//!       channel: ChannelSelection { Auto | Fixed(u16) },
//!   }
//! ```
//!
//! Candidates list (developer-curated):
//!
//! ```text
//!   Vec<MicCandidate { id, source: { Alsa | Mock }, channels: Vec<u16> }>
//! ```
//!
//! The catalogue and policy live in different layers: the
//! catalogue is read once from a launch-time TOML and is
//! immutable for the daemon's lifetime; the policy is hot-
//! reloadable + API-mutable.  They're bundled into one
//! [`MicSettings`] so the arbitrator's hot loop reads both with
//! a single `ArcSwap` load instead of two.
//!
//! ## Hot path
//!
//! One std thread (`mic-arbitrator`).  Per loop iteration:
//!
//!  1. [`MicSettingsStore::snapshot`] (vtable + ArcSwap
//!     load) the live [`MicSettings`].
//!  2. Resolve the desired mic.  `FirstAvailable` is
//!     sticky on the active source until it disappears or
//!     fails; only the fail path walks alternates, gated
//!     by `mic_failover_after`.
//!  3. If the desired mic differs from the active one,
//!     tear down + re-open (rate-limited warnings on
//!     persistent failure).  When no source is open the
//!     loop parks on `failover_retry_interval` instead of
//!     hot-spinning.
//!  4. Read one period of interleaved frames from the
//!     active source.  ALSA `try_recover` +
//!     `reset_per_channel_fir` runs here when `readi`
//!     returns an `Err`.
//!  5. Single-pass demux + sum-of-squares per whitelisted
//!     slot (each whitelisted-channel sample is read
//!     exactly once; the interleaved buffer stays
//!     L1-resident across the per-slot strided passes).
//!  6. Update per-slot EMA RMS state.
//!  7. Pick the active slot via [`ChannelSelection`] +
//!     hysteresis + dwell.
//!  8. If `device_rate == SampleRate::VALUE`: write the
//!     active slot's samples directly.  Else: feed every
//!     slot through its resampler so FIR states stay
//!     current and a channel switch is glitch-free; drain
//!     the active slot's output to the
//!     [`crate::audio_buffer::Writer`] and discard the
//!     others via [`Streaming::drop_output`].  In true
//!     fixed-channel mode non-active resamplers are reset
//!     and skipped instead.
//!
//! Steady-state cost per period (1024 frames at 44.1 kHz
//! = 23 ms wall): well under 100 us of CPU on a Pi 5,
//! comfortably below 0.5 % single-core utilisation.

// `types` carries the data model + validation surface;
// `store` introduces the [`MicSettingsStore`] read-only
// trait (production impl in
// [`crate::config::MicSettingsCell`]); `tests` holds the
// arbitrator's integration + unit test suite.
mod store;
#[cfg(test)]
mod tests;
mod types;

pub use store::{ArcSwapStore, MicSettingsStore};
pub use types::{
    CandidateError, CandidateSource, ChannelSelection, MAX_BUFFER_FRAMES, MAX_CHANNEL_INDEX,
    MAX_CHANNELS, MAX_PERIOD_FRAMES, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, MicCandidate, MicCatalogue,
    MicPolicy, MicSelection, MicSettings, PolicyValidationError,
};

use crate::audio_buffer::Writer;
use crate::audio_io::source::{ActiveSource, OpenError, open_source};
use crate::common::dims::SampleRate;
use crate::common::ids::MicId;
use crate::dsp::resample::Streaming;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Tunables for the mic-arbitrator's run loop.  Validate
/// via [`Self::validate`] before constructing a
/// [`MicArbitrator`].
#[derive(Clone, Debug)]
pub struct MicArbitratorConfig {
    /// Minimum dB margin a non-active channel must beat
    /// the active channel by before we switch.  3 dB ~=
    /// 1.41x linear.
    pub hysteresis_db: f32,
    /// Minimum time the active channel must hold before another
    /// switch is allowed.  Suppresses chatter on borderline RMS.
    pub dwell: Duration,
    /// EMA time constant for per-channel RMS.  The plan's "100 ms
    /// rolling RMS" mapped to a single-pole IIR.
    pub rms_window: Duration,
    /// If the active mic produces no fresh frames for this long,
    /// `MicSelection::FirstAvailable` triggers a failover walk over
    /// the candidate list.  `Fixed` ignores this -- it stays committed
    /// to its named mic and just keeps retrying open.
    pub mic_failover_after: Duration,
    /// On `FirstAvailable` failover, how long to sleep between
    /// retry-walks when no candidate opens.  Cheap; keeps the
    /// arbitrator responsive to a candidate coming back.
    pub failover_retry_interval: Duration,
    /// Pin the spawned mic-arbitrator thread to a specific
    /// CPU core via [`crate::sched::pin_to_core`].  `None`
    /// keeps the thread on the kernel scheduler's default
    /// placement.  Failure to pin (e.g. host has fewer cores)
    /// is logged at WARN and ignored; the arbitrator continues
    /// on default placement.
    pub sched_pin: Option<usize>,
    /// Switch the spawned mic-arbitrator thread to
    /// `SCHED_FIFO` with this priority via
    /// [`crate::sched::set_realtime`].  `None` keeps
    /// `SCHED_OTHER` (kernel default).  Production wiring
    /// uses priority 50; the binary needs `CAP_SYS_NICE`
    /// for the switch to succeed.  Failure is logged at
    /// WARN and ignored; the arbitrator continues at
    /// `SCHED_OTHER`, which produces occasional ALSA
    /// underruns under load but stays functional.
    pub sched_priority: Option<i32>,
    /// Optional shared timing anchor; when present, the run
    /// loop publishes a fresh
    /// [`crate::common::time::BufferTimingAnchor`] after each
    /// successful `Writer::push` so consumers (opus encoder,
    /// inference engine) can project a sample position to its
    /// capture monotonic time without going through the
    /// producer.  See `common::time::capture_us_for` for the
    /// projection math.  `None` (the default) is the
    /// no-anchor fallback: consumers stamp `CaptureTime::now()`
    /// at emit time -- a publish-time stamp masquerading as a
    /// capture-time stamp.  Production wiring constructs one
    /// anchor and threads it to the arbitrator, the opus
    /// encoder, and the inference engine; tests that do not
    /// assert capture-time semantics leave it `None` and the
    /// existing fallback runs unchanged.
    pub timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
}

impl Default for MicArbitratorConfig {
    fn default() -> Self {
        Self {
            hysteresis_db: 3.0,
            dwell: Duration::from_millis(250),
            rms_window: Duration::from_millis(100),
            mic_failover_after: Duration::from_secs(2),
            failover_retry_interval: Duration::from_secs(1),
            // Defaults are no-pin / no-realtime so unit tests +
            // macOS dev hosts work unchanged.  Daemon production
            // wiring overrides via `cfg.sched_pin = Some(1)` +
            // `cfg.sched_priority = Some(50)`.
            sched_pin: None,
            sched_priority: None,
            // No anchor by default so existing test fixtures
            // that build a config via struct-update syntax stay
            // unchanged; production daemon wiring sets `Some(...)`.
            timing_anchor: None,
        }
    }
}

impl MicArbitratorConfig {
    /// `10^(hysteresis_db / 20)`.  Linear ratio that a candidate
    /// channel's RMS must exceed the active channel's RMS by to
    /// trigger a switch.  The factor of 20 (not 10) is correct: RMS
    /// is an amplitude quantity, not a power quantity.
    ///
    /// Call once per arbitrator lifetime -- `run_loop` evaluates this
    /// before entering its main loop and threads the result through
    /// `process_period` / `pick_slot`.  `MicArbitratorConfig` is
    /// immutable for the arbitrator's lifetime by design (per the
    /// type's docstring), so there's no need to recompute, and no
    /// `powf` ever runs in the hot path.
    pub fn hysteresis_linear(&self) -> f32 {
        10f32.powf(self.hysteresis_db / 20.0)
    }

    /// Sanity-check operator-supplied tunables.  Invoked
    /// at the daemon's [`MicArbitrator::start`] site via an
    /// `expect` gate; the validator's preventive role
    /// survives a future move to operator-tunable
    /// launch-config.
    ///
    /// - `hysteresis_db >= 0.0`.  A negative threshold
    ///   inverts the "must beat by" comparison and would
    ///   oscillate on every sample.  NaN / Inf rejected.
    /// - `rms_window > Duration::ZERO`.  Used as the EMA
    ///   time constant; zero would divide by zero in the
    ///   smoothing coefficient.
    /// - `mic_failover_after > Duration::ZERO` and
    ///   `failover_retry_interval > Duration::ZERO`.  Both
    ///   gate blocking branches in the run loop; zero
    ///   would tight-loop.
    ///
    /// `dwell` is unsigned and unconstrained (a zero-dwell
    /// is legal; the EMA hysteresis still prevents chatter).
    pub fn validate(&self) -> Result<(), String> {
        if !(self.hysteresis_db.is_finite() && self.hysteresis_db >= 0.0) {
            return Err(format!(
                "mic_arbitrator: hysteresis_db must be finite and >= 0.0; got {}",
                self.hysteresis_db
            ));
        }
        if self.rms_window.is_zero() {
            return Err(
                "mic_arbitrator: rms_window must be > 0 (used as EMA time constant)".into(),
            );
        }
        if self.mic_failover_after.is_zero() {
            return Err("mic_arbitrator: mic_failover_after must be > 0".into());
        }
        if self.failover_retry_interval.is_zero() {
            return Err("mic_arbitrator: failover_retry_interval must be > 0".into());
        }
        Ok(())
    }
}

// MARK: Internal helpers + state

/// Per-block RMS.  Used both for the per-channel EMA
/// inside the arbitrator's `pick_slot` and by tests to
/// verify a captured signal's energy.  The f64 accumulator
/// preserves precision for long blocks at low amplitudes.
pub fn block_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// EMA alpha for a block of `block_dur` against an RMS window of
/// `window`.  `alpha = 1 - exp(-block_dur / window)`.  `window == 0`
/// disables smoothing (alpha = 1).
fn ema_alpha(block_dur: Duration, window: Duration) -> f32 {
    if window.is_zero() {
        return 1.0;
    }
    let ratio = block_dur.as_secs_f32() / window.as_secs_f32();
    1.0 - (-ratio).exp()
}

/// How long between two consecutive open-failure warnings before
/// we'll log another (per arbitrator instance, not per candidate).
/// Keeps `journalctl` readable when a USB unplug lasts 5 minutes.
const WARN_INTERVAL: Duration = Duration::from_secs(30);

/// Per-effective-whitelist-slot RMS state.  Per-slot, not per-
/// device-channel: when whitelist is `[0, 2]`, slot 0 -> device
/// channel 0, slot 1 -> device channel 2.
#[derive(Clone, Copy, Debug, Default)]
struct SlotState {
    /// EMA-smoothed RMS for this slot.
    rms: f32,
}

/// Internal mutable state of the arbitrator's run loop.  Packed
/// into a struct so the boot/teardown lifecycle is explicit and
/// scratch buffers are pre-allocated once per source.
struct ArbitratorState {
    active: Option<ActiveSource>,
    /// Cached snapshot of the active source's effective whitelist,
    /// captured at `boot()`.  Avoids per-period
    /// `active_src.effective_whitelist().to_vec()` (a small alloc
    /// chain through `Option<&ActiveSource>`'s borrow scope).
    /// Cleared by `tear_down()`.
    cached_whitelist: Vec<u16>,
    /// Cached `active_src.channels() as usize`.
    cached_n_channels: usize,
    /// Cached `active_src.rate()`.  Used when a source returns a
    /// short read and we need to compute the EMA alpha for the
    /// actual frame count; reading from a field avoids re-borrowing
    /// through `state.active`.
    cached_rate: u32,
    /// Cached `active_src.period_size()`.  Normal reads match this;
    /// ALSA can still return a smaller `Ok(frames)` on hot-unplug /
    /// partial-read edges, and those use a frame-accurate alpha.
    cached_period_frames: usize,
    /// EMA alpha for the source's `period_size / rate` block duration
    /// against `cfg.rms_window`, computed once at `boot()` and used
    /// for the overwhelmingly common full-period read.  Short reads
    /// recompute alpha from their actual frame count so partial ALSA
    /// reads do not over-weight the EMA.
    cached_alpha: f32,
    /// Per-slot RMS state.  `len == cached_whitelist.len()`.
    per_slot: Vec<SlotState>,
    /// Per-slot streaming resampler.  `Some` iff the source rate
    /// differs from [`SampleRate::VALUE`].  ALL `Some` slots
    /// are processed each period (so FIR state stays current); only
    /// the active slot's output is drained to the writer.
    resamplers: Vec<Option<Streaming>>,
    /// Slot index of the currently active channel.
    active_slot: Option<usize>,
    /// Wall-clock at last channel switch.  Drives the dwell check.
    last_switch_at: Option<Instant>,
    /// Wall-clock at last successful read.  Drives the
    /// mic-failover-after timer for `FirstAvailable`.
    last_data_at: Instant,
    /// Pre-allocated interleaved buffer: `period_size *
    /// channels` floats.  Re-sized at each `boot`; capacity persists.
    interleaved_scratch: Vec<f32>,
    /// Pre-allocated FLAT per-slot demuxed scratch.
    /// Layout: `slot * cached_period_frames + frame`, total
    /// length `n_slots * cached_period_frames`.  Short
    /// reads only fill the first `frames` of each slot's
    /// stride and leave a stale tail past `frames` (never
    /// read).  The flat layout avoids the outer-`Vec`
    /// pointer chase a `Vec<Vec<f32>>` would incur.
    slot_scratch_flat: Vec<f32>,
    /// Pre-allocated per-slot sum-of-squares accumulator
    /// (`len == per_slot.len()`); reset to 0 each period.
    /// f32 precision rationale: see
    /// [`single_pass_demux_and_rms`].
    sum_sq_scratch: Vec<f32>,
    /// Pre-allocated drain buffer for the active slot's resampler
    /// output.  Reused across periods so the hot path is alloc-free.
    out_scratch: Vec<f32>,
}

impl ArbitratorState {
    fn new() -> Self {
        Self {
            active: None,
            cached_whitelist: Vec::new(),
            cached_n_channels: 0,
            cached_rate: 0,
            cached_period_frames: 0,
            cached_alpha: 1.0,
            per_slot: Vec::new(),
            resamplers: Vec::new(),
            active_slot: None,
            last_switch_at: None,
            last_data_at: Instant::now(),
            interleaved_scratch: Vec::new(),
            slot_scratch_flat: Vec::new(),
            sum_sq_scratch: Vec::new(),
            out_scratch: Vec::new(),
        }
    }

    fn active_id(&self) -> Option<&MicId> {
        self.active.as_ref().map(|s| s.id())
    }

    /// Wire up state for a freshly-opened source.  Allocates
    /// per-slot scratch + (conditionally) per-slot resamplers, and
    /// caches the whitelist/channel-count/rate/alpha so the hot path
    /// reads them as struct fields rather than borrowing through
    /// `state.active`.
    fn boot(&mut self, src: ActiveSource, cfg: &MicArbitratorConfig) {
        let n_channels = src.channels() as usize;
        let period = src.period_size();
        let rate = src.rate();
        let needs_resample = rate != SampleRate::VALUE;

        // Cache static-for-this-source values up front so
        // `process_period` doesn't need to borrow through
        // `state.active`.
        self.cached_whitelist.clear();
        self.cached_whitelist
            .extend_from_slice(src.effective_whitelist());
        self.cached_n_channels = n_channels;
        self.cached_rate = rate;
        self.cached_period_frames = period;
        let nominal_block = Duration::from_secs_f64(period as f64 / rate as f64);
        self.cached_alpha = ema_alpha(nominal_block, cfg.rms_window);
        let n_slots = self.cached_whitelist.len();

        self.per_slot.clear();
        self.per_slot.resize(n_slots, SlotState::default());
        self.resamplers.clear();
        self.resamplers.extend((0..n_slots).map(|_| {
            if needs_resample {
                Some(Streaming::new(rate, SampleRate::VALUE))
            } else {
                None
            }
        }));
        // Flat scratch sized at `n_slots * period`.  Stride is the
        // constant `period` so per-slot offsets stay loop-invariant
        // through `process_period`.  Short reads (`frames < period`)
        // only fill the first `frames` of each per-slot stride.
        self.slot_scratch_flat.clear();
        self.slot_scratch_flat.resize(n_slots * period, 0.0);
        self.sum_sq_scratch.clear();
        self.sum_sq_scratch.resize(n_slots, 0.0);
        self.interleaved_scratch.clear();
        self.interleaved_scratch.resize(period * n_channels, 0.0);
        self.out_scratch.clear();
        self.active_slot = None;
        self.last_switch_at = None;
        self.last_data_at = Instant::now();
        self.active = Some(src);
    }

    fn tear_down(&mut self) {
        // Drop the source first; it owns the underlying PCM /
        // mock-source state.  Per-slot scratch capacity persists
        // for the next `boot` to avoid re-allocation churn on
        // back-to-back hot-plug events.  Cached fields are reset
        // to keep `state.active.is_some() == cached_*-populated`.
        self.active = None;
        self.cached_whitelist.clear();
        self.cached_n_channels = 0;
        self.cached_rate = 0;
        self.cached_period_frames = 0;
        self.cached_alpha = 1.0;
        self.per_slot.clear();
        self.resamplers.clear();
        self.active_slot = None;
        self.last_switch_at = None;
    }
}

/// Top-of-loop policy -> desired-candidate-index resolution.
///
/// `FirstAvailable` is **sticky**: if there's already an active
/// source AND it's still in the candidates list, return its index.
/// Only when there's no active source (boot or post-failover) does
/// `FirstAvailable` pick `candidates[0]`.  This prevents the loop
/// from flap-tearing the working source every iteration just
/// because the operator declared a different first candidate.
fn resolve_desired_idx(
    policy: &MicSelection,
    candidates: &[MicCandidate],
    current_active: Option<&MicId>,
) -> Option<usize> {
    match policy {
        MicSelection::Fixed { id } => candidates.iter().position(|c| &c.id == id),
        MicSelection::FirstAvailable => {
            if let Some(active) = current_active
                && let Some(idx) = candidates.iter().position(|c| &c.id == active)
            {
                return Some(idx);
            }
            if candidates.is_empty() { None } else { Some(0) }
        }
    }
}

/// Walk candidates from `start_idx` onward, return the first that
/// opens.  Used by `FirstAvailable` after the preferred candidate
/// failed to open.  Logs each failed open at warn (rate-limited
/// across the arbitrator's lifetime).
fn open_starting_from(
    candidates: &[MicCandidate],
    start_idx: usize,
    stop: Arc<AtomicBool>,
    last_warn_at: &mut Option<Instant>,
) -> Option<(usize, ActiveSource)> {
    for (i, cand) in candidates.iter().enumerate().skip(start_idx) {
        match open_source(cand, stop.clone()) {
            Ok(src) => return Some((i, src)),
            Err(e) => rate_limited_warn(last_warn_at, "open failed", &cand.id, &e),
        }
    }
    None
}

fn rate_limited_warn(
    last: &mut Option<Instant>,
    what: &'static str,
    id: &MicId,
    err: &dyn std::fmt::Display,
) {
    let now = Instant::now();
    let should = last.is_none_or(|t| now.duration_since(t) > WARN_INTERVAL);
    if should {
        tracing::warn!(
            target: "audio_io.mic_arbitrator",
            id = %id,
            err = %err,
            "{what}",
        );
        *last = Some(now);
    }
}

/// Reset every per-channel resampler's FIR history + accumulator.
/// Called by the run loop after a successful ALSA `try_recover`
/// (xrun / suspend) -- at that point the next `readi` returns
/// samples from a fresh capture session disconnected from each
/// resampler's prior FIR history.
///
/// Without the reset, the resampler would convolve new input
/// samples against a phantom tail of pre-xrun samples for ~3 ms
/// (sinc_len/2 ~= 128 samples at 44.1 k).  The reset substitutes a
/// zero-history transient of the same duration.  Audibly equivalent
/// (an xrun is already a click), but the discontinuity is now
/// honest: the post-recovery output starts from a known-clean state
/// rather than drifting through a phantom that no longer
/// corresponds to any real input.
///
/// `None` resamplers (native-rate sources) are skipped
/// via `flatten()`.  Per-slot EMA RMS + active-slot pick
/// are **deliberately preserved**: they carry signal
/// state that survives the xrun (e.g. "ch1 was loudest"),
/// and a reset there would discard the dwell timer and
/// could cause an audible channel switch on top of the
/// xrun click.
///
/// Lifted out of the run loop so it stays reachable from
/// a platform-independent unit test; the only production
/// caller is the ALSA recovery arm.
#[cfg_attr(not(all(target_os = "linux", feature = "alsa-real")), allow(dead_code))]
fn reset_per_channel_fir(resamplers: &mut [Option<Streaming>]) {
    for r in resamplers.iter_mut().flatten() {
        r.reset_after_discontinuity();
    }
}

/// Single-pass demux + per-slot sum-of-squares.  Each interleaved
/// sample for a whitelisted channel is read exactly once; per-slot
/// PCM output is written into the corresponding `stride`-sized
/// window of `slot_scratch_flat` (no caller-side pre-clear needed),
/// and `sum_sq_scratch[slot]` is overwritten with the f32
/// sum-of-squares for that period (no pre-zero needed).
///
/// Non-finite samples (NaN / +/-Inf) are clamped to 0.0 before being
/// written to the slot scratch and accumulated.  Without the clamp:
///
/// * the per-slot EMA RMS would absorb a NaN and stay NaN forever
///   (every subsequent `alpha * NaN + ... = NaN`), permanently
///   poisoning that slot for arbitration purposes;
/// * the audio buffer would propagate NaN to downstream consumers
///   (opus encoder, inference engine), confusing both.
///
/// In-domain f32 samples from real audio sources are always
/// finite, so the branch is essentially never taken -- branch
/// prediction makes the cost ~zero.  The clamp is defensive against
/// buggy USB drivers, future float ALSA backends, and any other
/// upstream that could leak a non-finite value.
///
/// ## Loop ordering
///
/// Outer over slots, inner over frames.  The natural "outer = frames,
/// inner = channels" ordering reads `interleaved` contiguously but
/// scatters writes across the per-slot scratches and forces a
/// memory read-modify-write of `sum_sq_scratch[slot]` per inner
/// iter.  Inverting makes `slot` loop-invariant in the hot inner
/// loop, so:
///
/// - `dst: &mut [f32]` is borrowed once per slot from
///   `slot_scratch_flat`: no per-iter bounds check on the
///   outer slice and no outer-`Vec` pointer chase.
/// - `sum_sq` accumulates in a local f32 register: no
///   per-iter memory write, NEON-eligible (f32 vectorises;
///   f64 does not).
/// - `dst[f] = s` is an index-store: no `Vec::push` length
///   / cap branch on every sample.
///
/// The strided read on `interleaved` is L1-resident for typical
/// period sizes (16 KB on a 4-channel 1024-frame block).
///
/// Hot path.  Inlined into the run loop.
#[inline]
fn single_pass_demux_and_rms(
    interleaved: &[f32],
    n_channels: usize,
    whitelist: &[u16],
    slot_scratch_flat: &mut [f32],
    sum_sq_scratch: &mut [f32],
    frames: usize,
    stride: usize,
) {
    debug_assert!(frames <= stride);
    debug_assert_eq!(slot_scratch_flat.len(), whitelist.len() * stride);
    debug_assert_eq!(sum_sq_scratch.len(), whitelist.len());
    debug_assert_eq!(interleaved.len(), frames * n_channels);
    for (slot_idx, &ch) in whitelist.iter().enumerate() {
        let offset = slot_idx * stride;
        let dst = &mut slot_scratch_flat[offset..offset + frames];
        let ch = ch as usize;
        let mut sum_sq: f32 = 0.0;
        for f in 0..frames {
            // SAFETY-BY-CONSTRUCTION: ch < n_channels was enforced
            // at source open time (Alsa intersected; Mock validated
            // by `MicCandidate::validate`).  Bounds-checked indexing
            // costs ~nothing here vs the resample work that
            // dominates the period -- keep the check.
            let raw = interleaved[f * n_channels + ch];
            let s = if raw.is_finite() { raw } else { 0.0 };
            dst[f] = s;
            sum_sq += s * s;
        }
        sum_sq_scratch[slot_idx] = sum_sq;
    }
}

/// EMA alpha for this read.  Full-period reads use the alpha cached
/// at `boot()`; short reads recompute from the actual frame count so
/// the EMA's time constant remains wall-clock accurate.
#[inline]
fn alpha_for_frames(
    frames: usize,
    nominal_period_frames: usize,
    rate: u32,
    cached_nominal_alpha: f32,
    window: Duration,
) -> f32 {
    debug_assert!(frames > 0, "process_period only handles non-zero reads");
    debug_assert!(nominal_period_frames > 0, "source period must be non-zero");
    debug_assert!(rate > 0, "source rate must be non-zero");
    if frames == nominal_period_frames {
        cached_nominal_alpha
    } else {
        let actual_block = Duration::from_secs_f64(frames as f64 / rate as f64);
        ema_alpha(actual_block, window)
    }
}

/// Choose the active slot for this period, given current per-slot
/// RMS, the channel policy, hysteresis, and dwell.  Pure function
/// for unit-testability -- called from the run loop with the
/// state's mutable fields passed in by reference.  Argument count is
/// the price of pure-fn testability over a bag-of-state struct.
#[allow(clippy::too_many_arguments)]
fn pick_slot(
    policy: &ChannelSelection,
    per_slot: &[SlotState],
    whitelist: &[u16],
    current_slot: Option<usize>,
    last_switched_at: Option<Instant>,
    now: Instant,
    hysteresis_linear: f32,
    dwell: Duration,
) -> Option<usize> {
    if per_slot.is_empty() {
        return None;
    }
    if let ChannelSelection::Fixed { channel } = policy
        && let Some(idx) = whitelist.iter().position(|&w| w == *channel)
    {
        return Some(idx);
    }
    // ChannelSelection::Auto, OR Fixed-but-channel-not-in-whitelist
    // (silently fall back to Auto rather than emit silence).
    let (loudest_idx, loudest_rms) = per_slot
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.rms))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty above");

    match current_slot {
        None => Some(loudest_idx),
        Some(active_idx) if active_idx == loudest_idx => Some(active_idx),
        Some(active_idx) => {
            let active_rms = per_slot[active_idx].rms;
            let above_threshold = loudest_rms > active_rms * hysteresis_linear;
            let dwell_satisfied = match last_switched_at {
                None => true,
                Some(t) => now.saturating_duration_since(t) >= dwell,
            };
            if above_threshold && dwell_satisfied {
                Some(loudest_idx)
            } else {
                Some(active_idx)
            }
        }
    }
}

/// Process one successfully-read period: demux + RMS + pick slot +
/// resample (if needed) + write active slot's output to the audio
/// buffer.  Called only when the active source returned a non-zero
/// frame count.
fn process_period(
    state: &mut ArbitratorState,
    writer: &mut Writer,
    frames: usize,
    channel_policy: &ChannelSelection,
    cfg: &MicArbitratorConfig,
    hysteresis_linear: f32,
) {
    // `state.active` is intentionally NOT borrowed here:
    // the mutable borrows of `slot_scratch_flat` /
    // `sum_sq_scratch` / `per_slot` / `resamplers` /
    // `out_scratch` need to interleave, which an outer
    // `&state.active` borrow would prevent.
    let n_channels = state.cached_n_channels;
    let n_slots = state.cached_whitelist.len();
    let stride = state.cached_period_frames;
    let interleaved = &state.interleaved_scratch[..frames * n_channels];

    // MARK: Demux + RMS (single pass)
    // Writes the first `frames` entries of each slot's `stride`-sized
    // window in `slot_scratch_flat` and overwrites `sum_sq_scratch[s]`
    // directly; no caller-side pre-clear needed.
    single_pass_demux_and_rms(
        interleaved,
        n_channels,
        &state.cached_whitelist,
        &mut state.slot_scratch_flat,
        &mut state.sum_sq_scratch,
        frames,
        stride,
    );

    // MARK: EMA-update RMS state
    // Full-period reads hit the cached alpha.  ALSA short reads are
    // rare, but when they happen (e.g. hot-unplug partial read), use
    // a frame-accurate alpha so the EMA does not over-weight that
    // partial block.
    let alpha = alpha_for_frames(
        frames,
        state.cached_period_frames,
        state.cached_rate,
        state.cached_alpha,
        cfg.rms_window,
    );
    for slot in 0..n_slots {
        let block_rms = (state.sum_sq_scratch[slot] / frames as f32).sqrt();
        state.per_slot[slot].rms = alpha * block_rms + (1.0 - alpha) * state.per_slot[slot].rms;
    }

    // MARK: Pick active slot
    let now = Instant::now();
    let prev_slot = state.active_slot;
    let new_slot = pick_slot(
        channel_policy,
        &state.per_slot,
        &state.cached_whitelist,
        prev_slot,
        state.last_switch_at,
        now,
        hysteresis_linear,
        cfg.dwell,
    );
    if new_slot != prev_slot {
        state.last_switch_at = Some(now);
        if let (Some(p), Some(n)) = (prev_slot, new_slot) {
            tracing::info!(
                target: "audio_io.mic_arbitrator",
                from_slot = p,
                from_channel = state.cached_whitelist[p],
                to_slot = n,
                to_channel = state.cached_whitelist[n],
                "channel switched",
            );
        } else if let Some(n) = new_slot {
            tracing::info!(
                target: "audio_io.mic_arbitrator",
                to_slot = n,
                to_channel = state.cached_whitelist[n],
                "channel selected (initial)",
            );
        }
    }
    state.active_slot = new_slot;

    let Some(active_idx) = state.active_slot else {
        return; // shouldn't happen with non-empty whitelist
    };

    // MARK: Emit through the writer
    let needs_resample = state.resamplers[active_idx].is_some();
    if !needs_resample {
        // Native rate.  Channel switch is glitch-free.
        let active_offset = active_idx * stride;
        writer.push(&state.slot_scratch_flat[active_offset..active_offset + frames]);
        publish_timing_anchor(writer, cfg);
    } else {
        // Resample-rate path.  In Auto mode (or Fixed-but-channel-not-
        // in-whitelist, where `pick_slot` falls back to Auto and the
        // active slot can change period-to-period), feed every slot's
        // resampler so all FIR states stay current -- a future channel
        // switch is glitch-free across the FIR group delay.
        //
        // In *truly* Fixed mode (Fixed { channel } with that channel
        // present in the whitelist), `active_idx` is locked for the
        // source's lifetime, so feeding non-active resamplers is pure
        // waste -- skip them to save ~(n_slots - 1)x sinc work.
        //
        // Skipped resamplers are reset each period rather than left
        // alone: `Streaming` retains a partial-chunk `accum` between
        // calls when `period_size != chunk_size` (e.g. a 512-frame
        // mock period feeding a 1024-sample chunker), and that
        // residue would otherwise glue pre-skip audio onto the first
        // post-flip chunk if the policy ever flipped Fixed->Auto.
        // Reset is idempotent (~200 ns: zeros rubato's input ring
        // ~=2.4 k floats + recomputes input/output lengths) versus
        // ~150 us per `process()` call -- net win for n_slots >= 2.
        // The only audible consequence is that the first post-flip
        // switch carries the FIR group-delay startup transient
        // (~6 ms @ 44.1 k), which is the documented cost of a manual
        // mode change.
        let truly_fixed = match channel_policy {
            ChannelSelection::Fixed { channel } => state.cached_whitelist.contains(channel),
            ChannelSelection::Auto => false,
        };
        for slot in 0..n_slots {
            if truly_fixed && slot != active_idx {
                let r = state.resamplers[slot].as_mut().expect("alloc'd in boot");
                r.reset_after_discontinuity();
                continue;
            }
            let r = state.resamplers[slot].as_mut().expect("alloc'd in boot");
            let slot_offset = slot * stride;
            // Panic-then-abort: see the `catch_unwind +
            // process::abort` wrapper in [`MicArbitrator::start`].
            r.process(&state.slot_scratch_flat[slot_offset..slot_offset + frames])
                .expect("Streaming::process invariant break -- see StreamingResampleError docs");
            if slot == active_idx {
                state.out_scratch.clear();
                r.drain_output_into(&mut state.out_scratch);
                if !state.out_scratch.is_empty() {
                    // Chunk by `max_push_len()` to honour the
                    // writer's safety-margin invariant under any
                    // future ring resize.  Today this loop runs
                    // once per drain at the canonical capacity.
                    let max_push = writer.max_push_len();
                    for chunk in state.out_scratch.chunks(max_push) {
                        writer.push(chunk);
                        publish_timing_anchor(writer, cfg);
                    }
                }
            } else {
                r.drop_output();
            }
        }
    }
}

/// Publish a fresh
/// [`crate::common::time::BufferTimingAnchor`] with the
/// post-push head and the current monotonic time, when the
/// arbitrator is configured with a shared anchor cell.  No-op
/// when `cfg.timing_anchor` is `None` (the no-anchor
/// fallback path: consumers stamp `CaptureTime::now()` at
/// emit time).
///
/// Cost when active: one `Arc::new` allocation + one
/// `ArcSwap::store`, plus the `CaptureTime::now()` syscall
/// (~10-50 ns).  Mic period is 5-23 ms typical; the
/// allocation is comfortably absorbed by mimalloc at the
/// producer's ~50-200 Hz update rate.
///
/// `sample_rate_hz` is the canonical buffer rate
/// (`SampleRate::VALUE` = 44_100); the producer-side resampler
/// has already converted device-rate audio to canonical
/// before this point, so consumers reading the anchor see
/// 44.1 kHz consistently regardless of the device's native
/// rate.
#[inline]
fn publish_timing_anchor(writer: &Writer, cfg: &MicArbitratorConfig) {
    let Some(cell) = cfg.timing_anchor.as_ref() else {
        return;
    };
    let anchor = crate::common::time::BufferTimingAnchor {
        head_pos: writer.head_pos(),
        captured_at: crate::common::time::CaptureTime::now(),
        sample_rate_hz: SampleRate::VALUE,
    };
    cell.store(std::sync::Arc::new(anchor));
}

// MARK: Public API

/// Running mic arbitrator thread.  Drop or call
/// [`MicArbitrator::stop`] to terminate.  Safe to drop the
/// [`crate::audio_buffer::AudioBuffer`] handle after
/// starting; [`Writer`] keeps the underlying ring alive.
#[derive(Debug)]
pub struct MicArbitrator {
    handle: Option<thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl MicArbitrator {
    /// Spin up the arbitrator thread.
    ///
    /// `settings` is the live policy + candidates surfaced
    /// via the [`MicSettingsStore`] trait.  Production
    /// wires [`crate::config::MicSettingsCell`]; tests
    /// substitute a mock cell or the [`ArcSwapStore`]
    /// adapter for direct [`arc_swap::ArcSwap`] shapes.
    /// The arbitrator reads via `settings.snapshot()` once
    /// per loop iteration; a successful API mutation
    /// becomes visible within ~one period.
    ///
    /// Panics if `cfg` fails [`MicArbitratorConfig::validate`]
    /// -- that is the contract: callers must supply a
    /// validated config, and the constructor enforces it
    /// intrinsically so a fresh call site or test fixture
    /// cannot bypass the gate.  Production literals are
    /// reviewed; operator-tunable fields are validated
    /// upstream (config schema + per-section `validate()`)
    /// before reaching here, so the panic is a debugging
    /// aid for development bugs rather than a runtime
    /// failure mode.
    pub fn start(
        writer: Writer,
        settings: Arc<dyn MicSettingsStore>,
        cfg: MicArbitratorConfig,
    ) -> Self {
        cfg.validate().unwrap_or_else(|e| {
            panic!("MicArbitrator::start: invalid MicArbitratorConfig: {e}");
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        // `run_loop` consumes `cfg`; snapshot the two
        // sched-config fields here so the spawn closure can
        // apply them after the move.
        let sched_pin = cfg.sched_pin;
        let sched_priority = cfg.sched_priority;
        let handle = thread::Builder::new()
            .name("mic-arbitrator".into())
            .spawn(move || {
                // Pin first: an unpinned RT thread can be
                // rebalanced by the kernel, so pinning
                // makes the realtime priority deterministic.
                // Both calls are best-effort; see the field
                // docs on [`MicArbitratorConfig`].
                if let Some(core) = sched_pin
                    && let Err(e) = crate::sched::pin_to_core(core)
                {
                    tracing::warn!(
                        target: "audio_io",
                        err = %e,
                        core = core,
                        "mic-arbitrator pin_to_core failed; continuing on default placement",
                    );
                }
                if let Some(prio) = sched_priority
                    && let Err(e) = crate::sched::set_realtime(prio)
                {
                    tracing::warn!(
                        target: "audio_io",
                        err = %e,
                        priority = prio,
                        "mic-arbitrator set_realtime failed (likely missing CAP_SYS_NICE); \
                         continuing at SCHED_OTHER",
                    );
                }
                // Fatal-thread policy: catch any panic,
                // log a structured record, then abort the
                // process so the supervisor restarts.  The
                // mic arbitrator produces all downstream
                // audio timing; without abort-on-panic the
                // audio_buffer head would freeze and
                // operators would see stale inference
                // frames flowing past silent failure.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_loop(writer, settings, cfg, stop_clone)
                }));
                if result.is_err() {
                    tracing::error!(
                        target: "audio_io",
                        "mic-arbitrator panicked; aborting process so operator can restart",
                    );
                    std::process::abort();
                }
            })
            .expect("spawn mic-arbitrator");
        Self {
            handle: Some(handle),
            stop,
        }
    }

    /// Signal the thread to stop without waiting for it to
    /// finish.  Idempotent.  Used by the daemon's shutdown
    /// sequence to silence the producer BEFORE draining
    /// audio consumers: once this call returns, the run
    /// loop exits within ~one capture period (~12 ms) and
    /// stops appending to the audio + lag buffers, so
    /// consumers drain into a quiet pipeline rather than
    /// racing against ongoing input.
    pub fn signal_stop(&self) {
        signal_only(&self.stop, self.handle.as_ref());
    }

    /// Signal the thread to stop and wait for it to finish.
    /// Idempotent with `signal_stop`: the second `store(true)` is a
    /// no-op so callers can `signal_stop()` early and then `stop()`
    /// later to join.
    pub fn stop(mut self) {
        signal_and_join(&self.stop, &mut self.handle);
    }
}

impl Drop for MicArbitrator {
    fn drop(&mut self) {
        // Idempotent: if `stop()` already consumed the handle this
        // is a no-op.
        signal_and_join(&self.stop, &mut self.handle);
    }
}

/// Set the stop flag and wake any park.  Does not join.
fn signal_only(stop: &AtomicBool, handle: Option<&thread::JoinHandle<()>>) {
    stop.store(true, Ordering::Release);
    if let Some(h) = handle {
        h.thread().unpark();
    }
}

/// Set the stop flag, wake any park, and join.  The `unpark` is what
/// makes shutdown prompt when the run loop is in the no-source
/// `park_timeout` branch -- a `thread::sleep` there would block teardown
/// for up to `failover_retry_interval` (1 s default).
fn signal_and_join(stop: &AtomicBool, handle: &mut Option<thread::JoinHandle<()>>) {
    stop.store(true, Ordering::Release);
    if let Some(h) = handle.take() {
        h.thread().unpark();
        let _ = h.join();
    }
}

fn run_loop(
    mut writer: Writer,
    settings: Arc<dyn MicSettingsStore>,
    cfg: MicArbitratorConfig,
    stop: Arc<AtomicBool>,
) {
    let hysteresis_linear = cfg.hysteresis_linear();
    let mut state = ArbitratorState::new();
    let mut last_warn_at: Option<Instant> = None;

    while !stop.load(Ordering::Acquire) {
        // `snapshot()` is a vtable dispatch + one Arc
        // load (~5 ns + ~1 ns).
        let snap = settings.snapshot();

        // Resolve desired source
        let desired_idx = resolve_desired_idx(
            &snap.policy.mic,
            &snap.catalogue.candidates,
            state.active_id(),
        );

        let need_switch = match (state.active_id(), desired_idx) {
            (None, None) => false,
            (None, Some(_)) => true,
            (Some(_), None) => true,
            (Some(active_id), Some(d)) => *active_id != snap.catalogue.candidates[d].id,
        };

        if need_switch {
            state.tear_down();
            if let Some(idx) = desired_idx {
                let cand = &snap.catalogue.candidates[idx];
                match open_source(cand, stop.clone()) {
                    Ok(src) => {
                        tracing::info!(
                            target: "audio_io.mic_arbitrator",
                            id = %cand.id,
                            "active mic opened",
                        );
                        state.boot(src, &cfg);
                    }
                    Err(OpenError::AlsaNotCompiledIn(id)) => {
                        rate_limited_warn(
                            &mut last_warn_at,
                            "alsa-real not compiled in",
                            &id,
                            &"build with --features alsa-real on Linux",
                        );
                        // Walk to alternates if FirstAvailable.
                        if matches!(&snap.policy.mic, MicSelection::FirstAvailable)
                            && let Some((_, src)) = open_starting_from(
                                &snap.catalogue.candidates,
                                idx + 1,
                                stop.clone(),
                                &mut last_warn_at,
                            )
                        {
                            state.boot(src, &cfg);
                        }
                    }
                    Err(e) => {
                        let id = match &e {
                            OpenError::InvalidCandidate(id, _) => id.clone(),
                            OpenError::AlsaNotCompiledIn(id) => id.clone(),
                            OpenError::SourceUnavailable(id, _) => id.clone(),
                        };
                        rate_limited_warn(&mut last_warn_at, "open failed", &id, &e);
                        if matches!(&snap.policy.mic, MicSelection::FirstAvailable)
                            && let Some((_, src)) = open_starting_from(
                                &snap.catalogue.candidates,
                                idx + 1,
                                stop.clone(),
                                &mut last_warn_at,
                            )
                        {
                            state.boot(src, &cfg);
                        }
                    }
                }
            }
        }

        // Mic-level failover timer for FirstAvailable
        // If the active source has been silent (no successful read)
        // for longer than the threshold, tear it down.  The next
        // iteration will re-resolve and pick a fresh candidate.
        // Fixed policy ignores the timer (it stays committed).
        if state.active.is_some()
            && matches!(&snap.policy.mic, MicSelection::FirstAvailable)
            && Instant::now().saturating_duration_since(state.last_data_at) > cfg.mic_failover_after
        {
            tracing::warn!(
                target: "audio_io.mic_arbitrator",
                id = ?state.active_id(),
                "no fresh data within mic_failover_after; failing over",
            );
            state.tear_down();
            // Don't sleep -- let the next iter try to open another
            // candidate immediately.
            continue;
        }

        // No source open: park and retry
        let Some(active_src) = state.active.as_mut() else {
            // Diagnose the persistent-no-source case for the operator.
            // `Fixed { id }` referencing a missing catalogue entry
            // resolves to `desired_idx = None`, the `need_switch`
            // arm doesn't open anything, and we end up parking here
            // every iteration -- silent without this.  Rate-limited
            // to once per 30 s so a long-standing typo doesn't spam
            // the journal but is still surfaced periodically.  Other
            // routes to this branch (FirstAvailable with all opens
            // failed; AlsaNotCompiledIn) already log their own
            // warns inside the `need_switch` arm above.
            if desired_idx.is_none()
                && let MicSelection::Fixed { id } = &snap.policy.mic
            {
                rate_limited_warn(
                    &mut last_warn_at,
                    "fixed mic id not in catalogue; staying inert",
                    id,
                    &"add the candidate or change policy",
                );
            }
            // `park_timeout` instead of `thread::sleep` so the
            // arbitrator's own `stop()` (or `Drop`) can `unpark` us
            // for prompt teardown -- `failover_retry_interval`
            // defaults to 1 s, which would otherwise block shutdown.
            // Spurious wakeups just round-trip the loop top, where
            // the `stop` check exits cleanly.  Top-of-loop already
            // saw `stop == false`, but the `unpark` token is held
            // until consumed, so a `stop()` call between the load
            // and the park still wakes us promptly.
            thread::park_timeout(cfg.failover_retry_interval);
            continue;
        };

        // The read dispatches through the [`MicSource`] trait into
        // the source's `read_interleaved`, returning the unified
        // [`ReadOutcome`].  Per-source recovery (ALSA's `try_recover`)
        // stays on a concrete match arm against [`ActiveSource`]
        // because mock has no analogue.  The trait method's per-period
        // bounded poll timeout (2x period at the negotiated rate) is
        // computed inside `AlsaSource::read_interleaved` itself rather
        // than passed in here.
        use crate::audio_io::source::{MicSource as _, ReadError, ReadOutcome};
        let frames_read = match active_src.read_interleaved(&mut state.interleaved_scratch) {
            Ok(ReadOutcome::Frames(n)) => n.get(),
            Ok(ReadOutcome::Timeout) | Ok(ReadOutcome::StopRequested) => {
                // Top-of-loop reads `stop` and exits if signalled;
                // otherwise we re-enter the read on the same source.
                continue;
            }
            Ok(ReadOutcome::EndOfStream) => {
                // ALSA `Frames(0)` is `EndOfStream`, which
                // tears down the source instead of silently
                // spinning on a dead PCM.
                tracing::warn!(
                    target: "audio_io.mic_arbitrator",
                    id = %active_src.id(),
                    "source returned EndOfStream; tearing down",
                );
                state.tear_down();
                continue;
            }
            Err(read_err) => match read_err {
                #[cfg(all(target_os = "linux", feature = "alsa-real"))]
                ReadError::Alsa(e) => {
                    let id = active_src.id().clone();
                    tracing::warn!(
                        target: "audio_io.mic_arbitrator",
                        id = %id,
                        err = %e,
                        "ALSA read error; attempting recovery",
                    );
                    if let ActiveSource::Alsa(a) = active_src {
                        if a.try_recover(e).is_err() {
                            state.tear_down();
                        } else {
                            reset_per_channel_fir(&mut state.resamplers);
                        }
                    }
                    continue;
                }
                ReadError::Mock(infallible) => match infallible {},
            },
        };

        state.last_data_at = Instant::now();

        process_period(
            &mut state,
            &mut writer,
            frames_read,
            &snap.policy.channel,
            &cfg,
            hysteresis_linear,
        );
    }

    state.tear_down();
}
