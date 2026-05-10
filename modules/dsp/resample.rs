//! Streaming sinc resampler wrapper.
//!
//! `crate::dsp::resample` is the workspace's single source
//! of truth for sinc-resampler configuration: [`sinc_resampler`]
//! holds the canonical [`rubato::SincInterpolationParameters`]
//! plus the [`rubato::Async::new_sinc`] arguments that the
//! classifier was validated against, and `preproc::wav_io`
//! re-exports those same symbols so training and inference
//! paths share the exact constants.  Mismatched params would
//! degrade top-1 accuracy silently; keeping one definition
//! makes drift impossible by construction.
//!
//! The neutral `dsp` umbrella is shared across capture,
//! preprocessing, and codec so none of them needs to depend
//! on the others to reach the canonical sinc params.
//!
//! The reference resampler in `preproc::wav_io::to_waveform`
//! is configured for one-shot use: it calls `r.reset()`
//! before processing each WAV, which is correct for batch
//! fine-tune but catastrophic for continuous streams --
//! `reset()` mid-stream zeros the FIR history and produces
//! a 256-sample click.
//!
//! [`Streaming`] (this module) instead constructs a
//! sinc-async resampler once per stream and feeds it
//! contiguously.  The wrapper handles the fact that rubato
//! wants exactly `input_frames_next()` samples per call: it
//! accumulates partial input until a full chunk is
//! available, then drains.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// Typed failure surface for [`Streaming::process`].
///
/// Each caller picks its own failure handling: `opus_stream`
/// propagates as `OpusError::ResamplerInternal` (closes the
/// subscriber cleanly); `mic_arbitrator` aborts (the audio
/// pipeline can't recover from a resampler invariant break).
///
/// Named distinctly from [`rubato::ResampleError`] (the
/// upstream name); the source-error fields wrap that
/// upstream type directly so a future rubato API change
/// surfaces with full context.
#[derive(Debug, thiserror::Error)]
pub enum StreamingResampleError {
    /// [`InterleavedSlice::new`] rejected the input slice
    /// (length or channel mismatch).
    #[error("rubato input adapter: {source}")]
    InputAdapter {
        #[source]
        source: audioadapter_buffers::SizeError,
    },
    /// [`InterleavedSlice::new_mut`] rejected the output
    /// scratch.
    #[error("rubato output adapter: {source}")]
    OutputAdapter {
        #[source]
        source: audioadapter_buffers::SizeError,
    },
    /// [`rubato::Async::process_into_buffer`] returned `Err`,
    /// typically a chunk-size invariant violation if a
    /// future refactor breaks the `chunk_size` cache.
    #[error("rubato process_into_buffer: {source}")]
    ProcessInto {
        #[source]
        source: rubato::ResampleError,
    },
}

impl crate::common::error::Categorized for StreamingResampleError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        // All variants are daemon-internal pipeline invariant
        // violations, never operator input.
        crate::common::error::ErrorKind::Internal
    }
}

/// Concrete sinc-async resampler type.  rubato 2.0 unified
/// the various `SincFixedIn` / `SincFixedOut` variants under
/// [`rubato::Async<T>`], with the fixed dimension chosen via
/// the [`rubato::FixedAsync`] constructor argument.  Our
/// pipeline is fixed-input-chunk (1024 frames per call).
///
/// `preproc::wav_io::SincResampler` is a re-export of this
/// type so training / inference share the exact same
/// concrete [`rubato::Async<f32>`].
pub type SincResampler = Async<f32>;

/// High-quality sinc-polyphase resampler matched to
/// `scipy.signal.resample_poly` on speech clips.  The
/// [`rubato::Fft`] (synchronous) alternative differs from
/// scipy by up to 0.9 on a +/-1.0 signal and can flip
/// borderline top-1; do **not** use it.
///
/// `preproc::wav_io::sinc_resampler` is a re-export of this
/// function; changing constants here changes them
/// everywhere.
pub fn sinc_resampler(from_sr: u32, to_sr: u32) -> SincResampler {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        // 0.95 of Nyquist: keep ~5 % guard-band against aliasing
        // images while leaving the audible band (up to ~22.8 kHz
        // at 48 kHz output) un-attenuated.  Lower values shave
        // more high-frequency content; higher values let the
        // alias-skirt creep into the passband.  Matches the
        // conservative defaults used by the scipy reference the
        // classifier was trained against.
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        // Polyphase table depth.  Higher values reduce the
        // residual interpolation error between phase taps at the
        // cost of resident memory (sinc_len * factor * 4 bytes ≈
        // 512 KiB at the current settings).  512 is rubato's
        // recommended high-quality value and matches scipy
        // resample_poly's defaults; halving would tighten the
        // memory budget but show up as audible imaging on
        // borderline ratios.
        oversampling_factor: 512,
        window: WindowFunction::BlackmanHarris2,
    };
    Async::<f32>::new_sinc(
        to_sr as f64 / from_sr as f64,
        2.0,
        &params,
        1024,
        1,
        FixedAsync::Input,
    )
    .expect("resampler init")
}

/// Continuous-mode wrapper around [`SincResampler`]
/// ([`rubato::Async<f32>`] in [`rubato::FixedAsync::Input`]
/// mode).  Owns the resampler + accumulators; never resets
/// mid-stream.
///
/// Construction picks up the canonical sinc params from
/// [`sinc_resampler`].
pub struct Streaming {
    resampler: SincResampler,
    /// Partial input chunk awaiting enough samples to feed the
    /// resampler (`input_frames_next()` per call).
    accum: Vec<f32>,
    /// Pending output not yet drained by the caller.
    out: Vec<f32>,
    /// `input_frames_next()` cached at construction.  With
    /// [`rubato::FixedAsync::Input`] and no runtime ratio
    /// changes, `input_frames_next()` is constant -- equal
    /// to the constructor's `chunk_size` argument (1024 for
    /// [`sinc_resampler`]).  Caching dodges a per-iteration
    /// trait-object dispatch.
    ///
    /// INVARIANT: this field MUST stay in sync with
    /// `self.resampler.input_frames_next()`.  The wrapper
    /// does not expose `set_resample_ratio` /
    /// `set_chunk_size`, so the only way it could drift is
    /// if a future refactor adds those, at which point this
    /// cache must be updated alongside the call.
    chunk_size: usize,
    /// Reusable per-call output scratch sized to the
    /// resampler's global maximum.  `output_frames_max()` is
    /// the upper bound over ALL legal ratios within
    /// `max_relative_ratio` (2.0 here), so it's safe across
    /// the lifetime of the resampler.  Allocating once at
    /// construction keeps the hot path alloc-free.
    out_scratch: Vec<f32>,
}

impl Streaming {
    /// Build a streaming resampler from `from_sr` to `to_sr`.
    pub fn new(from_sr: u32, to_sr: u32) -> Self {
        let resampler = sinc_resampler(from_sr, to_sr);
        let chunk_size = resampler.input_frames_next();
        let out_scratch = vec![0.0f32; resampler.output_frames_max()];
        Self {
            resampler,
            // Sized for the worst-case "two consecutive feeds
            // arrive between drains": one full chunk_size waiting
            // to be processed, plus up to chunk_size-1 carry-over
            // from the previous partial fill.  Rounded up to 2 ×
            // chunk_size so the steady state never reallocates.
            accum: Vec::with_capacity(chunk_size * 2),
            out: Vec::new(),
            chunk_size,
            out_scratch,
        }
    }

    /// Feed `input` samples into the resampler.  Returns
    /// the number of new output samples appended to the
    /// internal output buffer (drainable via
    /// [`Self::drain_output_into`]).
    ///
    /// Per chunk we wrap a chunk-sized window of `accum`
    /// and `out_scratch` as [`InterleavedSlice`] adapters
    /// (mono is the borrowed slice itself, zero copy at the
    /// API boundary).  The output scratch is reused across
    /// calls so the hot path is alloc-free.
    ///
    /// Consumed prefixes are coalesced into a single
    /// `accum.drain` at the end of the call.  A per-chunk
    /// `drain(..chunk_size)` would shift the trailing tail
    /// K times when processing K chunks, with total cost
    /// ~`K^2 * chunk_size / 2` element moves; coalescing
    /// makes it one shift of the residue.  For the typical
    /// call with `input.len() ~= chunk_size` (K = 1,
    /// residue = 0), both forms are zero-cost, but the
    /// coalesced form keeps cost bounded if a future caller
    /// passes a multi-period buffer in one go.
    pub fn process(&mut self, input: &[f32]) -> Result<usize, StreamingResampleError> {
        if input.is_empty() {
            return Ok(0);
        }
        // Anchor for the invariant documented on
        // `self.chunk_size`.  If a future refactor adds
        // `set_resample_ratio` / `set_chunk_size` without
        // updating the cache, this fires loudly in debug
        // builds.
        debug_assert_eq!(
            self.chunk_size,
            self.resampler.input_frames_next(),
            "chunk_size cache out of sync with resampler",
        );
        self.accum.extend_from_slice(input);
        let mut produced = 0;
        let mut consumed = 0;
        while consumed + self.chunk_size <= self.accum.len() {
            // Convert each rubato API failure to
            // [`StreamingResampleError`] rather than
            // panicking.  Callers (`opus_stream` graceful,
            // `mic_arbitrator` abort-on-error) pick their
            // own failure handling.
            let in_adapter = InterleavedSlice::new(
                &self.accum[consumed..consumed + self.chunk_size],
                1,
                self.chunk_size,
            )
            .map_err(|source| StreamingResampleError::InputAdapter { source })?;
            let max_out = self.out_scratch.len();
            let mut out_adapter = InterleavedSlice::new_mut(&mut self.out_scratch, 1, max_out)
                .map_err(|source| StreamingResampleError::OutputAdapter { source })?;
            let (n_in, n_out) = self
                .resampler
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                .map_err(|source| StreamingResampleError::ProcessInto { source })?;
            debug_assert_eq!(n_in, self.chunk_size);
            self.out.extend_from_slice(&self.out_scratch[..n_out]);
            produced += n_out;
            consumed += self.chunk_size;
        }
        if consumed > 0 {
            // Single shift of the (small) trailing residue.
            // For a typical call with `consumed ==
            // accum.len()` this is a no-op move.
            self.accum.drain(..consumed);
        }
        Ok(produced)
    }

    /// Drain pending output into a caller-owned [`Vec`],
    /// leaving the internal buffer empty (capacity
    /// preserved).  The internal `Vec`'s capacity persists
    /// across calls so a steady-state stream stops
    /// allocating once the buffer reaches its working size.
    pub fn drain_output_into(&mut self, sink: &mut Vec<f32>) {
        sink.append(&mut self.out);
    }

    /// Discard pending output without copying.  Used by the
    /// mic arbitrator on non-active per-channel resamplers:
    /// every whitelisted channel is `process`-ed each period
    /// (so its FIR state stays current and a switch is
    /// glitch-free), but only the active channel's output is
    /// written through to the audio buffer.  Without
    /// dropping, non-active `out` Vecs grow unboundedly.
    pub fn drop_output(&mut self) {
        self.out.clear();
    }

    /// Drain pending output, returning a freshly-allocated
    /// [`Vec`].  Convenience wrapper for one-shot callers;
    /// streaming hot paths should prefer
    /// [`Self::drain_output_into`] to avoid the per-call
    /// allocation that the returned `Vec`'s next push would
    /// otherwise force.
    pub fn take_output(&mut self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.out.len());
        out.append(&mut self.out);
        out
    }

    /// Number of output samples currently buffered (not yet drained).
    pub fn pending(&self) -> usize {
        self.out.len()
    }

    /// **Discontinuity reset.**  Clears the FIR history +
    /// accumulators.
    ///
    /// Call only when the upstream sample stream has a real
    /// discontinuity, i.e. the next samples fed to
    /// [`Self::process`] are not the natural continuation of
    /// the samples that produced the current FIR state.
    /// Calling on a continuous stream injects an audible
    /// click (~3 ms transient as the FIR rebuilds from zero
    /// state).
    ///
    /// # Legitimate callers
    ///
    /// - `crate::audio_buffer::Reader::seek_latest` after a
    ///   `Lagged` event: the listener has already accepted a
    ///   glitch, so the resampler reset substitutes one
    ///   transient for another.
    /// - The mic arbitrator's xrun-recovery path
    ///   (`reset_per_channel_fir`): after `pcm.try_recover`
    ///   returns Ok, the next `readi` returns samples from a
    ///   fresh capture session disconnected from the prior
    ///   FIR history.  Resetting substitutes a clean
    ///   zero-history transient for what would otherwise be
    ///   a phantom-tail-of-pre-xrun-stream transient of the
    ///   same duration.
    ///
    /// In both cases, the listener is already absorbing a
    /// glitch (Lagged event / xrun click), so the reset's
    /// transient is not an additional perceptual cost.
    pub fn reset_after_discontinuity(&mut self) {
        self.accum.clear();
        self.out.clear();
        self.resampler.reset();
    }
}

impl std::fmt::Debug for Streaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Streaming")
            .field("chunk_size", &self.chunk_size)
            .field("accum_len", &self.accum.len())
            .field("out_len", &self.out.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resampler built via [`sinc_resampler`] reports a
    /// fixed chunk size.  If this changes upstream the rest
    /// of the streaming logic needs revisiting.
    #[test]
    fn upstream_chunk_size_is_1024() {
        let s = Streaming::new(48_000, 44_100);
        assert_eq!(s.chunk_size, 1024);
    }

    /// Feed less than chunk_size; no output yet.
    #[test]
    fn process_under_chunk_buffers_no_output() {
        let mut s = Streaming::new(48_000, 44_100);
        let input = vec![0.5_f32; 512];
        let produced = s.process(&input).expect("test resampler invariant");
        assert_eq!(produced, 0);
        assert_eq!(s.pending(), 0);
        assert_eq!(s.take_output(), Vec::<f32>::new());
    }

    /// Feed exactly chunk_size; output appears.  Exact count depends on
    /// rubato's internal startup transient (sinc filter has internal
    /// state that bleeds out across the first few chunks); we just
    /// verify some output came out and it's plausible (less than the
    /// input chunk size, since the ratio is < 1).
    #[test]
    fn process_one_chunk_produces_output() {
        let mut s = Streaming::new(48_000, 44_100);
        let input = vec![0.0_f32; 1024];
        let produced = s.process(&input).expect("test resampler invariant");
        assert!(
            produced > 0 && produced < 1024,
            "produced {produced}; expected 0 < produced < 1024 (downsampling)",
        );
        assert_eq!(s.pending(), produced);
    }

    /// Feed many chunks contiguously; verify total output matches the
    /// rate ratio within a small startup-transient tolerance.
    #[test]
    fn streaming_total_output_matches_ratio() {
        let mut s = Streaming::new(48_000, 44_100);
        let total_in: usize = 48_000;
        let mut produced = 0;
        for chunk_start in (0..total_in).step_by(700) {
            let end = (chunk_start + 700).min(total_in);
            let input = vec![0.0_f32; end - chunk_start];
            produced += s.process(&input).expect("test resampler invariant");
        }
        // 48 000 in samples; expected ratio 44100/48000 =
        // 0.91875.  Rubato consumes only complete
        // 1024-sample chunks => 46 chunks consumed = 47 104
        // input samples.  Ideal output = 47 104 * 0.91875 =
        // 43 283.  Reality lags by the startup transient
        // (~120 samples).  Tolerance +/-200 covers both
        // rubato version drift and any residual unprocessed
        // accum.
        let expected_ideal = (47_104.0_f64 * 0.918_75) as usize;
        let lower = expected_ideal - 200;
        let upper = expected_ideal + 200;
        assert!(
            (lower..=upper).contains(&produced),
            "produced {produced}; expected ~{expected_ideal} (+/-200) for \
             48k to 44.1k of 48000 in",
        );
    }

    /// 44.1 to 44.1: no resampling, but the wrapper still
    /// buffers in 1024-sample chunks and emits ~1024 samples
    /// out per chunk.  Mainly verifies graceful degradation
    /// on the native rate.
    ///
    /// rubato 2.0's [`Async`] resampler reports a fixed sinc
    /// filter group delay (~128 samples for `sinc_len=256`);
    /// the test absorbs it with a small offset search so it
    /// stays alignment-agnostic to filter-design changes
    /// within rubato.
    #[test]
    fn identity_rate_round_trips_pattern_within_tolerance() {
        let mut s = Streaming::new(44_100, 44_100);
        // Input is a slow ramp; output should approximate it.
        let input: Vec<f32> = (0..2048).map(|i| i as f32 / 2048.0).collect();
        s.process(&input).expect("test resampler invariant");
        let out = s.take_output();
        // Out ~= in length (identity ratio is exact).
        assert!(
            (1900..=2050).contains(&out.len()),
            "identity ratio produced {} samples for 2048 in",
            out.len(),
        );
        // Compare a steady-state middle window: the sinc
        // filter introduces a fixed group delay that shifts
        // output relative to input.  Search +/-256 samples
        // (the sinc filter length) for the best-fit offset
        // and require the per-sample residual there to be
        // small.
        let win = 700..900;
        let mid_in = &input[win.clone()];
        let best = (-256i32..=256)
            .filter_map(|off| {
                let mut max_d = 0f32;
                for (i, &a) in mid_in.iter().enumerate() {
                    let j = (win.start as i32 + i as i32) + off;
                    if j < 0 || (j as usize) >= out.len() {
                        return None;
                    }
                    max_d = max_d.max((a - out[j as usize]).abs());
                }
                Some((off, max_d))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("at least one offset must be in-bounds");
        assert!(
            best.1 < 0.05,
            "identity ratio: best-fit offset {} max|Δ| = {} (>= 0.05)",
            best.0,
            best.1,
        );
    }

    #[test]
    fn reset_after_discontinuity_clears_buffers() {
        let mut s = Streaming::new(48_000, 44_100);
        s.process(&vec![1.0; 1024])
            .expect("test resampler invariant");
        assert!(s.pending() > 0);
        s.reset_after_discontinuity();
        assert_eq!(s.pending(), 0);
        // After reset we can still keep streaming.
        s.process(&vec![0.5; 1024])
            .expect("test resampler invariant");
        assert!(s.pending() > 0);
    }

    /// [`Streaming::drop_output`] clears pending output but
    /// **preserves the resampler's FIR state**: feeding more
    /// samples afterwards continues from where the resampler
    /// was, no transient.  This is the property the mic
    /// arbitrator depends on for non-active per-channel
    /// resamplers.
    #[test]
    fn drop_output_preserves_fir_state() {
        let mut canonical = Streaming::new(48_000, 44_100);
        let mut dropped = Streaming::new(48_000, 44_100);

        // Pump the same 4 chunks of input through both.  After each
        // chunk, drain `canonical`'s output (and keep it) and
        // `drop`-discard `dropped`'s output.
        let mut canonical_out: Vec<f32> = Vec::new();
        for _ in 0..4 {
            let input: Vec<f32> = (0..1024)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
                .collect();
            canonical.process(&input).expect("test resampler invariant");
            dropped.process(&input).expect("test resampler invariant");
            canonical.drain_output_into(&mut canonical_out);
            dropped.drop_output();
        }

        // Now feed both ANOTHER chunk; `dropped`'s output should be
        // bit-identical to the corresponding tail of canonical's
        // output.  If `drop_output` had reset FIR state, the next
        // chunk's output would carry a transient and diverge.
        let probe: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (4 * 1024 + i) as f32 / 48_000.0).sin())
            .collect();
        canonical.process(&probe).expect("test resampler invariant");
        dropped.process(&probe).expect("test resampler invariant");

        let probe_canonical = canonical.take_output();
        let probe_dropped = dropped.take_output();
        assert_eq!(
            probe_canonical.len(),
            probe_dropped.len(),
            "post-drop output length differs",
        );
        for (i, (a, b)) in probe_canonical.iter().zip(probe_dropped.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "drop_output altered FIR state at sample {i}: {a} vs {b}",
            );
        }
    }
}
