//! WebAudio-style audio preprocessing for the
//! `tm-my-audio-model` backbone.
//!
//! Replicates `sc_preproc_model` (TensorFlow SavedModel):
//!
//! - Input: 44032 mono PCM samples at 44100 Hz, f32 in
//!   `[-1, 1]`.
//! - Output: 43x232 f32 log-magnitude spectrogram,
//!   z-normalized over the plane.
//!
//! Framing (reverse-engineered against the reference
//! spectrogram):
//!
//! - frame_length = 2048, hop = 1024, n_frames = 43.
//! - offset = -1024: the input is virtually pre-padded with
//!   1024 zeros at the front.  Frame `i` covers samples
//!   `[(i-1)*1024, (i+1)*1024)`, with negative indices
//!   treated as zero.  Frame 0 is thus "1024 zeros + first
//!   1024 samples", frame 1 is exactly `samples[0..2048]`,
//!   and frame 42 is exactly `samples[41984..44032]`.
//! - Total virtual span = `(43-1)*1024 + 2048 = 45056`,
//!   minus 1024 front-pad = 44032 samples of real signal.
//!   No trailing padding needed.
//!
//! Window: *periodic* Blackman (denominator M=2048),
//! matching WebAudio's
//! `AnalyserNode.getFloatFrequencyData`.  NOT the symmetric
//! `numpy.blackman`.
//!
//! Log: natural `ln`, no epsilon.  Frames containing
//! exact-zero magnitude bins yield `-inf`, which then poisons
//! z-normalize to `NaN`.  This matches the TF graph and the
//! training notebook's `spectrogram_elements_finite` filter.

// WAV I/O + sinc resample utilities for the 44032-sample
// input pipeline.  Public so `training` can reach them as
// `preproc::wav_io::*`.  The canonical sinc-resampler params
// live in [`crate::dsp::resample`]; `wav_io` re-exports them
// so existing `preproc::wav_io::{SincResampler, sinc_resampler}`
// paths keep working.
pub mod wav_io;

use crate::common::dims::{HopSamples, NBins, NFrames, WaveformLen};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use std::sync::Arc;

/// FFT window length in samples.  Each spectrogram frame
/// covers `FRAME_LEN` samples post-windowing.
pub const FRAME_LEN: usize = 2048;
/// Virtual front-padding in samples (negative offset of
/// frame 0 relative to the first real sample).
pub const FRONT_PAD: usize = 1024;

/// Compile-time bounds invariant for the no-pad spectrogram
/// path; fires in release too so a constant tweak that breaks
/// the bound refuses to compile.
#[allow(dead_code)]
const FRAME_BOUNDS_INVARIANT: () = assert!(
    (NFrames::USIZE - 1) * HopSamples::USIZE - FRONT_PAD + FRAME_LEN <= WaveformLen::USIZE,
    "frame bounds violation: last pcm index would exceed WaveformLen",
);

/// Cached FFT plan + window + per-call scratch for repeated
/// calls.
///
/// Uses a real-input FFT ([`realfft`]) which does ~half the
/// work of a full complex FFT when the input is purely real
/// (audio PCM).  Output buffer length is `FRAME_LEN/2 + 1 =
/// 1025`, of which we read the first
/// [`NBins::VALUE`] = 232 bins; byte-identical to what the
/// equivalent complex FFT would produce at those indices.
///
/// [`Self::spectrogram`] takes `&mut self` because the
/// pre-allocated FFT output + scratch buffers live on the
/// struct (zero alloc in the hot path).  For multi-threaded
/// use (e.g. fine-tune's rayon parallel preproc), [`Clone`]
/// is provided: the FFT plan itself is [`Arc`]-shared and
/// each clone gets its own private scratch buffers.
pub struct Preproc {
    r2c: Arc<dyn RealToComplex<f32>>,
    window: [f32; FRAME_LEN],
    // Per-call working buffers, pre-allocated at `new()` so
    // the hot path doesn't allocate.
    //
    // `frame` is the windowed real-valued input that
    // `realfft` consumes (and overwrites) per frame.  Lives
    // on the struct, not the stack of `spectrogram_into`,
    // so the per-call 8 KB zero-init that
    // `let mut frame = [0.0; FRAME_LEN]` would emit is
    // amortized to zero: every cell is written before the
    // FFT reads it, so initial state is immaterial.
    frame: [f32; FRAME_LEN],
    spectrum: Vec<Complex32>, // FRAME_LEN/2 + 1 = 1025 complex bins
    scratch: Vec<Complex32>,  // rustfft internal scratch
}

impl Preproc {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FRAME_LEN);
        let spectrum = r2c.make_output_vec(); // Vec<Complex32> of length 1025
        let scratch = vec![Complex32::default(); r2c.get_scratch_len()];
        Self {
            r2c,
            window: load_bundled_window(),
            frame: [0.0; FRAME_LEN],
            spectrum,
            scratch,
        }
    }

    /// Compute the 43x232 spectrogram for a 44032-sample
    /// f32 waveform.
    ///
    /// Returns row-major `[NFrames][NBins]`.  NaN is
    /// returned for any frame that contains a zero-magnitude
    /// bin (matches TF behavior).
    ///
    /// Allocates one
    /// `Box<[[f32; NBins::USIZE]; NFrames::USIZE]>` (~40 KB)
    /// per call.  For zero-allocation streaming use cases
    /// (the daemon's inference hot loop), use
    /// [`Self::spectrogram_into`] instead.
    pub fn spectrogram(
        &mut self,
        pcm: &[f32; WaveformLen::USIZE],
    ) -> Box<[[f32; NBins::USIZE]; NFrames::USIZE]> {
        let mut out = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        self.spectrogram_into(pcm, &mut out);
        out
    }

    /// Compute the 43x232 spectrogram into a caller-provided
    /// buffer.  Zero allocation in steady state: every cell
    /// of `out` is overwritten before being read, so the
    /// caller can pass a buffer from any prior state.
    ///
    /// Numerically identical to [`Self::spectrogram`]; that
    /// wrapper just allocates the box and delegates here.
    /// Byte-identical equivalence is verified by the parity
    /// test suite.
    ///
    /// Intended for the streaming hot path: pre-allocate one
    /// scratch `Box<[[f32; NBins::USIZE]; NFrames::USIZE]>`
    /// outside the loop, then call this each frame.
    pub fn spectrogram_into(
        &mut self,
        pcm: &[f32; WaveformLen::USIZE],
        out: &mut [[f32; NBins::USIZE]; NFrames::USIZE],
    ) {
        // Bounds invariant enforced at compile time by
        // [`FRAME_BOUNDS_INVARIANT`].

        // Destructure once so the per-frame FFT call can hold an
        // `&self.r2c` borrow alongside disjoint `&mut` borrows of
        // `frame/spectrum/scratch` without confusing the borrow
        // checker.
        let Self {
            r2c,
            window,
            frame,
            spectrum,
            scratch,
        } = self;

        for (t, row) in out.iter_mut().enumerate() {
            // Frame t covers virtual indices `[t*hop,
            // t*hop + FRAME_LEN)` in the padded signal,
            // which maps to real sample indices `[t*hop -
            // FRONT_PAD, t*hop - FRONT_PAD + FRAME_LEN)`.
            // With current params (hop=1024, FRONT_PAD=1024),
            // only frame 0 straddles the front pad; every
            // later frame reads a contiguous in-bounds slice
            // of `pcm`.  Splitting the two cases removes a
            // per-iteration `if virt_idx < FRONT_PAD` branch
            // from the hot path.
            //
            // NOTE: Under `lto = "thin"` (the workspace
            // default), this loop's body is emitted as scalar
            // `fmul s0, s0, s1` despite being a textbook
            // SIMD-eligible kernel.  Disabling LTO
            // (`CARGO_PROFILE_RELEASE_LTO=off`) recovers the
            // expected 16-wide `fmul.4s` codegen.  The
            // pessimization is in LLVM's thin-LTO pass
            // pipeline, not in the source: reproduced both
            // inline (this form) and via an `#[inline(never)]`
            // helper.  Net effect at 4 Hz inference is ~20
            // us/spectrogram of leftover scalar work, dwarfed
            // by the ~200 us from libm `logf` in the
            // post-FFT loop.  Do NOT factor the multiply into
            // a helper "for SIMD"; verified to make no
            // difference under thin-LTO.
            let start = t * HopSamples::USIZE;
            if start < FRONT_PAD {
                let zeros = FRONT_PAD - start;
                let take = FRAME_LEN - zeros;
                // The bundled Blackman window is
                // non-negative, so `0.0 * w` produces +0.0
                // for every padded slot; filling with 0.0
                // directly is bit-identical (verified by the
                // parity test).  `fill` lowers to `bzero`
                // here.
                frame[..zeros].fill(0.0);
                let pcm_head = &pcm[..take];
                let win_tail = &window[zeros..];
                for ((f, &p), &w) in frame[zeros..].iter_mut().zip(pcm_head).zip(win_tail) {
                    *f = p * w;
                }
            } else {
                // Hot path (42 of 43 frames): contiguous
                // slice multiply.
                let s = start - FRONT_PAD;
                let pcm_slice = &pcm[s..s + FRAME_LEN];
                for ((f, &p), &w) in frame.iter_mut().zip(pcm_slice).zip(window.iter()) {
                    *f = p * w;
                }
            }

            // Real-to-complex FFT with a pre-allocated
            // scratch.  Sizes are fixed at `new()`; mismatch
            // would be a construction-time bug.
            r2c.process_with_scratch(frame, spectrum, scratch)
                .expect("realfft buffer sizes are fixed at Preproc::new");

            // log|z| = 0.5 * log(|z|^2).  Saves one sqrt
            // per bin (9976 sqrts per call); multiplying by
            // 0.5 is exact in fp32 (exponent shift).
            // Zero-magnitude bins still produce -inf here
            // (ln(0) = -inf), matching the
            // `spectrogram_elements_finite` filter the
            // training notebook relies on.
            for (r, s) in row.iter_mut().zip(&spectrum[..NBins::USIZE]) {
                *r = 0.5 * s.norm_sqr().ln();
            }
        }

        // Z-normalize over the entire 43x232 plane.
        // Accumulate in f32: NEON vectorises f32 but not f64.
        // The 9976-sample sum stays well inside f32's 24-bit
        // mantissa precision -- log magnitudes for normal
        // speech land in roughly `[-25, +5]`, so worst-case
        // `|sum| <= 9976 * 25 ~= 2.5e5`, five orders of
        // magnitude below f32's `2^24 ~= 1.7e7` lossless
        // integer ceiling.  The parity test is the gate
        // confirming drift stays inside the existing
        // tolerance.  TF `moments` uses population variance
        // (divide by N, not N-1).
        let count = (NFrames::USIZE * NBins::USIZE) as f32;
        let mut sum: f32 = 0.0;
        for row in out.iter() {
            for &v in row.iter() {
                sum += v;
            }
        }
        let mean = sum / count;
        let mut sq: f32 = 0.0;
        for row in out.iter() {
            for &v in row.iter() {
                let d = v - mean;
                sq += d * d;
            }
        }
        let std = (sq / count).sqrt();
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
    }
}

impl Default for Preproc {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Preproc {
    /// Clone shares the FFT plan ([`Arc`]) and window, but
    /// allocates fresh per-call scratch buffers for the
    /// clone.  Intended for giving each rayon worker its own
    /// [`Preproc`] via `map_init(|| base.clone(), ...)`.
    fn clone(&self) -> Self {
        Self {
            r2c: self.r2c.clone(),
            window: self.window,
            // `frame` is overwritten before every read, so
            // its initial contents don't matter; zero-init
            // is the cheapest legal value.
            frame: [0.0; FRAME_LEN],
            spectrum: vec![Complex32::default(); self.spectrum.len()],
            scratch: vec![Complex32::default(); self.scratch.len()],
        }
    }
}

/// Periodic Blackman window of length N, with denominator N
/// (not N-1).  Matches WebAudio's
/// `AnalyserNode.getFloatFrequencyData`.
///
/// We ship the exact bytes extracted from
/// `sc_preproc_model`'s graph rather than recomputing from
/// `0.42 - 0.5*cos(2*pi*n/N) + 0.08*cos(4*pi*n/N)`.  The two
/// agree to 1.19e-7 at every tap; harmless for most bins,
/// but on frames with near-zero mean the DC bin's magnitude
/// `~= |sum(signal * window)|` can land in the weeds and
/// `ln` amplifies any drift 10x+.  Using the bundled bytes
/// drops the worst-case idx=2 DC-bin drift from 8.9e-3 to
/// 8.0e-4: NOT negligible.
const WINDOW_BYTES: &[u8; 8192] = include_bytes!("preproc/window_blackman_2048.bin");

fn load_bundled_window() -> [f32; FRAME_LEN] {
    // 8192 bytes / 4 bytes-per-f32 = 2048 = FRAME_LEN coefficients.
    // The cardinality is a compile-time fact via WINDOW_BYTES'
    // `&[u8; 8192]` typing; chunks_exact(4) produces exactly 2048
    // 4-byte windows with no remainder, and FRAME_LEN bounds the
    // index range used to write `w`.
    const _: () = assert!(
        WINDOW_BYTES.len() == FRAME_LEN * std::mem::size_of::<f32>(),
        "bundled window byte count must equal FRAME_LEN * sizeof(f32)",
    );
    let mut w = [0.0f32; FRAME_LEN];
    for (i, chunk) in WINDOW_BYTES.chunks_exact(4).enumerate() {
        w[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_periodic_shape() {
        let w = load_bundled_window();
        // w[0] = 0.42 - 0.5 + 0.08 = 0 exactly.
        assert!(w[0].abs() < 1e-6, "w[0] = {}", w[0]);
        // Peak at n = N/2.
        assert!(
            (w[FRAME_LEN / 2] - 1.0).abs() < 1e-6,
            "w[N/2] = {}",
            w[FRAME_LEN / 2]
        );
        // At n = N/4: cos(pi/2)=0, cos(pi)=-1 => 0.42 - 0.08 = 0.34.
        assert!(
            (w[FRAME_LEN / 4] - 0.34).abs() < 1e-4,
            "w[N/4] = {}",
            w[FRAME_LEN / 4]
        );
    }

    #[test]
    fn output_shape() {
        let pcm = Box::new([0.1f32; WaveformLen::USIZE]);
        let mut p = Preproc::new();
        let s = p.spectrogram(&pcm);
        assert_eq!(s.len(), NFrames::USIZE);
        assert_eq!(s[0].len(), NBins::USIZE);
    }

    /// Silence input (all-zero PCM) must produce an all-NaN
    /// spectrogram.  The chain:
    ///
    /// - `|z|^2 = 0` -> `0.5 * ln(0) = -inf` for every bin
    /// - mean = -inf, std^2 = NaN
    /// - `(-inf - -inf) / NaN = NaN`.
    ///
    /// Locking this in protects the inference engine's
    /// silence filter from a future "helpful" log-domain
    /// epsilon that would silently turn silence into a
    /// finite spectrogram and route it into the classifier.
    #[test]
    fn silence_input_is_all_nan() {
        let pcm = Box::new([0.0f32; WaveformLen::USIZE]);
        let mut p = Preproc::new();
        let s = p.spectrogram(&pcm);
        for (t, row) in s.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(v.is_nan(), "silence: expected NaN at t={t} k={k}, got {v}");
            }
        }
    }
}
