//! Shared WAV-to-44032-sample-f32 pipeline used by training
//! and the classifier reference tools.
//!
//! Resampler configuration lives in
//! [`crate::dsp::resample`], the workspace's single source of
//! truth for the sinc params the classifier was validated
//! against.  This module re-exports [`SincResampler`] /
//! [`sinc_resampler`] so existing call sites keep a familiar
//! import path while every call constructs the same
//! [`rubato::Async<f32>`] with the same constants --
//! inference-time and fine-tune-time resamplers cannot drift.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::Resampler;
use std::path::Path;
use thiserror::Error;

pub use crate::dsp::resample::{SincResampler, sinc_resampler};

/// Target sample rate for the inference pipeline (Hz).
pub const TARGET_SR: u32 = 44100;
use crate::common::dims::WaveformLen;

/// Hard ceiling on accepted WAV input duration in seconds.
///
/// `WaveformLen::USIZE` ~= 1.0 s of audio at [`TARGET_SR`];
/// the inference pipeline tail-truncates to that single
/// window.  60 s gives operators a generous margin for
/// loosely-trimmed Speech-Commands-style clips while
/// rejecting hostile or accidental long recordings (concert
/// captures, full-track music) that would otherwise force a
/// large `Vec<f32>` allocation in [`read_wav_mono`] and burn
/// CPU resampling tail samples that [`to_waveform`]
/// immediately discards.
///
/// At 60 s, 44.1 kHz mono f32 is ~10 MiB; even the worst
/// supported 24-bit / 8-channel shape stays under ~85 MiB
/// pre-downmix, well inside the training-job envelope.
pub const MAX_INPUT_DURATION_SECS: u32 = 60;

/// Failure shapes for [`read_wav_mono`] and
/// [`to_waveform`].  Callers use `?`-style propagation and
/// explicit error classification rather than catching
/// panics; one bad file in a batch must not kill the
/// surrounding loop.
///
/// Categorised as
/// [`crate::common::error::ErrorKind::UserInput`] by the
/// wrapping modules' [`crate::common::error::Categorized`]
/// impls; the operator's input file is malformed.
#[derive(Debug, Error)]
pub enum PreprocError {
    /// [`hound::WavReader::open`] failed (file missing,
    /// permission denied, not a WAV, truncated header).
    #[error("open wav {path}: {source}")]
    WavOpen {
        path: String,
        #[source]
        source: hound::Error,
    },
    /// A per-sample decode returned `Err` mid-stream
    /// (truncated data, corrupt frame).
    #[error("decode wav {path} at sample {sample_idx}: {source}")]
    WavDecode {
        path: String,
        sample_idx: usize,
        #[source]
        source: hound::Error,
    },
    /// File parsed but the (format, bits-per-sample) pair
    /// isn't one of the four shapes we accept (PCM-i16,
    /// PCM-i24, PCM-i32, or IEEE-f32).
    #[error("unsupported wav format {format} / {bits} bits in {path}")]
    WavFormat {
        path: String,
        format: String,
        bits: u16,
    },
    /// Header parses but advertises a value outside the
    /// daemon's accepted range: zero-valued fields would trip
    /// a divide-by-zero downstream (sample_rate=0 crashes
    /// resampler init; channels=0 divides by zero in the
    /// mono downmix), and values outside
    /// [`crate::audio_io::mic_arbitrator::MIN_SAMPLE_RATE`] /
    /// [`crate::audio_io::mic_arbitrator::MAX_SAMPLE_RATE`] /
    /// [`crate::audio_io::mic_arbitrator::MAX_CHANNELS`] would
    /// pass capture-side validation gates the runtime relies
    /// on.  Reject up front so the diagnostic blames the
    /// header, not a deeper layer.  `got` echoes the
    /// offending value so the operator does not have to
    /// inspect the file by hand.
    #[error("invalid wav header in {path}: {field} = {got}")]
    WavInvalidHeader {
        path: String,
        field: &'static str,
        got: u64,
    },
    /// File parsed but a float-32 sample was non-finite
    /// (NaN, +Inf, -Inf).  Mirrors `opus_stream::OpusError::BadPcm`'s
    /// finite-only PCM contract: the daemon rejects non-finite
    /// floats at every ingest boundary instead of letting them
    /// propagate into resamplers / spectrograms / inference
    /// frames where they would silently produce NaN logits.
    /// Int-format WAVs (i16 / i24 / i32 -> f32) are finite by
    /// construction and skip this scan.
    #[error("non-finite wav sample in {path} at sample {sample_idx}: {value}")]
    BadWavSample {
        path: String,
        sample_idx: usize,
        value: f32,
    },
    /// Header parses but the file's announced duration
    /// exceeds [`MAX_INPUT_DURATION_SECS`].  Caught from the
    /// header's `num_samples / channels` count *before*
    /// allocating the decode buffer, so a 10-minute hostile
    /// upload doesn't get to allocate a hundred-MB `Vec<f32>`
    /// just to be discarded.
    #[error(
        "wav {path} duration {duration_secs:.1}s exceeds cap {max_secs}s \
         ({sample_rate} Hz x {frames} frames)"
    )]
    WavTooLong {
        path: String,
        sample_rate: u32,
        frames: u32,
        duration_secs: f32,
        max_secs: u32,
    },
    /// Resampler init or per-chunk processing returned
    /// `Err`.  Triggers on adapter shape mismatches or
    /// resampler-internal failures (rare; mostly caught by
    /// `audio_io`'s own validation before reaching here).
    #[error("resample: {0}")]
    Resample(String),
}

/// Read a `.wav` file, returning `(sample_rate, mono f32
/// samples in [-1, 1])`.  Downmixes to mono by averaging
/// channels if needed.
///
/// Header-declared duration is checked against
/// [`MAX_INPUT_DURATION_SECS`] *before* the per-sample
/// decode loop runs, so a hostile or accidentally-long file
/// is rejected with [`PreprocError::WavTooLong`] without
/// growing the decode buffer past the cap.
///
/// # Errors
///
/// See [`PreprocError`] for the shape of each failure.
/// Callers can drop a single bad file via `?` propagation
/// rather than `catch_unwind`.
pub fn read_wav_mono(path: &Path) -> Result<(u32, Vec<f32>), PreprocError> {
    let r = hound::WavReader::open(path).map_err(|e| PreprocError::WavOpen {
        path: path.display().to_string(),
        source: e,
    })?;
    let s = r.spec();
    let n_chan = s.channels as usize;
    let path_string = path.display().to_string();

    // Reject header fields outside the daemon's accepted
    // range BEFORE they reach the resampler (sr=0 -> divide
    // by zero), the downmix (channels=0 -> chunks(0) panic),
    // or the runtime (a sample rate / channel count below
    // capture-side caps would parse here but fail downstream
    // with a less actionable error).  Hound itself accepts
    // anything; the gate is ours.  The accepted ranges match
    // the canonical capture caps in
    // `audio_io::mic_arbitrator::{MIN_SAMPLE_RATE,
    // MAX_SAMPLE_RATE, MAX_CHANNELS}` so WAV ingest and live
    // capture refuse the same set of pathological metadata.
    use crate::audio_io::mic_arbitrator::{MAX_CHANNELS, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
    if s.sample_rate == 0 {
        return Err(PreprocError::WavInvalidHeader {
            path: path_string,
            field: "sample_rate",
            got: 0,
        });
    }
    if s.channels == 0 {
        return Err(PreprocError::WavInvalidHeader {
            path: path_string,
            field: "channels",
            got: 0,
        });
    }
    if s.sample_rate < MIN_SAMPLE_RATE {
        return Err(PreprocError::WavInvalidHeader {
            path: path_string,
            field: "sample_rate",
            got: s.sample_rate as u64,
        });
    }
    if s.sample_rate > MAX_SAMPLE_RATE {
        return Err(PreprocError::WavInvalidHeader {
            path: path_string,
            field: "sample_rate",
            got: s.sample_rate as u64,
        });
    }
    if (s.channels as usize) > MAX_CHANNELS {
        return Err(PreprocError::WavInvalidHeader {
            path: path_string,
            field: "channels",
            got: s.channels as u64,
        });
    }

    // Pre-decode duration cap.  `r.duration()` is per-channel
    // sample count read from the WAV header (no scan), so the
    // check is O(1) and runs before any
    // `Vec<f32>::with_capacity`-shaped growth.  Saturating
    // multiply guards against pathological headers that
    // claim u32::MAX samples; the comparison still rejects
    // them as too long.
    let frames = r.duration();
    let max_frames = (MAX_INPUT_DURATION_SECS as u64).saturating_mul(s.sample_rate as u64);
    if (frames as u64) > max_frames {
        let duration_secs = if s.sample_rate > 0 {
            frames as f32 / s.sample_rate as f32
        } else {
            f32::INFINITY
        };
        return Err(PreprocError::WavTooLong {
            path: path_string,
            sample_rate: s.sample_rate,
            frames,
            duration_secs,
            max_secs: MAX_INPUT_DURATION_SECS,
        });
    }

    let mut r = r;
    fn collect<S, F>(
        path: &str,
        iter: impl Iterator<Item = Result<S, hound::Error>>,
        scale: F,
    ) -> Result<Vec<f32>, PreprocError>
    where
        F: Fn(S) -> f32,
    {
        let mut out = Vec::new();
        for (i, sample) in iter.enumerate() {
            let s = sample.map_err(|e| PreprocError::WavDecode {
                path: path.to_string(),
                sample_idx: i,
                source: e,
            })?;
            out.push(scale(s));
        }
        Ok(out)
    }
    // Per-format full-scale magnitudes (`2^(bits-1)`) used to map
    // signed integer PCM into the canonical [-1.0, 1.0] f32 range.
    // Naming the divisors (rather than inlining 32768.0 / 8_388_608.0
    // / 2_147_483_648.0) keeps the intent unambiguous and lets the
    // constants be referenced elsewhere if a future ingest path
    // (e.g., MP3 decoded to int) needs the same scaling.
    const SCALE_I16: f32 = (1u32 << 15) as f32;
    const SCALE_I24: f32 = (1u32 << 23) as f32;
    const SCALE_I32: f32 = (1u64 << 31) as f32;
    let samples: Vec<f32> = match (s.sample_format, s.bits_per_sample) {
        // Int formats (i16 / i24 / i32) decode to a finite f32
        // by construction: the source is a bounded integer
        // divided by a non-zero power of two, so no NaN or Inf
        // can appear.  The finite scan only runs on the
        // float-32 path below to avoid paying for a check that
        // would always pass.
        (hound::SampleFormat::Int, 16) => {
            collect(&path_string, r.samples::<i16>(), |v| v as f32 / SCALE_I16)?
        }
        (hound::SampleFormat::Int, 24) => {
            collect(&path_string, r.samples::<i32>(), |v| v as f32 / SCALE_I24)?
        }
        (hound::SampleFormat::Int, 32) => {
            collect(&path_string, r.samples::<i32>(), |v| v as f32 / SCALE_I32)?
        }
        (hound::SampleFormat::Float, 32) => {
            // Finite-only PCM contract.  Fail on the first
            // non-finite sample with its index so the operator
            // can locate the bad frame; no range clamp.  Sibling
            // contract: `opus_stream::process_pcm` rejects the
            // same shape on streaming PCM ingress.
            let raw = collect(&path_string, r.samples::<f32>(), |v| v)?;
            if let Some(idx) = raw.iter().position(|s| !s.is_finite()) {
                return Err(PreprocError::BadWavSample {
                    path: path_string,
                    sample_idx: idx,
                    value: raw[idx],
                });
            }
            raw
        }
        (fmt, bits) => {
            return Err(PreprocError::WavFormat {
                path: path_string,
                format: format!("{fmt:?}"),
                bits,
            });
        }
    };
    let mono: Vec<f32> = if n_chan == 1 {
        samples
    } else {
        samples
            .chunks(n_chan)
            .map(|c| c.iter().sum::<f32>() / n_chan as f32)
            .collect()
    };
    Ok((s.sample_rate, mono))
}

/// Rate-keyed cache slot for [`to_waveform`].
///
/// Wraps the source sample rate alongside the
/// [`SincResampler`] so callers reusing one slot across many
/// files automatically rebuild the resampler whenever the
/// next file's `sr` differs from the cached one.  The prior
/// `Option<SincResampler>` shape silently reused the first
/// file's rate ratio for every later file -- a mixed-rate
/// dataset (16 kHz + 44.1 kHz + 48 kHz, common in the wild)
/// produced numerically valid but semantically wrong
/// spectrograms with no error surfaced.
///
/// The slot is `pub` so call sites can declare the cache
/// type without importing the generic `Option<(u32, _)>`
/// shape.  Construct as [`ResamplerCache::empty`] (or
/// `ResamplerCache::default()`) and pass `&mut` into
/// [`to_waveform`].
#[derive(Debug, Default)]
pub struct ResamplerCache {
    inner: Option<(u32, SincResampler)>,
}

impl ResamplerCache {
    /// Empty cache; the first non-target-rate
    /// [`to_waveform`] call lazy-initialises a
    /// [`SincResampler`] keyed by that file's `sr`.
    #[inline]
    pub const fn empty() -> Self {
        Self { inner: None }
    }

    /// Source sample rate of the cached resampler, or `None`
    /// if no resampler has been constructed yet.  Exposed
    /// for tests and diagnostic logging; production callers
    /// don't need to inspect this.
    #[inline]
    pub fn cached_rate(&self) -> Option<u32> {
        self.inner.as_ref().map(|(sr, _)| *sr)
    }

    /// Drop the cached resampler so the next call rebuilds.
    /// Used by callers on resampler-internal errors where
    /// FIR history may be partial; an explicit clear is
    /// safer than letting `reset()` (which clears history
    /// but preserves the ratio) try to recover.
    #[inline]
    pub fn clear(&mut self) {
        self.inner = None;
    }

    /// Borrow the cached resampler for `sr`, building (or
    /// rebuilding) it if the cache is empty or holds a
    /// different rate.  Always `reset()`s the FIR history so
    /// stateful taps from a prior file don't bleed into the
    /// current one.
    fn get_or_init(&mut self, sr: u32) -> &mut SincResampler {
        match &self.inner {
            // Hot path: cache hit on the same rate.  `reset`
            // is still required because the prior call left
            // FIR history populated.
            Some((cached, _)) if *cached == sr => {}
            // Miss: empty slot or a different rate.  Build
            // (or rebuild) before borrowing mutably.  The
            // sinc-table precompute is ~6 ms; happens once
            // per (worker, source-rate) pair, not per file.
            _ => self.inner = Some((sr, sinc_resampler(sr, TARGET_SR))),
        }
        let (_, r) = self
            .inner
            .as_mut()
            .expect("inner populated by the match arm above");
        r.reset();
        r
    }
}

/// Turn a mono f32 waveform at sample rate `sr` into
/// exactly [`WaveformLen::USIZE`] samples at [`TARGET_SR`].
/// Resamples via [`sinc_resampler`] if `sr != TARGET_SR`;
/// otherwise pads or truncates.
///
/// `cache` is a mutable [`ResamplerCache`] slot whose stored
/// resampler is keyed by source sample rate: a call with a
/// different `sr` than the cached one rebuilds before use,
/// so a single slot can safely process a mixed-rate dataset
/// (16 kHz + 44.1 kHz + 48 kHz) without silently applying
/// the first file's rate ratio to later files.  The cached
/// resampler is `reset()` on every call to drop FIR history
/// from the prior file.
///
/// **Tail truncation.**  If the resampled stream is longer
/// than [`WaveformLen::USIZE`], samples past the limit are
/// dropped from the tail without warning.  The resample
/// loop early-exits once the output buffer is full, so a
/// long input doesn't burn CPU resampling samples that get
/// dropped.  Pre-trim upstream if you need a centered or
/// otherwise-aligned window.
pub fn to_waveform(
    sr: u32,
    mono: Vec<f32>,
    cache: &mut ResamplerCache,
) -> Result<Box<[f32; WaveformLen::USIZE]>, PreprocError> {
    let resampled: Vec<f32> = if sr == TARGET_SR {
        mono
    } else {
        let r = cache.get_or_init(sr);
        let in_per = r.input_frames_next();
        let max_out = r.output_frames_max();
        let mut padded = mono;
        let pad = (in_per - (padded.len() % in_per)) % in_per;
        padded.extend(std::iter::repeat_n(0.0, pad));
        // Pre-size the destination to the
        // tail-truncation cap rather than the upstream
        // ratio-based estimate: even if the input is longer
        // than `WaveformLen` after resampling we never need
        // more than `WaveformLen + max_out - 1` samples
        // (one chunk's worth of overshoot before the loop
        // breaks), so cap allocation conservatively at
        // `WaveformLen + max_out`.  Avoids a worst-case
        // `padded.len() * TARGET_SR / sr` allocation for
        // long inputs.
        let cap = WaveformLen::USIZE.saturating_add(max_out);
        let mut out: Vec<f32> = Vec::with_capacity(cap);
        // Single per-call output scratch; sized by
        // `output_frames_max` so it's safe across the whole
        // loop.  Mono = 1 channel = the adapter borrows the
        // contiguous slice directly.
        let mut out_scratch = vec![0.0f32; max_out];
        for c in padded.chunks(in_per) {
            let in_adapter = InterleavedSlice::new(c, 1, in_per)
                .map_err(|e| PreprocError::Resample(format!("input adapter: {e}")))?;
            let mut out_adapter = InterleavedSlice::new_mut(&mut out_scratch, 1, max_out)
                .map_err(|e| PreprocError::Resample(format!("output adapter: {e}")))?;
            let (_, n_out) = r
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                .map_err(|e| PreprocError::Resample(format!("process chunk: {e}")))?;
            out.extend_from_slice(&out_scratch[..n_out]);
            // Tail-truncation early exit: once we've produced
            // enough output to fill the post-truncation
            // window, additional input chunks would only
            // contribute samples we'd discard.  Saves CPU
            // (and bounds latency) on inputs that pass the
            // duration cap but are still longer than the
            // 1-second inference window.
            if out.len() >= WaveformLen::USIZE {
                break;
            }
        }
        out
    };
    let mut arr = Box::new([0f32; WaveformLen::USIZE]);
    let n = resampled.len().min(WaveformLen::USIZE);
    arr[..n].copy_from_slice(&resampled[..n]);
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`read_wav_mono`] returns
    /// [`PreprocError::WavOpen`] for a missing path rather
    /// than panicking.
    #[test]
    fn read_wav_mono_missing_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/.acoustics_lab/no-such.wav");
        let err = read_wav_mono(path).expect_err("missing file must be Err");
        match err {
            PreprocError::WavOpen { path: p, .. } => {
                assert!(p.contains("no-such"), "diagnostic should embed path: {p}");
            }
            other => panic!("expected WavOpen, got {other:?}"),
        }
    }

    /// A file whose header parses but bytes don't form a
    /// valid WAV (truncated / random bytes / wrong magic)
    /// returns [`PreprocError::WavOpen`] (hound surfaces
    /// the parse failure at open time).
    #[test]
    fn read_wav_mono_garbage_bytes_returns_err() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("garbage.wav");
        // Test fixture only; no atomicity / 0o600 concerns
        // apply here.  The `disallowed_methods` lint is
        // enforced for production code paths via the
        // workspace clippy config.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(&path, b"not a wav file at all, just garbage bytes").unwrap();
        let err = read_wav_mono(&path).expect_err("garbage bytes must be Err");
        // Could be `WavOpen` (parse failure at open) or
        // `WavDecode` (header parses, samples don't); both
        // are valid mappings of the failure shape, so
        // matching either prevents the test from depending
        // on hound's exact dispatch.
        assert!(
            matches!(
                err,
                PreprocError::WavOpen { .. }
                    | PreprocError::WavDecode { .. }
                    | PreprocError::WavFormat { .. }
            ),
            "expected WavOpen / WavDecode / WavFormat, got {err:?}",
        );
    }

    /// Helper: write a mono 16-bit PCM WAV with `n_frames`
    /// samples at `sr` Hz; returns the path inside the given
    /// tempdir.  All samples are zero -- the duration cap
    /// gate runs before per-sample decode, so the value
    /// doesn't matter.
    fn write_silence_wav(dir: &std::path::Path, name: &str, sr: u32, n_frames: u32) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for _ in 0..n_frames {
            w.write_sample(0i16).expect("write sample");
        }
        w.finalize().expect("finalize wav");
        path
    }

    use std::path::PathBuf;

    /// A WAV whose header announces a duration > the
    /// [`MAX_INPUT_DURATION_SECS`] cap is rejected with
    /// [`PreprocError::WavTooLong`] *before* the per-sample
    /// decode runs.  Serves as the resource guard against
    /// accidental long recordings dominating training
    /// preprocessing.
    #[test]
    fn read_wav_mono_rejects_overlong_input() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Just past the cap so the test stays cheap to
        // build.  At 16 kHz the cap is `60 * 16000 = 960_000`
        // frames; one frame past triggers the gate.
        let sr: u32 = 16_000;
        let n_frames = MAX_INPUT_DURATION_SECS * sr + 1;
        let path = write_silence_wav(dir.path(), "long.wav", sr, n_frames);
        let err = read_wav_mono(&path).expect_err("overlong wav must be Err");
        match err {
            PreprocError::WavTooLong {
                sample_rate,
                frames,
                max_secs,
                ..
            } => {
                assert_eq!(sample_rate, sr);
                assert_eq!(frames, n_frames);
                assert_eq!(max_secs, MAX_INPUT_DURATION_SECS);
            }
            other => panic!("expected WavTooLong, got {other:?}"),
        }
    }

    /// Helper: write a mono float-32 WAV with the supplied
    /// samples at `sr` Hz.  Used to stage non-finite-sample
    /// fixtures for the finite-only PCM contract tests --
    /// hound writes f32 directly so we control the exact
    /// bit pattern (including NaN / Inf).
    fn write_float_wav(dir: &std::path::Path, name: &str, sr: u32, samples: &[f32]) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for s in samples {
            w.write_sample(*s).expect("write sample");
        }
        w.finalize().expect("finalize wav");
        path
    }

    /// Helper: write an N-channel int-16 WAV with `n_frames`
    /// frames per channel.  Used to stage the
    /// `channels > MAX_CHANNELS` rejection.
    fn write_silence_multichannel_wav(
        dir: &std::path::Path,
        name: &str,
        sr: u32,
        channels: u16,
        n_frames: u32,
    ) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for _ in 0..n_frames {
            for _ in 0..channels {
                w.write_sample(0i16).expect("write sample");
            }
        }
        w.finalize().expect("finalize wav");
        path
    }

    /// Finite-only PCM contract for WAV float-32 ingest:
    /// NaN samples are rejected with `BadWavSample` carrying
    /// the offending sample index.  Mirrors
    /// `opus_stream::OpusError::BadPcm`'s contract on
    /// streaming ingress.  Int-format WAVs are finite by
    /// construction and skip this scan, so the test stages a
    /// dedicated float-32 fixture.
    #[test]
    fn read_wav_mono_rejects_nan_in_float() {
        let dir = tempfile::tempdir().expect("tempdir");
        let samples = [0.0f32, 0.5, f32::NAN, -0.25];
        let path = write_float_wav(dir.path(), "nan.wav", 16_000, &samples);
        let err = read_wav_mono(&path).expect_err("NaN sample must reject");
        match err {
            PreprocError::BadWavSample {
                sample_idx, value, ..
            } => {
                assert_eq!(sample_idx, 2, "NaN was at index 2");
                assert!(value.is_nan(), "rejected value must be NaN, got {value}");
            }
            other => panic!("expected BadWavSample, got {other:?}"),
        }
    }

    /// Same finite-only contract for +Inf samples; pinned
    /// because the operator-facing diagnostic must distinguish
    /// "non-finite sample" from "header rejection".
    #[test]
    fn read_wav_mono_rejects_inf_in_float() {
        let dir = tempfile::tempdir().expect("tempdir");
        let samples = [0.0f32, f32::INFINITY, 0.5];
        let path = write_float_wav(dir.path(), "inf.wav", 16_000, &samples);
        let err = read_wav_mono(&path).expect_err("Inf sample must reject");
        match err {
            PreprocError::BadWavSample {
                sample_idx, value, ..
            } => {
                assert_eq!(sample_idx, 1);
                assert!(value.is_infinite() && value.is_sign_positive());
            }
            other => panic!("expected BadWavSample, got {other:?}"),
        }
    }

    /// WAV header advertising a sample rate below
    /// `audio_io::mic_arbitrator::MIN_SAMPLE_RATE` is rejected
    /// at the metadata gate, before any per-sample decode runs.
    /// The diagnostic carries the offending value so the
    /// operator does not have to inspect the file by hand.
    #[test]
    fn read_wav_mono_rejects_sub_min_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 100 Hz is well below the 8 kHz floor.
        let path = write_silence_wav(dir.path(), "slow.wav", 100, 50);
        let err = read_wav_mono(&path).expect_err("100 Hz must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "sample_rate");
                assert_eq!(got, 100);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    /// Sibling cap on the upper end: a sample rate above
    /// `audio_io::mic_arbitrator::MAX_SAMPLE_RATE` (192 kHz)
    /// is rejected at the metadata gate.
    #[test]
    fn read_wav_mono_rejects_super_max_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 384 kHz is twice the 192 kHz ceiling.
        let path = write_silence_wav(dir.path(), "fast.wav", 384_000, 50);
        let err = read_wav_mono(&path).expect_err("384 kHz must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "sample_rate");
                assert_eq!(got, 384_000);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    /// WAV header with channel count above
    /// `audio_io::mic_arbitrator::MAX_CHANNELS` is rejected at
    /// the metadata gate.  The capture-side validator already
    /// refuses these candidates; WAV ingest now refuses the
    /// same set so the daemon's PCM admission contract is
    /// unified across live and offline paths.
    #[test]
    fn read_wav_mono_rejects_oversized_channels() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 9 > MAX_CHANNELS (8).
        let path = write_silence_multichannel_wav(dir.path(), "wide.wav", 16_000, 9, 50);
        let err = read_wav_mono(&path).expect_err("9-channel must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "channels");
                assert_eq!(got, 9);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    /// A WAV exactly at the cap is accepted.  Prevents the
    /// duration gate from being off-by-one against
    /// [`MAX_INPUT_DURATION_SECS`].
    #[test]
    fn read_wav_mono_accepts_at_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        // `MAX_INPUT_DURATION_SECS * sr` frames is exactly
        // the cap; must succeed.  Use 8 kHz to keep the
        // file small (<=  ~960 KiB on disk).
        let sr: u32 = 8_000;
        let n_frames = MAX_INPUT_DURATION_SECS * sr;
        let path = write_silence_wav(dir.path(), "at_cap.wav", sr, n_frames);
        let (got_sr, mono) = read_wav_mono(&path).expect("at-cap wav must succeed");
        assert_eq!(got_sr, sr);
        assert_eq!(mono.len(), n_frames as usize);
    }

    /// `to_waveform` with a fresh cache builds a resampler
    /// keyed by the first file's `sr`; a follow-up call at a
    /// different `sr` rebuilds rather than reusing the wrong
    /// rate ratio.  This is the contract the prior
    /// `Option<SincResampler>` shape silently violated --
    /// mixed-rate datasets would then carry the first file's
    /// ratio for every later file.
    #[test]
    fn to_waveform_cache_rebuilds_on_rate_change() {
        // Two short fake mono signals at different source
        // rates.  Length is irrelevant for the cache contract;
        // the assertion is over `cache.cached_rate()` after
        // each call.
        let mono_16k: Vec<f32> = vec![0.0; 16_000];
        let mono_48k: Vec<f32> = vec![0.0; 48_000];
        let mut cache = ResamplerCache::empty();
        assert_eq!(cache.cached_rate(), None, "fresh cache must be empty");

        // First call at 16 kHz: cache populates with sr=16000.
        let _ = to_waveform(16_000, mono_16k.clone(), &mut cache)
            .expect("16 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(16_000),
            "after first call, cache must hold the 16 kHz resampler",
        );

        // Second call at 48 kHz must rebuild; the prior
        // shape would have silently kept the 16->44.1 ratio.
        let _ = to_waveform(48_000, mono_48k.clone(), &mut cache)
            .expect("48 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(48_000),
            "cache must rebuild on rate change; got rate {:?}",
            cache.cached_rate(),
        );

        // Third call back at 16 kHz must rebuild again
        // (cached rate is currently 48 kHz).
        let _ = to_waveform(16_000, mono_16k, &mut cache)
            .expect("repeat 16 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(16_000),
            "cache must rebuild when switching back to a previously-seen rate",
        );

        // At-target rate (44.1 kHz) does not touch the
        // cache: no resample needed.
        let prev = cache.cached_rate();
        let mono_44k: Vec<f32> = vec![0.0; 44_100];
        let _ = to_waveform(TARGET_SR, mono_44k, &mut cache)
            .expect("target-rate to_waveform must succeed without cache use");
        assert_eq!(
            cache.cached_rate(),
            prev,
            "target-rate path must not modify the cache",
        );
    }

    /// `to_waveform` produces a `WaveformLen`-sized buffer
    /// even when the input is much longer than the
    /// 1-second inference window.  The early-exit in the
    /// resample loop must NOT truncate the output below
    /// the window size for an input that, after
    /// resampling, comfortably exceeds the window.
    ///
    /// Uses `sr == TARGET_SR` so the test doesn't depend on
    /// rubato's exact ratio-output count; the
    /// pad-or-truncate path inside `to_waveform` still runs.
    #[test]
    fn to_waveform_truncates_long_input_at_target_rate() {
        // Twice the inference window worth of samples.
        let n = WaveformLen::USIZE * 2;
        let mono: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.001).collect();
        let mut cache = ResamplerCache::empty();
        let arr = to_waveform(TARGET_SR, mono.clone(), &mut cache)
            .expect("at-target to_waveform must succeed");
        assert_eq!(arr.len(), WaveformLen::USIZE);
        // The first `WaveformLen` samples must come through
        // unchanged when no resampling is needed.
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, mono[i], "expected pass-through at sample {i}");
        }
    }
}
