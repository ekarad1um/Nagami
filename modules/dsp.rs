//! Neutral DSP utilities shared across capture, preprocessing,
//! and codec paths.
//!
//! `dsp` is the workspace's single source of truth for shared
//! signal-processing configuration that is NOT owned by any
//! single subsystem.  Today the only inhabitant is
//! [`resample`]: the canonical sinc-resampler parameters used
//! by streaming capture (`audio_io::mic_arbitrator`), Opus
//! broadcast (`opus_stream`), and offline WAV ingest
//! (`preproc::wav_io`).
//!
//! Pre-relocation the resampler lived under `audio_io` even
//! though `preproc` and `opus_stream` consumed it.  That
//! created two upward layer-graph edges
//! (`preproc -> audio_io`, `opus_stream -> audio_io`) for
//! reasons unrelated to capture.  Moving the module here
//! removes those edges: capture, preproc, and codec each
//! depend on `dsp` only, and `dsp` depends on neither.
//!
//! This module remains intentionally small.  Cross-cutting
//! preprocessing logic (STFT, windowing, WAV ingest) stays in
//! `preproc` because it is not shared by capture or codec; the
//! `dsp` umbrella only collects items that genuinely have
//! more than one consumer outside `preproc`.

pub mod resample;
