//! Synthetic waveform shape used by
//! [`crate::audio_io::mic_arbitrator::CandidateSource::Mock`].
//!
//! The mock-source implementation (multi-channel synthesis +
//! real-time pacing) lives in
//! `crate::audio_io::source::mock`; this module exists
//! solely to home the [`Waveform`] enum so it can be
//! constructed by config + tests without pulling the source
//! impl in.

/// Synthetic waveform shape.  Adding a new variant only
/// requires updating the per-channel synthesis loop in
/// `crate::audio_io::source::mock`.
///
/// Mock-in-config is intended for tests + dev environments;
/// production deployments use
/// [`crate::audio_io::mic_arbitrator::CandidateSource::Alsa`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Waveform {
    /// All-zero PCM.  Useful as a low-RMS reference for
    /// channel-arbitration tests.
    Silence,
    /// Sine wave with continuous phase across blocks.
    Sine {
        /// Frequency in Hz.
        freq_hz: f32,
        /// Linear amplitude in `[-amplitude, amplitude]`.
        /// 0.5 is a sensible monitoring level; 1.0 saturates
        /// downstream nonlinearities.
        amplitude: f32,
    },
    /// Deterministic white noise via a 64-bit LCG.  Seeded
    /// for reproducibility: same `seed` produces
    /// bit-identical samples.
    WhiteNoise { amplitude: f32, seed: u64 },
    /// Index-as-f32 ramp; mostly for tests where each sample
    /// must identify itself unambiguously.  The ramp is
    /// `(absolute_sample_idx & 0xFFFF) as f32` so it doesn't
    /// grow unboundedly.
    Counter,
    /// Sine wave whose amplitude alternates between
    /// `high_amp` and `low_amp` on a 50/50 duty cycle
    /// (`half_period_samples` per half).  Designed for
    /// end-to-end tests of the arbitrator's channel
    /// auto-switch: pair two channels with opposite
    /// `inverted` values so they take turns being the louder
    /// one and the arbitrator must follow the flip across
    /// dwell + hysteresis.  The phase of the underlying sine
    /// is continuous across amplitude transitions (no
    /// click), so any audible discontinuity in downstream
    /// output isolates an arbitrator-side issue.
    ///
    /// `half_period_samples = 0` is clamped to 1 so the
    /// modulo in synthesis can't divide by zero.  That
    /// degenerate case alternates high/low every sample;
    /// pass a sane positive value for realistic tests.
    PingPongSine {
        freq_hz: f32,
        high_amp: f32,
        low_amp: f32,
        /// Length of one half of the cycle in samples; total
        /// cycle is `2 * half_period_samples`.  E.g.
        /// `sample_rate / 2` for a 0.5 s loud / 0.5 s quiet
        /// alternation.
        half_period_samples: u32,
        /// `false` -- start in the `high_amp` half; `true`
        /// -- start in the `low_amp` half.  Setting opposite
        /// values on two channels makes them anti-phase
        /// loud-vs-quiet.
        inverted: bool,
    },
}
