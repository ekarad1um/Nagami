//! Audio capture, intra-mic channel arbitration, and streaming
//! resample.
//!
//! # Architecture
//!
//! One arbitrator thread ([`mic_arbitrator::MicArbitrator`])
//! owns the active capture source, demuxes its interleaved
//! frames into per-channel slots, picks the active channel via
//! RMS arbitration with hysteresis + dwell, resamples to 44.1
//! kHz if needed, and writes the result to the daemon's
//! [`crate::audio_buffer::AudioBuffer`] via its
//! [`crate::audio_buffer::Writer`].
//!
//! Two source types implement the capture surface:
//!
//! - `source::AlsaSource` -- Linux only (gated by
//!   `target_os = "linux"` AND the `alsa-real` crate feature).
//!   Multi-channel `snd_pcm_readi` with xrun + ENODEV recovery.
//! - `source::MockSource` -- cross-platform synthetic source
//!   for tests, the daemon's `--mock-audio` path, and macOS
//!   dev iteration.  Wired up via `audio_io::mock::Waveform`
//!   in the mic catalogue.
//!
//! # Why no cross-mic arbitration
//!
//! The arbitrator's model is **intra-mic channel
//! arbitration**: a single mic with multiple raw channels
//! (e.g. a beamform-less array) is opened, its whitelisted
//! channels are RMS-tracked in software, and the loudest one
//! feeds the audio buffer.  There is no automatic
//! switching between mics based on signal level: operators
//! either fix a mic by id or accept "first available"
//! failover.  See the module-level docs of
//! [`mic_arbitrator`] for rationale.

#![warn(missing_debug_implementations)]

pub mod mic_arbitrator;
pub mod mock;
// `source` is `pub(crate)`: only `mic_arbitrator` (sibling)
// and source children consume it.
pub(crate) mod source;
