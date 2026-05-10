//! Continuous Opus encoder pipeline.
//!
//! Topology:
//!
//! ```text
//!   crate::audio_buffer::Reader  --(44.1 kHz mono PCM)-->  Streaming resampler
//!                                                          |
//!                                              48 kHz mono PCM (steady state)
//!                                                          |
//!                                                  20 ms framer (960 samples)
//!                                                          |
//!                                                  opus Encoder
//!                                                          |
//!                                                  20 ms Opus packet
//!                                                          |
//!                                            tokio::sync::broadcast::Sender
//!                                                          |
//!                                          UDS / WS subscribers
//! ```
//!
//! ## Subscriber-driven pause
//!
//! `run` accepts a `watch::Receiver<usize>` whose value is the count of
//! active `/stream/audio` connections.  When the count is 0 the task
//! parks on `subscribers.changed()` and consumes nothing from the audio
//! buffer (which keeps producing audio for the inference engine -- the
//! reader's `seek_latest` on resume drops the unread backlog).  When
//! the count goes positive we **rebuild** the encoder + resampler and
//! `seek_latest(BACKLOG_SAMPLES)` on the reader; the resulting fresh
//! encoder + a few hundred ms of pre-roll smooths the connect at the
//! cost of a small extra CPU spike on resume.
//!
//! ## Resampler reset semantics
//!
//! The streaming resampler uses an internal FIR with ~256 samples of
//! state.  Calling `reset()` mid-stream produces a click (zeroes the
//! FIR history, which propagates through subsequent output).  Rules:
//!
//! - **Steady-state playback**: NEVER reset.  The `Streaming` wrapper
//!   handles that automatically -- only the engine's pause path even
//!   has access to the reset method.
//! - **After `Lagged` from the AudioBuffer**: DO reset.  The reader has
//!   already `seek_latest`'d past unread samples, so the listener is
//!   about to hear a discontinuity anyway; the FIR transient is
//!   masked by the larger glitch.
//!
//! Implementation rule: `reset()` is paired with `seek_latest()`,
//! never separated.  See [`OpusEngine::reset_after_discontinuity`].

#![warn(missing_debug_implementations)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use opus::{Application, Bitrate, Channels, Encoder};
use thiserror::Error;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use crate::audio_buffer::{ReadStatus, Reader};
use crate::dsp::resample::Streaming;

// MARK: Opus tuning knobs
//
// All tuned for the daemon's Pi 5 target: a single mono speech/music
// stream encoded alongside the inference engine on the same SoC.  The
// philosophy is "trade encoder cycles for inference headroom" — we
// sit comfortably above Opus's transparent-speech floor (~24 kbps) at
// the chosen bitrate, drop encoder complexity from libopus's default
// (9–10) to 5 so the encoder costs ~30 % less CPU for an audibly
// indistinguishable result, and pick a 20 ms frame for the standard
// latency/efficiency balance.  Raising complexity back to 9–10, or
// the bitrate above 32 kbps, is a one-line change if a future host
// can spare the cycles.

/// Output sample rate of the streaming pipeline.  Opus supports
/// 8/12/16/24/48 kHz; we pick 48 to match high-quality monitoring.
pub const OUT_RATE_HZ: u32 = 48_000;

/// Source sample rate of the [`crate::audio_buffer::AudioBuffer`] feed
/// (written by [`crate::audio_io::mic_arbitrator`]).
pub const IN_RATE_HZ: u32 = 44_100;

/// Opus frame duration in milliseconds.  20 ms is the canonical balance
/// between codec efficiency and end-to-end latency.  Other Opus-legal
/// values: 2.5, 5, 10, 20, 40, 60.
pub const FRAME_MS: u32 = 20;

/// Number of mono samples in one Opus encoder input frame at 48 kHz.
/// `48_000 * 20 / 1000 = 960`.
pub const FRAME_SAMPLES: usize = (OUT_RATE_HZ as usize) * (FRAME_MS as usize) / 1000;

/// Per-call output buffer cap recommended by libopus for one mono frame
/// at 48 kHz.  The Opus RFC specifies an absolute upper bound of 1275
/// bytes for any single packet, but the libopus reference doc
/// recommends 4000 to leave headroom for repacketized output.  We size
/// at 4000 because (a) the cost is negligible (~16 KB across four
/// scratch buffers) and (b) it removes any worry about VBR spikes.
pub const MAX_PACKET_BYTES: usize = 4000;

/// Target encoder bitrate, in bits/sec.  32 kbps sits comfortably above
/// Opus's transparent threshold for speech (~24 kbps) and well inside
/// the "good" zone for music at 48 kHz mono -- halving the wire
/// bandwidth vs libopus's auto default (~64 kbps for this configuration)
/// with no perceptually relevant cost on mic content.
///
/// Average packet size at this bitrate is ~80 B (32_000 bits/s x 0.020 s
/// / 8 = 80 B); peaks under VBR remain well below `MAX_PACKET_BYTES`.
pub const BITRATE_BPS: i32 = 32_000;

/// Encoder complexity (0..=10).  libopus defaults to 9-10 (deepest
/// search).  We pin to 5 for a ~30 % encode-CPU reduction on the
/// daemon's Pi 5 target, validated as audibly indistinguishable from
/// 9 at the chosen bitrate; raising back to 9-10 is a one-line change
/// if a future host can afford the extra cycles.
pub const COMPLEXITY: i32 = 5;

/// Per-pull chunk size from the audio buffer.  1024 samples = ~23 ms at
/// 44.1 kHz, matching the resampler's internal `input_frames_next()`.
/// Exactly one chunk -> one resampler `process` call -> ~1115 output
/// samples -> one or two complete 20 ms windows ready to encode.
pub const PCM_PULL_CHUNK: usize = 1024;

/// Pre-roll on resume.  We drop the backlog accumulated while paused
/// (the listener wasn't listening) and seek to a position
/// `BACKLOG_SAMPLES` behind the live edge.  This gives the encoder
/// ~100 ms of audio before the live edge so the first packet is full,
/// not silence-padded.
pub const BACKLOG_SAMPLES: usize = 4096;

/// Per-iteration timeout when waiting for audio in the active state.
/// Short enough to react quickly to subscriber changes (we won't park
/// for longer than this); long enough not to spin.
const ACTIVE_TICK: Duration = Duration::from_millis(10);

/// Pipeline failures from libopus init/encode/reset and from
/// the upstream resampler.  The underlying `opus::Error` is boxed
/// behind `Box<dyn Error + Send + Sync>` so an `opus` crate
/// version bump is not a SemVer break for `opus_stream` consumers;
/// the `EncoderInternal` `stage` tag preserves log-grep ability.
#[derive(Debug, Error)]
pub enum OpusError {
    /// Libopus FFI returned an error at encoder construction,
    /// per-frame encode, or reset_after_discontinuity.  `stage`
    /// is one of `"init"`, `"encode"`, or `"reset"`; `source`
    /// is the boxed underlying error, opaque to consumers.
    #[error("opus encoder ({stage}) failed: {source}")]
    EncoderInternal {
        stage: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("encode produced 0 bytes (empty packet) -- encoder state corrupt?")]
    EmptyEncode,
    /// Non-finite (NaN / +-Inf) PCM sample at `index`.  The
    /// whole slice is rejected: libopus `encode_float` is
    /// undefined on non-finite, and the sinc resampler FIR
    /// would smear contamination across subsequent frames.
    /// Surfaced as a closed subscriber (1011), not a daemon
    /// abort.
    #[error("non-finite PCM sample at index {index} (NaN/Inf rejected before encode)")]
    BadPcm { index: usize },
    /// The upstream sinc resampler returned a typed failure
    /// (rubato or audioadapter-buffers API contract break).
    /// Routing through `OpusError` lets the broadcast path close
    /// the subscriber cleanly with 1011 instead of killing the
    /// process; the mic_arbitrator caller still aborts via its
    /// `catch_unwind + process::abort` wrapper, which is the
    /// right shape for the audio-timing producer.
    #[error("opus resampler internal: {source}")]
    ResamplerInternal {
        #[source]
        source: crate::dsp::resample::StreamingResampleError,
    },
}

impl OpusError {
    /// Boundary helper: wrap any `opus::Error` (or other
    /// underlying encoder failure) at a known stage.  Keeps the
    /// `Box::new` away from call sites so the wrapping shape can
    /// change without touching each `map_err`.
    fn encoder_internal(
        stage: &'static str,
        e: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OpusError::EncoderInternal {
            stage,
            source: Box::new(e),
        }
    }
}

impl crate::common::error::Categorized for OpusError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        // Every variant is daemon-internal: encoder
        // failures surface from libopus on a known-good
        // input stream the arbitrator already validated.
        crate::common::error::ErrorKind::Internal
    }
}

/// CPU-side encoder pipeline: 44.1 kHz f32 PCM in -> 20 ms Opus packets
/// out.  Stateless across pause/resume -- callers rebuild a fresh
/// `OpusEngine` after a discontinuity to avoid carrying stale FIR or
/// encoder predictor state into a fresh stream.
///
/// `process_pcm` accepts arbitrary-length input slices and emits zero
/// or more 20 ms packets per call -- output cadence depends on how
/// much input has accumulated relative to FRAME_SAMPLES.
pub struct OpusEngine {
    encoder: Encoder,
    resampler: Streaming,
    /// Pre-allocated 4 KiB scratch for one encoded packet (reused
    /// across calls; freezed-out as `Bytes` to the broadcast channel).
    encode_scratch: Vec<u8>,
    /// Resampler output ringlet.  The encoder reads `FRAME_SAMPLES`
    /// at a time from the front of this buffer and we then `drain`
    /// the consumed prefix; no separate frame scratch is needed
    /// (`encode_float` takes `&[f32]`, so we avoid the 3.8 KB / packet
    /// memcpy that an intermediate frame Vec would force).
    out_pcm: Vec<f32>,
}

impl std::fmt::Debug for OpusEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEngine")
            .field("resampler", &self.resampler)
            .field("out_pcm_len", &self.out_pcm.len())
            .finish_non_exhaustive()
    }
}

impl OpusEngine {
    pub fn new() -> Result<Self, OpusError> {
        let mut encoder = Encoder::new(OUT_RATE_HZ, Channels::Mono, Application::Audio)
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        // Pin bitrate + complexity.  Both are CTL calls on
        // the freshly built encoder; failure means the FFI
        // handle is broken, so we route through
        // `EncoderInternal { stage: "init" }` rather than
        // a separate variant.
        encoder
            .set_bitrate(Bitrate::Bits(BITRATE_BPS))
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        encoder
            .set_complexity(COMPLEXITY)
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        Ok(Self {
            encoder,
            resampler: Streaming::new(IN_RATE_HZ, OUT_RATE_HZ),
            encode_scratch: vec![0u8; MAX_PACKET_BYTES],
            out_pcm: Vec::with_capacity(FRAME_SAMPLES * 4),
        })
    }

    /// Feed `pcm_44k_in` into the resampler and drain as many complete
    /// 20 ms Opus packets as become ready.  Returns the number of
    /// packets pushed into `out`.
    ///
    /// Per-frame allocations:
    ///   * 1 `Bytes` per emitted packet (sized exactly to `n`).
    ///   * `Streaming::drain_output_into` moves samples into
    ///     `out_pcm` without allocating (both `Vec`s' capacities
    ///     persist across calls), and `Streaming::process` itself
    ///     writes into a reusable scratch owned by the resampler.
    pub fn process_pcm(
        &mut self,
        pcm_44k_in: &[f32],
        out: &mut Vec<Bytes>,
    ) -> Result<usize, OpusError> {
        // Reject non-finite PCM up front (libopus undefined; sinc
        // FIR would smear).
        if let Some(index) = pcm_44k_in.iter().position(|s| !s.is_finite()) {
            return Err(OpusError::BadPcm { index });
        }
        // `Streaming::process` short-circuits on empty input itself.
        // Typed Result lets a future rubato API
        // change degrade as a closed subscriber (1011) instead of
        // a daemon abort.
        self.resampler
            .process(pcm_44k_in)
            .map_err(|source| OpusError::ResamplerInternal { source })?;
        // Pull newly-resampled samples into the local ringlet
        // without allocating: `drain_output_into` moves elements and
        // preserves capacities on both sides.
        self.resampler.drain_output_into(&mut self.out_pcm);

        let mut emitted = 0usize;
        while self.out_pcm.len() >= FRAME_SAMPLES {
            // Encode directly out of the ringlet.  `&self.out_pcm[..]`,
            // `&mut self.encoder`, and `&mut self.encode_scratch` borrow
            // three disjoint fields of `*self`, so the split-borrow is
            // sound.  The `drain` after encode shifts the unconsumed
            // tail forward in place; in production the caller feeds
            // one PCM_PULL_CHUNK at a time -> resampler emits ~1115
            // samples -> loop exits after a single drain with a
            // sub-FRAME_SAMPLES tail.
            let n = self
                .encoder
                .encode_float(&self.out_pcm[..FRAME_SAMPLES], &mut self.encode_scratch)
                .map_err(|e| OpusError::encoder_internal("encode", e))?;
            if n == 0 {
                return Err(OpusError::EmptyEncode);
            }
            // Copy out the encoded prefix as an owned `Bytes`.  The
            // copy is unavoidable (the broadcast receivers outlive
            // `encode_scratch`); `Bytes::copy_from_slice` allocates
            // exactly `n` bytes -- no headroom-padding waste.
            out.push(Bytes::copy_from_slice(&self.encode_scratch[..n]));
            self.out_pcm.drain(..FRAME_SAMPLES);
            emitted += 1;
        }
        Ok(emitted)
    }

    /// Reset internal FIR + encoder state after a `Lagged` event.  The
    /// listener has already accepted a glitch; we drop the resampler
    /// history (256-sample FIR transient is masked) and call
    /// `Encoder::reset_state` (libopus `OPUS_RESET_STATE` ctl) to clear
    /// the SILK/CELT predictor state without rebuilding the encoder.
    pub fn reset_after_discontinuity(&mut self) -> Result<(), OpusError> {
        self.resampler.reset_after_discontinuity();
        self.out_pcm.clear();
        self.encoder
            .reset_state()
            .map_err(|e| OpusError::encoder_internal("reset", e))?;
        Ok(())
    }
}

/// Async run loop.  Spawned by the daemon as a regular tokio task (no
/// `spawn_blocking` needed -- both libopus encode and rubato resample
/// are us-range CPU per call, suitable for inline async).
///
/// State machine:
///
/// 1. **Paused**: `subscribers == 0`.  Park on `subscribers.changed()`.
///    The OpusEngine is dropped; reader is held but not read.
/// 2. **Active**: `subscribers > 0`.  Build a fresh OpusEngine, seek
///    the reader to `BACKLOG_SAMPLES` behind the live edge, then loop
///    pulling PCM, feeding the engine, and broadcasting packets.
///
/// `packets_encoded` is bumped (Relaxed) once per Opus packet
/// the encoder emits and hands to the broadcast channel.  The
/// counter reflects encoder progress, not delivery: it
/// increments even when zero subscribers are attached (the
/// `broadcast::Sender::send` result is discarded), so the
/// daemon's heartbeat reads it to detect a stalled encoder
/// (subscribers > 0 but no new encoded frames for >= 2
/// heartbeat periods -> unhealthy heartbeat).  Pre-rename this
/// counter was `packets_emitted`; renamed to make the encoder-
/// progress (not delivery) semantics self-documenting.
///
/// `timing_anchor`, when present, plumbs the producer-side
/// [`crate::common::time::BufferTimingAnchor`] into each
/// emitted `AudioFrame.t_us_capture_monotonic` so the field
/// reflects the capture monotonic time of the FIRST 44.1 kHz
/// sample of the chunk that produced this packet (window-start
/// semantic).  When `None`, the encoder falls back to
/// stamping `CaptureTime::now()` at emit time -- correct
/// clock domain, but a publish-time stamp masquerading as a
/// capture-time stamp.  See
/// [`crate::common::time::capture_us_for`] for the projection
/// math.
///
/// Returns `Ok(())` on shutdown, `Err(OpusError)` if the encoder
/// fails irrecoverably (the daemon supervisor restarts the task).
pub async fn run(
    mut reader: Reader,
    mut subscribers: watch::Receiver<usize>,
    out: broadcast::Sender<Bytes>,
    shutdown: CancellationToken,
    packets_encoded: Arc<AtomicU64>,
    timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
) -> Result<(), OpusError> {
    // Scratch reused across packet bursts.  Holds the Bytes packets
    // emitted by `process_pcm`; we drain them into the broadcast
    // channel each iteration.
    let mut packet_scratch: Vec<Bytes> = Vec::with_capacity(4);
    let mut pcm_scratch = vec![0.0f32; PCM_PULL_CHUNK];
    // Envelope-encode scratch reused
    // across every emitted AudioFrame via `wrap_audio_into`.
    // `BytesMut::split().freeze()` returns the head as an
    // Arc-backed `Bytes` (zero-copy fan-out across the broadcast
    // channel's clones) and re-uses the residual capacity for
    // the next packet.  After a few packets the underlying
    // allocation is reused indefinitely; the per-packet
    // `wrap_audio` Vec alloc that this replaces was the
    // dominant heap activity in the 50 Hz audio path.  1 KiB
    // initial capacity covers a typical 20 ms Opus packet
    // (~80 B) + envelope tax (~4 B) with comfortable headroom.
    let mut encode_buf: bytes::BytesMut = bytes::BytesMut::with_capacity(1024);
    // Per-packet seq counter.  Stamped on every emitted
    // `AudioFrame`; restarts from 0 on daemon reboot (matches
    // `InferenceFrame.seq` semantics).
    let mut audio_seq: u64 = 0;
    // Last `peek_into` outcome.  Drives the
    // conditional `sleep(ACTIVE_TICK)` below: only sleep when the
    // ring genuinely has no new samples (`Wait`).  On `Ready` /
    // `Lagged` the next iteration pumps immediately, eliminating
    // the unconditional 100 Hz wakeup that otherwise prevented the
    // kernel from entering deeper cpuidle C-states (~50-100 mW
    // continuous draw).  Initial `Wait` sleeps once before the
    // first peek so the very-first tick after subscriber arrival
    // gives audio_buffer a moment to fill.
    let mut last_status = ReadStatus::Wait;

    loop {
        if shutdown.is_cancelled() {
            return Ok(());
        }

        // Paused
        // `borrow()` does NOT mark-seen on the watch receiver -- only
        // `changed()` does.  So if a subscriber arrives in the gap
        // between this read and the `changed()` await below, the
        // pending change still wakes us (changed() returns Ok
        // immediately for any unseen value).
        if *subscribers.borrow() == 0 {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                changed = subscribers.changed() => {
                    if changed.is_err() {
                        // Watch channel closed: subscribers gone forever
                        // -> end of life for this task.
                        return Ok(());
                    }
                }
            }
            continue;
        }

        // Active: build a fresh engine, drop the backlog
        let mut engine = OpusEngine::new()?;
        reader.seek_latest(BACKLOG_SAMPLES);
        tracing::info!(
            target: "opus_stream",
            subscribers = *subscribers.borrow(),
            "audio stream resumed; encoder + resampler rebuilt",
        );

        // Active loop
        loop {
            if shutdown.is_cancelled() {
                return Ok(());
            }
            if *subscribers.borrow() == 0 {
                tracing::info!(target: "opus_stream", "audio stream paused; dropping encoder");
                break; // back to outer paused branch
            }

            // Only sleep when the previous peek
            // returned `Wait` (no new samples in the ring).  On
            // `Ready` / `Lagged` we have data to pump *now*; the
            // unconditional 100 Hz wakeup of the shape
            // prevented the kernel from reaching deeper cpuidle
            // C-states even when audio was flowing steadily.
            //
            // Each select arm here is cancel-safe: `cancelled()` is
            // idempotent, `changed()` only marks-seen on success, and
            // `sleep(ACTIVE_TICK)` carries no state.  The loop-top
            // re-check is load-bearing -- if a future refactor moves
            // the count check below the select, a 0->1->0 toggle
            // within one tick could be missed.
            if matches!(last_status, ReadStatus::Wait) {
                tokio::select! {
                    biased;
                    _ = shutdown.cancelled() => return Ok(()),
                    changed = subscribers.changed() => {
                        if changed.is_err() {
                            return Ok(());
                        }
                        // Loop top will re-check the count.
                        continue;
                    }
                    _ = tokio::time::sleep(ACTIVE_TICK) => {}
                }
            }

            // Pull a chunk; feed; emit.
            last_status = reader.peek_into(&mut pcm_scratch);
            match last_status {
                ReadStatus::Wait => continue,
                ReadStatus::Lagged { by } => {
                    tracing::warn!(
                        target: "opus_stream",
                        by_samples = by,
                        "audio reader lagged in encoder; resync + reset",
                    );
                    reader.seek_latest(BACKLOG_SAMPLES);
                    engine.reset_after_discontinuity()?;
                    continue;
                }
                ReadStatus::Ready => {}
            }
            // Snapshot tail BEFORE advance: the stamp must
            // reference the chunk's first sample (window-start).
            // Two slack sources: resampler input buffering
            // (~360 us), and an occasional pull yielding >1 opus
            // packet -- in that drain every packet shares this
            // tail, so the second drifts one opus frame.
            let pull_start_tail = reader.tail();
            reader.advance(PCM_PULL_CHUNK);

            // `packet_scratch` is empty here -- the prior `drain(..)`
            // emptied it and error paths return from `run`.
            engine.process_pcm(&pcm_scratch, &mut packet_scratch)?;
            for pkt in packet_scratch.drain(..) {
                // Wrap each Opus packet in
                // `AudioFrame { codec: Opus(pkt), ... }` then in
                // `Envelope { schema_version: 1, payload: Audio(...) }`.
                // ~4 bytes envelope tax per packet (negligible
                // at 50 Hz audio); receivers decode the Envelope
                // first and dispatch on its `payload` oneof.
                audio_seq = audio_seq.wrapping_add(1);
                // `t_us_capture_monotonic` projects the chunk's
                // first-sample tail position back to the capture
                // monotonic time the producer recorded for that
                // sample (window-start convention).  The daemon
                // wires a [`crate::common::time::BufferTimingAnchor`]
                // through; tests and adversarial harnesses pass
                // `None`, in which case we fall back to a
                // publish-time `CaptureTime::now()` stamp.
                let t_us_capture_monotonic = match timing_anchor.as_ref() {
                    Some(cell) => {
                        let anchor = **cell.load();
                        crate::common::time::capture_us_for(anchor, pull_start_tail)
                    }
                    None => crate::common::time::CaptureTime::now().as_micros(),
                };
                let frame = crate::proto::AudioFrame {
                    seq: audio_seq,
                    t_us_capture_monotonic: Some(t_us_capture_monotonic),
                    t_us_publish_unix: crate::common::time::WallTime::now().map(|w| w.as_micros()),
                    sample_rate: Some(48_000),
                    frame_duration_ms: Some(20),
                    codec: Some(crate::proto::audio_frame::Codec::Opus(pkt)),
                };
                // `wrap_audio_into`
                // reuses `encode_buf`'s allocation across
                // packets.  After warm-up the per-packet
                // envelope-encode allocation rate drops to
                // zero; the returned Bytes is Arc-backed so
                // `broadcast::Sender::send` clones it by
                // bumping the Arc, never copying the bytes.
                let envelope_bytes = crate::proto::framing::wrap_audio_into(&mut encode_buf, frame);
                // SendError(_) means no receivers; that's fine.
                // The subscriber count is technically the upstream
                // signal for this, but a brief race window is OK
                // (count drops to 0 just as we're emitting).
                let _ = out.send(envelope_bytes);
                // Bumped after `out.send` so the counter only
                // reflects packets the encoder produced and
                // handed to the broadcast channel.  This is
                // encoder-progress, NOT delivery: a packet that
                // no subscriber received still counts here.
                // The heartbeat consumer wants encoder-progress
                // (proof the encode loop is making forward
                // progress); see the renamed parameter docs.
                packets_encoded.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: 960 samples x 20 ms x 50 = 1 s.  Constants align.
    #[test]
    fn frame_samples_is_960() {
        assert_eq!(FRAME_SAMPLES, 960);
    }

    /// OpusEngine creates without panicking.  Encoder init can fail
    /// only if the system libopus is missing or mismatched -- neither
    /// is true on host CI (audiopus_sys vendors libopus).
    #[test]
    fn opus_engine_constructs() {
        let _e = OpusEngine::new().expect("engine init");
    }

    /// `OpusEngine::new` actually applies the bitrate + complexity CTLs.
    /// Without this, a refactor that drops one of the `set_*` calls
    /// would silently revert to libopus's auto defaults -- packet sizes
    /// would balloon (~2x wire bandwidth) and CPU would jump on the
    /// Pi 5 target without a single failing test.  We round-trip via
    /// the encoder's own getters rather than asserting payload sizes
    /// to keep the check independent of input shape.
    #[test]
    fn encoder_pins_bitrate_and_complexity() {
        let mut e = OpusEngine::new().expect("engine init");
        match e.encoder.get_bitrate().expect("get_bitrate") {
            Bitrate::Bits(n) => assert_eq!(n, BITRATE_BPS, "bitrate not pinned"),
            other => panic!("expected Bits({BITRATE_BPS}), got {other:?}"),
        }
        assert_eq!(
            e.encoder.get_complexity().expect("get_complexity"),
            COMPLEXITY,
            "complexity not pinned",
        );
    }

    /// 1 kHz sine through encode -> decode round-trip.  Since Opus is
    /// lossy and the sinc resampler introduces non-integer phase
    /// drift, sample-domain RMSE is too phase-brittle for a pure tone
    /// (any sub-sample misalignment between input and output produces
    /// double-digit RMSE percentages even for a perfect codec).  We
    /// instead measure spectral fidelity:
    ///
    /// 1. Body energy is preserved within 5 % of input RMS.
    /// 2. At least 95 % of body energy lies in a narrow band around
    ///    1 kHz.
    ///
    /// Together these test "decodes cleanly" + "without
    /// blurring the tone into broadband noise" without
    /// brittle sample-level comparisons.
    #[test]
    fn round_trip_1khz_tone_spectral_energy_at_1khz() {
        use opus::{Channels, Decoder};

        const TONE_HZ: f32 = 1000.0;
        const TONE_AMPL: f32 = 0.5;

        // 1 second of TONE_HZ at IN_RATE_HZ.
        let n_in = IN_RATE_HZ as usize;
        let pcm: Vec<f32> = (0..n_in)
            .map(|i| {
                let t = i as f32 / IN_RATE_HZ as f32;
                TONE_AMPL * (2.0 * std::f32::consts::PI * TONE_HZ * t).sin()
            })
            .collect();

        let mut engine = OpusEngine::new().expect("engine");
        let mut packets: Vec<Bytes> = Vec::new();
        for chunk in pcm.chunks(PCM_PULL_CHUNK) {
            engine.process_pcm(chunk, &mut packets).expect("encode");
        }
        assert!(packets.len() > 30, "too few packets: {}", packets.len());

        let mut decoder = Decoder::new(OUT_RATE_HZ, Channels::Mono).expect("decoder");
        let mut decoded = Vec::with_capacity(packets.len() * FRAME_SAMPLES);
        let mut frame_buf = vec![0f32; FRAME_SAMPLES];
        for pkt in &packets {
            let n = decoder
                .decode_float(pkt.as_ref(), &mut frame_buf, false)
                .expect("decode");
            // The encoder is configured for 20 ms / 48 kHz / mono,
            // so every packet decodes to exactly FRAME_SAMPLES.  A
            // surprise here would mean a non-20-ms packet snuck in
            // and `frame_buf` would have been undersized.
            assert!(
                n <= FRAME_SAMPLES,
                "decoded {n} > FRAME_SAMPLES ({FRAME_SAMPLES})"
            );
            decoded.extend_from_slice(&frame_buf[..n]);
        }
        assert!(
            decoded.len() >= FRAME_SAMPLES * 30,
            "decoded too short: {}",
            decoded.len(),
        );

        // Skip 100 ms head (encoder + sinc lookahead) and a short tail
        // (decoder closes its predictor) so the body is steady-state.
        let skip_head = (OUT_RATE_HZ as usize) * 100 / 1000;
        let skip_tail = (OUT_RATE_HZ as usize) * 20 / 1000;
        let body = &decoded[skip_head..decoded.len() - skip_tail];
        assert!(
            body.len() > 4 * FRAME_SAMPLES,
            "comparison window too small ({})",
            body.len(),
        );

        // Goertzel-style energy at 1 kHz (sin/cos correlation).
        let omega = 2.0_f64 * std::f64::consts::PI * (TONE_HZ as f64) / (OUT_RATE_HZ as f64);
        let mut acc_cos = 0.0f64;
        let mut acc_sin = 0.0f64;
        let mut total_sq = 0.0f64;
        for (i, &v) in body.iter().enumerate() {
            let phase = omega * i as f64;
            acc_cos += v as f64 * phase.cos();
            acc_sin += v as f64 * phase.sin();
            total_sq += (v as f64).powi(2);
        }
        let n = body.len() as f64;
        // For a pure sin(omegat)*A signal: sumv.cos(omegat) ~= 0, sumv.sin(omegat) ~= N.A/2
        // -> power_at_target = A^2/4.  Multiply by 2 to express as power
        // (mean of v^2 for that component).
        let target_power = 2.0 * (acc_cos.powi(2) + acc_sin.powi(2)) / (n * n);
        let total_power = total_sq / n;
        let body_rms = total_power.sqrt() as f32;
        let in_band_frac = target_power / total_power;

        let ref_rms = TONE_AMPL / 2.0_f32.sqrt(); // 0.3536
        let amp_drift = ((body_rms - ref_rms) / ref_rms).abs();

        eprintln!(
            "round_trip_1khz: body_rms={body_rms:.4} ref_rms={ref_rms:.4} \
             amp_drift={:.2}% in_band@1kHz={:.2}% ({:.0} samples)",
            amp_drift * 100.0,
            in_band_frac * 100.0,
            n,
        );

        // (1) RMS preserved within 5 %.
        assert!(
            amp_drift < 0.05,
            "amplitude drift {:.2}% > 5% (body_rms={body_rms} vs ref_rms={ref_rms})",
            amp_drift * 100.0,
        );
        // (2) >= 95 % of body energy is in the 1 kHz spectral bin.
        // This is the spectrally-honest analogue of "<= 5 % RMSE":
        // <= 5 % of energy went anywhere other than 1 kHz.
        assert!(
            in_band_frac > 0.95,
            "only {:.1}% of body energy at 1 kHz; expected >= 95%",
            in_band_frac * 100.0,
        );
    }

    /// Soak-style: 5 seconds of white noise -> all packets decode
    /// cleanly without error.
    #[test]
    fn five_second_white_noise_all_packets_decode() {
        use opus::{Channels, Decoder};

        let n_in = (IN_RATE_HZ as usize) * 5;
        // Deterministic LCG for reproducibility (we don't depend on
        // a separate RNG crate just for tests).
        let mut s: u32 = 0xdeadbeef;
        let pcm: Vec<f32> = (0..n_in)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 0xFFFFFF as f32) * 0.5 - 0.25
            })
            .collect();

        let mut engine = OpusEngine::new().expect("engine");
        let mut packets: Vec<Bytes> = Vec::new();
        for chunk in pcm.chunks(PCM_PULL_CHUNK) {
            engine.process_pcm(chunk, &mut packets).expect("encode");
        }

        let mut decoder = Decoder::new(OUT_RATE_HZ, Channels::Mono).expect("decoder");
        let mut frame_buf = vec![0f32; FRAME_SAMPLES];
        let mut total_decoded = 0usize;
        for (i, pkt) in packets.iter().enumerate() {
            let n = decoder
                .decode_float(pkt.as_ref(), &mut frame_buf, false)
                .unwrap_or_else(|e| panic!("packet {i} decode failed: {e}"));
            assert_eq!(
                n, FRAME_SAMPLES,
                "packet {i} decoded {n} samples (expected 960)"
            );
            total_decoded += n;
        }
        assert!(
            packets.len() >= 240,
            "expected >=240 packets in 5 s, got {}",
            packets.len()
        );
        assert!(total_decoded >= 240 * FRAME_SAMPLES);
    }

    /// Property-style: every encoded packet -- across silence, a
    /// pure 1 kHz sine, deterministic white noise, and a 200 Hz
    /// saturated square wave -- must fit inside `MAX_PACKET_BYTES`.
    /// The wire-format card in the
    /// frontend appendix and the WS frame-cap pin a 4 000-byte
    /// upper bound; this test ensures that contract holds for any
    /// plausible PCM input the daemon's microphone could feed in.
    /// A future codec swap that violates the bound would now fail
    /// here BEFORE shipping a packet that the frontend would drop
    /// at decode time.
    #[test]
    fn every_packet_fits_max_packet_bytes_across_input_shapes() {
        // 0.5 s of audio per case -- enough to amortize the encoder's
        // VBR steady-state behaviour and produce ~25 packets.
        let n_in = (IN_RATE_HZ as usize) / 2;

        // Case 1: pure silence.
        let silence = vec![0.0f32; n_in];

        // Case 2: 1 kHz sine at 0.5 amplitude.
        let sine: Vec<f32> = (0..n_in)
            .map(|i| {
                let t = i as f32 / IN_RATE_HZ as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        // Case 3: white noise (deterministic LCG so the test is
        // reproducible across CI runs).
        let mut s: u32 = 0x1234_5678;
        let noise: Vec<f32> = (0..n_in)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 0xFF_FFFF as f32) * 1.8 - 0.9
            })
            .collect();

        // Case 4: saturated +1.0 / -1.0 square wave at 200 Hz --
        // worst-case for a low-bit-rate codec because it triggers
        // VBR's spike behaviour at every transition.
        let period = (IN_RATE_HZ as usize) / 200;
        let square: Vec<f32> = (0..n_in)
            .map(|i| {
                if (i / (period / 2)).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        for (label, pcm) in [
            ("silence", silence),
            ("sine_1k", sine),
            ("white_noise", noise),
            ("square_200", square),
        ] {
            let mut engine = OpusEngine::new().expect("engine");
            let mut packets: Vec<Bytes> = Vec::new();
            for chunk in pcm.chunks(PCM_PULL_CHUNK) {
                engine.process_pcm(chunk, &mut packets).expect("encode");
            }
            assert!(
                !packets.is_empty(),
                "{label}: no packets emitted for {} input samples",
                pcm.len()
            );
            for (i, pkt) in packets.iter().enumerate() {
                assert!(
                    pkt.len() <= MAX_PACKET_BYTES,
                    "{label}: packet {i} = {} bytes > MAX_PACKET_BYTES ({})",
                    pkt.len(),
                    MAX_PACKET_BYTES,
                );
                // Also assert non-empty -- the encoder should never
                // emit a zero-byte packet, and OpusError::EmptyEncode
                // exists to catch this in the engine; double-check
                // here so a future code path that goes around the
                // engine's check is still constrained.
                assert!(
                    !pkt.is_empty(),
                    "{label}: packet {i} is empty (encoder corrupt?)",
                );
            }
        }
    }

    /// NaN-laden PCM is rejected with a typed `BadPcm { index }`
    /// error before any resample / encode work happens, and the
    /// engine remains usable for valid input afterwards.  Without
    /// this guard, `opus::Encoder::encode_float` is undefined on
    /// non-finite samples and the streaming sinc FIR would smear
    /// the contamination across subsequent frames; here we assert
    /// the boundary check fires first, no panic, no corrupt frame.
    #[test]
    fn process_pcm_rejects_nan_with_index_and_does_not_emit() {
        let mut engine = OpusEngine::new().expect("engine");
        let mut pcm = vec![0.1f32; PCM_PULL_CHUNK];
        // Place NaN at a non-zero index so we also prove the index
        // is reported faithfully.
        pcm[7] = f32::NAN;
        let mut packets: Vec<Bytes> = Vec::new();
        let err = engine
            .process_pcm(&pcm, &mut packets)
            .expect_err("non-finite PCM must surface as Err");
        match err {
            OpusError::BadPcm { index } => assert_eq!(index, 7, "BadPcm index mismatch"),
            other => panic!("expected BadPcm {{ index: 7 }}, got {other:?}"),
        }
        assert!(
            packets.is_empty(),
            "no packets must be emitted when the input slice is rejected (got {})",
            packets.len(),
        );

        // Engine is not poisoned by a rejected slice -- the bad
        // input never reached the resampler, so a follow-up clean
        // slice still encodes.  Two PCM_PULL_CHUNKs guarantees at
        // least one complete 20 ms frame regardless of FIR warmup.
        let clean = vec![0.1f32; PCM_PULL_CHUNK * 2];
        let n = engine
            .process_pcm(&clean, &mut packets)
            .expect("encode after rejected slice");
        assert!(n > 0, "engine remained usable: expected >0 packets, got 0");
    }

    /// +Inf and -Inf are also rejected; the index of the FIRST
    /// non-finite sample is reported (matching `Iterator::position`
    /// semantics).  Covers both infinity polarities to guard
    /// against an accidental `is_nan()`-only check creeping into
    /// the boundary scan during a future refactor.
    #[test]
    fn process_pcm_rejects_pos_and_neg_inf() {
        for (label, bad) in [("+inf", f32::INFINITY), ("-inf", f32::NEG_INFINITY)] {
            let mut engine = OpusEngine::new().expect("engine");
            let mut pcm = vec![0.0f32; PCM_PULL_CHUNK];
            pcm[42] = bad;
            // Salt a second bad value later in the slice to confirm
            // the FIRST index wins.
            pcm[100] = f32::NAN;
            let mut packets: Vec<Bytes> = Vec::new();
            let err = engine
                .process_pcm(&pcm, &mut packets)
                .expect_err("non-finite PCM must surface as Err");
            match err {
                OpusError::BadPcm { index } => assert_eq!(
                    index, 42,
                    "{label}: expected first non-finite index 42, got {index}"
                ),
                other => panic!("{label}: expected BadPcm, got {other:?}"),
            }
            assert!(packets.is_empty(), "{label}: must not emit packets");
        }
    }

    /// `reset_after_discontinuity` clears state and lets us continue
    /// encoding after.
    #[test]
    fn reset_then_continue_works() {
        let mut engine = OpusEngine::new().expect("engine");
        let pcm = vec![0.1f32; PCM_PULL_CHUNK * 2];
        let mut packets: Vec<Bytes> = Vec::new();
        engine.process_pcm(&pcm, &mut packets).expect("encode");
        assert!(!packets.is_empty());

        engine.reset_after_discontinuity().expect("reset");
        packets.clear();
        engine
            .process_pcm(&pcm, &mut packets)
            .expect("encode after reset");
        assert!(!packets.is_empty(), "no packets emitted after reset");
    }
}
