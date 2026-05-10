//! Multi-channel synthetic capture source.
//!
//! Cross-platform; used by tests, the daemon's `--mock-audio` path,
//! and macOS dev iteration.  Generates one [`Waveform`] per channel
//! and interleaves them into the buffer the arbitrator reads.
//!
//! ## Real-time pacing
//!
//! `read_interleaved` paces in wall-clock so the arbitrator
//! observes the same cadence it would from real ALSA hardware.
//! Without pacing, `read_interleaved` would return at memory speed
//! and the audio buffer's head would advance thousands of times
//! faster than wall-clock -- downstream consumers (opus encoder,
//! inference engine) would observe "future" audio.  Pacing is
//! enforced **inside** `read_interleaved`, not by a separate
//! synthesis thread, because the new arbitrator design has only
//! one thread; the source must therefore block in line.
//!
//! ### Skew clamp on consumer stalls
//!
//! If the consumer stalls long enough that `next_block_at` would
//! otherwise fall more than one `block_dur` behind real time (e.g.
//! a long GC pause, scheduler stall, or a test that holds the
//! thread), the pacing target is reset to `now` rather than
//! emitting a burst of synthesized audio at memory speed to "catch
//! up." This matches ALSA's behaviour after an xrun + recover: the
//! next read returns samples timestamped from now, not from the
//! stale buffer position.  Without the clamp, a 1-second consumer
//! stall would cause the source to emit ~43 periods @ 23 ms back-
//! to-back, swamping the audio buffer's ring with synthesized data
//! that has stale wall-clock timestamps -- exactly the dishonesty
//! we'd flag in production.
//!
//! ## Stop responsiveness during sleep
//!
//! The arbitrator's `stop: Arc<AtomicBool>` is observed inside the
//! pacing sleep in 2 ms slices.  Worst-case stop latency from inside
//! `read_interleaved` is ~2 ms (one slice), regardless of how long
//! the period is.  This matters for clean test teardown.
//!
//! ## Determinism
//!
//! `Waveform::WhiteNoise { seed }` produces bit-identical samples
//! across runs for the same `seed`.  Per-channel rng state is
//! independent -- channel 0 and channel 1 with seeds (1, 2) generate
//! distinct streams that don't drift between runs.

use crate::audio_io::mic_arbitrator::{CandidateSource, MicCandidate};
use crate::audio_io::mock::Waveform;
use crate::audio_io::source::{ReadError, ReadOutcome};
use crate::common::ids::MicId;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Synthetic multi-channel capture source.
#[derive(Debug)]
pub struct MockSource {
    id: MicId,
    waveforms: Vec<Waveform>,
    /// Whitelist captured from the candidate at open.  Validated by
    /// [`MicCandidate::validate`]; for mock the device's channel
    /// count equals `waveforms.len()` so no intersection is needed.
    /// Sorted + deduped for parity with `AlsaSource`.
    effective_whitelist: Vec<u16>,
    period_size: usize,
    sample_rate: u32,
    /// LCG state per device-channel (one per entry of `waveforms`).
    /// Initialized from the channel's `WhiteNoise.seed` (or 1 for
    /// non-noise variants); preserved across periods so noise is
    /// continuous.
    rng_states: Vec<u64>,
    /// Wall-clock target for the next period.  Advanced by exactly
    /// one `block_dur` per successful `read_interleaved`.  Small
    /// drift (a slightly slow consumer, scheduler jitter) is
    /// allowed; if drift would exceed one `block_dur`, the
    /// module-level skew clamp resets this field to `now` to avoid
    /// emitting a burst of audio with stale wall-clock timestamps
    /// (see the module-level "Skew clamp on consumer stalls"
    /// section for the full rationale).
    next_block_at: Instant,
    /// Total samples-per-channel synthesized so far.  Index into
    /// the waveform's analytic form so phase / counter stay
    /// continuous across periods.
    absolute_idx: u64,
    /// Held alongside the source so the pacing sleep can break
    /// when the arbitrator asks to stop.
    stop: Arc<AtomicBool>,
}

impl MockSource {
    /// Open a mock source for `candidate`.  Crate-private;
    /// external callers go through [`super::open_source`].
    /// Re-validates as a guard for in-crate test callers that
    /// bypass the dispatcher -- the synthesis path's `expect`s
    /// on `NonZeroUsize::new(period_size)` and the rate divide
    /// rely on the static invariants holding.
    pub(crate) fn open(candidate: &MicCandidate, stop: Arc<AtomicBool>) -> Result<Self, String> {
        candidate
            .validate()
            .map_err(|e| format!("MockSource::open candidate failed validation: {e}"))?;
        let (waveforms, period_size, sample_rate) = match &candidate.source {
            CandidateSource::Mock {
                waveforms,
                period_size,
                sample_rate,
            } => (waveforms.clone(), *period_size, *sample_rate),
            CandidateSource::Alsa { .. } => {
                return Err("MockSource::open called on an Alsa candidate".into());
            }
        };
        let rng_states = waveforms
            .iter()
            .map(|w| match w {
                // Match today's MockCapture seed-zero handling: an
                // LCG seeded with 0 produces the all-zero stream,
                // useless as "noise." Bump to 1.
                Waveform::WhiteNoise { seed, .. } => (*seed).max(1),
                _ => 1,
            })
            .collect();
        let mut effective_whitelist = candidate.channels.clone();
        effective_whitelist.sort_unstable();
        effective_whitelist.dedup();
        Ok(Self {
            id: candidate.id.clone(),
            waveforms,
            effective_whitelist,
            period_size,
            sample_rate,
            rng_states,
            next_block_at: Instant::now(),
            absolute_idx: 0,
            stop,
        })
    }

    pub fn id(&self) -> &MicId {
        &self.id
    }

    /// Number of interleaved channels the source produces.  Equal to
    /// the number of waveforms -- the candidate's whitelist may
    /// reference any subset of these.
    pub fn channels(&self) -> u16 {
        self.waveforms.len() as u16
    }

    pub fn rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn period_size(&self) -> usize {
        self.period_size
    }

    /// Whitelist as configured for this source.  Stored sorted +
    /// deduped at open time.  Mirror of `AlsaSource::effective_whitelist`
    /// so the arbitrator has a single accessor regardless of variant.
    /// (Plain code rather than an intra-doc link -- the alsa source
    /// is feature-gated and the link wouldn't resolve in default-
    /// feature doc builds.)
    pub fn effective_whitelist(&self) -> &[u16] {
        &self.effective_whitelist
    }

    /// Synthesize one period of interleaved frames into
    /// `out` and return the [`ReadOutcome`] (the
    /// [`super::MicSource::read_interleaved`] impl
    /// delegates here).
    ///
    /// `out` must be at least `period_size * channels`
    /// samples long.  Sleeps until the next block's
    /// wall-clock target before returning, observing
    /// `stop` in 2 ms slices so a teardown is responsive.
    /// Returns [`ReadOutcome::StopRequested`] only when
    /// `stop` was observed during the sleep; the
    /// arbitrator's run loop re-checks stop and exits on
    /// the next iteration.  On a normal cycle returns
    /// [`ReadOutcome::Frames`] of `NonZero(period_size)`.
    /// [`MockSource`] cannot fail mid-stream, so the
    /// [`ReadError::Mock`] arm is uninhabited.
    pub fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        let n_channels = self.waveforms.len();
        let needed = self.period_size * n_channels;
        debug_assert!(
            out.len() >= needed,
            "mock read buffer too small: {} < {needed}",
            out.len(),
        );

        // Pace
        // First call: `next_block_at` is `Instant::now()` from open;
        // skip the sleep so the first read returns immediately.
        // Subsequent calls sleep until the wall-clock target.
        let block_dur = Duration::from_secs_f64(self.period_size as f64 / self.sample_rate as f64);
        let now = Instant::now();
        // Skew clamp: a long consumer stall (GC pause, scheduler
        // slice, test holding the thread) leaves `next_block_at` far
        // in the past, which would otherwise cause this source to
        // emit one period per call with no sleep until pacing
        // catches up -- a burst of synthesized audio with stale
        // wall-clock timestamps.  Reset to `now` instead.  Threshold
        // is one `block_dur` so legitimate small drift (a slow
        // single iteration) falls through unchanged.
        if now > self.next_block_at + block_dur {
            self.next_block_at = now;
        }
        if now < self.next_block_at {
            let mut remaining = self.next_block_at - now;
            // 2 ms slices so the stop flag is observed
            // promptly.  Short enough that very tight test
            // assertions ("got blocks within 200 ms")
            // aren't perturbed by the slice cadence.
            const STOP_POLL_SLICE: Duration = Duration::from_millis(2);
            while remaining > STOP_POLL_SLICE {
                if self.stop.load(Ordering::Acquire) {
                    return Ok(ReadOutcome::StopRequested);
                }
                thread::sleep(STOP_POLL_SLICE);
                let now2 = Instant::now();
                remaining = self.next_block_at.saturating_duration_since(now2);
            }
            if remaining > Duration::ZERO {
                thread::sleep(remaining);
            }
        }
        self.next_block_at += block_dur;

        // Synthesize
        // Direct interleaved generation: writes `out[f * C + ch]`
        // for each (frame, channel) without an intermediate
        // per-channel scratch.  Cache pattern is contiguous in
        // `out`, which is what downstream RMS+demux passes prefer.
        let target = &mut out[..needed];
        // Zero-fill once; per-waveform loops only overwrite (no
        // explicit `fill(0)` needed) -- but `Silence` relies on
        // pre-zeroed memory.
        target.fill(0.0);
        for (ch, waveform) in self.waveforms.iter().enumerate() {
            let rng = &mut self.rng_states[ch];
            generate_channel_into(
                target,
                ch,
                n_channels,
                self.period_size,
                *waveform,
                self.sample_rate,
                self.absolute_idx,
                rng,
            );
        }
        self.absolute_idx += self.period_size as u64;
        // `period_size` is validated nonzero at `MicCandidate::validate`
        // (per L1 invariants) -- `unwrap()` cannot fire under any
        // production path.  Tests that construct `MockSource` directly
        // must use a nonzero period_size.
        Ok(ReadOutcome::Frames(
            std::num::NonZeroUsize::new(self.period_size)
                .expect("MockSource::period_size must be nonzero"),
        ))
    }
}

impl super::MicSource for MockSource {
    fn id(&self) -> &MicId {
        MockSource::id(self)
    }
    fn channels(&self) -> u16 {
        MockSource::channels(self)
    }
    fn rate(&self) -> u32 {
        MockSource::rate(self)
    }
    fn period_size(&self) -> usize {
        MockSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        MockSource::effective_whitelist(self)
    }
    fn read_interleaved(
        &mut self,
        out: &mut [f32],
    ) -> Result<super::ReadOutcome, super::ReadError> {
        MockSource::read_interleaved(self, out)
    }
}

/// Write `period_size` samples of `waveform` into the per-channel
/// strided slots of an interleaved buffer.  Argument count is the
/// price of being a pure synthesis helper (no struct ownership of
/// the rng/scratch); called once per channel in the read hot path.
#[allow(clippy::too_many_arguments)]
fn generate_channel_into(
    interleaved: &mut [f32],
    channel: usize,
    n_channels: usize,
    period_size: usize,
    waveform: Waveform,
    sample_rate: u32,
    start: u64,
    rng_state: &mut u64,
) {
    match waveform {
        Waveform::Silence => {
            // Pre-filled to 0; nothing to do.
        }
        Waveform::Sine { freq_hz, amplitude } => {
            // Compute phase in f64: `f32` only represents successive
            // integers exactly up to 2^24 (~6 minutes at 44.1 kHz),
            // after which the sine quantizes audibly. f64 covers
            // 2^53 -- comfortably beyond any realistic mock-stream
            // duration.  The cast to f32 happens at the final sample
            // boundary, where downstream cares about f32 precision
            // (~24 bits >> audio dynamic range) anyway.
            let omega = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
            let amp = amplitude as f64;
            for f in 0..period_size {
                let n = (start + f as u64) as f64;
                interleaved[f * n_channels + channel] = (amp * (omega * n).sin()) as f32;
            }
        }
        Waveform::WhiteNoise { amplitude, .. } => {
            // Knuth's MMIX LCG.  Reproducible; not cryptographic.
            const MUL: u64 = 6_364_136_223_846_793_005;
            const ADD: u64 = 1_442_695_040_888_963_407;
            for f in 0..period_size {
                *rng_state = rng_state.wrapping_mul(MUL).wrapping_add(ADD);
                let bits = (*rng_state >> 32) as u32;
                let signed = bits as i32;
                let normalized = signed as f32 / i32::MAX as f32;
                interleaved[f * n_channels + channel] = normalized * amplitude;
            }
        }
        Waveform::Counter => {
            for f in 0..period_size {
                let absolute = start + f as u64;
                interleaved[f * n_channels + channel] = (absolute & 0xFFFF) as f32;
            }
        }
        Waveform::PingPongSine {
            freq_hz,
            high_amp,
            low_amp,
            half_period_samples,
            inverted,
        } => {
            // Phase of the underlying sine is continuous across all
            // samples -- only the amplitude envelope flips at half-cycle
            // boundaries.  Same f64-phase precision as `Sine` so a
            // long-running test doesn't audibly drift.
            let omega = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
            let high = high_amp as f64;
            let low = low_amp as f64;
            // `.max(1)` guards against divide-by-zero in the modulo
            // when `half_period_samples == 0`.  With `half = 1`, the
            // condition `cycle_pos < 1` is true only at cycle_pos=0
            // (and false at cycle_pos=1), so the output collapses to
            // alternating-per-sample, which is degenerate but won't
            // panic -- operators are expected to pass a sane value.
            let half = (half_period_samples as u64).max(1);
            let cycle = half.saturating_mul(2);
            for f in 0..period_size {
                let n = start + f as u64;
                let cycle_pos = n % cycle;
                let in_high_half = cycle_pos < half;
                // XOR with `inverted` flips which half is loud.
                let amp = if in_high_half ^ inverted { high } else { low };
                let n_f = n as f64;
                interleaved[f * n_channels + channel] = (amp * (omega * n_f).sin()) as f32;
            }
        }
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::MicId;

    fn mock_candidate(channels: Vec<u16>, waveforms: Vec<Waveform>) -> MicCandidate {
        MicCandidate {
            id: MicId::from_static("mock:test"),
            source: CandidateSource::Mock {
                waveforms,
                period_size: 256,
                sample_rate: 44_100,
            },
            channels,
        }
    }

    /// Test helper: collapse `Result<ReadOutcome, _>` back
    /// to `usize` semantics for assertions that only care
    /// about "how many frames did I get".  `Frames(n)`
    /// returns `n.get()`; every other variant returns 0.
    /// Tests that need to discriminate between variants
    /// `match` directly.
    fn frames_or_zero(r: Result<ReadOutcome, ReadError>) -> usize {
        match r.expect("mock read cannot fail") {
            ReadOutcome::Frames(n) => n.get(),
            ReadOutcome::StopRequested | ReadOutcome::Timeout | ReadOutcome::EndOfStream => 0,
        }
    }

    /// Open with valid inputs; the source reflects them.
    #[test]
    fn open_round_trips_id_and_shape() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = mock_candidate(vec![0, 1], vec![Waveform::Silence, Waveform::Silence]);
        let s = MockSource::open(&cand, stop).expect("open");
        assert_eq!(s.id(), &MicId::from_static("mock:test"));
        assert_eq!(s.channels(), 2);
        assert_eq!(s.rate(), 44_100);
        assert_eq!(s.period_size(), 256);
    }

    /// Opening a Mock source on an Alsa candidate is rejected.
    #[test]
    fn open_rejects_alsa_candidate() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("alsa-impostor"),
            source: CandidateSource::Alsa {
                hw_spec: "hw:1,0".into(),
                period_size: 1024,
                buffer_size: 4096,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject");
        assert!(err.contains("Alsa"), "err = {err:?}");
    }

    /// `MockSource::open` runs candidate validation before
    /// constructing.  `period_size = 0` would later panic on
    /// `NonZeroUsize::new(self.period_size).expect(...)` in
    /// `read_interleaved`, so the constructor must reject it up
    /// front.  Guards in-crate test callers that bypass
    /// `open_source`.
    #[test]
    fn open_rejects_zero_period_size() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("zero-period"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 0,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject zero period");
        assert!(
            err.contains("validation") || err.contains("period_size"),
            "err = {err:?}",
        );
    }

    /// `MockSource::open` rejects a sample_rate outside the
    /// supported range, defending the resampler ratio against
    /// pathological values.
    #[test]
    fn open_rejects_sample_rate_out_of_range() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("bad-rate"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 512,
                // Above MAX_SAMPLE_RATE (192 kHz).
                sample_rate: 1_000_000,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject extreme rate");
        assert!(
            err.contains("validation") || err.contains("sample_rate"),
            "err = {err:?}",
        );
    }

    /// First read returns immediately; subsequent reads pace to
    /// real-time.  Verifies that two reads' wall-clock spacing is
    /// at least most of one block duration.
    ///
    /// We use a 4096-frame period (~93 ms @ 44.1 k) to make the
    /// pacing duration much larger than scheduler-noise slack on
    /// a busy CI box.  The lower bound is 80 % of `block_dur`
    /// because (a) on macOS / Linux desktop, `thread::sleep` is
    /// known to over-deliver by a few ms but rarely cuts short;
    /// (b) any *real* pacing bug would either skip the sleep
    /// entirely (<< 80 % of block_dur) or sleep way too long
    /// (caught by the upper bound).  80 % comfortably distinguishes
    /// "scheduler noise" from "broken pacing."
    #[test]
    fn read_paces_to_real_time() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:pace"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size() * s.channels() as usize];

        let block_dur = Duration::from_secs_f64(s.period_size() as f64 / s.rate() as f64);
        let t0 = Instant::now();
        let n0 = frames_or_zero(s.read_interleaved(&mut buf));
        let t1 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let t2 = Instant::now();

        assert_eq!(n0, s.period_size());
        assert_eq!(n1, s.period_size());

        // First read: ~immediate (no pacing target yet).  5 ms
        // remains a tight bound: there is no sleep on the first
        // call, only synthesis (microseconds).
        let first_dur = t1 - t0;
        assert!(
            first_dur < Duration::from_millis(5),
            "first read should be ~immediate; took {first_dur:?}",
        );
        // Second read: at least 80 % of one block duration.
        let second_dur = t2 - t1;
        let lower = block_dur.mul_f32(0.80);
        assert!(
            second_dur >= lower,
            "second read should pace to >= {lower:?} (80% of {block_dur:?}); took {second_dur:?}",
        );
        // And not WAY too long either -- bound to 2x block_dur to
        // catch a sleep-too-much regression.
        assert!(
            second_dur < block_dur * 2,
            "second read paced too long: {second_dur:?} > 2x {block_dur:?}",
        );
    }

    /// Multi-channel demux: per-channel waveforms produce the
    /// expected samples at the right interleaved positions.
    #[test]
    fn read_interleaves_per_channel_waveforms() {
        let stop = Arc::new(AtomicBool::new(false));
        // ch0: silence; ch1: counter; ch2: sine-amplitude-1.0
        let cand = mock_candidate(
            vec![0, 1, 2],
            vec![
                Waveform::Silence,
                Waveform::Counter,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 1.0,
                },
            ],
        );
        let mut s = MockSource::open(&cand, stop).expect("open");
        let n_ch = s.channels() as usize;
        let mut buf = vec![1234.0f32; s.period_size() * n_ch];
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, s.period_size());

        // Channel 0: all zeros (Silence).
        for f in 0..s.period_size() {
            assert_eq!(buf[f * n_ch], 0.0, "ch0 frame {f}");
        }
        // Channel 1: counter, frame f -> (f & 0xFFFF) for absolute_idx=0.
        for f in 0..s.period_size() {
            let expected = (f as u64 & 0xFFFF) as f32;
            assert_eq!(buf[f * n_ch + 1], expected, "ch1 frame {f}");
        }
        // Channel 2: sine, bounded by amplitude=1.0.
        let max = (0..s.period_size())
            .map(|f| buf[f * n_ch + 2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max <= 1.0 + 1e-6, "ch2 sine max {max} > 1.0");
        let min = (0..s.period_size())
            .map(|f| buf[f * n_ch + 2])
            .fold(f32::INFINITY, f32::min);
        assert!(min >= -1.0 - 1e-6, "ch2 sine min {min} < -1.0");
    }

    /// Counter + sine waveforms are continuous across `read_*`
    /// calls -- `absolute_idx` advances by `period_size` per period.
    #[test]
    fn waveforms_are_continuous_across_reads() {
        let stop = Arc::new(AtomicBool::new(false));
        // Tiny period to keep the test wall-clock under a second.
        let cand = MicCandidate {
            id: MicId::from_static("mock:cont"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Counter],
                period_size: 64,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; 64];

        let mut concat: Vec<f32> = Vec::new();
        for _ in 0..3 {
            let _ = s.read_interleaved(&mut buf);
            concat.extend_from_slice(&buf);
        }
        for (i, &v) in concat.iter().enumerate() {
            let expected = (i as u64 & 0xFFFF) as f32;
            assert_eq!(v, expected, "discontinuity at absolute idx {i}");
        }
    }

    /// White-noise determinism: identical seed -> bit-identical
    /// samples across separately-opened sources.
    #[test]
    fn white_noise_is_seed_deterministic() {
        let cand = MicCandidate {
            id: MicId::from_static("mock:noise"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::WhiteNoise {
                    amplitude: 1.0,
                    seed: 42,
                }],
                period_size: 128,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut a = MockSource::open(&cand, Arc::new(AtomicBool::new(false))).expect("a");
        let mut b = MockSource::open(&cand, Arc::new(AtomicBool::new(false))).expect("b");
        let mut buf_a = vec![0.0f32; 128];
        let mut buf_b = vec![0.0f32; 128];
        let _ = a.read_interleaved(&mut buf_a);
        let _ = b.read_interleaved(&mut buf_b);
        for (i, (x, y)) in buf_a.iter().zip(buf_b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "noise drift at sample {i}: {x} vs {y}",
            );
        }
    }

    /// Stop signaled during pacing sleep returns 0 promptly -- no
    /// long wait holding teardown back.  Worst case <= 5 ms.
    #[test]
    fn stop_during_sleep_returns_zero_quickly() {
        let stop = Arc::new(AtomicBool::new(false));
        // Pick a long period so the second read has a real sleep
        // to interrupt: 4096 frames @ 44.1 k = ~93 ms.
        let cand = MicCandidate {
            id: MicId::from_static("mock:stop"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop.clone()).expect("open");
        let mut buf = vec![0.0f32; 4096];
        let n0 = frames_or_zero(s.read_interleaved(&mut buf)); // immediate (first call)
        assert_eq!(n0, 4096);

        // Schedule a stop a few ms in; the second read is mid-sleep.
        let stop_clone = stop.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            stop_clone.store(true, Ordering::Release);
        });
        let t0 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let elapsed = t0.elapsed();
        t.join().unwrap();

        assert_eq!(n1, 0, "stop signaled mid-sleep should return 0");
        // Period was 93 ms; we should exit within 15 ms (10 ms
        // sleep before signaling + 2 ms slice + slop).
        assert!(
            elapsed < Duration::from_millis(20),
            "stop response too slow: {elapsed:?}",
        );
    }

    /// Stop signaled mid-pacing-sleep returns the
    /// [`ReadOutcome::StopRequested`] variant (not `Frames`
    /// and not `EndOfStream`); the typed enum makes the
    /// intent explicit and disambiguates from ALSA's
    /// `EndOfStream` (closed PCM).
    #[test]
    fn stop_during_sleep_returns_stop_requested_variant() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:stop_variant"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop.clone()).expect("open");
        let mut buf = vec![0.0f32; 4096];
        let _ = s.read_interleaved(&mut buf); // prime pacing
        let stop_clone = stop.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            stop_clone.store(true, Ordering::Release);
        });
        let outcome = s.read_interleaved(&mut buf).expect("mock infallible");
        t.join().unwrap();
        assert_eq!(
            outcome,
            ReadOutcome::StopRequested,
            "mid-sleep stop must surface as StopRequested, not Frames/EndOfStream",
        );
    }

    /// Stop already set before entering `read_interleaved` is
    /// honoured (ANY pacing sleep, even a short one, sees the flag).
    #[test]
    fn read_with_stop_preset_returns_zero() {
        let stop = Arc::new(AtomicBool::new(true));
        let cand = mock_candidate(vec![0], vec![Waveform::Silence]);
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size()];
        // First call has no pacing sleep -- but bypass that by
        // making a second call (which DOES pace).
        let _ = s.read_interleaved(&mut buf);
        let t0 = Instant::now();
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, 0);
        assert!(
            t0.elapsed() < Duration::from_millis(5),
            "stop pre-set should return ~immediately",
        );
    }

    /// Skew clamp: after a long stall (here, simulated by
    /// pre-rolling `next_block_at` far into the past), the next
    /// read does NOT return immediately and then back-to-back
    /// burst until catching up.  Instead, pacing resumes from `now`
    /// and the *next* call paces a normal `block_dur`.  Without the
    /// clamp, a stalled consumer would see N periods of synthetic
    /// audio dumped at memory speed.
    #[test]
    fn read_clamps_skew_after_long_stall() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:skew"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size()];

        // First read primes pacing (next_block_at = open + block_dur).
        let n0 = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n0, s.period_size());

        // Simulate a long stall by rolling `next_block_at` 1 s into
        // the past.  Without the skew clamp the next two reads would
        // both return immediately (no sleep) until pacing caught up.
        let block_dur = Duration::from_secs_f64(s.period_size() as f64 / s.rate() as f64);
        s.next_block_at = Instant::now() - Duration::from_secs(1);

        // First post-stall read: clamped -> returns ~immediately.
        let t0 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let first_post = t0.elapsed();
        assert_eq!(n1, s.period_size());
        assert!(
            first_post < Duration::from_millis(5),
            "first post-stall read should return ~immediately (clamp resets pacing); \
             took {first_post:?}",
        );

        // Second post-stall read: pacing has resumed from "now"
        // (set by the clamp on the first call), so this read MUST
        // sleep approximately one block_dur.  If the clamp isn't
        // doing its job, this read would also return immediately
        // and we'd be in burst territory.
        let t1 = Instant::now();
        let n2 = frames_or_zero(s.read_interleaved(&mut buf));
        let second_post = t1.elapsed();
        assert_eq!(n2, s.period_size());
        let lower = block_dur.mul_f32(0.80);
        assert!(
            second_post >= lower,
            "second post-stall read should pace >= {lower:?} (80% of \
             {block_dur:?}); took {second_post:?} (clamp regression?)",
        );
    }

    /// `PingPongSine` alternates between high and low amplitudes on
    /// the configured half-period boundary, with `inverted` swapping
    /// which half is loud.  Verified directly: pump enough frames to
    /// span at least one full cycle, then check that the per-frame
    /// peak amplitude crosses both levels and switches polarity at
    /// the half-period boundary.
    #[test]
    fn ping_pong_sine_alternates_amplitude_at_half_period() {
        let stop = Arc::new(AtomicBool::new(false));
        // 3 channels: ch0 high-first, ch1 low-first (inverted), ch2 silence
        // for contrast.  Half-period 64 samples -> full cycle 128.
        let cand = MicCandidate {
            id: MicId::from_static("mock:pp"),
            source: CandidateSource::Mock {
                waveforms: vec![
                    Waveform::PingPongSine {
                        freq_hz: 4_410.0, // 0.1 cycles per sample at 44.1k -> easy to detect
                        high_amp: 0.8,
                        low_amp: 0.05,
                        half_period_samples: 64,
                        inverted: false,
                    },
                    Waveform::PingPongSine {
                        freq_hz: 4_410.0,
                        high_amp: 0.8,
                        low_amp: 0.05,
                        half_period_samples: 64,
                        inverted: true,
                    },
                    Waveform::Silence,
                ],
                period_size: 256, // 2 full cycles
                sample_rate: 44_100,
            },
            channels: vec![0, 1, 2],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let n_ch = s.channels() as usize;
        let mut buf = vec![0.0f32; s.period_size() * n_ch];
        let _ = s.read_interleaved(&mut buf);

        // Compute peak |sample| over a 32-sample window centered in
        // each half-cycle; the sine is fast enough relative to the
        // half-period that 32 samples reliably contain a peak.
        let peak_in = |ch: usize, start: usize, len: usize| -> f32 {
            (start..start + len)
                .map(|f| buf[f * n_ch + ch].abs())
                .fold(0.0_f32, f32::max)
        };
        // ch0: high in [0, 64), low in [64, 128), high in [128, 192), low in [192, 256).
        let ch0_high0 = peak_in(0, 16, 32);
        let ch0_low = peak_in(0, 80, 32);
        let ch0_high1 = peak_in(0, 144, 32);
        assert!(ch0_high0 > 0.5, "ch0 high half 0: peak {ch0_high0}");
        assert!(ch0_low < 0.1, "ch0 low half: peak {ch0_low}");
        assert!(ch0_high1 > 0.5, "ch0 high half 1: peak {ch0_high1}");

        // ch1 inverted: low in [0, 64), high in [64, 128).
        let ch1_low = peak_in(1, 16, 32);
        let ch1_high = peak_in(1, 80, 32);
        assert!(ch1_low < 0.1, "ch1 low half (inverted): peak {ch1_low}");
        assert!(ch1_high > 0.5, "ch1 high half (inverted): peak {ch1_high}");

        // Silence stays silent.
        let ch2_peak = peak_in(2, 0, 256);
        assert_eq!(ch2_peak, 0.0);
    }

    /// `half_period_samples = 0` must not panic -- the synthesis path
    /// guards via `.max(1)`.  Output is degenerate but must terminate
    /// cleanly so a misconfigured operator setup doesn't crash the
    /// arbitrator thread.
    #[test]
    fn ping_pong_sine_zero_half_period_does_not_panic() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:pp-zero"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::PingPongSine {
                    freq_hz: 1_000.0,
                    high_amp: 0.5,
                    low_amp: 0.05,
                    half_period_samples: 0,
                    inverted: false,
                }],
                period_size: 64,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; 64];
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, 64);
        assert!(buf.iter().all(|s| s.is_finite()));
    }

    /// Open + close many sources without leaking.  Mostly sanity:
    /// no internal state on the heap that outlives the source.
    #[test]
    fn many_open_close_cycles() {
        let cand = mock_candidate(vec![0], vec![Waveform::Silence]);
        for _ in 0..50 {
            let stop = Arc::new(AtomicBool::new(false));
            let mut s = MockSource::open(&cand, stop).expect("open");
            let mut buf = vec![0.0f32; s.period_size()];
            let _ = s.read_interleaved(&mut buf);
            // Source dropped at end of scope.
        }
    }
}
