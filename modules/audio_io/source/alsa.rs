//! Multi-channel ALSA capture source.
//!
//! Linux-only (compiled iff `target_os = "linux" && feature =
//! "alsa-real"`).  On macOS and on Linux without the feature this
//! module isn't compiled and [`super::open_source`] returns
//! [`super::OpenError::AlsaNotCompiledIn`] for ALSA candidates.
//!
//! ## Why multi-channel
//!
//! For a multi-channel mic with no beamformer, an ALSA-side
//! downmix to mono would lose the per-channel signal needed for
//! RMS arbitration.  This source negotiates at the device's
//! actual channel count and lets the arbitrator do RMS-based
//! channel selection in software.
//!
//! ## Channel negotiation
//!
//! 1. Open the PCM and probe `HwParams::get_channels_max`.
//! 2. Prefer `set_channels(whitelist.max() + 1)` -- minimum needed to
//!    expose every whitelisted index.  Falls back to
//!    `set_channels(channels_max)` if the device refuses (some USB
//!    class-compliant devices snap to fixed counts like 2 only).
//! 3. After successful negotiation, intersect the candidate's
//!    whitelist with `[0..actual_channels)`.  An empty intersection
//!    means the candidate is misconfigured for this hardware ->
//!    `Err`.  Out-of-range indices in the original whitelist log a
//!    warning so operators can fix their config.
//!
//! ## Format negotiation
//!
//! Float LE preferred (lossless mapping to f32); s16 LE fallback for
//! USB class-compliant mics that don't expose float.  The s16 path
//! converts via `/32_768.0` in the read hot path (preserves
//! negative-domain symmetry that `i16::MAX` division would not).
//!
//! ## Recovery
//!
//! [`AlsaSource::try_recover`] wraps `pcm.try_recover(e, true)` for
//! the standard xrun/suspend recovery.  Persistent failures (ENODEV
//! after USB hot-unplug) surface as `try_recover` itself returning
//! `Err`; the arbitrator handles those at the policy level
//! (`FirstAvailable` walks to the next candidate; `Fixed` retries
//! forever with rate-limited logs).

use crate::audio_io::mic_arbitrator::{CandidateSource, MicCandidate};
use crate::audio_io::source::{ReadError, ReadOutcome};
use crate::common::dims::SampleRate;
use crate::common::ids::MicId;
use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, PollDescriptors, ValueOr};
use std::num::NonZeroUsize;
use std::time::Duration;

/// Outcome of a non-blocking ALSA read attempt.  Surfaces
/// the "no data within `timeout`" case as an explicit value
/// so the arbitrator's run loop can re-check the stop flag
/// instead of blocking indefinitely on `readi`.  Without
/// this enum a hung USB device would keep the arbitrator
/// thread unkillable from outside Rust
/// (the run-loop's stop check happens *before* the read; a blocking
/// read never returned).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AlsaReadOutcome {
    /// `n` interleaved frames were read into `out[..n * channels]`.
    Frames(usize),
    /// `poll` returned 0 ready events within `timeout`.  The
    /// arbitrator should re-check its stop flag and either continue
    /// or exit.  Frames read so far this period are not lost -- `out`
    /// is untouched on Timeout.
    Timeout,
}

/// Configured ALSA capture source.
pub struct AlsaSource {
    id: MicId,
    pcm: PCM,
    /// Whitelist intersected with the device's actual channel count,
    /// sorted, deduped.  The arbitrator drives its per-channel RMS
    /// state off this slice (not the candidate's original whitelist,
    /// which may have asked for indices the device doesn't expose).
    effective_whitelist: Vec<u16>,
    /// Channel count ALSA actually negotiated.  Frames in the
    /// interleaved buffer are `actual_channels` samples wide.
    actual_channels: u16,
    /// Frames per `readi` call.  Memory for [`read_interleaved`] is
    /// `period_size * actual_channels` samples; the arbitrator
    /// pre-allocates this once at open time.
    period_size: usize,
    /// True iff the device negotiated `Format::float()`.  Determines
    /// which `IO<T>` the read path uses.
    is_float: bool,
    /// Sample rate ALSA actually negotiated.  The arbitrator
    /// constructs a per-channel resampler iff this differs from
    /// [`SampleRate::VALUE`].
    rate: u32,
    /// Reusable scratch for the s16 fallback path so the read hot
    /// path is alloc-free when `is_float == false`.  Empty Vec on
    /// `is_float == true`.
    i16_scratch: Vec<i16>,
    /// Cache of the PCM's pollfds, populated at `open()`
    /// after `pcm.start()`.  Calling `self.pcm.get()?` per
    /// period (~43 Hz) would allocate a fresh
    /// `Vec<libc::pollfd>` on the real-time audio thread,
    /// and the resulting malloc-lock contention with the
    /// inference allocator could spike the worst-case
    /// period to multi-ms under load.  ALSA's pollfds are
    /// stable from `prepare()` until `drop`, so reusing
    /// this cached vec (with `revents` reset to 0 each
    /// call) is safe.
    pollfds: Vec<alsa::poll::pollfd>,
}

// `alsa::PCM` doesn't implement `Debug`; hand-roll one that prints
// the operator-relevant negotiated values.
impl std::fmt::Debug for AlsaSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlsaSource")
            .field("id", &self.id)
            .field("actual_channels", &self.actual_channels)
            .field("effective_whitelist", &self.effective_whitelist)
            .field("period_size", &self.period_size)
            .field("rate", &self.rate)
            .field("is_float", &self.is_float)
            .finish()
    }
}

impl AlsaSource {
    /// Open the candidate's ALSA device + negotiate channels,
    /// format, rate, and period.  Errors are stringified for
    /// propagation through [`super::OpenError`] without pulling
    /// `alsa::Error` into the crate-level enum.  Crate-private;
    /// external callers go through [`super::open_source`], and
    /// re-validating here guards in-crate test callers.
    pub(crate) fn open(candidate: &MicCandidate) -> Result<Self, String> {
        candidate
            .validate()
            .map_err(|e| format!("AlsaSource::open candidate failed validation: {e}"))?;
        let (hw_spec, period_size, buffer_size) = match &candidate.source {
            CandidateSource::Alsa {
                hw_spec,
                period_size,
                buffer_size,
            } => (hw_spec.clone(), *period_size, *buffer_size),
            CandidateSource::Mock { .. } => {
                return Err("AlsaSource::open called on a Mock candidate".into());
            }
        };

        // Open in non-blocking mode (3rd arg = nonblock).
        // `readi` then returns EAGAIN immediately when no
        // data is ready instead of blocking indefinitely;
        // the run loop polls the PCM's file descriptors
        // with a bounded timeout in `read_interleaved`
        // below so a wedged USB device can no
        // longer pin the arbitrator thread.
        let pcm = PCM::new(&hw_spec, Direction::Capture, true)
            .map_err(|e| format!("PCM::new({hw_spec}): {e}"))?;

        // Channel negotiation
        // 1. Decide the channel count to ask for.  We need enough to
        //    expose every whitelisted index, but no more -- wider
        //    channel counts cost bandwidth and per-period work.
        let whitelist_max = candidate
            .channels
            .iter()
            .copied()
            .max()
            .ok_or("candidate channels whitelist is empty")?;
        let needed = whitelist_max as u32 + 1;

        // 2. Probe + negotiate inside a HwParams scope.  We attempt
        //    `set_channels(needed)` exact first; if the device
        //    refuses (USB class-compliant mics often only accept
        //    fixed values like 2 or 4), fall back to whatever the
        //    device exposes via `channels_max`.
        let (actual_channels, is_float, actual_rate, actual_period);
        {
            let hwp = HwParams::any(&pcm).map_err(|e| format!("HwParams::any: {e}"))?;
            hwp.set_access(Access::RWInterleaved)
                .map_err(|e| format!("set_access: {e}"))?;

            // Prefer FloatLE; fall back to S16 LE if the device
            // doesn't support float (USB class-compliant mics often
            // don't).
            is_float = if hwp.set_format(Format::float()).is_ok() {
                true
            } else {
                hwp.set_format(Format::s16())
                    .map_err(|e| format!("set_format(s16): {e}"))?;
                false
            };

            // Channel negotiation: try exact, fall back to max.
            // The `as u16` casts below are guarded by
            // `MicCandidate::validate`'s [`MAX_CHANNEL_INDEX`] cap
            // (which keeps `whitelist_max + 1 <= 1024 <= u16::MAX`)
            // plus the `max <= u16::MAX` debug-assert: real audio
            // hardware never reports anywhere near that count, but
            // we'd rather panic in debug than silently truncate to
            // 0 if a future ALSA backend returned a stupid value.
            if hwp.set_channels(needed).is_ok() {
                debug_assert!(
                    needed <= u16::MAX as u32,
                    "negotiated channel count {needed} exceeds u16",
                );
                actual_channels = needed as u16;
            } else {
                let max = hwp
                    .get_channels_max()
                    .map_err(|e| format!("get_channels_max: {e}"))?;
                if max == 0 {
                    return Err(format!("device {hw_spec} reports 0 channels"));
                }
                if max > u16::MAX as u32 {
                    return Err(format!(
                        "device {hw_spec} reports {max} channels -- implausible, refusing",
                    ));
                }
                hwp.set_channels(max)
                    .map_err(|e| format!("set_channels({max}) fallback: {e}"))?;
                actual_channels = max as u16;
                tracing::warn!(
                    target: "audio_io.source.alsa",
                    device = %candidate.id,
                    hw_spec = %hw_spec,
                    requested = needed,
                    actual = max,
                    "device refused requested channel count; using device max",
                );
            }

            hwp.set_rate(SampleRate::VALUE, ValueOr::Nearest)
                .map_err(|e| format!("set_rate: {e}"))?;
            // `set_period_size_near` snaps to the nearest supported value;
            // the non-`_near` form requires an exact match (the `dir`
            // arg is just a metadata hint, not a search direction).  USB
            // class-compliant mics often quantize to 480/960/1920, so an
            // exact-only set with `period_size = 1024` fails on those
            // devices.  The actually-negotiated value is read back via
            // `get_period_size` after `hw_params` (subsequent buffer_size
            // / channel constraints can further refine).
            hwp.set_period_size_near(period_size as alsa::pcm::Frames, ValueOr::Nearest)
                .map_err(|e| format!("set_period_size_near: {e}"))?;
            hwp.set_buffer_size_near(buffer_size as alsa::pcm::Frames)
                .map_err(|e| format!("set_buffer_size_near: {e}"))?;
            pcm.hw_params(&hwp).map_err(|e| format!("hw_params: {e}"))?;
            actual_rate = hwp.get_rate().map_err(|e| format!("get_rate: {e}"))?;
            // Read back the negotiated period size: with `ValueOr::Nearest`
            // the device may snap to a different value (USB class-compliant
            // mics often quantize to 480/960/1920).  The arbitrator sizes
            // its `interleaved_scratch` off `period_size()`, and our
            // `i16_scratch` below is sized off this same value -- using
            // the *requested* value would undersize both whenever the
            // device negotiated a larger period, causing chronic short
            // reads.  `Frames` is `i64` in alsa-rs; reject non-positive
            // values defensively.
            let neg_period = hwp
                .get_period_size()
                .map_err(|e| format!("get_period_size: {e}"))?;
            if neg_period <= 0 {
                return Err(format!(
                    "device {hw_spec} reported non-positive period size {neg_period}",
                ));
            }
            actual_period = neg_period as usize;
        }

        // 3. Intersect the candidate's whitelist with what the
        //    device actually exposes.  Out-of-range entries are
        //    dropped with a single consolidated warning (one log
        //    line lists all dropped indices -- a per-entry loop
        //    would spam journalctl when an operator misconfigures
        //    a multi-entry whitelist).  An empty intersection is a
        //    fatal config error for this hardware.
        let dropped: Vec<u16> = candidate
            .channels
            .iter()
            .copied()
            .filter(|&ch| ch >= actual_channels)
            .collect();
        if !dropped.is_empty() {
            tracing::warn!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                dropped_channels = ?dropped,
                actual_channels,
                "whitelist entries exceed device channel count; dropping",
            );
        }
        let effective_whitelist = intersect_whitelist(&candidate.channels, actual_channels);
        if effective_whitelist.is_empty() {
            return Err(format!(
                "candidate {} whitelist {:?} has no entries within \
                 device's {} channels -- refusing to open",
                candidate.id, candidate.channels, actual_channels,
            ));
        }

        pcm.prepare().map_err(|e| format!("prepare: {e}"))?;
        pcm.start().map_err(|e| format!("start: {e}"))?;

        // Capture the PCM's pollfds once.  Stable for the
        // source's lifetime (until drop), so subsequent
        // `read_with_timeout` calls reset `revents` and
        // reuse this vec instead of allocating a fresh
        // `Vec<pollfd>` per period.
        let pollfds = pcm
            .get()
            .map_err(|e| format!("PollDescriptors::get: {e}"))?;

        if actual_period != period_size {
            tracing::info!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                requested_period = period_size,
                actual_period,
                "device snapped to a different period size",
            );
        }

        // Warn loudly when the device negotiates a non-native
        // rate.  The arbitrator then has to instantiate a per-
        // slot rubato resampler (~140 KB sinc table per slot)
        // and pay extra CPU per period plus L2 contention
        // against any co-located inference engine.  USB class-
        // compliant mics often quantize to 48 kHz; prefer a mic
        // that supports 44.1 kHz natively if you see this warning.
        if actual_rate != SampleRate::VALUE {
            tracing::warn!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                hw_spec = %hw_spec,
                requested_rate = SampleRate::VALUE,
                actual_rate,
                "device negotiated a non-44.1 kHz rate; arbitrator will run a per-slot \
                 resampler -- pick 44.1 kHz-native hardware to avoid this overhead",
            );
        }

        let i16_scratch = if is_float {
            Vec::new()
        } else {
            vec![0i16; actual_period * actual_channels as usize]
        };

        tracing::info!(
            target: "audio_io.source.alsa",
            device = %candidate.id,
            hw_spec = %hw_spec,
            actual_channels,
            actual_rate,
            is_float,
            actual_period,
            effective_whitelist = ?effective_whitelist,
            "ALSA source opened",
        );
        Ok(Self {
            id: candidate.id.clone(),
            pcm,
            effective_whitelist,
            actual_channels,
            period_size: actual_period,
            is_float,
            rate: actual_rate,
            i16_scratch,
            pollfds,
        })
    }

    pub fn id(&self) -> &MicId {
        &self.id
    }

    pub fn channels(&self) -> u16 {
        self.actual_channels
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn period_size(&self) -> usize {
        self.period_size
    }

    /// Whitelist as actually usable on this device (subset of the
    /// candidate's `channels` after intersection with
    /// `[0..actual_channels)`).  The arbitrator drives its per-slot
    /// RMS state off this slice.
    pub fn effective_whitelist(&self) -> &[u16] {
        &self.effective_whitelist
    }

    /// Default per-call poll timeout: two periods at the
    /// negotiated rate.  Healthy devices return data within
    /// one period; the 2x factor absorbs scheduler jitter
    /// so we don't spuriously bounce off
    /// [`AlsaReadOutcome::Timeout`].  Used by the trait impl
    /// below; the inherent [`Self::read_with_timeout`]
    /// accepts a caller-supplied value for tests that want a
    /// tighter or looser bound.
    pub fn default_read_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.period_size as f64 / self.rate as f64).saturating_mul(2)
    }

    /// Read up to one period of interleaved frames into
    /// `out`, polling the PCM's file descriptors with a
    /// bounded `timeout` first so a wedged USB device can't
    /// pin the arbitrator thread.
    ///
    /// `timeout` should be >= one period duration (typically
    /// `period_size / rate`, ~21 ms at 1024 frames / 48
    /// kHz) plus some slack, so a healthy device that just
    /// had to wait for the next period boundary returns
    /// [`AlsaReadOutcome::Frames`] rather than
    /// [`AlsaReadOutcome::Timeout`] on every call.  The
    /// arbitrator passes `period_dur * 2` to give that
    /// headroom.
    ///
    /// `out` must be at least `period_size * channels()`
    /// samples long.  Short reads (partial frame at EOF on
    /// a hot-unplug edge) return `Frames(n)` with the
    /// actual count; the arbitrator processes whatever was
    /// read.
    pub fn read_with_timeout(
        &mut self,
        out: &mut [f32],
        timeout: Duration,
    ) -> alsa::Result<AlsaReadOutcome> {
        let needed = self.period_size * self.actual_channels as usize;
        debug_assert!(
            out.len() >= needed,
            "alsa read buffer too small: {} < {needed}",
            out.len(),
        );

        // Poll first.  `alsa-rs` exposes `poll::poll(&mut
        // [pollfd], timeout_ms)` which wraps `libc::poll`
        // directly -- the same primitive `nix` uses, but
        // reachable without an extra dep.  The PCM's
        // pollfds were captured once at `open()` and are
        // stable for the source's lifetime; we just need to
        // clear any sticky `revents` bits from the previous
        // poll so this call only reports fds that are ready
        // NOW.  On Timeout we return early so the
        // arbitrator's run loop can re-check its stop flag; on
        // EINTR (the run loop's `unpark` interrupting the syscall)
        // we also return Timeout.
        for p in &mut self.pollfds {
            p.revents = 0;
        }
        // alsa-rs's `poll` takes `i32` ms; clamp to i32::MAX
        // (~24 days) to absorb a pathological long timeout
        // without overflow.
        let timeout_ms: i32 = timeout.as_millis().min(i32::MAX as u128) as i32;
        let n_ready = match alsa::poll::poll(&mut self.pollfds, timeout_ms) {
            Ok(n) => n,
            Err(e) => {
                // alsa::Error wraps a raw errno; -EINTR is benign
                // (a signal interrupted the syscall -- typically
                // the `unpark` from `signal_stop`).  Treat as
                // Timeout so the run loop's stop check fires.
                if e.errno() == libc::EINTR {
                    return Ok(AlsaReadOutcome::Timeout);
                }
                return Err(e);
            }
        };
        if n_ready == 0 {
            return Ok(AlsaReadOutcome::Timeout);
        }

        let target = &mut out[..needed];
        // EAGAIN/EWOULDBLOCK can fire post-poll on nonblocking
        // PCMs (plugin chains + poll-wakeup races); classify as
        // Timeout so `try_recover` doesn't reset a healthy
        // session.  Other errnos (EPIPE, ESTRPIPE, ENODEV)
        // still surface.
        let frames = if self.is_float {
            let io = self.pcm.io_f32()?;
            match io.readi(target) {
                Ok(n) => n,
                Err(e) if is_eagain(&e) => return Ok(AlsaReadOutcome::Timeout),
                Err(e) => return Err(e),
            }
        } else {
            let io = self.pcm.io_i16()?;
            // Read into the i16 scratch (sized at open time).
            let frames = match io.readi(&mut self.i16_scratch) {
                Ok(n) => n,
                Err(e) if is_eagain(&e) => return Ok(AlsaReadOutcome::Timeout),
                Err(e) => return Err(e),
            };
            // Convert frames * channels samples. /32_768.0 (i.e.
            // /2^15) preserves the negative-domain symmetry that
            // /i16::MAX would not -- i16 has one more negative value
            // than positive; /32768 maps -32768 to -1.0 exactly,
            // /32767 wouldn't.
            let n_samples = frames * self.actual_channels as usize;
            for (dst, &src) in target[..n_samples]
                .iter_mut()
                .zip(self.i16_scratch[..n_samples].iter())
            {
                *dst = src as f32 / 32_768.0;
            }
            frames
        };
        Ok(AlsaReadOutcome::Frames(frames))
    }

    /// Standard ALSA xrun/suspend recovery.  Returns `Ok(())` if the
    /// PCM is now ready for another `read_interleaved`; `Err` if the
    /// failure is unrecoverable (typically ENODEV -- the arbitrator
    /// drops the source and either fails over or stays inert per
    /// policy).
    pub fn try_recover(&mut self, e: alsa::Error) -> alsa::Result<()> {
        // `silent = true`: don't print to stderr (we have tracing).
        self.pcm.try_recover(e, true)
    }
}

// [`super::MicSource`] impl.  Maps the inherent
// [`AlsaReadOutcome`] surface onto the unified
// [`super::ReadOutcome`] enum:
//
// - `Frames(0)` -> `EndOfStream` (PCM hit EOF on a closed
//   device; the arbitrator tears the source down rather
//   than silently keeping a dead PCM live).
// - `Frames(n>0)` -> `Frames(NonZero(n))`.
// - `Timeout` -> `Timeout`.
// - `Err` -> `Err(ReadError::Alsa(e))`.  The arbitrator's
//   match arm still calls `try_recover` directly on the
//   concrete `ActiveSource::Alsa` variant; recovery isn't
//   on the trait surface (mock has no analogue).
impl super::MicSource for AlsaSource {
    fn id(&self) -> &MicId {
        AlsaSource::id(self)
    }
    fn channels(&self) -> u16 {
        AlsaSource::channels(self)
    }
    fn rate(&self) -> u32 {
        AlsaSource::rate(self)
    }
    fn period_size(&self) -> usize {
        AlsaSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        AlsaSource::effective_whitelist(self)
    }
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        let timeout = self.default_read_timeout();
        match self.read_with_timeout(out, timeout) {
            Ok(AlsaReadOutcome::Timeout) => Ok(ReadOutcome::Timeout),
            Ok(AlsaReadOutcome::Frames(0)) => Ok(ReadOutcome::EndOfStream),
            Ok(AlsaReadOutcome::Frames(n)) => Ok(ReadOutcome::Frames(
                NonZeroUsize::new(n).expect("n > 0 in this match arm"),
            )),
            Err(e) => Err(ReadError::Alsa(e)),
        }
    }
}

/// True iff `e` is `EAGAIN` / `EWOULDBLOCK` (equal on Linux;
/// both checked for portability).  Pure function so the unit
/// tests can exercise it without an ALSA PCM.
fn is_eagain(e: &alsa::Error) -> bool {
    let code = e.errno();
    code == libc::EAGAIN || code == libc::EWOULDBLOCK
}

/// Compute the open-time effective whitelist as the intersection
/// of `raw` (operator-supplied) with `[0..actual_channels)`,
/// sorted + deduped.  Pure function so the intersection logic is
/// testable without an ALSA environment.
///
/// Out-of-range entries are silently filtered here -- `open()`'s
/// caller is responsible for emitting per-entry diagnostic logs
/// before calling this (which is exactly what
/// [`AlsaSource::open`] does).
fn intersect_whitelist(raw: &[u16], actual_channels: u16) -> Vec<u16> {
    let mut effective: Vec<u16> = raw
        .iter()
        .copied()
        .filter(|&ch| ch < actual_channels)
        .collect();
    effective.sort_unstable();
    effective.dedup();
    effective
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::MicId;

    /// Open against a hw_spec that is essentially guaranteed not to
    /// exist.  Skipped by default -- runs ALSA syscalls; safer to
    /// gate behind `--ignored` so it only fires on Linux CI with a
    /// real ALSA environment.
    #[test]
    #[ignore = "requires a Linux ALSA environment with no bogus123 PCM"]
    fn opening_invalid_device_returns_error() {
        let cand = MicCandidate {
            id: MicId::from_static("test-bogus"),
            source: CandidateSource::Alsa {
                hw_spec: "bogus_device_does_not_exist_42".into(),
                period_size: 1024,
                buffer_size: 4096,
            },
            channels: vec![0],
        };
        let res = AlsaSource::open(&cand);
        assert!(res.is_err(), "expected open to fail; got {res:?}");
    }

    // Direct tests of the open-time helper -- no ALSA env required.
    // These exercise the actual function used by `open()`, not a
    // hand-mirrored copy, so refactoring the helper can't silently
    // pass while breaking the production path.

    #[test]
    fn intersect_whitelist_drops_out_of_range_entries() {
        let raw: Vec<u16> = vec![0, 1, 5, 7];
        let effective = intersect_whitelist(&raw, 2);
        assert_eq!(effective, vec![0_u16, 1]);
    }

    #[test]
    fn intersect_whitelist_returns_empty_when_all_entries_are_out_of_range() {
        let raw: Vec<u16> = vec![5, 7];
        let effective = intersect_whitelist(&raw, 2);
        assert!(effective.is_empty(), "all entries should be dropped");
    }

    #[test]
    fn intersect_whitelist_sorts_and_dedups() {
        // Operator-supplied order is not necessarily sorted, and a
        // duplicate at the same index would corrupt per-slot RMS.
        let raw: Vec<u16> = vec![3, 0, 2, 0, 3];
        let effective = intersect_whitelist(&raw, 4);
        assert_eq!(effective, vec![0_u16, 2, 3]);
    }

    #[test]
    fn intersect_whitelist_passes_through_in_range_entries() {
        let raw: Vec<u16> = vec![0, 1, 2, 3];
        let effective = intersect_whitelist(&raw, 4);
        assert_eq!(effective, vec![0_u16, 1, 2, 3]);
    }

    /// Pin [`AlsaReadOutcome`]'s shape so a future refactor
    /// can't silently rename / merge the variants and
    /// break the arbitrator's pattern-match.  The enum is
    /// the public contract between the source and the
    /// arbitrator's run loop; the test fails to compile if
    /// a variant disappears.
    #[test]
    fn alsa_read_outcome_variants_pin_shape() {
        let f = AlsaReadOutcome::Frames(42);
        let t = AlsaReadOutcome::Timeout;
        assert_ne!(f, t);
        match f {
            AlsaReadOutcome::Frames(n) => assert_eq!(n, 42),
            AlsaReadOutcome::Timeout => panic!("expected Frames"),
        }
        match t {
            AlsaReadOutcome::Frames(_) => panic!("expected Timeout"),
            AlsaReadOutcome::Timeout => {}
        }
    }

    /// `is_eagain` recognises both `EAGAIN` and `EWOULDBLOCK`
    /// errno values, so the `read_with_timeout` classifier maps
    /// either name to [`AlsaReadOutcome::Timeout`].  On Linux
    /// these are the same constant; the test guards against a
    /// future libc port that distinguishes them.
    ///
    /// Other errnos (EPIPE = xrun, ENODEV = hot-unplug)
    /// must NOT be classified as EAGAIN; the arbitrator
    /// relies on `try_recover` for those.
    #[test]
    fn is_eagain_recognises_eagain_and_ewouldblock_only() {
        assert!(
            is_eagain(&alsa::Error::new("test", libc::EAGAIN)),
            "EAGAIN must be classified as eagain",
        );
        assert!(
            is_eagain(&alsa::Error::new("test", libc::EWOULDBLOCK)),
            "EWOULDBLOCK must be classified as eagain",
        );
        // Negative cases: errors that must surface to
        // `try_recover` rather than being swallowed as
        // Timeout.
        for code in [
            libc::EPIPE,    // xrun
            libc::ESTRPIPE, // suspended
            libc::ENODEV,   // hot-unplug
            libc::EINTR,    // signal (handled separately by poll)
            libc::EIO,      // generic I/O
        ] {
            assert!(
                !is_eagain(&alsa::Error::new("test", code)),
                "errno {code} must NOT be classified as eagain",
            );
        }
    }
}
