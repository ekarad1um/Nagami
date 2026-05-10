//! Two clock-domain time newtypes.
//!
//! Cross-process consumers (web client, future Python clients)
//! cannot reliably interpret a timestamp without knowing whether
//! it is a monotonic or a wall-clock value.  Two distinct types
//! make the choice load-bearing:
//!
//! - [`CaptureTime`] -- microseconds since process boot, sampled
//!   from a monotonic clock.  Stable under wall-clock skew (NTP,
//!   suspend/resume), comparable across this process's lifetime,
//!   meaningless to other processes.  Used for intra-process
//!   relative measurements: hop deadlines, frame jitter,
//!   subsystem heartbeats.
//! - [`WallTime`] -- microseconds since the Unix epoch.  Sampled
//!   from a wall-clock; jumps on NTP correction.  Comparable
//!   across processes / hosts.  Used for cross-process absolute
//!   timestamps on emitted frames.
//!
//! The proto wire format carries the corresponding pair
//! (`t_us_capture_monotonic` and `t_us_publish_unix`), and the
//! `.proto` schema pins these Rust types as the canonical
//! conversion targets.
//!
//! # Resolution rationale
//!
//! Microseconds, not nanoseconds:
//!
//! - fits a `u64` for >580k years (vs ~580 years for ns), so
//!   overflow is not a concern;
//! - matches the `Frame.t_us_*` field names in the proto;
//! - is sufficient for inference cadence (~250 ms hops) and
//!   per-frame jitter analysis (~100 us scheduler granularity).
//!
//! # What this module does NOT provide
//!
//! - **Durations / arithmetic.** Use [`std::time::Duration`] for
//!   intervals; [`CaptureTime::since`] returns `Option<Duration>`
//!   so a saturating-subtract can't silently misbehave.
//! - **Formatted serialization.** Both types are `u64`-transparent
//!   under serde; downstream DTOs that need an RFC3339 string
//!   format wall-times at the boundary.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;

/// Microseconds since this process's boot, sampled from a
/// monotonic clock.  Stable under wall-clock skew.  Comparable
/// only within this process's lifetime.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct CaptureTime(u64);

impl CaptureTime {
    /// Sample the monotonic clock now.  Cheap (~10-50 ns on
    /// commodity x86 / aarch64).
    ///
    /// The reference [`Instant`] (boot anchor) is captured on
    /// first call via a process-local `OnceLock`; subsequent
    /// calls are `Instant::elapsed()` against that anchor.
    /// Worst-case skew from "true boot time" is the latency
    /// between `main()` starting and the first `now()` call
    /// (typically <100 ms).
    pub fn now() -> Self {
        Self(elapsed_us_since_boot_anchor())
    }

    /// Construct from a raw microsecond count.  For tests and
    /// deserialization paths; production code prefers
    /// [`Self::now`].
    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us)
    }

    /// Microseconds since the boot anchor.
    #[inline]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Difference between two [`CaptureTime`]s as a
    /// [`Duration`], or `None` if `self < other` (would
    /// underflow).
    pub const fn since(self, other: Self) -> Option<Duration> {
        if self.0 < other.0 {
            None
        } else {
            Some(Duration::from_micros(self.0 - other.0))
        }
    }
}

/// Microseconds since the Unix epoch (1970-01-01 00:00:00
/// UTC), sampled from a wall-clock.  Jumps on NTP correction;
/// comparable across processes and hosts.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct WallTime(u64);

impl WallTime {
    /// Sample the wall-clock now.  Returns `None` only on a
    /// system whose clock is set before the Unix epoch; in
    /// practice only on bare-metal boots without an RTC.
    pub fn now() -> Option<Self> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_micros()).ok())
            .map(Self)
    }

    /// Construct from a raw microsecond count.
    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us)
    }

    /// Microseconds since the Unix epoch.
    #[inline]
    pub const fn as_micros(self) -> u64 {
        self.0
    }
}

// MARK: BufferTimingAnchor

/// Snapshot of the producer's monotonic clock at a known
/// absolute head position in an audio ring buffer.
///
/// The audio producer (today: `audio_io::mic_arbitrator`)
/// updates the anchor after each `Writer::push` so consumers
/// holding a sample-position cursor (`Reader::tail`,
/// `Reader::peek_into` callers) can convert that cursor into
/// the capture monotonic time of the underlying audio without
/// a second hop through the producer.
///
/// Pre-anchor the daemon stamped `t_us_capture_monotonic` at
/// engine emit time -- correct as a clock domain
/// (CLOCK_MONOTONIC) but wrong as a semantic (encode/publish
/// time, not capture time).  The anchor closes the semantic
/// gap by carrying the producer-side timestamp at a known
/// position; consumers project from their own read position
/// to capture time via [`capture_us_for`].
///
/// Accuracy is bounded by:
///
/// - the producer's `Instant::now()` granularity (~ns to µs
///   on modern x86 / aarch64);
/// - the producer's push period (~5-23 ms at typical 256-1024
///   sample ALSA periods on 44.1 kHz capture); the anchor
///   reflects the moment the producer FINISHED reading the
///   period, so samples within the same push share the same
///   anchor and their per-sample timestamp is interpolated
///   linearly at `sample_rate_hz`;
/// - mic-swap boundaries: the anchor reflects the producer
///   currently writing.  A reader processing samples that
///   span a mic swap interpolates across the discontinuity
///   for the overlap window (~tens of ms).  Uncorrected
///   residual.
///
/// `Copy` so consumers can dereference the `ArcSwap` once and
/// pass the value by-value through math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferTimingAnchor {
    /// Absolute head position when this anchor was sampled.
    /// Matches `audio_buffer::AudioBuffer::head` /
    /// `Reader::tail` u64 sample counters.
    pub head_pos: u64,
    /// Monotonic time the producer recorded for this push.
    pub captured_at: CaptureTime,
    /// Producer-side sample rate (Hz) when this anchor was
    /// set.  Always 44.1 kHz canonical today; the field is
    /// recorded per-anchor so a future variable-rate producer
    /// can surface the rate alongside the anchor without a
    /// secondary lookup.
    pub sample_rate_hz: u32,
}

impl BufferTimingAnchor {
    /// Boot-time placeholder: `head_pos = 0`,
    /// `captured_at = 0`, `sample_rate_hz = 44_100`.
    /// Initialises the `SharedTimingAnchor` cell before the
    /// producer's first push so consumers see a
    /// self-consistent linear projection from a zero baseline
    /// (`capture_us_for(p, N) = N * 1e6 / 44_100`) rather
    /// than uninitialised state.  A zero `captured_at` is
    /// "no real anchor yet"; the producer overwrites it on
    /// the first push.
    pub const fn boot_placeholder() -> Self {
        Self {
            head_pos: 0,
            captured_at: CaptureTime::from_micros(0),
            sample_rate_hz: 44_100,
        }
    }
}

/// Wait-free, ArcSwap-backed cell so the producer can publish
/// a fresh anchor without locking and consumers can `load_full`
/// it in their hot loop (~5 ns).  One Arc allocation per push
/// at the producer's period rate (~50 Hz for canonical mic
/// periods); mimalloc absorbs.
///
/// Production wiring constructs one `SharedTimingAnchor`
/// initialised with [`BufferTimingAnchor::boot_placeholder`]
/// and threads it through to the producer (`mic_arbitrator`)
/// and every consumer (`opus_stream::run`,
/// `inference::engine`).  Tests substitute either a stub
/// producer that controls the anchor or `None`-shaped
/// consumers that fall back to the pre-anchor stamp.
pub type SharedTimingAnchor = Arc<ArcSwap<BufferTimingAnchor>>;

/// Construct a fresh `SharedTimingAnchor` initialised to
/// [`BufferTimingAnchor::boot_placeholder`].  Production
/// callers wire one of these from `daemon::main_body`.
pub fn shared_timing_anchor() -> SharedTimingAnchor {
    Arc::new(ArcSwap::from_pointee(BufferTimingAnchor::boot_placeholder()))
}

/// Project an absolute read position to its capture monotonic
/// time, given the latest [`BufferTimingAnchor`] published by
/// the producer.
///
/// `read_pos` is signed-relative to `anchor.head_pos`: reads
/// before the anchor head get a past timestamp; reads at or
/// after the anchor head get a present-or-future timestamp
/// (bounded by reality at one push period since the anchor
/// reflects the moment the producer finished its last push).
///
/// Saturates at 0 if the projection would underflow (e.g. a
/// reader holding a position from before the anchor's
/// `captured_at`).  Saturation is the right behaviour for the
/// daemon's purposes: a downstream consumer compares this
/// timestamp against its own `CaptureTime::now()` reading, and
/// a saturated 0 surfaces as "older than process boot," which
/// is the correct interpretation for samples captured before
/// any anchor existed.
pub fn capture_us_for(anchor: BufferTimingAnchor, read_pos: u64) -> u64 {
    // Convert `(read_pos - anchor.head_pos)` to microseconds,
    // signed.  At 44.1 kHz a u64 sample counter holds ~225
    // days of sample positions before overflow; the daemon
    // restarts well before then.
    let head_pos = anchor.head_pos as i128;
    let read = read_pos as i128;
    let sr = anchor.sample_rate_hz as i128;
    if sr == 0 {
        // Defensive: a zero sample_rate_hz anchor is malformed.
        // Surface as the anchor's captured_at so callers see
        // a deterministic value rather than a divide-by-zero
        // panic.
        return anchor.captured_at.as_micros();
    }
    let delta_samples = read - head_pos;
    // i128 multiplication keeps the intermediate product in
    // range even at extreme sample positions: 1e9 samples *
    // 1e6 us / 4.41e4 Hz = 2.27e10, well within i128.
    let delta_us = delta_samples * 1_000_000 / sr;
    let projected = anchor.captured_at.as_micros() as i128 + delta_us;
    projected.max(0) as u64
}

// MARK: Internals

/// Process-local boot anchor for [`CaptureTime::now`].  Captured
/// lazily on first call; subsequent calls compute
/// `Instant::now() - anchor`.
fn elapsed_us_since_boot_anchor() -> u64 {
    use std::sync::OnceLock;
    static ANCHOR: OnceLock<Instant> = OnceLock::new();
    let anchor = ANCHOR.get_or_init(Instant::now);
    u64::try_from(anchor.elapsed().as_micros()).unwrap_or(u64::MAX)
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn capture_time_now_is_monotonic() {
        let a = CaptureTime::now();
        thread::sleep(Duration::from_micros(100));
        let b = CaptureTime::now();
        assert!(b.as_micros() >= a.as_micros());
    }

    #[test]
    fn capture_time_since_returns_duration() {
        let a = CaptureTime::from_micros(1_000);
        let b = CaptureTime::from_micros(3_500);
        assert_eq!(b.since(a), Some(Duration::from_micros(2_500)));
        // Reversed direction underflows to None (no silent
        // saturating-sub footgun).
        assert_eq!(a.since(b), None);
    }

    #[test]
    fn wall_time_now_post_dates_2000() {
        // Any test environment sampled here should be running
        // well past the Unix epoch.  30 yr * 365.25 * 86400
        // ~= 9.467e8 s ~= 9.467e14 us.
        let now = WallTime::now().expect("clock past unix epoch");
        const Y2000_US: u64 = 946_684_800_000_000;
        assert!(now.as_micros() > Y2000_US);
    }

    #[test]
    fn from_micros_round_trips() {
        let c = CaptureTime::from_micros(42);
        let w = WallTime::from_micros(42);
        assert_eq!(c.as_micros(), 42);
        assert_eq!(w.as_micros(), 42);
    }

    /// The two types do NOT share a common `From`/`Into` -- the
    /// wire format must explicitly choose one when emitting a
    /// timestamp.
    #[test]
    fn types_are_distinct_at_compile_time() {
        fn _accepts_capture(_: CaptureTime) {}
        fn _accepts_wall(_: WallTime) {}
        // _accepts_capture(WallTime::from_micros(0));   // <-- compile error
        // _accepts_wall(CaptureTime::from_micros(0));   // <-- compile error
    }

    // MARK: BufferTimingAnchor tests

    /// Read at the anchor's exact head position projects to
    /// the anchor's `captured_at` -- the boundary case the
    /// rest of the math interpolates around.
    #[test]
    fn capture_us_for_at_anchor_head_returns_captured_at() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 44_100,
        };
        assert_eq!(capture_us_for(anchor, 44_100), 1_000_000);
    }

    /// Reads BEHIND the anchor head get a past timestamp:
    /// 44_100 samples back at 44.1 kHz = exactly 1 second.
    #[test]
    fn capture_us_for_behind_anchor_interpolates_back() {
        let anchor = BufferTimingAnchor {
            head_pos: 88_200,
            captured_at: CaptureTime::from_micros(2_000_000),
            sample_rate_hz: 44_100,
        };
        // 1 s back.
        assert_eq!(capture_us_for(anchor, 44_100), 1_000_000);
        // 0.5 s back (22_050 samples).
        assert_eq!(capture_us_for(anchor, 66_150), 1_500_000);
    }

    /// Reads AHEAD of the anchor head extrapolate forward.
    /// The anchor is updated once per producer push, so a
    /// reader observing samples slightly past the recorded
    /// head_pos applies the linear sample-rate projection.
    #[test]
    fn capture_us_for_ahead_of_anchor_extrapolates_forward() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 44_100,
        };
        // 22_050 samples ahead = 0.5 s ahead.
        assert_eq!(capture_us_for(anchor, 66_150), 1_500_000);
    }

    /// Underflow saturates at 0: a read position that would
    /// project to a negative timestamp clamps to 0 (older
    /// than process boot, the correct interpretation).
    #[test]
    fn capture_us_for_saturates_on_underflow() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(500_000),
            sample_rate_hz: 44_100,
        };
        // Read at sample 0 would project to 500_000 us minus
        // 44_100 samples * 1_000_000 us / 44_100 samples =
        // -500_000 us; saturates to 0.
        assert_eq!(capture_us_for(anchor, 0), 0);
    }

    /// Sample-rate variance changes the interpolation slope.
    /// Pin both 22.05 kHz and 48 kHz for the same delta.
    #[test]
    fn capture_us_for_respects_anchor_sample_rate() {
        let half = BufferTimingAnchor {
            head_pos: 22_050,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 22_050,
        };
        // 22_050 samples back at 22.05 kHz = 1 s.
        assert_eq!(capture_us_for(half, 0), 0);

        let high = BufferTimingAnchor {
            head_pos: 48_000,
            captured_at: CaptureTime::from_micros(2_000_000),
            sample_rate_hz: 48_000,
        };
        // 24_000 samples back at 48 kHz = 0.5 s.
        assert_eq!(capture_us_for(high, 24_000), 1_500_000);
    }

    /// Defensive: a malformed anchor with `sample_rate_hz = 0`
    /// returns the anchor's `captured_at` instead of dividing
    /// by zero.  Production paths never produce such an
    /// anchor (the producer always sets a non-zero rate); the
    /// guard lets a future bug surface as a deterministic
    /// stale stamp rather than a hot-path panic.
    #[test]
    fn capture_us_for_zero_sample_rate_returns_captured_at() {
        let bad = BufferTimingAnchor {
            head_pos: 100,
            captured_at: CaptureTime::from_micros(7_777_777),
            sample_rate_hz: 0,
        };
        assert_eq!(capture_us_for(bad, 0), 7_777_777);
        assert_eq!(capture_us_for(bad, 1_000_000), 7_777_777);
    }

    /// `boot_placeholder` initialises a self-consistent
    /// anchor that reads as "process boot, no audio yet."
    /// Pinned because the daemon constructs one before the
    /// arbitrator's first push to give consumers a value
    /// other than uninitialised memory.
    #[test]
    fn boot_placeholder_is_self_consistent() {
        let p = BufferTimingAnchor::boot_placeholder();
        assert_eq!(p.head_pos, 0);
        assert_eq!(p.captured_at.as_micros(), 0);
        assert_eq!(p.sample_rate_hz, 44_100);
        // Reading at position 0 projects to 0.
        assert_eq!(capture_us_for(p, 0), 0);
        // Reading at any position projects to that position's
        // capture time relative to a captured_at = 0 baseline,
        // which is monotonically non-negative.
        assert!(capture_us_for(p, 22_050) > 0);
    }

    /// `shared_timing_anchor()` returns a fresh ArcSwap
    /// initialised to the boot placeholder; storing a fresh
    /// anchor via `.store(Arc::new(...))` is observed by
    /// subsequent `.load_full()` calls.  This is the
    /// concurrency primitive consumers will use in their hot
    /// loop.
    #[test]
    fn shared_timing_anchor_round_trips() {
        let cell = shared_timing_anchor();
        let initial = **cell.load();
        assert_eq!(initial, BufferTimingAnchor::boot_placeholder());

        let fresh = BufferTimingAnchor {
            head_pos: 12_345,
            captured_at: CaptureTime::from_micros(67_890),
            sample_rate_hz: 48_000,
        };
        cell.store(Arc::new(fresh));
        let observed = **cell.load();
        assert_eq!(observed, fresh);
    }
}
