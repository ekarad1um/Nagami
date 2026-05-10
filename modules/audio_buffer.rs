//! Single-writer / multi-reader f32 sample ring buffer with copy-out
//! reads and a seqlock-style recheck guarding against the writer
//! overwriting cells during the read.
//!
//! # Threading model
//!
//! ```text
//!                          +------------------------------+
//!  std-thread (arbitrator) |  capacity samples, mod-cap   |
//!  Writer::push  --------> |  data: Box<[AtomicU32]>      |
//!                          |  head: AtomicU64 (padded)    |
//!                          +-------------^----------------+
//!                                        |  Acquire load on head
//!                                        |  + per-cell Relaxed load
//!                          tokio task #1 |  + recheck head
//!                          Reader::peek_into(&mut out)
//!                                        |
//!                          tokio task #2 |
//!                          Reader::peek_into(&mut out)   (independent tail)
//! ```
//!
//! - Writer is unique (panics if a second `take_writer` is called
//!   while one is alive).  Holds `Send + !Sync + !Clone`.
//! - Readers are owned per task; created cheaply via `reader()` /
//!   `reader_at()`.  Each holds its own `tail: u64`.  `Send + !Sync`.
//! - The buffer itself (`AudioBuffer`) is `Send + Sync + Clone`
//!   (cheap Arc bump) -- pass it freely to whoever needs handles.
//!
//! # Memory ordering
//!
//! - Writer: per-cell `store(f32::to_bits(s), Relaxed)` x n ->
//!   `head.store(head + n, Release)`.  (Plain store on head, not
//!   `fetch_add` -- single-writer makes the RMW superfluous; same
//!   publication edge at lower cost: `mov` vs `lock xadd` on x86-64,
//!   `stlr` vs `ldaxr/stlxr` loop on aarch64.)
//! - Reader: `head.load(Acquire)` -> bounds-check -> per-cell
//!   `load(Relaxed)` -> `f32::from_bits` -> re-check head.
//!
//! The Release/Acquire pair on `head` establishes the happens-before
//! edge that makes the cell stores visible to the reader.  The cells
//! themselves are atomic (`AtomicU32`), so per-cell access is never
//! torn -- the seqlock recheck guards only against the writer
//! *overwriting* a cell while we were reading it, not against
//! byte-level tearing.
//!
//! # Soundness
//!
//! Two invariants protect the ring against stale-mixed reads:
//!
//! 1. **Single-writer**: `take_writer` swaps `writer_taken` and
//!    panics if it was already set.  `Writer::drop` clears the flag.
//!    Concurrent writers would race on the per-cell stores; while
//!    each cell store is atomic, two writers would interleave at
//!    the cell level and produce gibberish samples.
//!
//! 2. **Safety margin**: at peek time we require
//!    `head - tail <= capacity * 3/4`, leaving a `cap/4`
//!    cushion.  The post-copy recheck reloads head and
//!    reports `Lagged` if the writer surged past the
//!    cushion during the per-cell loads; the read is
//!    *discarded* rather than returned with possibly mixed
//!    old / new samples.  The recheck is strictly tighter
//!    than the actual overlap boundary (the writer would
//!    need to push `cap - avail + 1` more samples to lap
//!    the reader's first cell, while the recheck trips
//!    after just `cmm - avail + 1`), so false-positive
//!    `Lagged` is possible; false-negative `Ready` is not.
//!    No internal retry: re-running `peek_into` without
//!    moving tail would re-observe `avail > cmm` and lag
//!    immediately, so the caller must `seek_latest` to
//!    recover.
//!
//! # Latency note for downstream consumers
//!
//! Readers that poll on `Wait` (e.g. the streaming inference engine)
//! pay polling latency -- up to one sleep interval between "writer
//! published enough samples" and "reader noticed".  The plan's
//! suggested `min(hop / 4, 50ms)` gives <= 50 ms slack.  If tighter
//! latency is required, expose a `tokio::sync::Notify`-style
//! signal that's bumped after every `Writer::push` and have the
//! reader `select!` between push events + a long timeout.  Out of
//! scope for the standalone `audio_buffer` crate (no async dep);
//! intended to be wired up by the inference engine.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

use std::cell::UnsafeCell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Cache-line-aligned wrapper. 64 bytes covers x86-64 and aarch64
/// (and is the conservative-correct floor; some targets -- Apple
/// Silicon, ppc64 -- use 128-byte "destructive interference" lines,
/// where this still helps but doesn't fully isolate).  Used to keep
/// the contended `head` atomic out of the same line as the
/// otherwise-immutable `Inner` fields, so the writer's per-push
/// store doesn't invalidate readers' cached metadata.
#[repr(align(64))]
struct CacheLineAligned<T>(T);

impl<T> Deref for CacheLineAligned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Outcome of a [`Reader::peek_into`] call.
///
/// `Ready` means `out` was fully populated.  `Wait` means there are
/// not yet `out.len()` unconsumed samples and the caller should
/// retry after some delay (one hop interval is sensible for the
/// inference loop).  `Lagged` means the writer has gotten too far
/// ahead and the reader should `seek_latest` to resync.
#[must_use = "ignoring ReadStatus risks reading uninitialized samples on Wait/Lagged"]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ReadStatus {
    /// `out` was fully written; tail is unchanged.  Caller
    /// decides how many samples to consume via
    /// [`Reader::advance`].
    Ready,
    /// Not enough new data yet (`head - tail < out.len()`).
    /// `out` is unchanged.
    Wait,
    /// Reader fell more than `capacity * 3/4` samples
    /// behind head.  `out` may be partially or fully
    /// clobbered (don't trust it).  Caller should call
    /// [`Reader::seek_latest`] to resync.
    Lagged {
        /// Distance `head - tail` at the time of detection.
        by: u64,
    },
}

/// Internal: the sample ring + atomic coordination state.  Held inside
/// an `Arc` and shared between the AudioBuffer handle, the writer,
/// and any number of readers.
///
/// All fields are atomic (or immutable after construction), so `Inner`
/// is auto-`Sync`; no manual `unsafe impl` required.  The
/// happens-before edge between writer and reader is established by
/// `head`'s Release/Acquire pair, not by the per-cell ordering
/// (which is `Relaxed`).
struct Inner {
    /// Fixed-size ring of `capacity` cells, each holding the bit
    /// pattern of one f32 sample (via `f32::to_bits`/`from_bits`).
    /// `AtomicU32` matches f32's size and alignment exactly, so the
    /// memory layout is identical to a `Box<[f32]>`.
    data: Box<[AtomicU32]>,
    /// Cached `data.len()`.
    capacity: usize,
    /// `capacity - capacity / 4`.  Reader's "safe to read" condition is
    /// `head - tail <= capacity_minus_margin`.
    capacity_minus_margin: usize,
    /// `capacity / 4`.  Maximum samples a single [`Writer::push`]
    /// may publish without breaking the seqlock recheck (see
    /// [`Writer::push`] for the proof).
    safety_margin: usize,
    /// True iff a [`Writer`] currently exists for this buffer.
    writer_taken: AtomicBool,
    /// Total samples ever written.  Monotonic, never wraps in any
    /// realistic uptime (2^64 / 44100 ~= 13 million years).
    ///
    /// Cache-line padded: this is the only contended atomic in
    /// `Inner` (writer stores per-push, readers Acquire-load twice
    /// per peek).  Padding it onto its own line stops the writer's
    /// `Release`-store from invalidating readers' cached copies of
    /// `capacity`/`capacity_minus_margin`/`data` (which never
    /// change after construction).
    head: CacheLineAligned<AtomicU64>,
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inner")
            .field("capacity", &self.capacity)
            .field("capacity_minus_margin", &self.capacity_minus_margin)
            .field("head", &self.head.load(Ordering::Relaxed))
            .field("writer_taken", &self.writer_taken.load(Ordering::Relaxed))
            .finish()
    }
}

/// Handle to a sample ring buffer.  Cheap to clone (Arc bump); pass
/// freely to whoever needs to spawn readers or take the writer.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    inner: Arc<Inner>,
}

impl AudioBuffer {
    /// Create a fresh buffer with `capacity` sample slots.
    ///
    /// `capacity` must be a power of two and >= 4:
    ///
    /// - **Power of two**: the wrap-index inside `push`/`peek_into`
    ///   then collapses to `head & (capacity - 1)` -- a single-cycle
    ///   AND instead of a 20-30-cycle 64-bit modulo.  This is the
    ///   universal ring-buffer idiom for a reason.
    /// - **>= 4**: the seqlock invariant requires a positive safety
    ///   margin (`cap/4 >= 1`); below this the post-copy recheck
    ///   cannot strictly distinguish "writer hasn't lapped" from
    ///   "writer just lapped."
    ///
    /// Callers are also expected to size the buffer for **>= 2 x
    /// the largest expected peek window**.  For the daemon at
    /// 44.1 kHz with `WaveformLen = 44 032` samples, **262 144**
    /// (`= 2^18`, ~= 5.94 s, 1 MB) is the canonical choice -- the
    /// next power of two above the original `5 x 44 100 = 220 500`
    /// target, giving ~= 2.97 x headroom over `WaveformLen`.
    ///
    /// **What capacity does NOT do**: it does NOT govern inference
    /// latency.  As long as `capacity >= 2 x peek_window`, every
    /// successful `peek_into` returns `Ready` immediately -- the
    /// reader's wait time is bounded only by the *audio rate*
    /// (44 100 samples/sec) and the writer's pacing (ALSA
    /// `period_size` granularity).  Increasing capacity beyond ~5 s
    /// just buys longer recovery margin against transient consumer
    /// stalls (training spike, GC pause); it does NOT speed up
    /// inference.  Set capacity to "the smallest power of two that
    /// gives enough headroom against the slowest stall you care to
    /// recover from."
    ///
    /// # Panics
    /// Panics if `capacity` is not a power of two, or if
    /// `capacity < 4`.  Use [`usize::next_power_of_two`] to round
    /// up an arbitrary target sample count.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity >= 4 && capacity.is_power_of_two(),
            "AudioBuffer capacity must be a power of two and >= 4 (got {capacity}); \
             the wrap-index `head & (capacity - 1)` requires this. Use \
             `usize::next_power_of_two()` to round up; the canonical production \
             value is 262144 (= 2^18, ~= 5.94 s at 44.1 kHz)",
        );
        let data: Box<[AtomicU32]> = (0..capacity).map(|_| AtomicU32::new(0)).collect();
        let safety_margin = capacity / 4;
        let capacity_minus_margin = capacity - safety_margin;
        Self {
            inner: Arc::new(Inner {
                data,
                capacity,
                capacity_minus_margin,
                safety_margin,
                writer_taken: AtomicBool::new(false),
                head: CacheLineAligned(AtomicU64::new(0)),
            }),
        }
    }

    /// Total ring capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Largest peek window that can be served (= `capacity - capacity/4`).
    /// Peeks of `out.len() > safe_peek_window()` will never return
    /// `Ready`; typically this is a bug -- sized your capacity wrong.
    #[inline]
    pub fn safe_peek_window(&self) -> usize {
        self.inner.capacity_minus_margin
    }

    /// Maximum samples a single [`Writer::push`] may publish
    /// (= `capacity / 4`, the seqlock safety margin).  Callers
    /// with larger batches must split into `<= max_push_len()`
    /// chunks; see [`Writer::push`] for the soundness proof.
    ///
    /// Cross-module invariant: the canonical production
    /// capacity is 262 144 samples, so this returns 65 536.
    /// The capture-side validator
    /// [`crate::audio_io::mic_arbitrator::MAX_PERIOD_FRAMES`]
    /// = 8 192 stays 8x below this margin.  If the canonical
    /// capacity ever shrinks, that constant must be revisited
    /// in lockstep so admitted ALSA periods do not panic
    /// `Writer::push`.  The two values are reviewed together;
    /// a future deployment that shrinks `AudioBuffer::new(...)`
    /// must update `MAX_PERIOD_FRAMES` in the same diff.
    #[inline]
    pub fn max_push_len(&self) -> usize {
        self.inner.safety_margin
    }

    /// Current head (samples written so far).  Mostly useful for
    /// tests and for instrumentation; the daemon's hot paths shouldn't
    /// need this.
    #[inline]
    pub fn head(&self) -> u64 {
        self.inner.head.load(Ordering::Acquire)
    }

    /// Acquire the unique [`Writer`].  Single-instance enforced via an
    /// internal flag; the flag is cleared when the Writer is dropped.
    ///
    /// # Panics
    /// Panics if a Writer for this buffer already exists.
    pub fn take_writer(&self) -> Writer {
        // Acquire (not AcqRel): we need to synchronize-with the prior
        // `Writer::drop`'s Release-store on the flag, but we have no
        // prior writes of our own to publish at this swap.
        if self.inner.writer_taken.swap(true, Ordering::Acquire) {
            panic!("audio_buffer: a Writer already exists for this AudioBuffer");
        }
        Writer {
            inner: Arc::clone(&self.inner),
            _not_sync: PhantomData,
        }
    }

    /// Spawn a reader at the live edge -- `tail = head` at call time.
    /// Subsequent `peek_into` calls return `Wait` until the writer
    /// has produced enough new samples.
    ///
    /// This is the right factory for readers that don't care about
    /// historical audio (inference, streaming Opus during active
    /// session).
    pub fn reader(&self) -> Reader {
        let tail = self.inner.head.load(Ordering::Acquire);
        Reader {
            inner: Arc::clone(&self.inner),
            tail,
            _not_sync: PhantomData,
        }
    }

    /// Spawn a reader `behind_head` samples behind the live edge --
    /// `tail = head - behind_head` (saturating).  Useful when starting
    /// up a stream that wants a small backlog so its first packet
    /// doesn't depend on the very next push (Opus resume).
    pub fn reader_at(&self, behind_head: usize) -> Reader {
        let head = self.inner.head.load(Ordering::Acquire);
        let tail = head.saturating_sub(behind_head as u64);
        Reader {
            inner: Arc::clone(&self.inner),
            tail,
            _not_sync: PhantomData,
        }
    }
}

/// Unique sample writer.  `Send + !Sync + !Clone`.  Pushes new samples
/// at the head, advancing it with `Release` ordering so readers see a
/// consistent slice of the ring after their `Acquire` load.
pub struct Writer {
    inner: Arc<Inner>,
    /// `PhantomData<UnsafeCell<()>>` is `Send + !Sync`; this marker
    /// prevents the Writer from being shared by reference across
    /// threads (per the contract: single-writer is required for
    /// soundness).
    _not_sync: PhantomData<UnsafeCell<()>>,
}

impl fmt::Debug for Writer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("head", &self.inner.head.load(Ordering::Relaxed))
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

impl Writer {
    /// Same as [`AudioBuffer::max_push_len`]; exposed on
    /// [`Writer`] so producers can chunk without holding the
    /// [`AudioBuffer`] handle.
    #[inline]
    pub fn max_push_len(&self) -> usize {
        self.inner.safety_margin
    }

    /// Current absolute head position (total samples written
    /// so far across all pushes, modulo a u64 wraparound that
    /// the daemon's restart cadence makes unreachable).
    /// Reads `inner.head` with `Acquire` to synchronize-with
    /// the latest internal `Release` store from `push`; on
    /// the same thread that owns the [`Writer`] (single-
    /// writer invariant) the value reflects the most recent
    /// push.
    ///
    /// Used by the producer-side
    /// [`crate::common::time::BufferTimingAnchor`] update in
    /// `audio_io::mic_arbitrator::process_period`: after each
    /// `Writer::push` the producer captures the post-push
    /// head plus the current monotonic time so consumers can
    /// project a sample position back to its capture
    /// monotonic time without going through the producer.
    #[inline]
    pub fn head_pos(&self) -> u64 {
        self.inner.head.load(Ordering::Acquire)
    }

    /// Append `samples` to the ring.  Wraps around silently;
    /// samples that fall outside `[head - capacity, head)` are
    /// overwritten and become unreadable.
    ///
    /// # Invariant
    /// `samples.len() <= capacity / 4`
    /// (= [`AudioBuffer::max_push_len`]).  Cells are written
    /// `Relaxed` BEFORE `head`'s `Release` store, so a push
    /// larger than the reader's safety margin could overwrite
    /// cells inside a reader's currently-safe window before the
    /// recheck observes `head` advancing past
    /// `capacity_minus_margin`, returning torn samples.
    ///
    /// O(samples.len()) -- one Relaxed `AtomicU32::store` per
    /// sample.  LLVM does NOT vectorize atomic stores (would
    /// change observed atomicity); scalar `mov`s, well within
    /// the audio-rate budget.
    ///
    /// # Panics
    /// Panics if `samples.len() > capacity / 4` -- hard assert
    /// in every build because silent corruption is harder to
    /// debug than a loud failure.
    #[inline]
    pub fn push(&mut self, samples: &[f32]) {
        let n = samples.len();
        if n == 0 {
            return;
        }
        let cap = self.inner.capacity;
        let max_push = self.inner.safety_margin;
        assert!(
            n <= max_push,
            "Writer::push: samples.len()={n} exceeds max_push_len={max_push} \
             (= {cap}/4); split the batch.",
        );
        // Relaxed is fine -- only the writer modifies
        // head, and the `Release` store below synchronises
        // visibility.
        let head = self.inner.head.load(Ordering::Relaxed);
        let start = (head & (cap as u64 - 1)) as usize;
        // Split into the contiguous head and the
        // wrap-around tail.  `n <= cap` (asserted above)
        // keeps both slices in range; the regions are
        // disjoint by construction.
        let first = (cap - start).min(n);
        let (head_part, wrap_part) = samples.split_at(first);
        let data = &self.inner.data;
        for (cell, &s) in data[start..start + first].iter().zip(head_part) {
            cell.store(s.to_bits(), Ordering::Relaxed);
        }
        for (cell, &s) in data[..wrap_part.len()].iter().zip(wrap_part) {
            cell.store(s.to_bits(), Ordering::Relaxed);
        }
        // Release-store publishes all prior Relaxed cell
        // stores; the reader's Acquire-load on `head`
        // synchronizes-with this store.
        self.inner
            .head
            .store(head.wrapping_add(n as u64), Ordering::Release);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.inner.writer_taken.store(false, Ordering::Release);
    }
}

/// Sample reader.  Each reader owns its `tail`; multiple readers
/// progress at independent rates.  Readers do not lend out slices into
/// the ring -- they always copy out into a caller-provided buffer.
/// `Send + !Sync`.
pub struct Reader {
    inner: Arc<Inner>,
    tail: u64,
    _not_sync: PhantomData<UnsafeCell<()>>,
}

impl fmt::Debug for Reader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("tail", &self.tail)
            .field("head", &self.inner.head.load(Ordering::Relaxed))
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

impl Reader {
    /// Copy `out.len()` of the oldest unconsumed samples into `out`,
    /// **without** advancing tail.  Returns `Ready` on success;
    /// `Wait` if not enough data is available; `Lagged { by }` if
    /// the writer has gotten more than `capacity * 3/4` samples
    /// ahead.
    ///
    /// Single-iteration seqlock: load head -> bounds-check -> per-cell
    /// Relaxed loads -> re-load head.  If the writer overwrote any of
    /// our cells during the loads (i.e. its head crossed
    /// `tail + cap`), the recheck reports `Lagged` and the data in
    /// `out` is to be discarded -- the caller is expected to
    /// `seek_latest` and retry.  Per-cell loads are atomic, so each
    /// individual sample is either pre- or post-overwrite, never
    /// torn at the byte level; the seqlock only guards against the
    /// *mix* of old and new samples that an overlapping write would
    /// produce.
    ///
    /// No retry loop: head is monotonic, so a second iteration after
    /// a recheck failure would observe `avail > cmm` at the very
    /// first precheck and return `Lagged` immediately -- i.e. the
    /// retry would be a strict no-op.  The recheck alone suffices for
    /// deterministic safety.
    #[must_use = "the read may have failed (Wait/Lagged); inspect the status before consuming `out`"]
    #[inline]
    pub fn peek_into(&self, out: &mut [f32]) -> ReadStatus {
        let n = out.len();
        if n == 0 {
            return ReadStatus::Ready;
        }
        // Hard-assert in every build so a wedged consumer
        // surfaces immediately.  A `debug_assert!` would let
        // release builds silently accept oversized peeks and
        // stall in `Wait` forever, since `avail < n_u64` is
        // unreachable when `n > capacity`.  Caught by the
        // daemon's task-drain log path and by the supervisor
        // as a heartbeat downgrade.
        assert!(
            n <= self.inner.capacity_minus_margin,
            "peek_into called with out.len()={n} > safe_peek_window={}; \
             the buffer's capacity ({}) is sized too small for this peek",
            self.inner.capacity_minus_margin,
            self.inner.capacity,
        );

        let cap = self.inner.capacity;
        let mask = (cap - 1) as u64;
        let cmm = self.inner.capacity_minus_margin as u64;
        let n_u64 = n as u64;

        let h0 = self.inner.head.load(Ordering::Acquire);
        // saturating_sub handles the (unlikely) case where the
        // reader was constructed at a future tail (e.g. seek_latest
        // before the writer caught up); avail = 0 -> Wait.
        let avail = h0.saturating_sub(self.tail);
        if avail < n_u64 {
            return ReadStatus::Wait;
        }
        if avail > cmm {
            return ReadStatus::Lagged { by: avail };
        }

        // avail <= cmm = cap - cap/4, so [tail, tail+n) is wholly
        // within the writer's "valid past" [h0 - cap, h0).  The
        // writer cannot overwrite any cell in our range without
        // first pushing >= cap/4 more samples -- and any progress
        // past `cmm` flips the recheck below to Lagged, which
        // discards whatever mixture we may have read.
        let start = (self.tail & mask) as usize;
        let first = (cap - start).min(n);
        let (out_head, out_wrap) = out.split_at_mut(first);
        let data = &self.inner.data;
        for (slot, cell) in out_head.iter_mut().zip(&data[start..start + first]) {
            *slot = f32::from_bits(cell.load(Ordering::Relaxed));
        }
        let wrap_len = out_wrap.len();
        for (slot, cell) in out_wrap.iter_mut().zip(&data[..wrap_len]) {
            *slot = f32::from_bits(cell.load(Ordering::Relaxed));
        }

        // Pin the per-cell Relaxed loads in program order
        // before the recheck's Acquire load.  A
        // `compiler_fence(Acquire)` would NOT suffice on
        // weakly-ordered architectures (aarch64): LDAR (the
        // recheck's Acquire load) is one-way -- it forbids
        // subsequent loads from being hoisted above it, but
        // permits prior Relaxed loads to sink past it.
        // Without this CPU load-load barrier the cell loads
        // could observe a writer state newer than `h1`
        // (the recheck), so a writer lap during a long
        // reader stall would slip past `avail1 <= cmm`.
        // Unreachable in practice under SCHED_FIFO +
        // bounded loop timing, but
        // correctness-by-construction is cheaper than
        // reasoning about preemption windows.  The full
        // `fence(Acquire)` emits `dmb ishld` on aarch64;
        // negligible at audio-rate cadence.
        std::sync::atomic::fence(Ordering::Acquire);

        // Recheck: did the writer overwrite any of our cells during
        // the loads?
        let h1 = self.inner.head.load(Ordering::Acquire);
        let avail1 = h1.saturating_sub(self.tail);
        if avail1 <= cmm {
            ReadStatus::Ready
        } else {
            ReadStatus::Lagged { by: avail1 }
        }
    }

    /// Advance tail by `n` samples.  Caller is responsible
    /// for not over-advancing; this asserts against head in
    /// every build.
    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.tail = self.tail.saturating_add(n as u64);
        // Hard-assert (not `debug_assert!`) so an
        // over-advance surfaces as a panic in release rather
        // than a permanent dataflow stall: the
        // `saturating_add` above caps tail silently, after
        // which the reader sits forever in `Wait`
        // (`avail = 0`).
        let head = self.inner.head.load(Ordering::Relaxed);
        assert!(
            self.tail <= head,
            "Reader::advance({n}): tail={} now exceeds head={head}; over-advanced",
            self.tail,
        );
    }

    /// Resync tail to `head - n` (saturating).  After a `Lagged`
    /// outcome, call this to drop the backlog and resume from the
    /// freshest `n` samples.
    #[inline]
    pub fn seek_latest(&mut self, n: usize) {
        let head = self.inner.head.load(Ordering::Acquire);
        self.tail = head.saturating_sub(n as u64);
    }

    /// Current tail (samples consumed so far on this reader).
    #[inline]
    pub fn tail(&self) -> u64 {
        self.tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_zero_capacity_panics() {
        let _ = AudioBuffer::new(0);
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_capacity_below_minimum_panics() {
        // cap=3 fails both checks (not pow2, and below the
        // safety-margin floor of 4 where cap/4 < 1).
        let _ = AudioBuffer::new(3);
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_non_power_of_two_capacity_panics() {
        // cap=100 satisfies >=4 but not pow2; the bitwise wrap
        // index `head & (cap - 1)` only equals `head % cap` for
        // pow2 capacities, so non-pow2 must reject.
        let _ = AudioBuffer::new(100);
    }

    /// Lock down the layout invariant the cache-line padding depends
    /// on: `head` must be on a 64-byte boundary inside `Inner`, and
    /// the wrapper must occupy a full 64-byte line.  If a future
    /// struct edit silently regresses this (e.g. by stripping
    /// `#[repr(align(64))]` or reordering fields in a way that
    /// shadows the alignment), the writer's per-push store will
    /// invalidate readers' cached metadata -- the test catches that
    /// at build time instead of as a slow regression in the daemon.
    #[test]
    fn head_is_isolated_on_its_own_cache_line() {
        use std::mem;
        assert_eq!(mem::size_of::<CacheLineAligned<AtomicU64>>(), 64);
        assert_eq!(mem::align_of::<CacheLineAligned<AtomicU64>>(), 64);
        assert_eq!(mem::offset_of!(Inner, head) % 64, 0);
        // No other Inner field shares head's line.  With head's 64-byte
        // size+alignment, the line covered by head is fully head's.
        let head_off = mem::offset_of!(Inner, head);
        let head_line_end = head_off + 64;
        for (name, off) in [
            ("data", mem::offset_of!(Inner, data)),
            ("capacity", mem::offset_of!(Inner, capacity)),
            (
                "capacity_minus_margin",
                mem::offset_of!(Inner, capacity_minus_margin),
            ),
            ("safety_margin", mem::offset_of!(Inner, safety_margin)),
            ("writer_taken", mem::offset_of!(Inner, writer_taken)),
        ] {
            assert!(
                off < head_off || off >= head_line_end,
                "field {name} at offset {off} overlaps head's cache line [{head_off}, {head_line_end})",
            );
        }
    }

    /// `cap = 4` is the smallest legal capacity (any smaller would
    /// give `safety_margin = 0` and break the seqlock recheck).
    /// Verifies the bitwise wrap (`mask = 3`) handles the smallest
    /// non-trivial ring correctly.  At cap=4 the seqlock bound is
    /// `max_push_len = 1`, so each `push` writes one sample.
    #[test]
    fn cap_4_smallest_valid_capacity_works() {
        let buf = AudioBuffer::new(4); // safety_margin=1, cmm=3
        assert_eq!(buf.safe_peek_window(), 3);
        assert_eq!(buf.max_push_len(), 1);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        // Fill three cells one sample at a time (max_push_len=1).
        for s in [1.0_f32, 2.0, 3.0] {
            w.push(&[s]);
        }
        let mut out = [0.0; 3];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        // Advance and push more to exercise the wrap-around with
        // mask=3 (positions 0..3 then position 0), still one sample
        // per call.
        r.advance(3);
        for s in [4.0_f32, 5.0] {
            w.push(&[s]); // cells 3, 0 -- wraps the ring boundary
        }
        let mut out2 = [0.0; 2];
        assert_eq!(r.peek_into(&mut out2), ReadStatus::Ready);
        assert_eq!(out2, [4.0, 5.0]);
    }

    /// Peek of exactly `safe_peek_window` samples should succeed.
    /// Boundary case for the seqlock soundness argument.  Filled in
    /// `safety_margin`-sized batches because a single push is
    /// bounded by `max_push_len = safety_margin`.
    #[test]
    fn peek_at_exactly_cmm_is_ready() {
        let buf = AudioBuffer::new(16); // safety_margin=4, cmm=12
        assert_eq!(buf.max_push_len(), 4);
        let mut w = buf.take_writer();
        let r = buf.reader();
        // Fill 12 cells in three pushes of 4 (= max_push_len).
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        let mut out = [0.0; 12];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        // Exactly cmm samples returned correctly.
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    /// A single push larger than the ring would wrap past itself
    /// in the wrap branch (e.g. cap=10, head=5, push of 20 ->
    /// the second slice would overwrite cells we just wrote).
    /// `max_push_len` rejects this loudly.
    #[test]
    #[should_panic(expected = "exceeds max_push_len")]
    fn push_larger_than_capacity_panics() {
        let buf = AudioBuffer::new(8); // safety_margin=2
        let mut w = buf.take_writer();
        // Prime head off zero so wrap-arithmetic in push is exercised.
        w.push(&[0.0; 2]);
        // 20 > max_push_len=2 -> must panic, not corrupt memory.
        let big: Vec<f32> = (0..20).map(|i| i as f32).collect();
        w.push(&big);
    }

    /// `push` of `safety_margin + 1` panics (the seqlock bound
    /// is the strict cap, enforced in every build).
    #[test]
    #[should_panic(expected = "exceeds max_push_len")]
    fn push_exceeds_safety_margin_panics() {
        let buf = AudioBuffer::new(64); // safety_margin = 16
        assert_eq!(buf.max_push_len(), 16);
        let mut w = buf.take_writer();
        // safety_margin + 1 -- one over the seqlock bound, well
        // under the capacity bound (17 < 64).  Must still panic.
        let just_over: Vec<f32> = (0..17).map(|i| i as f32).collect();
        w.push(&just_over);
    }

    /// Boundary positive: a push of exactly `safety_margin`
    /// samples is the largest legal single publication and must
    /// succeed.  Pinning this prevents an off-by-one regression
    /// in the new bound (`<=` vs `<`).
    #[test]
    fn push_at_exactly_safety_margin_succeeds() {
        let buf = AudioBuffer::new(64); // safety_margin = 16
        let mut w = buf.take_writer();
        let r = buf.reader();
        let exact: Vec<f32> = (0..16).map(|i| i as f32).collect();
        w.push(&exact); // exactly max_push_len -- legal
        let mut out = [0.0; 16];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    #[should_panic(expected = "Writer already exists")]
    fn take_writer_twice_panics() {
        let buf = AudioBuffer::new(64);
        let _w1 = buf.take_writer();
        let _w2 = buf.take_writer();
    }

    #[test]
    fn take_writer_after_drop_ok() {
        let buf = AudioBuffer::new(64);
        {
            let _w1 = buf.take_writer();
        } // _w1 dropped, flag cleared
        let _w2 = buf.take_writer();
    }

    #[test]
    fn reader_starts_at_live_edge() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        // Reader spawned AFTER 4 samples were already pushed.
        let r = buf.reader();
        let mut out = [0.0; 2];
        // No new samples since reader was created -> Wait.
        assert_eq!(r.peek_into(&mut out), ReadStatus::Wait);
        // Push more samples, reader sees them.
        w.push(&[5.0, 6.0]);
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [5.0, 6.0]);
    }

    #[test]
    fn reader_at_with_backlog_returns_ready_immediately() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        // Reader_at(2) starts 2 samples behind head=4, so tail=2,
        // and there are 2 samples in [2, 4) that the reader can see.
        let r = buf.reader_at(2);
        let mut out = [0.0; 2];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [3.0, 4.0]);
    }

    #[test]
    fn simple_push_and_peek_round_trip() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut out = [0.0; 5];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn peek_does_not_advance() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[10.0, 20.0, 30.0]);
        let mut out = [0.0; 3];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0, 30.0]);
        // Second peek with no advance returns the same samples.
        out.fill(0.0);
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn advance_consumes_samples() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        w.push(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut out = [0.0; 2];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0]);
        r.advance(2); // tail=2
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [30.0, 40.0]);
        r.advance(2); // tail=4 -> only sample [50] remains
        let mut last = [0.0; 2];
        assert_eq!(r.peek_into(&mut last), ReadStatus::Wait);
        let mut one = [0.0; 1];
        assert_eq!(r.peek_into(&mut one), ReadStatus::Ready);
        assert_eq!(one, [50.0]);
    }

    #[test]
    fn peek_wait_when_short() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[1.0, 2.0]);
        let mut out = [0.0; 5];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Wait);
        // out is left untouched.
        assert_eq!(out, [0.0; 5]);
    }

    #[test]
    fn peek_lagged_when_writer_far_ahead() {
        // capacity 16; safety_margin = 4; capacity_minus_margin = 12.
        let buf = AudioBuffer::new(16);
        let mut w = buf.take_writer();
        let r = buf.reader(); // tail=0
        // Push 13 samples in 4-sample batches (= max_push_len) plus
        // one trailing 1-sample push -> head=13, head - tail = 13
        // > 12 -> Lagged.
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        w.push(&[12.0]);
        let mut out = [0.0; 4];
        match r.peek_into(&mut out) {
            ReadStatus::Lagged { by: 13 } => {}
            other => panic!("expected Lagged{{by:13}}, got {other:?}"),
        }
    }

    #[test]
    fn seek_latest_resets_tail_near_head() {
        let buf = AudioBuffer::new(16); // safety_margin = 4
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        // 13 samples in batches of 4 + a trailing 1 (max_push_len=4).
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        w.push(&[12.0]);
        let mut out = [0.0; 4];
        // Confirm we're Lagged.
        assert!(matches!(r.peek_into(&mut out), ReadStatus::Lagged { .. }));
        // Resync -- request the freshest 4 samples.
        r.seek_latest(4);
        assert_eq!(r.tail(), 9); // head=13, behind=4 -> tail=9
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [9.0, 10.0, 11.0, 12.0]);
    }

    /// Push enough samples to wrap the ring multiple times; verify the
    /// reader sees the right samples on either side of the wrap.
    /// Pushes are split into `max_push_len`-sized chunks
    /// (= safety_margin = 4 at cap=16) so the seqlock-bound
    /// assert is not the thing under test here -- we want pure
    /// wrap-around correctness across the bitwise-mod boundary.
    #[test]
    fn wrap_around_correctness() {
        let cap = 16;
        let buf = AudioBuffer::new(cap); // safety_margin = 4
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        // Push 8 samples in two 4-sample batches.  tail=0, head=8.
        let first: Vec<f32> = (0..8).map(|i| i as f32).collect();
        for chunk in first.chunks(4) {
            w.push(chunk);
        }
        let mut out = [0.0; 8];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out.to_vec(), first);
        r.advance(8); // tail=8

        // Push 10 samples in three batches (4 + 4 + 2) -- wraps the
        // ring (head goes 8 -> 18, capacity 16, so cells 0..2 get
        // overwritten).  tail=8 is still in valid range
        // [head-cap, head) = [2, 18).
        let second: Vec<f32> = (100..110).map(|i| i as f32).collect();
        for chunk in second.chunks(4) {
            w.push(chunk);
        }
        // Now head=18, tail=8, head-tail=10, capacity_minus_margin=12.  OK.
        let mut out2 = [0.0; 10];
        assert_eq!(r.peek_into(&mut out2), ReadStatus::Ready);
        assert_eq!(
            out2.to_vec(),
            second,
            "reader should see exactly the second batch starting at tail=8",
        );
    }

    /// Two readers progress at independent rates against the same writer.
    #[test]
    fn multi_reader_independent_tails() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let mut r1 = buf.reader();
        let r2 = buf.reader();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        let mut o1 = [0.0; 2];
        let mut o2 = [0.0; 4];
        assert_eq!(r1.peek_into(&mut o1), ReadStatus::Ready);
        assert_eq!(o1, [1.0, 2.0]);
        r1.advance(2);
        assert_eq!(r2.peek_into(&mut o2), ReadStatus::Ready);
        assert_eq!(o2, [1.0, 2.0, 3.0, 4.0]); // r2 still at tail=0
        let mut o1b = [0.0; 2];
        assert_eq!(r1.peek_into(&mut o1b), ReadStatus::Ready);
        assert_eq!(o1b, [3.0, 4.0]); // r1 now at tail=2
    }

    /// Concurrent writer + reader on separate threads.  Verifies the
    /// **memory-ordering** edge -- every sample written by the writer
    /// shows up at the reader with the right value (would fail under a
    /// torn read or wrong Acquire/Release).
    ///
    /// **Why no Lag-recovery in this test**: scheduler fairness varies
    /// wildly across hosts (especially `release` mode where the writer
    /// is faster).  To keep the test deterministic we size capacity so
    /// the writer's entire output fits without lap -- the test then
    /// purely exercises ordering, not scheduling.  Wrap-around is
    /// covered by [`wrap_around_correctness`] separately.  Lag detection
    /// is covered by [`peek_lagged_when_writer_far_ahead`].
    #[test]
    fn concurrent_writer_reader_ordering() {
        const TOTAL: u64 = 50_000;
        const CHUNK: usize = 256;
        // capacity > TOTAL guarantees `head - tail < cap` throughout
        // the test -> never Lagged, regardless of scheduling. 131072
        // (= 2^17) is the smallest pow2 >= TOTAL x 2 = 100 000.
        let buf = AudioBuffer::new(131_072);
        let mut r = buf.reader_at(0); // tail=0 right now

        let buf_w = buf.clone();
        let writer_thread = thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut next: u64 = 0;
            while next < TOTAL {
                let upto = (next + CHUNK as u64).min(TOTAL);
                let batch: Vec<f32> = (next..upto).map(|i| i as f32).collect();
                w.push(&batch);
                next = upto;
            }
        });

        let reader_thread = thread::spawn(move || {
            let mut got = 0u64;
            let mut chunk = vec![0f32; CHUNK];
            while got < TOTAL {
                match r.peek_into(&mut chunk) {
                    ReadStatus::Ready => {
                        for (i, &v) in chunk.iter().enumerate() {
                            assert_eq!(
                                v,
                                (got + i as u64) as f32,
                                "drift at sample {} (got={got})",
                                got + i as u64,
                            );
                        }
                        r.advance(CHUNK);
                        got += CHUNK as u64;
                    }
                    ReadStatus::Wait => {
                        let remaining = TOTAL - got;
                        if remaining < CHUNK as u64 && remaining > 0 {
                            let mut tail_chunk = vec![0f32; remaining as usize];
                            if r.peek_into(&mut tail_chunk) == ReadStatus::Ready {
                                for (i, &v) in tail_chunk.iter().enumerate() {
                                    assert_eq!(v, (got + i as u64) as f32);
                                }
                                r.advance(remaining as usize);
                                got += remaining;
                                break;
                            }
                        }
                        thread::sleep(Duration::from_micros(10));
                    }
                    ReadStatus::Lagged { by } => {
                        panic!(
                            "capacity > TOTAL guarantees no Lag -- got by={by}, got={got}; \
                             this is a code bug, not a flake",
                        );
                    }
                }
            }
            got
        });

        writer_thread.join().expect("writer thread");
        let got = reader_thread.join().expect("reader thread");
        assert_eq!(got, TOTAL);
    }

    /// Concurrent test that **forces** wrap-around and recovers from
    /// transient Lag via `seek_latest`.  Verifies (a) wrap-around in
    /// the multi-threaded path doesn't corrupt data, (b) `seek_latest`
    /// correctly resyncs after Lag.  Asserts *every sample observed*
    /// matches the index-as-f32 pattern; doesn't assert "no samples
    /// dropped" because Lag is allowed by design.
    #[test]
    fn concurrent_wrap_around_with_lag_recovery() {
        const TOTAL: u64 = 200_000;
        const CHUNK: usize = 256;
        // Small capacity -> forces dozens of wraps + plenty of Lag.
        let buf = AudioBuffer::new(2_048);
        let mut r = buf.reader_at(0);
        let writer_done = Arc::new(AtomicBool::new(false));

        let buf_w = buf.clone();
        let writer_done_w = writer_done.clone();
        let writer_thread = thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut next: u64 = 0;
            while next < TOTAL {
                let upto = (next + CHUNK as u64).min(TOTAL);
                let batch: Vec<f32> = (next..upto).map(|i| i as f32).collect();
                w.push(&batch);
                next = upto;
            }
            writer_done_w.store(true, Ordering::Release);
        });

        let writer_done_r = writer_done.clone();
        let reader_thread = thread::spawn(move || {
            let mut chunk = vec![0f32; CHUNK];
            let mut samples_in_pattern = 0u64;
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if std::time::Instant::now() > deadline {
                    panic!("timeout: tail={}", r.tail());
                }
                match r.peek_into(&mut chunk) {
                    ReadStatus::Ready => {
                        let base = r.tail();
                        for (i, &v) in chunk.iter().enumerate() {
                            assert_eq!(
                                v,
                                (base + i as u64) as f32,
                                "pattern drift at sample {}",
                                base + i as u64,
                            );
                        }
                        r.advance(CHUNK);
                        samples_in_pattern += CHUNK as u64;
                    }
                    ReadStatus::Wait => {
                        // Writer is done AND no fresh samples -> exit.
                        if writer_done_r.load(Ordering::Acquire) {
                            break;
                        }
                        thread::sleep(Duration::from_micros(20));
                    }
                    ReadStatus::Lagged { by: _ } => {
                        // Resync to live edge; the samples we skip
                        // simply weren't observed (acceptable).
                        r.seek_latest(CHUNK);
                    }
                }
            }
            samples_in_pattern
        });

        writer_thread.join().expect("writer thread");
        let observed = reader_thread.join().expect("reader thread");
        // We don't assert observed == TOTAL -- Lag is expected.  We do
        // assert we observed at least *some* samples (sanity); the
        // per-sample pattern check above guarantees correctness of
        // whatever we did observe.
        assert!(
            observed > 0,
            "reader observed nothing -- pattern check never ran",
        );
    }

    #[test]
    fn safe_peek_window_matches_capacity_three_quarters() {
        let buf = AudioBuffer::new(128);
        // capacity_minus_margin = 128 - 128/4 = 96
        assert_eq!(buf.safe_peek_window(), 96);
    }

    /// `peek_into` with empty `out` slice returns Ready trivially.
    #[test]
    fn peek_into_empty_buffer_is_ready_no_op() {
        let buf = AudioBuffer::new(16);
        let r = buf.reader();
        let mut out: [f32; 0] = [];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
    }

    /// Compile-time check that handle types have the expected
    /// auto-traits.  If any of these fail to compile, the API contract
    /// has been broken.  (Errors will surface at `cargo build` rather
    /// than runtime.)
    #[test]
    fn auto_trait_assertions() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn assert_send_static<T: Send + 'static>() {}

        assert_send::<AudioBuffer>();
        assert_sync::<AudioBuffer>();
        assert_send_static::<AudioBuffer>();
        assert_send_static::<Writer>();
        assert_send_static::<Reader>();
        // Negative checks via inability to compile; not runtime-testable
        // without unstable.  Documented invariants in the type-level
        // marker fields.
    }

    /// Stress test: a single writer and TWO readers running in
    /// parallel, both at slow cadences relative to the writer so
    /// they're force-lagged repeatedly.  Both must
    /// (a) eventually advance their tails (proves recovery
    ///     re-arms the ready path), and
    /// (b) experience at least one `Lagged` event (proves the
    ///     test actually exercised the recovery path -- without
    ///     this assertion a faster machine could pass trivially).
    /// Combined with no torn reads (samples are written as a
    /// monotone f32 sequence; readers verify any Ready window
    /// is ascending), this pins the seqlock-based reader's
    /// behavior under concurrent lag -- the contract that
    /// inference and opus_stream both depend on simultaneously.
    #[test]
    fn concurrent_two_readers_resync_under_writer_pressure() {
        // Capacity tight enough that both readers will lag if
        // they sleep (safe_peek_window = 768).
        let buf = AudioBuffer::new(1024);
        // Pre-create readers BEFORE the writer thread starts so
        // their tails are pinned at 0.  If we constructed them
        // inside the reader threads (`buf.reader()` then), a fast
        // machine could let the writer finish all 200 K pushes
        // before either reader thread schedules in -- both readers
        // would then start at tail = head = 200 000, never observe
        // any new samples, and the `lags > 0` assertion below would
        // spuriously fail.  tail = 0 with a 1024-sample ring +
        // 200 K-sample writer guarantees the contract is exercised.
        let r1 = buf.reader_at(0);
        let r2 = buf.reader_at(0);
        let buf_w = buf.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_r1 = stop.clone();
        let stop_r2 = stop.clone();

        // Writer pushes 200_000 monotonically increasing samples
        // in 200-sample bursts, with no sleep -- outpaces both
        // readers.  Total: ~200 KiB written.
        let writer = std::thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut counter: f32 = 0.0;
            for _ in 0..1_000 {
                let chunk: Vec<f32> = (0..200).map(|i| counter + i as f32).collect();
                counter += 200.0;
                w.push(&chunk);
            }
        });

        // Reader 1: 64-sample window, advance 32 per Ready, sleep
        // 1 ms between successful peeks (slow consumer).
        let r1_handle = std::thread::spawn(move || {
            let mut r = r1;
            let mut window = [0.0_f32; 64];
            let mut total_advanced = 0_u64;
            let mut lag_count = 0_u64;
            let mut max_seen: f32 = -1.0;
            while !stop_r1.load(Ordering::Relaxed) {
                match r.peek_into(&mut window) {
                    ReadStatus::Ready => {
                        // Within a Ready window, samples are a
                        // contiguous slice of the monotone sequence
                        // -> strictly ascending.
                        for w_pair in window.windows(2) {
                            assert!(
                                w_pair[1] > w_pair[0],
                                "torn read: {:?} not ascending",
                                w_pair
                            );
                        }
                        // Across Ready windows max-seen must not
                        // *decrease* -- only seek_latest jumps it
                        // forward.  (Strict monotone across windows
                        // would require no advance gaps, which
                        // isn't true here.)
                        if window[0] >= max_seen {
                            max_seen = window[63];
                        }
                        r.advance(32);
                        total_advanced += 32;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    ReadStatus::Wait => std::thread::sleep(Duration::from_micros(50)),
                    ReadStatus::Lagged { .. } => {
                        r.seek_latest(64);
                        lag_count += 1;
                    }
                }
            }
            (total_advanced, lag_count, max_seen)
        });

        // Reader 2: 32-sample window, advance 16 per Ready, no
        // sleep but at a stricter peek rate.  Both readers
        // independently observe lag.
        let r2_handle = std::thread::spawn(move || {
            let mut r = r2;
            let mut window = [0.0_f32; 32];
            let mut total_advanced = 0_u64;
            let mut lag_count = 0_u64;
            let mut max_seen: f32 = -1.0;
            while !stop_r2.load(Ordering::Relaxed) {
                match r.peek_into(&mut window) {
                    ReadStatus::Ready => {
                        for w_pair in window.windows(2) {
                            assert!(w_pair[1] > w_pair[0], "torn read on reader 2: {w_pair:?}");
                        }
                        if window[0] >= max_seen {
                            max_seen = window[31];
                        }
                        r.advance(16);
                        total_advanced += 16;
                    }
                    ReadStatus::Wait => std::thread::sleep(Duration::from_micros(20)),
                    ReadStatus::Lagged { .. } => {
                        r.seek_latest(32);
                        lag_count += 1;
                    }
                }
            }
            (total_advanced, lag_count, max_seen)
        });

        writer.join().expect("writer panicked");
        // Give readers a small grace to drain the tail end.
        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        let (r1_adv, r1_lags, r1_max) = r1_handle.join().expect("r1 panicked");
        let (r2_adv, r2_lags, r2_max) = r2_handle.join().expect("r2 panicked");

        assert!(r1_adv > 0, "reader 1 never advanced");
        assert!(r2_adv > 0, "reader 2 never advanced");
        assert!(
            r1_lags > 0 || r2_lags > 0,
            "neither reader observed Lagged -- test didn't exercise the contract"
        );
        // Final values seen must be in the monotone-write range.
        // Writer pushed up to ~200_000 inclusive; readers can be at
        // most that large.
        assert!(r1_max < 200_000.0, "r1 saw out-of-range value {r1_max}");
        assert!(r2_max < 200_000.0, "r2 saw out-of-range value {r2_max}");
    }

    /// [`Reader::advance`] past head panics in every build.
    /// A `debug_assert!` would silently cap via
    /// `saturating_add` in release and stall the reader
    /// forever in `Wait`.
    #[test]
    fn reader_advance_past_head_panics() {
        let buf = AudioBuffer::new(64);
        let mut writer = buf.take_writer();
        // head = 4 after one push.
        writer.push(&[0.0, 0.1, 0.2, 0.3]);
        let mut r = buf.reader_at(0);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // 100 >> 4 -- must panic, not saturate.
            r.advance(100);
        }));
        assert!(
            result.is_err(),
            "advance past head must panic; got Ok which means saturating_add silently capped",
        );
    }

    /// [`Reader::peek_into`] with `out.len() >
    /// safe_peek_window` panics in every build.  A
    /// `debug_assert!` would silently stall in `Wait`
    /// because `avail < n_u64` is unreachable when `n`
    /// exceeds the buffer capacity.
    #[test]
    fn reader_peek_into_oversize_panics() {
        let buf = AudioBuffer::new(64);
        let _writer = buf.take_writer();
        let r = buf.reader_at(0);
        // safe_peek_window for cap=64 is 48 (= cap - cap/4); 100
        // is well past it.
        let mut out = vec![0.0f32; 100];
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.peek_into(&mut out)));
        assert!(
            result.is_err(),
            "peek_into past safe_peek_window must panic; got Ok which means \
             release silently accepted the oversized peek",
        );
    }
}
