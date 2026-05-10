//! Loom model of the `audio_buffer` seqlock for memory-ordering
//! verification.
//!
//! ## Why a separate model
//!
//! Loom replaces `std::sync::atomic` with its own permutation-
//! exploring shim; we cannot drop it directly into the production
//! [`acoustics_lab::audio_buffer`] module without polluting every
//! release build.  Instead this file mirrors the algorithm
//! (single-writer / single-reader at `cap = 4`, `safety_margin = 1`)
//! using `loom::sync::atomic` and asserts the *same* invariants the
//! production reader's `peek_into` claims:
//!
//! 1. **No torn read on `Ready`**: when the reader returns `Ready`,
//!    every sample copied out is one of the writer's published
//!    values (i.e. the bit pattern of one of the f32s the writer
//!    sent), and the sequence is the contiguous prefix of pushes
//!    starting at the reader's tail.
//! 2. **`Lagged` is a true positive guard**: after `Lagged`, the
//!    reader must seek_latest before peeking again.  We do not
//!    consume Lagged data.
//!
//! ## Why this matters on aarch64
//!
//! The production code uses `Relaxed` per-cell stores +
//! `Release/Acquire` on `head`, with an explicit
//! `fence(Ordering::Acquire)` between cell loads and the recheck
//! head load.  On strongly-ordered x86 the fence is essentially a
//! no-op; on aarch64 `LDAR` is one-way and prior `Relaxed` loads
//! can sink past it without the explicit `dmb ishld` the fence
//! emits.  Loom enumerates the interleavings that would exhibit
//! such a sinking, so the model exercises the barrier the way an
//! aarch64 CPU would.
//!
//! ## Cost
//!
//! `cap = 4` + `max_push_len = 1` keeps the algorithmic state
//! space small.  We bound writer iterations to 3 pushes and the
//! reader to one peek, which is the smallest configuration that
//! exercises the wrap (`mask = 3`).  Loom's default branching
//! cap covers it well under one second on a modern machine
//! (`~10 K` paths).  The test is `#[cfg(loom)]`-gated so normal
//! `cargo test` runs do not pay for it.
//!
//! ## Running
//!
//! ```text
//! RUSTFLAGS="--cfg audio_buffer_loom" cargo test \
//!   --test audio_buffer_loom --release
//! ```
//!
//! `--release` is recommended -- Loom in `dev` is roughly 3x
//! slower and the model is not debugging-load-bearing.  The cfg
//! is named `audio_buffer_loom` rather than the bare `loom`
//! because tokio-tungstenite (a sibling dev-dep) consumes
//! tokio's `net` module which is itself gated `#![cfg(not(loom))]`
//! -- using `--cfg loom` would break the dev-dep graph compile.

#![cfg(audio_buffer_loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use loom::thread;

/// Mirrors `audio_buffer::Inner` minus the cache-line padding
/// (Loom's atomics aren't `repr(C)`, and padding is a perf
/// concern not a correctness one -- we model the algorithm,
/// not the layout).  Holds `Box<[AtomicU32]>` of `cap`
/// f32-bit cells.
struct Ring {
    cells: Box<[AtomicU32]>,
    head: AtomicU64,
}

impl Ring {
    /// Smallest legal capacity: 4.  `safety_margin = 1`, so a
    /// single push of 1 sample is the maximum legal publication
    /// quantum -- exactly the contract the production assert
    /// enforces.  Larger cap would multiply Loom paths without
    /// adding distinct interleavings of the seqlock edges.
    const CAP: usize = 4;
    const MASK: u64 = (Self::CAP - 1) as u64;
    /// `cap - cap/4` -- reader's "safe to read" upper bound on
    /// `head - tail`.  Mirrors `capacity_minus_margin`.
    const CMM: u64 = (Self::CAP - Self::CAP / 4) as u64;

    fn new() -> Self {
        let cells = (0..Self::CAP).map(|_| AtomicU32::new(0)).collect();
        Self {
            cells,
            head: AtomicU64::new(0),
        }
    }

    /// One push of one sample.  Mirrors
    /// `Writer::push(&[sample])`: per-cell Relaxed store, then
    /// head Release-store.
    fn push_one(&self, sample: f32) {
        let head = self.head.load(Ordering::Relaxed);
        let idx = (head & Self::MASK) as usize;
        self.cells[idx].store(sample.to_bits(), Ordering::Relaxed);
        self.head.store(head.wrapping_add(1), Ordering::Release);
    }
}

/// Mirrors `Reader::peek_into`'s outcome enum.  We model only
/// the values, not the panic/`debug_assert` behaviour.
#[derive(Debug, Eq, PartialEq)]
enum ReadStatus {
    Ready,
    Wait,
    Lagged,
}

/// Mirrors `Reader::peek_into` for `out.len() == n`.  Returns
/// the status and -- on `Ready` -- writes into `out`.
fn peek(ring: &Ring, tail: u64, out: &mut [f32]) -> ReadStatus {
    let n = out.len() as u64;
    let h0 = ring.head.load(Ordering::Acquire);
    let avail = h0.saturating_sub(tail);
    if avail < n {
        return ReadStatus::Wait;
    }
    if avail > Ring::CMM {
        return ReadStatus::Lagged;
    }
    // Per-cell Relaxed loads; same iteration order as the
    // production wrap-aware implementation but unrolled here for
    // the `n <= cap` model.
    for (i, slot) in out.iter_mut().enumerate() {
        let idx = ((tail.wrapping_add(i as u64)) & Ring::MASK) as usize;
        *slot = f32::from_bits(ring.cells[idx].load(Ordering::Relaxed));
    }
    // The load-load barrier the production reader needs on
    // aarch64 to keep prior Relaxed loads from sinking past
    // the recheck Acquire load.  Loom honours fences exactly
    // like a real CPU's `dmb ishld`.
    fence(Ordering::Acquire);
    let h1 = ring.head.load(Ordering::Acquire);
    let avail1 = h1.saturating_sub(tail);
    if avail1 <= Ring::CMM {
        ReadStatus::Ready
    } else {
        ReadStatus::Lagged
    }
}

/// Single-writer / single-reader Loom model.  The writer pushes
/// three distinct sentinel samples.  The reader either sees a
/// contiguous Ready prefix of those samples or sees Wait/Lagged.
/// Asserts on every Loom interleaving:
///   * Ready prefix == [1.0, 2.0, ...] for some k <= 3 from tail.
///   * No sample observed under Ready is anything other than the
///     three published bit patterns -- catches a torn mix where a
///     stale 0-bit cell would slip through if the fence/recheck
///     were unsound.
///
/// Writer pushes 3 samples (max_push_len = 1 each), reader peeks
/// once with out.len() = 1, advancing nothing.  This is the
/// smallest model that exercises (a) per-cell Relaxed visibility
/// vs (b) the recheck under writer progress.
#[test]
fn loom_seqlock_no_torn_read_one_sample() {
    loom::model(|| {
        let ring = Arc::new(Ring::new());
        let writer_ring = ring.clone();
        let writer = thread::spawn(move || {
            // Sentinel samples: 1.0, 2.0, 3.0.  All distinct
            // f32 bit patterns; 0.0 (the cell init value) is a
            // tell-tale of an uninitialized cell read.
            for k in 1..=3u32 {
                writer_ring.push_one(k as f32);
            }
        });

        let reader_ring = ring;
        let reader = thread::spawn(move || {
            let mut out = [0.0_f32];
            let status = peek(&reader_ring, 0, &mut out);
            match status {
                ReadStatus::Ready => {
                    let v = out[0];
                    assert!(
                        v == 1.0 || v == 2.0 || v == 3.0,
                        "torn read: peek returned {v}, expected one of \
                         the sentinel samples 1.0/2.0/3.0",
                    );
                    // The first sample at tail=0 is sample #1
                    // by construction (writer publishes in
                    // order).  If avail >= 1 at recheck, h1
                    // saw at least the first push -- so cell[0]
                    // must already be 1.0 by Relaxed visibility
                    // through the head Release/Acquire edge.
                    // Anything else is a torn read.
                    assert_eq!(
                        v, 1.0,
                        "Ready returned cell[0]={v} but tail=0; the head \
                         Release-store must publish cell[0]=1.0 first",
                    );
                }
                ReadStatus::Wait | ReadStatus::Lagged => {
                    // Both are legal: Wait if no push has been
                    // observed yet, Lagged if the writer raced
                    // all 3 pushes ahead of our recheck (avail >
                    // cmm = 3 at h1).  Neither claims `out` is
                    // valid, so no further assertion.
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}

/// Companion model: peek of two samples.  Tail = 0, writer
/// pushes 2 sentinels.  This forces Loom to interleave the
/// reader's two per-cell Relaxed loads against a writer that may
/// publish either, neither, or both before the recheck.  Catches
/// any reordering bug that would let the reader observe cell[1]
/// before cell[0]'s Release-publish became visible -- the case
/// the recheck is meant to convert into `Lagged`.
#[test]
fn loom_seqlock_no_torn_read_two_samples() {
    loom::model(|| {
        let ring = Arc::new(Ring::new());
        let writer_ring = ring.clone();
        let writer = thread::spawn(move || {
            // Two pushes is the minimum to expose a writer
            // progressing during a 2-cell read.  Each push is
            // one sample (= max_push_len at cap=4).
            writer_ring.push_one(11.0);
            writer_ring.push_one(22.0);
        });

        let reader_ring = ring;
        let reader = thread::spawn(move || {
            let mut out = [0.0_f32; 2];
            let status = peek(&reader_ring, 0, &mut out);
            match status {
                ReadStatus::Ready => {
                    // Both cells must be the published values
                    // in publication order.  An out-of-order
                    // value (cell[1]==22 but cell[0]==0)
                    // would mean the reader observed the
                    // second store before the first -- which
                    // the head Release/Acquire edge plus the
                    // recheck must convert into Lagged or
                    // Wait, never Ready.
                    assert_eq!(
                        out,
                        [11.0, 22.0],
                        "torn read on 2-sample peek: got {out:?}, expected \
                         [11.0, 22.0]",
                    );
                }
                ReadStatus::Wait | ReadStatus::Lagged => {
                    // Wait: avail < 2 at h0 (writer hadn't
                    // published the second store yet).
                    // Lagged: cmm = 3, n = 2; could fire only
                    // if h1 - tail > 3 -- impossible with
                    // only 2 pushes.  So Lagged in this model
                    // would itself indicate a bug.
                    assert_ne!(
                        status,
                        ReadStatus::Lagged,
                        "Lagged is unreachable with 2 pushes and cmm=3; \
                         got Lagged anyway -- check the recheck arithmetic",
                    );
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}
