//! Hot-state versioning primitives.
//!
//! - [`ResourceVersion`] -- monotonically-increasing counter
//!   stamped on each mutation.
//! - [`SwapReceipt`] -- returned by mutation methods, carries
//!   the post-mutation version.
//! - [`VersionedSwap<T>`] -- concrete generic backing a
//!   wait-free-read / writer-mutex-serialised hot-swap of an
//!   `Arc<T>`.  Backs every domain-side `try_swap` /
//!   `try_mutate` impl (`inference::HotHead`,
//!   `config::MicSettingsCell`, ...).
//!
//! # Why `VersionedSwap<T>` is concrete, not a trait
//!
//! A trait-shaped surface for `try_mutate<R>(&self, f: impl
//! FnOnce(&T) -> Result<(T, R), _>)` is **not object-safe** --
//! the `<R>` generic and the `impl FnOnce` both forbid
//! `Arc<dyn HotState<T>>`.  Concrete hot-swap surfaces
//! (`HeadStore::try_swap`, `MicSettingsHandle::try_mutate`)
//! live per-resource with object-safe signatures (e.g.
//! `fn try_swap(&self, candidate: HeadCandidate) ->
//! Result<SwapReceipt, _>`); each delegates to a private
//! [`VersionedSwap::try_mutate`] call.

/// Monotonically-increasing version of a hot-swappable
/// resource.
///
/// Each successful mutation of a [`VersionedSwap<T>`] bumps
/// the underlying counter and stamps the new version onto the
/// returned [`SwapReceipt`].  Read-your-write HTTP semantics
/// compare an operator-supplied `?min_version=N` against the
/// current resource version.
///
/// `u64` is sized for sustained 1 us-per-mutation for ~580k
/// years; overflow is a non-concern.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct ResourceVersion(u64);

impl ResourceVersion {
    /// Initial / sentinel version.  Used as the seed for a
    /// fresh [`VersionedSwap<T>`]; the first successful
    /// mutation produces version `1`.
    pub const ZERO: Self = Self(0);

    /// Construct from a raw u64.
    #[inline]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    /// Extract the inner u64.  Used by the serde layer (DTOs
    /// expose a `version: u64` field) and by HTTP query-param
    /// parsing (`?min_version=N`).
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Return the next version after `self`.  Saturates at
    /// [`u64::MAX`].
    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl Default for ResourceVersion {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for ResourceVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Returned by every successful mutation of a hot-swappable
/// resource (`HeadStore::try_swap`,
/// `MicSettingsHandle::try_mutate`,
/// [`VersionedSwap::try_mutate`]).  Carries the post-mutation
/// version so the API layer can include it in the response
/// body for read-your-write semantics.
///
/// `#[must_use]` because dropping the receipt throws away the
/// only handle on "the version that just landed": no caller
/// can issue a `?min_version=N` round trip against it.  Bind
/// explicitly (or `let _ =`) when the version is genuinely
/// not needed.
#[must_use = "the SwapReceipt carries the post-mutation version; drop only when no caller needs read-your-write semantics"]
#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SwapReceipt {
    /// The version of the resource after the mutation.
    /// Reading the resource with `?min_version=version`
    /// guarantees the returned value reflects this mutation
    /// or a later one.
    pub version: ResourceVersion,
}

// MARK: VersionedSwap

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// One snapshot of the live value plus the version that
/// produced it.  Wrapped in `Arc` so readers (`snapshot`) get
/// a refcount-bump load rather than copying the value out.
#[derive(Debug)]
struct Versioned<T> {
    version: ResourceVersion,
    value: Arc<T>,
}

/// Wait-free atomic hot-swap of an `Arc<T>`, with a monotonic
/// [`ResourceVersion`] stamped on every successful mutation.
///
/// Reads ([`Self::snapshot`], [`Self::version`]) are wait-free
/// `ArcSwap::load` calls (single atomic load + refcount bump,
/// ~5 ns).  Writes ([`Self::try_mutate`]) serialise through an
/// internal `parking_lot::Mutex<()>` so the read-modify-write
/// cycle is linearisable; concurrent writers queue, never lose
/// updates.
///
/// The writer mutex covers the in-memory swap **only** --
/// callers that also need to persist the new value (the
/// daemon's `MicSettingsCell` writes a TOML file after the
/// swap) MUST do the persistence outside `try_mutate`'s
/// closure.  The closure runs under the lock; an fsync inside
/// it would block every subsequent writer for the duration of
/// the disk I/O.  Persistence after lock-release is safe: the
/// new value is already published via `ArcSwap`, so a
/// concurrent reader sees the post-mutation state immediately,
/// while the writer's persistence happens at its own cadence.
///
/// `T: Send + Sync + 'static` because both the inner
/// `Arc<T>` and the wrapping `Versioned<T>` carrier (private
/// to this module; it pairs the `Arc<T>` with its
/// `ResourceVersion`) flow through `ArcSwap`, which requires
/// its payload to satisfy those bounds.
#[derive(Debug)]
pub struct VersionedSwap<T> {
    inner: ArcSwap<Versioned<T>>,
    /// Serialises writers.  The `()` payload is a marker; the
    /// lock itself is what matters, and `try_mutate`'s
    /// critical section runs under it.
    writer: Mutex<()>,
    /// `fetch_add` returns the prior value; the assigned
    /// post-mutation version is `prior + 1`.
    counter: AtomicU64,
}

impl<T: Send + Sync + 'static> VersionedSwap<T> {
    /// Construct with `initial` as the v0 value.  The version
    /// counter starts at `0` (the seed); the first successful
    /// mutation stamps version `1` per
    /// [`ResourceVersion::next`].
    pub fn new(initial: T) -> Self {
        Self {
            inner: ArcSwap::from_pointee(Versioned {
                version: ResourceVersion::ZERO,
                value: Arc::new(initial),
            }),
            writer: Mutex::new(()),
            counter: AtomicU64::new(0),
        }
    }

    /// Wait-free read of the current value.  Returns an
    /// `Arc<T>` aliasing the live snapshot; safe to hold
    /// across further mutations -- the old `Arc<T>` stays
    /// alive until its last refcount drops, even after a
    /// swap.
    pub fn snapshot(&self) -> Arc<T> {
        self.inner.load_full().value.clone()
    }

    /// Wait-free read of the current version.
    pub fn version(&self) -> ResourceVersion {
        self.inner.load().version
    }

    /// Read both the current value and its version atomically.
    /// Equivalent to `(snapshot(), version())` but avoids the
    /// race where a mutation slips between the two reads --
    /// useful for `?min_version=N` GET handlers that need the
    /// version that goes with the value they return.
    pub fn snapshot_with_version(&self) -> (Arc<T>, ResourceVersion) {
        let g = self.inner.load_full();
        (g.value.clone(), g.version)
    }

    /// Run `f` against the current value (under the writer
    /// mutex) and atomically install the result.  Returns the
    /// new [`SwapReceipt`] plus the closure's `R` payload.
    ///
    /// `f` receives `&T` (no clone).  It returns either:
    ///
    /// - `Ok((new_value, ret))` -- install `new_value` at the
    ///   next version; return `(SwapReceipt, ret)`.
    /// - `Err(e)` -- bail; the in-memory value stays at the
    ///   pre-call version; no version bump.
    ///
    /// The `Arc<T>` in the Ok branch is the caller's choice
    /// of allocation strategy: `Arc::new(...)` for "fresh
    /// value", or `current.clone()` if the closure realised
    /// no change is needed but still wants to bump the
    /// version (rare; caller's choice).
    pub fn try_mutate<R, E>(
        &self,
        f: impl FnOnce(&T) -> Result<(Arc<T>, R), E>,
    ) -> Result<(SwapReceipt, R), E> {
        let _g = self.writer.lock();
        let cur = self.inner.load_full();
        let (new_value, ret) = f(&cur.value)?;
        // `fetch_add` returns the prior value; the assigned
        // version is `prior + 1`.  AcqRel because the swap
        // below synchronises-with subsequent loads via
        // ArcSwap.
        let v = ResourceVersion::new(self.counter.fetch_add(1, Ordering::AcqRel) + 1);
        self.inner.store(Arc::new(Versioned {
            version: v,
            value: new_value,
        }));
        Ok((SwapReceipt { version: v }, ret))
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_version_zero_is_canonical_seed() {
        assert_eq!(ResourceVersion::ZERO.get(), 0);
        assert_eq!(ResourceVersion::default(), ResourceVersion::ZERO);
    }

    #[test]
    fn resource_version_next_monotonic() {
        let v0 = ResourceVersion::ZERO;
        let v1 = v0.next();
        let v2 = v1.next();
        assert_eq!(v0.get(), 0);
        assert_eq!(v1.get(), 1);
        assert_eq!(v2.get(), 2);
        assert!(v0 < v1);
        assert!(v1 < v2);
    }

    #[test]
    fn resource_version_next_saturates_at_u64_max() {
        let max = ResourceVersion::new(u64::MAX);
        assert_eq!(max.next(), max);
    }

    #[test]
    fn swap_receipt_carries_version() {
        let r = SwapReceipt {
            version: ResourceVersion::new(42),
        };
        assert_eq!(r.version.get(), 42);
    }

    #[test]
    fn display_writes_inner() {
        assert_eq!(format!("{}", ResourceVersion::new(7)), "7");
    }

    #[test]
    fn versioned_swap_initial_state() {
        let s: VersionedSwap<u32> = VersionedSwap::new(42);
        assert_eq!(s.version(), ResourceVersion::ZERO);
        assert_eq!(*s.snapshot(), 42);
        let (val, v) = s.snapshot_with_version();
        assert_eq!(*val, 42);
        assert_eq!(v, ResourceVersion::ZERO);
    }

    /// `try_mutate` bumps the version, installs the new
    /// value atomically, and returns the receipt + the
    /// closure's `R` payload.
    #[test]
    fn versioned_swap_try_mutate_bumps_version() {
        let s: VersionedSwap<u32> = VersionedSwap::new(10);
        let (receipt, ret) = s
            .try_mutate(|cur| Ok::<_, ()>((Arc::new(*cur + 1), "ok")))
            .expect("mutate");
        assert_eq!(receipt.version.get(), 1);
        assert_eq!(ret, "ok");
        assert_eq!(*s.snapshot(), 11);
        assert_eq!(s.version().get(), 1);
    }

    /// `try_mutate`'s closure can bail; on `Err` the in-memory
    /// value stays at the pre-call version (no version bump on
    /// failure).
    #[test]
    fn versioned_swap_try_mutate_err_does_not_bump() {
        let s: VersionedSwap<u32> = VersionedSwap::new(7);
        let res: Result<(SwapReceipt, ()), &'static str> = s.try_mutate(|_cur| Err("nope"));
        assert!(matches!(res, Err("nope")));
        assert_eq!(*s.snapshot(), 7);
        assert_eq!(s.version(), ResourceVersion::ZERO);
    }

    /// Concurrent writers serialise through the writer mutex;
    /// versions are strictly monotonic and the total count
    /// matches the concurrent task count.
    #[test]
    fn versioned_swap_concurrent_writers_monotonic() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let s: StdArc<VersionedSwap<u32>> = StdArc::new(VersionedSwap::new(0));
        let n_tasks = 100;
        let handles: Vec<_> = (0..n_tasks)
            .map(|_| {
                let s = s.clone();
                thread::spawn(move || {
                    let (receipt, _) = s
                        .try_mutate(|cur| Ok::<_, ()>((Arc::new(*cur + 1), ())))
                        .expect("mutate");
                    receipt.version.get()
                })
            })
            .collect();
        let mut versions: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        versions.sort_unstable();
        // Strict monotonic: every version 1..=n_tasks appears
        // exactly once.
        assert_eq!(versions, (1..=n_tasks as u64).collect::<Vec<_>>());
        assert_eq!(s.version().get(), n_tasks as u64);
        assert_eq!(*s.snapshot(), n_tasks);
    }
}
