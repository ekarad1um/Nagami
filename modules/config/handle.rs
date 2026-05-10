//! Object-safe `ConfigHandle` trait + `ConfigGuard` pattern.
//!
//! Cross-crate consumers (api handlers, test rebinders) hold
//! `Arc<dyn ConfigHandle>`; the concrete impl is
//! [`crate::config::ConfigCell`].
//!
//! ## Atomicity contract
//!
//! Read-modify-write must run the mutator AND the after-hook while
//! holding the mutate lock.  Handlers that update both the on-disk
//! config AND a runtime ArcSwap (e.g. `inference_cfg`, `mic_settings`)
//! need both writes serialised against concurrent mutators; otherwise
//! two callers can interleave and leave runtime state out of sync with
//! disk.  [`ConfigGuard::commit`] preserves this: the after-hook
//! closure runs while still inside the lock-protected critical section.
//! Dropping the guard without calling `commit` rolls back (releases
//! the lock without persisting).
//!
//! ## Why a closure, not a method, on `commit`
//!
//! The after-hook is `Box<dyn FnOnce(&Config) + Send>` rather than
//! a separate `commit_with_after` method, because:
//! - it keeps the trait single-method (smaller vtable, easier to mock);
//! - callers that don't need an after-hook pass a no-op `Box::new(|_| ())`
//!   (one heap allocation per mutate; mutates are REST-cadence not
//!   hot-path, so the allocation is in the noise).

use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::config::error::ConfigError;

/// Object-safe handle on the daemon configuration.  Every consumer
/// outside the `config` crate (api handlers, mic_settings rebinders
/// in test fixtures) holds an `Arc<dyn ConfigHandle>` so the
/// concrete impl is substitutable.
///
/// Production impl: [`crate::config::ConfigCell`] (file-backed,
/// reads/writes TOML, runs a notify watcher).  Tests substitute
/// in-memory mocks satisfying the trait without touching the disk.
///
/// `Send + Sync` are required because the handle is shared across
/// the tokio runtime (api handlers + the watcher worker thread).
pub trait ConfigHandle: Send + Sync {
    /// Read the current snapshot.  Wait-free (`ArcSwap::load_full`
    /// on the file-backed impl); ~5 ns.  Hold for as long as needed.
    fn snapshot(&self) -> Arc<Config>;

    /// Open a write guard.  Acquires the mutate lock; the lock is
    /// held until the returned [`ConfigGuard`] is committed or
    /// rolled back.  Returns [`ConfigError::ReentrantMutate`] if
    /// the calling thread is already inside another guard
    /// (re-entrant locking would deadlock).
    ///
    /// The `'_` lifetime ties the guard to `&self` so the guard
    /// cannot outlive the handle.  The boxed return is required for
    /// object safety.
    fn open_mutation(&self) -> Result<Box<dyn ConfigGuard + '_>, ConfigError>;

    /// Where the handle reads + writes.  Used by diagnostics
    /// (`AppState`'s Debug impl, error messages that want to
    /// surface the offending file path).  In-memory test mocks may
    /// return a synthetic path or `Path::new("(in-memory)")` --
    /// the caller is required to treat the result as opaque.
    fn path(&self) -> &Path;
}

/// RAII write guard returned by [`ConfigHandle::open_mutation`].
/// Holds the mutate lock for its lifetime; consume via
/// [`ConfigGuard::commit`] (validates + persists + atomically
/// swaps the in-memory snapshot, runs the after-hook under the
/// lock) or [`ConfigGuard::rollback`] (releases the lock without
/// persisting; the in-memory snapshot is unchanged).
///
/// Dropping the guard without calling either method is equivalent
/// to `rollback` -- the lock is released and no state changes.  This
/// matches today's `mutate_then` behaviour when the mutator panics
/// before the disk write reaches it.
pub trait ConfigGuard {
    /// Mutable access to the cloned snapshot under the guard.
    /// Multiple field updates can be applied in sequence; nothing
    /// is persisted until [`Self::commit`] is called.
    fn config(&mut self) -> &mut Config;

    /// Validate the mutated config, write a fresh TOML file
    /// atomically (tempfile + rename), then store the new
    /// `Arc<Config>` into the in-memory snapshot.  Finally, invoke
    /// `after` with a borrow of the new snapshot WHILE STILL
    /// HOLDING the mutate lock -- this is the atomicity hook for
    /// callers that need to update a runtime ArcSwap (e.g.
    /// `inference_cfg`, `mic_settings`) consistently with the
    /// on-disk state.
    ///
    /// Use `Box::new(|_| ())` when no after-hook is needed; the
    /// allocation is negligible at REST cadence.
    fn commit(self: Box<Self>, after: Box<dyn FnOnce(&Config) + Send>) -> Result<(), ConfigError>;

    /// Discard the guard without persisting.  Releases the mutate
    /// lock; the in-memory snapshot stays at its prior value.
    /// Equivalent to dropping the guard, but makes intent explicit
    /// at the call site.
    fn rollback(self: Box<Self>);
}
