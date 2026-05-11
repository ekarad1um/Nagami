//! Acoustics Lab configuration: TOML on disk -> `Arc<ArcSwap<Config>>`
//! in memory.
//!
//! ## Source of truth
//!
//! The TOML file at the configured path is the durable source of
//! truth.  The in-memory `ConfigHandle` is a snapshot.  Every mutation
//! goes through `ConfigCell::mutate`, which:
//!
//! 1. Clones the current snapshot,
//! 2. Applies the caller's mutation,
//! 3. Writes a fresh TOML file via tempfile + atomic `persist`,
//! 4. Stores the new `Arc<Config>` into the `ArcSwap`.
//!
//! Step 3 happens BEFORE step 4 so that:
//!   * If the write fails (disk full, permissions), the in-memory
//!     state stays consistent with the on-disk state.
//!   * If the process dies between 3 and 4, the next startup loads
//!     the new config -- the user's change is preserved.
//!
//! ## Hot reload
//!
//! `ConfigCell::watch(path)` spawns a `notify::RecommendedWatcher`
//! thread.  On each fs event matching the config file's `file_name`,
//! we re-parse the file and `ArcSwap::store` the result.  The
//! watcher subscribes to the parent directory (not the file
//! directly) so vim-style "write tmp + rename" edits surface even
//! when the inode is replaced; events for sibling files are
//! filtered out in the notify callback by `file_name()` comparison.
//! Self-writes from `mutate` re-enter `try_reload`, but the value-
//! equality short-circuit (`*prev == cfg`) elides the redundant
//! `ArcSwap::store` so reload_count is not bumped for our own
//! writes.  Comments are NOT round-tripped through TOML -- the TOML
//! write strips them.  This is acceptable for a daemon config
//! (operators edit comments rarely; they edit values often).
//!
//! Reload errors (parse failure, type mismatch) are logged via
//! `tracing::warn!` and the in-memory snapshot is left unchanged.  The
//! operator gets a chance to fix the TOML; the daemon keeps running.
//!
//! ## Type classification
//!
//! `config` mixes schema, runtime projection, and live-store
//! responsibilities (controlled coupling debt for a
//! single-binary monolith).  The split is pragmatic, not
//! extracted into separate sub-crates; the taxonomy below is
//! doc-only so a maintainer can see at a glance which kind
//! of type they are touching.
//!
//! | Type | Kind | Why |
//! |---|---|---|
//! | [`Config`] | Schema DTO | Top-level TOML mapping; `Serialize + Deserialize`; one-to-one with `<workspace>/config.toml`.  Carries only the fields that are hot-reloadable AND API-mutable (mic policy + inference cadence). |
//! | [`LaunchConfig`] | Schema DTO | Boot-time TOML; mic catalogue + backbone catalogue + stream listener settings + default-head file pair + training defaults + file caps. |
//! | [`TrainingDefaults`], [`FileCfg`], [`StreamCfg`] | Schema DTO | Sub-tables in `LaunchConfig`; pure schema, all `Serialize + Deserialize`; startup-only, never hot-reloaded. |
//! | [`crate::audio_io::mic_arbitrator::MicPolicy`] / `MicSelection` / `ChannelSelection` (re-exported through `Config.mic`) | Schema DTO (cross-module) | Owned by `audio_io` but lives in the TOML; cross-validated against `LaunchConfig.mic`. |
//! | [`crate::inference::InferenceCfg`] (re-exported through `Config.inference`) | Schema DTO (cross-module) | Inference cadence; same dual-ownership pattern. |
//! | [`ConfigError`] | Diagnostic type | Internal failure shape mapped to HTTP statuses via `Categorized`. |
//! | [`mic_settings::MicError`] | Diagnostic type | Same shape; mic-policy validation failures. |
//! | [`ConfigHandle`] / [`ConfigGuard`] | Live-store port (trait) | Object-safe trait surface; `Arc<dyn ConfigHandle>` flows into `api::AppState`. |
//! | [`ConfigCell`] | Live store | Concrete impl: `Arc<ArcSwap<Config>>` + mutate-lock + reload counter; daemon constructs one and dyn-coerces. |
//! | [`MicSettingsHandle`] / [`MicSettingsCell`] | Live store (mic specialization) | Wraps `VersionedSwap<MicSettings>` + persistence back-pointer to `ConfigCell`; mic-arbitrator reads via the trait. |
//! | `mic_settings::MicSettings` (read-shape, internal) | Runtime projection | Combines the launch catalogue (immutable) with the live `MicPolicy` (mutable) into one read DTO; the arbitrator hot-loop reads this. |
//! | [`WatcherGuard`] | Lifecycle handle | Drop-driven cleanup of the notify-watcher thread. |
//!
//! Adding a new type:
//!
//! - **Schema DTO**: derive `Serialize + Deserialize`; live in
//!   `domain.rs` or its own sub-module; never hold runtime
//!   handles (Arc / channels / Senders).  Validation belongs to
//!   a `validate(&self) -> Result<(), String>` method on the
//!   DTO; aggregators call it from [`Config::validate`], which
//!   wraps each leaf in [`ConfigValidationError`] for typed
//!   discrimination at the watcher / callback boundary.
//! - **Runtime projection**: lives in the consuming crate, not
//!   in `config`.  Built from a Schema DTO at boot or on
//!   reload.  Holds runtime-only state (resolved paths, derived
//!   caps).  The mic-arbitrator's `MicSettings` is the
//!   reference example.
//! - **Live store**: wraps a Schema DTO in an `ArcSwap` /
//!   `VersionedSwap` / `parking_lot::Mutex`; exposes an
//!   object-safe trait so cross-crate consumers depend on the
//!   trait, not the concrete type.  Mutation goes through a
//!   `ConfigGuard`-style read-modify-commit surface so disk
//!   and memory stay in lockstep.

#![warn(missing_debug_implementations)]

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;

// Thread-local re-entrancy sentinel for `mutate_then`.
// `parking_lot::Mutex` is non-reentrant, so a callback that
// re-entered the same handle's `mutate_then` (typically via the
// `after` hook calling `config.mutate(...)` recursively) would
// deadlock the worker silently.  This `Cell<bool>` tracks whether
// the current thread is *inside* a `mutate_then` body; the entry
// guard returns `Err(ConfigError::ReentrantMutate)` when set
// rather than blocking on the mutex.
//
// Per-thread (not per-handle) is the right granularity: the
// deadlock condition is "this same thread is re-entering the
// same lock on the same handle." A different thread holding
// `mutate_lock` is a normal blocking case and must wait, not
// fail.  The guard's drop unconditionally clears the flag so a
// panic in the mutator + after-hook doesn't poison the slot.
thread_local! {
    static IN_MUTATE: Cell<bool> = const { Cell::new(false) };
}

// Sub-modules carved out of `lib.rs` to bring the file
// under the 1,500-LoC L2 layer-gate.  Public types are re-exported
// here so consumers' existing import paths
// (`config::ConfigError` etc.) continue to work without churn.
mod domain;
mod error;
mod handle;
mod launch;
// `MicSettingsCell` + the `MicSettingsStore` /
// `MicSettingsHandle` traits that wrap a `VersionedSwap<MicSettings>`
// + persistence.  Public; consumers (api, daemon) hold trait-object
// Arcs.
pub mod mic_settings;
mod watcher;

pub use domain::{FileCfg, StreamCfg, TrainingDefaults};
pub use error::{ConfigError, ConfigValidationError};
use error::{parse_err, read_err};
// Object-safe `ConfigHandle` trait + `ConfigGuard`
// pattern.  The trait is the public surface every cross-crate
// consumer holds (`Arc<dyn ConfigHandle>`); the concrete impl is
// `ConfigCell` below.
pub use handle::{ConfigGuard, ConfigHandle};
pub use launch::{
    DefaultHeadRef, HeadLaunchConfig, LaunchConfig, validate_policy_against_catalogue,
};
pub use mic_settings::{MicError, MicSettingsCell, MicSettingsHandle};

use crate::audio_io::mic_arbitrator::{ChannelSelection, MicPolicy, MicSelection};
use crate::inference::InferenceCfg;
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Trait-object form of the user-supplied `watch_with` callback.
/// Used by the debounce worker; aliased so the lengthy type
/// signature isn't repeated at every passing point.
type ReloadCallback = dyn Fn(&Config) -> Result<(), ConfigValidationError> + Send + Sync;

/// Top-level user-preference config.  Loaded once from
/// `<workspace>/config.toml`; mutated through `ConfigCell::mutate`
/// (API routes) and via the hot-reload watcher (operator edits).
///
/// Carries only fields that are BOTH hot-reloadable AND
/// API-mutable.  Boot-time constants (mic catalogue, stream binds,
/// training defaults, file-service caps, etc.) live in
/// [`LaunchConfig`] instead.  The workspace root is operator-
/// supplied via the daemon's `--workspace` CLI flag and is not
/// persisted in this TOML -- a stored copy would only ever drift
/// from the CLI on the next boot.
///
/// `deny_unknown_fields` enforces the redesign §10 migration
/// contract: a legacy `[head_active]` block (or any other
/// retired key) fails closed at boot with a serde "unknown
/// field" diagnostic rather than silently accumulating dead
/// state.  `docs/BUILD.md` documents the named-field rejection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// User preference for mic + channel selection.  Mutable at
    /// runtime via API (`POST /mic/policy`) and via hot-reload of
    /// this TOML file.  Cross-validated at every entry point
    /// against the launch-time [`LaunchConfig::mic`] catalogue,
    /// which is immutable for the daemon's lifetime.
    pub mic: MicPolicy,
    /// Inference cadence (hop samples + top-k).  Mutable at
    /// runtime via `POST /inference` and via hot-reload.  The
    /// engine reads the live ArcSwap on every iteration so edits
    /// take effect within one frame.
    pub inference: InferenceCfg,
}

impl Config {
    /// Aggregate validator.  Walks every operator-tunable
    /// sub-section that has its own `validate(&self)` predicate,
    /// returning the *first* failure with section-prefixed context.
    /// `MicPolicy` cross-validation against the launch catalogue is
    /// not included here -- it requires the catalogue Arc, which the
    /// `config` crate cannot reach by itself; callers run that step
    /// separately via [`validate_policy_against_catalogue`].
    ///
    /// Called from every entry point that materializes a `Config`:
    /// [`ConfigCell::load`] (TOML on disk),
    /// [`ConfigCell::from_value`] (in-memory bootstrap),
    /// [`ConfigCell::mutate_then`] (post-mutator, pre-persist),
    /// and the watcher's `try_reload` path.  A previously-accepted
    /// config that fails any predicate now refuses at boot/reload
    /// rather than silently corrupting runtime state.
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        self.inference
            .validate()
            .map_err(ConfigValidationError::Inference)?;
        Ok(())
    }

    /// Default user-preference [`Config`] for a fresh install.  See
    /// [`LaunchConfig::default_for`] for the matching deployment
    /// manifest (which provisions the synthetic mock candidate so
    /// `mic.policy = FirstAvailable + Auto` resolves against
    /// something on the first boot).
    ///
    /// The user-pref TOML now ships only the policy + inference
    /// cadence; the catalogue (which mics + channels exist),
    /// training defaults, file-service caps, and stream binds all
    /// live in the launch-time [`LaunchConfig`] file.  The active
    /// head is persisted under `<workspace_root>/active/` (per
    /// redesign §2); the legacy `head_active` TOML contract is
    /// removed.
    pub fn default_for() -> Self {
        Self {
            mic: MicPolicy {
                mic: MicSelection::FirstAvailable,
                channel: ChannelSelection::Auto,
            },
            inference: InferenceCfg::default(),
        }
    }
}

/// Owned config snapshot + atomic mutation surface.  `Clone` is cheap
/// (one Arc bump) -- pass to handlers freely.  The contained
/// `ArcSwap<Config>` provides wait-free reads.
///
/// ## Concurrency
///
/// Reads (`snapshot`) are wait-free atomic loads.
///
/// Writes (`mutate`, `mutate_then`) are serialized through a sync
/// `parking_lot::Mutex<()>`.  Two concurrent `mutate` calls don't
/// race the snapshot+modify+write+store cycle -- without the lock,
/// two callers could each clone the OLD snapshot, mutate disjoint
/// fields, and then last-writer-wins on both disk and memory,
/// silently dropping the earlier writer's change.
///
/// The lock is held briefly: snapshot clone (us), closure (us),
/// atomic file rename (us-ms), `ArcSwap::store` (~5 ns).  For a
/// daemon whose configuration mutates only via REST (rare events),
/// this is in the noise.  Hot-path readers never touch the lock.
///
/// Concrete impl of the [`ConfigHandle`] trait.  Daemon constructs
/// `Arc<ConfigCell>` and dyn-coerces to `Arc<dyn ConfigHandle>` at
/// the api boundary; in-crate callers (mic_settings, watcher) hold
/// `Arc<ConfigCell>` directly to access impl-only methods (`watch`,
/// `persist`, `path()`, `reload_count()`).  Naming follows the
/// `MicSettingsCell` pattern (trait `MicSettingsHandle`, concrete
/// `MicSettingsCell`).
#[derive(Clone, Debug)]
pub struct ConfigCell {
    inner: Arc<ArcSwap<Config>>,
    /// The on-disk path.  We use it for `mutate` (atomic write) and
    /// `watch` (notify watcher); held internally so the public API
    /// is `(&self)` everywhere.
    path: Arc<PathBuf>,
    /// Serializes the read-modify-write cycle in `mutate`.  Held
    /// briefly; never on the hot read path.
    mutate_lock: Arc<parking_lot::Mutex<()>>,
    /// Cumulative count of successful reloads from the watcher.
    /// Bumped only when the swapped-in value differs from the prior
    /// snapshot.  Tests use this to assert debounce coalescing; ops
    /// dashboards can chart it for "config thrashed" alerts.
    reload_count: Arc<std::sync::atomic::AtomicU64>,
}

impl ConfigCell {
    /// Load from `path`, returning a handle.  Does NOT spawn the
    /// hot-reload watcher; call `watch()` separately.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| read_err(path.display(), e))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| parse_err(path.display(), e))?;
        // Reject operator-supplied values that would corrupt runtime
        // behaviour (inference: zero hop / top_k out of range,
        // file caps at zero, invalid training defaults, etc.).
        // The engine clamps some of these defensively at runtime,
        // but failing loudly at boot is
        // preferable to a silently-corrected config that drifts from
        // what's on disk.  Mic policy cross-validation against the
        // launch catalogue runs at the daemon boundary (see
        // [`validate_policy_against_catalogue`]) -- `Config`-alone
        // can't reach the immutable catalogue.
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Ok(Self {
            inner: Arc::new(ArcSwap::from_pointee(cfg)),
            path: Arc::new(path.to_path_buf()),
            mutate_lock: Arc::new(parking_lot::Mutex::new(())),
            reload_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Build a handle around an in-memory `Config`.  The `path` is
    /// where future `mutate` calls will write.  Useful for tests +
    /// first-boot bootstrapping (write the default cfg to disk
    /// without first reading it).
    ///
    /// Validates `cfg` at the same level as [`ConfigCell::load`]
    /// -- a caller passing an in-memory config that fails any
    /// sub-section predicate gets `ConfigError::Invalid` immediately
    /// rather than having the daemon boot into an inconsistent
    /// state.  Tests that deliberately want an unvalidated handle
    /// should construct `Self { inner, path, ... }` directly inside
    /// the test module (the field-level constructor is private to
    /// the crate).
    pub fn from_value(cfg: Config, path: PathBuf) -> Result<Self, ConfigError> {
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Ok(Self {
            inner: Arc::new(ArcSwap::from_pointee(cfg)),
            path: Arc::new(path),
            mutate_lock: Arc::new(parking_lot::Mutex::new(())),
            reload_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Persist the in-memory snapshot to disk.  Used at first boot to
    /// materialize a default config.
    pub fn persist(&self) -> Result<(), ConfigError> {
        let snap = self.snapshot();
        crate::config::watcher::write_toml_atomically(&self.path, &snap)?;
        Ok(())
    }

    /// One ArcSwap::load_full -> an `Arc<Config>` aliasing the current
    /// snapshot. ~5 ns.  Hold for as long as needed.
    pub fn snapshot(&self) -> Arc<Config> {
        self.inner.load_full()
    }

    /// Apply `f` to a clone of the snapshot, then atomically write
    /// the result to disk and store it in the ArcSwap.  Serialized
    /// against concurrent `mutate` / `mutate_then` calls.
    pub fn mutate<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut Config),
    {
        self.mutate_then(
            |c| {
                f(c);
            },
            |_| (),
        )
    }

    /// Like [`Self::mutate`], but
    ///
    /// 1. The mutator returns a value `R` propagated to the caller --
    ///    handy for capturing "what did I just compute" without an
    ///    `Option<R>` + post-call `.expect()` dance.
    /// 2. An `after` closure runs while STILL HOLDING the mutate
    ///    lock.  Use this when an in-memory runtime ArcSwap (e.g. the
    ///    daemon's `mic_policy`) MUST stay consistent with the
    ///    on-disk config across concurrent updates.
    ///
    /// Without (2), an API handler that does
    /// `config.mutate(...)?; runtime.store(...)` can interleave with
    /// a second concurrent handler:
    ///
    /// ```text
    ///   T1: config.mutate(mic = X) -> disk has X, mem has X
    ///   T2: config.mutate(mic = Y) -> disk has Y, mem has Y
    ///   T2: runtime.store(Y) -> runtime = Y
    ///   T1: runtime.store(X) -> runtime = X
    ///   --- end: disk Y, mem Y, runtime X (drift) ---
    /// ```
    ///
    /// `mutate_then` collapses both ops into one critical section,
    /// preventing drift.
    pub fn mutate_then<F, G, R>(&self, mutator: F, after: G) -> Result<R, ConfigError>
    where
        F: FnOnce(&mut Config) -> R,
        G: FnOnce(&Config),
    {
        // Re-entrancy check.  Surfaces a structured error
        // instead of a silent deadlock when a callback re-enters.
        // `Cell::replace(true)` returns the prior value: if it was
        // already `true`, this is a re-entry; if `false`, we're the
        // outermost frame and the RAII guard clears it on drop.
        struct ResetOnDrop;
        impl Drop for ResetOnDrop {
            fn drop(&mut self) {
                IN_MUTATE.with(|c| c.set(false));
            }
        }
        let was_in_mutate = IN_MUTATE.with(|c| c.replace(true));
        if was_in_mutate {
            return Err(ConfigError::ReentrantMutate);
        }
        let _reset = ResetOnDrop;
        let _guard = self.mutate_lock.lock();
        let mut cfg = (*self.snapshot()).clone();
        let result = mutator(&mut cfg);
        // Gate persistence on validation.  A mutator that
        // produces an invalid `Config` (e.g. an API handler that
        // accepts an out-of-range field, or a programmer error in a
        // composite update) must NOT reach disk.  Run validation here
        // BEFORE `write_toml_atomically` so the on-disk + in-memory
        // state both stay at the prior snapshot on rejection.
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: self.path.display().to_string(),
            msg: e.to_string(),
        })?;
        crate::config::watcher::write_toml_atomically(&self.path, &cfg)?;
        // Move `cfg` into the Arc and `clone` only the Arc (~5 ns
        // ref-bump) for the store.  The `after` hook reads through
        // the same Arc -- no second deep-clone of `Config`.  Saves
        // one full `Config::clone()` (paths + strings + nested
        // structs) per mutation.
        let arc = Arc::new(cfg);
        self.inner.store(arc.clone());
        after(&arc);
        Ok(result)
    }

    /// Spawn a notify watcher that reloads the in-memory snapshot
    /// whenever the on-disk file changes.  Returns a guard that owns
    /// the watcher thread; drop to stop watching.
    ///
    /// Equivalent to [`watch_with`](Self::watch_with) with a
    /// trivially-accepting callback.  Use `watch_with` when the
    /// daemon needs to update downstream live state (e.g.
    /// ArcSwap'd policy clones the arbitrator/inference engine
    /// read) on each successful reload, AND/OR cross-validate the
    /// reloaded config against state the `config` crate doesn't
    /// know about.
    pub fn watch(&self) -> Result<WatcherGuard, ConfigError> {
        self.watch_with(|_| Ok(()))
    }

    /// Like [`Self::watch`] but invokes `on_reload(&Config) ->
    /// Result<(), ConfigValidationError>` **before** committing the
    /// parsed config to the inner snapshot.  The callback gets a
    /// chance to veto: returning `Err(diagnostic)` causes the worker to log
    /// the diagnostic at `warn!` and **discard** the reload --
    /// `inner` stays at its prior value.  This is the hook used
    /// by the daemon to cross-validate the user-pref policy
    /// against the launch catalogue before exposing it to the
    /// arbitrator.
    ///
    /// On `Ok`, the callback's side effects (e.g. live-ArcSwap
    /// stores) are visible BEFORE `inner` is updated.  The window
    /// between "live ArcSwap stored" and "inner stored" is
    /// microseconds, all inside `mutate_lock`.  Other writers (api
    /// handlers via `mutate_then`, the watcher's own next reload)
    /// take the same lock and serialize, so they never observe
    /// the divergence.  Read-only consumers that touch ONLY one
    /// side (e.g. the arbitrator's hot loop reading the live
    /// ArcSwap, or a status endpoint reading `snapshot()`) are
    /// trivially fine -- they don't cross-correlate.  The only
    /// observer that could legitimately notice is one that reads
    /// BOTH `inner` and a live ArcSwap in the same operation
    /// without taking the lock; no such code path exists today.
    ///
    /// Callback contract: do not apply side effects on the `Err`
    /// path; the worker will not commit, and a half-applied state
    /// would leave inner / live inconsistent.
    ///
    /// The callback runs on the debounce worker thread, **with
    /// `mutate_lock` held** -- see the implementation note below
    /// for why this matters.
    ///
    /// ## Errors during reload
    ///
    /// Logged at `warn!`; the in-memory snapshot is preserved.
    ///
    /// ## Debouncing
    ///
    /// Many editors emit several FS events per save (`vim` does
    /// truncate -> write -> rename -> chmod, >=3 events in
    /// milliseconds).  Without debouncing each one reads + parses
    /// the file, costing a parse + atomic store per spurious event.
    /// We coalesce events with a 100 ms quiet window: the notify
    /// callback nudges a worker thread, which drains all pending
    /// nudges that arrive within 100 ms before performing exactly
    /// one reload.  The window is short enough to feel instant to a
    /// human operator and long enough to absorb editor write-bursts.
    ///
    /// ## Why the worker holds `mutate_lock`
    ///
    /// Without it, an API mutation interleaved with a file edit
    /// could lose the API's write: the worker reads the file
    /// before the API's `write_toml_atomically` finishes, then
    /// stores its (stale) parsed value back into the inner
    /// ArcSwap, overwriting the API's just-stored value.  Holding
    /// `mutate_lock` for the read+parse+validate+store cycle
    /// serializes file-edit reloads with API mutations.
    ///
    /// The lock is held briefly: one `fs::read_to_string` on a
    /// small TOML (~ms on rotational disk, microseconds on SSD),
    /// one parse, one validate, one `inner.store`.  API mutations
    /// pay the same lock cost, so amplification is limited to
    /// "file editor's save now waits for any concurrent API
    /// write" -- fine at human edit cadence.
    ///
    /// ## Callback contract
    ///
    /// `on_reload` MUST NOT call back into `mutate` /
    /// `mutate_then` / re-emit a file write -- those would
    /// re-acquire `mutate_lock` from the same thread (parking_lot
    /// is non-reentrant by default -> deadlock).  The callback IS
    /// safe to call `snapshot()` (wait-free atomic load) or update
    /// other ArcSwaps that the callback owns.
    ///
    /// **Panics.** A panicking callback is caught by
    /// `std::panic::catch_unwind` inside the worker thread and
    /// logged at `error!`; the panic is treated as `Err`-equivalent
    /// (reload discarded).  The worker continues, so subsequent
    /// reloads still fire.  The callback's panic-payload is dropped
    /// -- log enough context inside the callback that an operator
    /// can diagnose from the error log alone.  Don't rely on
    /// catch_unwind: the only safe assumption is "panics may be
    /// silently absorbed; design the callback to never panic." A
    /// callback that uses `UnwindSafe`-violating types (raw
    /// mutex/refcell access, etc.) may still leave them in an
    /// inconsistent state -- keep the callback short and obvious.
    pub fn watch_with<F>(&self, on_reload: F) -> Result<WatcherGuard, ConfigError>
    where
        F: Fn(&Config) -> Result<(), ConfigValidationError> + Send + Sync + 'static,
    {
        use notify::{Event, RecursiveMode, Watcher};

        let path = self.path.clone();
        let inner = self.inner.clone();
        let reload_count = self.reload_count.clone();
        let mutate_lock = self.mutate_lock.clone();
        let on_reload: Arc<ReloadCallback> = Arc::new(on_reload);

        // Channel from the notify worker to the debounce thread.
        // Bounded depth would risk stalling the notify thread; we
        // expect << 1k events/s in production, so unbounded is fine.
        let (kick_tx, kick_rx) = std::sync::mpsc::channel::<()>();

        // Captured for the notify callback's per-event path filter.
        // We watch the parent directory (so rename-replace edits
        // surface even when the inode swaps) and gate kicks on
        // `event.paths` containing an entry whose `file_name()`
        // matches our target.  `file_name()` comparison sidesteps
        // platform-specific canonicalization (notably macOS
        // `/tmp` <-> `/private/tmp`) that breaks full-path equality
        // and is unambiguous because two distinct files in the
        // same directory cannot share a file name.  Events with no
        // file_name (e.g. dir-level rescans) are dropped -- the
        // worker would re-read the file anyway and the value-
        // equality short-circuit absorbs no-op reloads, but
        // dropping them avoids an unnecessary lock+read+parse
        // cycle on every unrelated dir event.
        let target_name = self.path.file_name().map(|n| n.to_owned());

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<Event>| match res {
                Ok(event) => {
                    let touches_target = match &target_name {
                        // Empty paths: be conservative and kick.
                        // notify can emit pathless events (rescan
                        // signals on some platforms, error-recovery
                        // metadata) that legitimately mean "you may
                        // have missed an event"; dropping them
                        // would lose hot-reload coverage during
                        // backend transients.  The value-equality
                        // short-circuit in `try_reload` absorbs
                        // the false-positive at the cost of one
                        // lock+read+parse cycle.
                        Some(name) => {
                            event.paths.is_empty()
                                || event
                                    .paths
                                    .iter()
                                    .any(|p| p.file_name() == Some(name.as_os_str()))
                        }
                        // Pathological: target has no file_name
                        // (e.g. "/").  Don't filter -- preserves the
                        // behaviour of forwarding everything.
                        None => true,
                    };
                    if !touches_target {
                        return;
                    }
                    // `send` only fails if the receiver hung up,
                    // i.e. the worker thread already exited (e.g.
                    // because the WatcherGuard was dropped).  Ignore.
                    let _ = kick_tx.send(());
                }
                Err(e) => {
                    tracing::warn!(target: "config", err = %e, "watcher error");
                }
            })?;

        // Debounce worker.  Owns the path + ArcSwap clone + lock +
        // callback; loops until all senders are dropped (which
        // happens when the notify watcher inside `WatcherGuard` is
        // dropped).
        let path_for_worker = path.clone();
        let worker = std::thread::Builder::new()
            .name("config-reload-debounce".into())
            .spawn(move || {
                crate::config::watcher::debounce_reload_worker(
                    path_for_worker,
                    inner,
                    reload_count,
                    mutate_lock,
                    on_reload,
                    kick_rx,
                )
            })
            .map_err(|e| ConfigError::ThreadSpawn {
                path: path.display().to_string(),
                source: e,
            })?;

        // Watch the parent directory so vim-style "write to tmp, rename"
        // edits also surface (notify on the file itself misses them on
        // some platforms when the inode is replaced).
        let parent = self
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        watcher.watch(&parent, RecursiveMode::NonRecursive)?;
        Ok(WatcherGuard {
            watcher: Some(Box::new(watcher)),
            worker: Some(worker),
        })
    }

    /// Number of times the watcher has successfully reloaded the
    /// config (i.e. swapped a different value into the ArcSwap).
    /// Used by ops dashboards + by the debounce test to assert a
    /// burst of writes coalesces into a small number of reloads.
    pub fn reload_count(&self) -> u64 {
        self.reload_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Path the handle reads + writes.  Useful for diagnostics.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// MARK: `ConfigHandle` trait + `ConfigGuard` impls

/// Concrete guard returned by `ConfigCell::open_mutation`.  Holds the
/// mutate-lock guard, the cloned config under mutation, and the
/// re-entrancy sentinel.  Drop without `commit` releases the lock and
/// the IN_MUTATE flag without persisting (rollback semantics).
///
/// `'h` is the borrowed lifetime of the parent `ConfigCell`.  The
/// type is private to the crate; callers see it only through the
/// `Box<dyn ConfigGuard + 'h>` returned by the trait method.
struct CellGuard<'h> {
    /// Borrow of the parent cell's inner ArcSwap + path.  Used in
    /// `commit` to atomic-write + atomic-swap.
    cell: &'h ConfigCell,
    /// Mutate-lock guard.  Drop releases the lock; held until commit
    /// or rollback.
    _lock: parking_lot::MutexGuard<'h, ()>,
    /// Cloned snapshot the caller mutates via `config()`.  `Some`
    /// until `commit`/`rollback` consumes the guard.
    cfg: Option<Config>,
    /// Re-entrancy sentinel reset on drop.  Mirrors the local
    /// `ResetOnDrop` shape inside `mutate_then`.
    _reset: ResetOnDrop,
}

/// Per-thread re-entrancy reset.  Cleared on drop so a panic mid-
/// guard doesn't leave the thread permanently locked out of
/// future mutations.  Mirrored from `mutate_then`'s inline shape.
struct ResetOnDrop;
impl Drop for ResetOnDrop {
    fn drop(&mut self) {
        IN_MUTATE.with(|c| c.set(false));
    }
}

impl<'h> handle::ConfigGuard for CellGuard<'h> {
    fn config(&mut self) -> &mut Config {
        self.cfg
            .as_mut()
            .expect("config() after commit/rollback is a guard misuse")
    }

    fn commit(
        mut self: Box<Self>,
        after: Box<dyn FnOnce(&Config) + Send>,
    ) -> Result<(), ConfigError> {
        let cfg = self
            .cfg
            .take()
            .expect("commit on a guard already committed/rolled-back");
        // Validate before writing -- same gating as `mutate_then` so
        // an invalid mutator doesn't reach disk.
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: self.cell.path.display().to_string(),
            msg: e.to_string(),
        })?;
        crate::config::watcher::write_toml_atomically(&self.cell.path, &cfg)?;
        // Move into Arc; clone Arc (~5 ns ref-bump) so the after
        // hook reads through the same pointer the inner ArcSwap
        // now holds -- saves a second deep-clone of `Config`.
        let arc = Arc::new(cfg);
        self.cell.inner.store(arc.clone());
        after(&arc);
        // _lock + _reset drop here; lock released, IN_MUTATE cleared.
        Ok(())
    }

    fn rollback(self: Box<Self>) {
        // Explicit drop. _lock + _reset Drop impls do the actual work.
        // `cfg` (if Some) is discarded along with the boxed guard.
        drop(self);
    }
}

impl handle::ConfigHandle for ConfigCell {
    fn snapshot(&self) -> Arc<Config> {
        ConfigCell::snapshot(self)
    }

    fn path(&self) -> &Path {
        ConfigCell::path(self)
    }

    fn open_mutation(&self) -> Result<Box<dyn handle::ConfigGuard + '_>, ConfigError> {
        // Re-entrancy check -- same as `mutate_then`.  The guard's
        // `_reset: ResetOnDrop` clears the flag when the guard is
        // dropped (committed or rolled back, or panic-unwound).
        let was_in = IN_MUTATE.with(|c| c.replace(true));
        if was_in {
            return Err(ConfigError::ReentrantMutate);
        }
        // Acquire lock AFTER setting IN_MUTATE so the panic-on-
        // wait case (a re-entry from inside an `after` callback)
        // surfaces as ReentrantMutate rather than a deadlock.
        let _reset = ResetOnDrop;
        let lock = self.mutate_lock.lock();
        let cfg = (*self.snapshot()).clone();
        Ok(Box::new(CellGuard {
            cell: self,
            _lock: lock,
            cfg: Some(cfg),
            _reset,
        }))
    }
}

/// RAII guard for the notify watcher thread + debounce worker.
/// Dropping the guard stops watching: the notify watcher is freed,
/// which closes the kick channel, which unblocks the debounce
/// thread's `recv` with `Disconnected` and lets it exit.  We `join`
/// the debounce thread on drop so tests + supervised restarts see
/// deterministic teardown rather than a thread that may still be
/// in flight when the next test starts.
///
/// **Drop order matters.** The worker blocks on the kick channel,
/// whose only sender lives inside the notify watcher's callback.
/// Joining the worker BEFORE dropping the watcher would deadlock.
/// The custom Drop impl explicitly drops the watcher first, then
/// joins.
pub struct WatcherGuard {
    watcher: Option<Box<dyn notify::Watcher + Send + Sync>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for WatcherGuard {
    fn drop(&mut self) {
        // 1) Free the notify watcher -> its callback (and the
        //    `kick_tx` it owns) is dropped -> the kick channel
        //    closes -> the worker's `rx.recv*` returns Disconnected.
        if let Some(w) = self.watcher.take() {
            drop(w);
        }
        // 2) Wait for the worker to exit.  Ignore panics -- a
        //    panicked worker has already logged the cause.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl std::fmt::Debug for WatcherGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatcherGuard").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    // Test code: writes TOML fixtures with `std::fs::write` and uses
    // `std::sync::Mutex` to count callback invocations from a watcher
    // worker.  Both are off-limits to production code per `clippy.toml`;
    // the production constraint doesn't apply here.
    #![allow(clippy::disallowed_methods, clippy::disallowed_types)]
    use super::*;
    // Types pulled by name in test bodies that no longer
    // appear in lib.rs's free imports (they live in `launch.rs` /
    // `domain.rs` now).
    use crate::common::ids::MicId;
    use crate::inference::{BackboneCatalogue, BackboneKind, BackboneRef};

    /// Test fixture: production-default hot config.  Stream binds
    /// live in `LaunchConfig` now, so this can validate without
    /// test-side socket path patching.
    fn fresh_default() -> Config {
        Config::default_for()
    }

    fn fresh_stream() -> StreamCfg {
        StreamCfg {
            uds_path: std::env::temp_dir().join("acoustics_lab_test.sock"),
            ..StreamCfg::default()
        }
    }

    fn fresh_launch_for_load(dir: &Path) -> LaunchConfig {
        let mut launch = LaunchConfig::default_for();
        launch.stream.uds_path = dir.join("launch.sock");
        launch
    }

    /// TOML round-trip preserves all fields.
    #[test]
    fn config_round_trips_through_toml() {
        let cfg = fresh_default();
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        let back: Config = toml::from_str(&s).expect("parse");
        assert_eq!(cfg, back);
    }

    /// Defaults populate distinct `tcp_policy` /
    /// `uds_policy`.  TCP gets the strict capped policy
    /// (32 connections, subprotocol required); UDS relaxes the
    /// subprotocol gate (filesystem perms are the auth boundary).
    /// The TOML round-trip preserves the distinction.
    #[test]
    fn stream_cfg_defaults_per_listener_policy() {
        let stream = StreamCfg::default();
        assert_eq!(stream.tcp_policy.max_connections_per_stream, 32);
        assert!(
            stream.tcp_policy.require_subprotocol,
            "TCP default must keep the strict subprotocol gate",
        );
        assert_eq!(stream.uds_policy.max_connections_per_stream, 32);
        assert!(
            !stream.uds_policy.require_subprotocol,
            "UDS default must relax the subprotocol gate",
        );

        // Round-trip preserves both policies.
        let s = toml::to_string_pretty(&stream).expect("serialize");
        let back: StreamCfg = toml::from_str(&s).expect("parse");
        assert_eq!(stream.tcp_policy, back.tcp_policy);
        assert_eq!(stream.uds_policy, back.uds_policy);
    }

    /// Pre-S4 TOML files (no `[stream.tcp_policy]` /
    /// `[stream.uds_policy]` tables) load cleanly via the
    /// `#[serde(default)]` fallbacks; the resulting policies are
    /// the same defaults `Config::default_for` would have written.
    #[test]
    fn stream_cfg_legacy_toml_loads_with_default_policies() {
        // Hand-rolled minimal TOML mirrors the pre-S4 shape: stream
        // table has only the four legacy fields.
        let stream = fresh_stream();
        let mut text = toml::to_string_pretty(&stream).expect("ser");
        // Strip the new policy tables from the on-disk text to
        // simulate a pre-S4 config file.
        for marker in ["[stream.tcp_policy]", "[stream.uds_policy]"] {
            if let Some(start) = text.find(marker) {
                // Truncate at the first marker; both tables are
                // serialized at the end of `[stream]`'s block, so
                // we drop everything from the first one onward and
                // know the file ends cleanly.
                text.truncate(start);
                break;
            }
        }
        let parsed: StreamCfg = toml::from_str(&text).expect("parse legacy shape");
        assert_eq!(parsed.tcp_policy, stream.tcp_policy);
        assert_eq!(parsed.uds_policy, stream.uds_policy);
    }

    /// `load()` reads a TOML file and yields the parsed config.
    #[test]
    fn load_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let h = ConfigCell::load(&path).expect("load");
        assert_eq!(*h.snapshot(), cfg);
    }

    /// Loading a config with an out-of-range `inference.hop_samples`
    /// returns `ConfigError::Invalid` rather than silently clamping
    /// to a runtime-safe value.  The engine still belt-and-suspenders
    /// `.max(1)`s in its hot loop, but boot-time validation lets the
    /// operator see the typo via systemd logs instead of debugging
    /// "why did my hop disagree with the config file?" later.
    /// `Config::validate` walks every operator-tunable
    /// sub-section.  Each predicate is exercised here so a future
    /// loosening of an individual predicate (or its outright
    /// removal) is caught.
    #[test]
    fn config_validate_walks_every_subsection() {
        // Default-for is by construction valid.
        fresh_default().validate().expect("default must validate");

        // Inference (delegated) -- zero hop, already covered by
        // `load_rejects_invalid_inference_cfg`; ensure the aggregate
        // walker reaches inference too.  Training-defaults and the
        // file-service caps moved to `LaunchConfig` during the
        // launch/user-pref split; their validators are exercised
        // via `launch_load_rejects_invalid_training_defaults` /
        // `launch_load_rejects_invalid_file_cfg` below.
        let mut cfg = fresh_default();
        cfg.inference.hop_samples = 0;
        let err = cfg
            .validate()
            .expect_err("zero hop must reject")
            .to_string();
        assert!(err.contains("inference"), "{err}");
    }

    /// Production default `tcp_bind` is loopback.  The daemon
    /// trusts every request that reaches it (auth lives in the
    /// front-proxy), but a loopback default is the safest shape
    /// for a fresh install -- operators who want to expose the
    /// listener to the network must flip the bind explicitly.
    #[test]
    fn default_tcp_bind_is_loopback() {
        let launch = LaunchConfig::default_for();
        assert_eq!(launch.stream.tcp_bind, "127.0.0.1:8787");
    }

    /// `StreamCfg::validate` accepts any well-formed `host:port`,
    /// including non-loopback (`0.0.0.0`, a NIC address) and the
    /// IPv6 unspecified form.  Auth/exposure is the operator's
    /// responsibility; the validator only rejects shapes that
    /// cannot bind at all (empty host, missing colon, non-u16
    /// port -- those have their own dedicated tests).
    #[test]
    fn stream_cfg_accepts_any_well_formed_tcp_bind() {
        for ok in [
            "127.0.0.1:8787",
            "127.0.0.1:0",
            "127.5.5.5:8787",
            "[::1]:8787",
            "localhost:8787",
            "Localhost:9000",
            "0.0.0.0:8787",
            "[::]:8787",
            "192.168.1.10:8787",
            "8.8.8.8:8787",
            "example.com:8787",
        ] {
            let mut stream = fresh_stream();
            stream.tcp_bind = ok.into();
            if let Err(e) = stream.validate() {
                panic!("{ok:?} should validate but got {e}");
            }
        }
    }

    /// `StreamCfg::validate` rejects an empty-host
    /// `tcp_bind` like `":8787"` -- previously parsed as
    /// "valid port, empty host" and only failed at `SocketAddr`
    /// parse time inside the daemon's bind future.  Surfacing it
    /// in the operator's TOML diagnostic at boot is the desired
    /// shape.
    #[test]
    fn stream_cfg_rejects_empty_host_tcp_bind() {
        let mut stream = fresh_stream();
        stream.tcp_bind = ":8787".into();
        let err = stream
            .validate()
            .expect_err("empty-host tcp_bind must reject")
            .to_string();
        assert!(
            err.contains("empty host"),
            "diagnostic should name the empty-host shape: {err}",
        );
    }

    /// `StreamCfg::validate_uds_path` rejects a regular file at
    /// `uds_path`.  `stream_io::bind_uds` owns race-safe deletion
    /// at bind time; the static check at config-load surfaces the
    /// typo before the daemon starts.
    #[test]
    fn stream_cfg_rejects_uds_path_pointing_at_regular_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let regular_file = dir.path().join("not-a-socket.txt");
        std::fs::write(&regular_file, b"hello").expect("write regular file");
        let mut stream = fresh_stream();
        stream.uds_path = regular_file.clone();
        let err = stream
            .validate()
            .expect_err("regular-file uds_path must reject")
            .to_string();
        assert!(
            err.contains("regular file"),
            "diagnostic should name the regular-file shape: {err}",
        );
        assert!(
            err.contains(&regular_file.display().to_string()),
            "diagnostic should include the offending path: {err}",
        );
    }

    /// `StreamCfg::validate_uds_path` rejects a symlink at
    /// `uds_path` even when the symlink target is itself a
    /// socket.  Following the symlink at bind time would expose
    /// the unlink to a TOCTOU on the symlink target.
    #[cfg(unix)]
    #[test]
    fn stream_cfg_rejects_uds_path_pointing_at_symlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("real.sock");
        std::fs::write(&target, b"x").expect("write target");
        let link = dir.path().join("link.sock");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let mut stream = fresh_stream();
        stream.uds_path = link;
        let err = stream
            .validate()
            .expect_err("symlink uds_path must reject")
            .to_string();
        assert!(
            err.contains("symlink"),
            "diagnostic should name the symlink shape: {err}",
        );
    }

    /// `StreamCfg::validate_uds_path` rejects a path whose
    /// parent directory does not exist.  An operator who
    /// fat-fingers a directory name would otherwise see a
    /// confusing `bind: No such file or directory` from inside
    /// the daemon's bind future; rejecting at config-load time
    /// surfaces the typo with the offending path in the
    /// diagnostic.
    #[test]
    fn stream_cfg_rejects_uds_path_with_missing_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut stream = fresh_stream();
        stream.uds_path = dir.path().join("does-not-exist").join("a.sock");
        let err = stream
            .validate()
            .expect_err("missing-parent uds_path must reject")
            .to_string();
        assert!(
            err.contains("parent directory") && err.contains("does not exist"),
            "diagnostic should name the missing-parent shape: {err}",
        );
    }

    /// `StreamCfg::validate_uds_path` rejects a bare filename
    /// (no parent component).  Binding into CWD is undefined for
    /// a daemon-supervised process; require an explicit parent.
    #[test]
    fn stream_cfg_rejects_uds_path_without_parent() {
        let mut stream = fresh_stream();
        stream.uds_path = PathBuf::from("acoustics_lab.sock");
        let err = stream
            .validate()
            .expect_err("bare-filename uds_path must reject")
            .to_string();
        assert!(
            err.contains("parent directory"),
            "diagnostic should name the missing-parent shape: {err}",
        );
    }

    /// `StreamCfg::validate_uds_path` accepts a path whose
    /// parent exists and whose target does not yet exist (the
    /// normal first-boot case -- bind will create the socket).
    #[test]
    fn stream_cfg_accepts_uds_path_in_existing_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut stream = fresh_stream();
        stream.uds_path = dir.path().join("a.sock");
        stream
            .validate()
            .expect("uds_path under existing dir is fine");
    }

    /// `TrainingDefaults::validate` rejects every
    /// runtime-fatal shape (`0` on any of the three numeric
    /// fields) and is reachable through the aggregate
    /// `Config::validate`.  The runtime
    /// `TrainingConfig::validate` already catches the same
    /// shapes at job-spawn time; this test pins the boot-time
    /// gate so an operator-edited `epochs = 0` is rejected at
    /// the TOML parser, not after the first `POST /train`.
    #[test]
    fn training_defaults_validator_rejects_zero_fields() {
        // epochs.
        let td = TrainingDefaults {
            epochs: 0,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("zero epochs must reject");
        assert!(err.contains("epochs"), "{err}");

        // batch_size.
        let td = TrainingDefaults {
            batch_size: 0,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("zero batch_size must reject");
        assert!(err.contains("batch_size"), "{err}");

        // learning_rate_e6.
        let td = TrainingDefaults {
            learning_rate_e6: 0,
            ..TrainingDefaults::default()
        };
        let err = td
            .validate()
            .expect_err("zero learning_rate_e6 must reject");
        assert!(err.contains("learning_rate"), "{err}");
    }

    /// `TrainingDefaults::validate` rejects values past the
    /// sanity caps so a typo (`epochs = 100000`) surfaces at
    /// boot rather than wedging the daemon for an hour
    /// reporting "training in progress" on a job that will
    /// never finish.
    #[test]
    fn training_defaults_validator_rejects_oversized_fields() {
        let td = TrainingDefaults {
            epochs: 1_000_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd epochs must reject");
        assert!(err.contains("epochs"), "{err}");

        let td = TrainingDefaults {
            batch_size: 100_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd batch_size must reject");
        assert!(err.contains("batch_size"), "{err}");

        let td = TrainingDefaults {
            learning_rate_e6: 5_000_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd lr must reject");
        assert!(err.contains("learning_rate"), "{err}");
    }

    /// Default `TrainingDefaults` validates and round-trips
    /// through TOML cleanly even when an operator omits the
    /// `[training_defaults]` block entirely (covered by
    /// `legacy_config_without_training_defaults_loads_defaults`)
    /// AND when they spell out every field with the production
    /// defaults.
    #[test]
    fn training_defaults_default_validates() {
        TrainingDefaults::default()
            .validate()
            .expect("default training_defaults must validate");
    }

    /// `LaunchConfig::load` rejects an invalid
    /// `[training_defaults]` block (e.g. `epochs = 0`) at boot.
    /// Training defaults moved from the hot user-pref TOML to the
    /// launch manifest during the split; this pins the boot-time
    /// gate so a typo surfaces in systemd logs rather than at the
    /// first `POST /train`.
    #[test]
    fn launch_load_rejects_invalid_training_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load(dir.path());
        launch.training_defaults.epochs = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero epochs must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("training_defaults") && msg.contains("epochs"),
                "diagnostic should name the offending field: {msg}",
            );
        }
    }

    /// A `mutate_then` callback that re-enters
    /// `mutate_then` on the same handle from the same thread now
    /// returns `Err(ConfigError::ReentrantMutate)` instead of
    /// deadlocking on `parking_lot::Mutex`.  The deadlock would
    /// have been silent and indistinguishable from a slow worker
    /// -- one of the worst failure modes (no panic, no log, no
    /// recovery without a process restart).
    ///
    /// Test runs the inner `mutate` from inside the outer
    /// mutator's body; the inner call hits the re-entrancy
    /// sentinel before reaching the lock and returns the
    /// structured error.  Outer mutator captures that result via
    /// a closure-shared `Option`.
    #[test]
    fn mutate_then_rejects_reentry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");
        let h_arc = Arc::new(h);

        let inner_result = Arc::new(parking_lot::Mutex::new(None::<Result<(), ConfigError>>));
        let inner_for_outer = inner_result.clone();
        let h_inner = h_arc.clone();
        h_arc
            .mutate(move |c| {
                // Trivial outer mutation; the load-bearing call is
                // the inner `mutate` below.
                c.inference.top_k = 5;
                let r = h_inner.mutate(|c2| {
                    c2.inference.top_k = 7;
                });
                *inner_for_outer.lock() = Some(r);
            })
            .expect("outer mutate must succeed");

        let inner = inner_result.lock().take().expect("inner ran");
        match inner {
            Err(ConfigError::ReentrantMutate) => {}
            other => panic!("expected ReentrantMutate from inner re-entry, got {other:?}"),
        }

        // After both mutates, the outer's value (top_k = 5) must
        // have committed; the inner's would-be value (7) must NOT
        // have, because re-entry rejected before the inner reached
        // the lock or write step.
        assert_eq!(h_arc.snapshot().inference.top_k, 5);

        // Sentinel is reset after the outer returns: a fresh
        // top-level `mutate` must succeed (would fail with
        // ReentrantMutate if the thread-local stayed `true`).
        h_arc
            .mutate(|c| {
                c.inference.top_k = 11;
            })
            .expect("post-reentry mutate must succeed");
        assert_eq!(h_arc.snapshot().inference.top_k, 11);
    }

    /// `mutate_then` rejects a mutator that produces an
    /// invalid `Config`.  The on-disk + in-memory state must stay at
    /// the prior snapshot; nothing is half-written.
    #[test]
    fn mutate_rejects_invalid_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");

        let prior_disk = std::fs::read_to_string(&path).expect("read disk");
        let prior_mem = (*h.snapshot()).clone();

        let err = h
            .mutate(|c| {
                c.inference.top_k = 0;
            })
            .expect_err("invalid mutator result must reject");
        assert!(matches!(err, ConfigError::Invalid { .. }), "got {err:?}");

        // Disk + memory both unchanged.
        let post_disk = std::fs::read_to_string(&path).expect("read disk");
        assert_eq!(prior_disk, post_disk, "disk must not change on rejection");
        assert_eq!(
            prior_mem,
            *h.snapshot(),
            "snapshot must not change on rejection"
        );
    }

    #[test]
    fn load_rejects_invalid_inference_cfg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let mut cfg = fresh_default();
        cfg.inference.hop_samples = 0;
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = ConfigCell::load(&path).expect_err("zero hop must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}"
        );

        // top_k=0 also rejected.
        let mut cfg = fresh_default();
        cfg.inference.top_k = 0;
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = ConfigCell::load(&path).expect_err("zero top_k must reject");
        assert!(matches!(err, ConfigError::Invalid { .. }));
    }

    /// `LaunchConfig::load` rejects a catalogue whose candidates
    /// fail `MicCandidate::validate` (here: an empty channel
    /// whitelist).  Without this, the malformed catalogue would
    /// silently propagate to the arbitrator and surface only as
    /// a runtime tracing warn at first open.
    #[test]
    fn launch_load_rejects_invalid_mic_candidate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load(dir.path());
        // Empty channels whitelist -- caught by catalogue validate.
        launch.mic.candidates[0].channels.clear();
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("empty channels must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("launch mic catalogue"),
                "error message should identify the launch catalogue, got {msg}",
            );
        }
    }

    /// `LaunchConfig` round-trips through TOML preserving every
    /// catalogue field.  Catches future serde tag/rename issues.
    #[test]
    fn launch_config_round_trips_through_toml() {
        let original = LaunchConfig::default_for();
        let s = toml::to_string_pretty(&original).expect("ser");
        let back: LaunchConfig = toml::from_str(&s).expect("de");
        assert_eq!(original, back);
    }

    /// First-boot defaults include a cross-platform mock mic but no
    /// backbone paths.  Backbone artifacts are deployment-specific;
    /// operators must specify them in launch.toml instead of relying
    /// on daemon-hardcoded filesystem assumptions.
    #[test]
    fn launch_default_backbone_catalogue_is_empty() {
        let l = LaunchConfig::default_for();
        assert!(l.backbone.is_empty());
    }

    /// The checked-in dev fixtures are a PAIR of distinct schemas:
    /// `misc/etc/config.toml` carries hot-reloadable user
    /// preferences (mirrors `<workspace>/config.toml` shape), while
    /// `misc/etc/launch.toml` carries the immutable launch
    /// catalogues.  Pin them together so the smoke command from
    /// `docs/BUILD.md` does not regress to a policy/catalogue or
    /// backbone-path mismatch.
    #[test]
    fn bundled_etc_configs_parse_and_cross_validate() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config_path = root.join("misc/etc/config.toml");
        let launch_path = root.join("misc/etc/launch.toml");

        std::fs::create_dir_all(root.join("misc/share")).expect("bundled UDS parent");
        let launch = LaunchConfig::load(&launch_path).expect("load bundled launch.toml");
        let text = std::fs::read_to_string(&config_path).expect("read bundled config.toml");
        let cfg: Config = toml::from_str(&text).expect("parse bundled config.toml");
        cfg.validate().expect("bundled config.toml validates");
        validate_policy_against_catalogue(&cfg.mic, &launch.mic, &config_path)
            .expect("bundled mic policy matches bundled launch catalogue");
        let paths: Vec<_> = launch
            .backbone
            .candidates
            .iter()
            .map(|c| c.path.as_path())
            .collect();
        assert_eq!(
            paths,
            vec![
                Path::new("misc/backbones/backbone.rknn"),
                Path::new("misc/backbones/backbone.mpk"),
            ],
            "bundled launch.toml must specify local dev backbone paths",
        );
        assert_eq!(
            launch.head.default.as_ref().map(|h| h.path.as_path()),
            Some(Path::new("misc/heads/default/head.mpk")),
            "bundled launch.toml must specify the local dev default head mpk",
        );
        assert_eq!(
            launch
                .head
                .default
                .as_ref()
                .map(|h| h.labels_path.as_path()),
            Some(Path::new("misc/heads/default/labels.txt")),
            "bundled launch.toml must specify the local dev default labels",
        );
        assert_eq!(
            launch.stream.uds_path,
            PathBuf::from("misc/share/acousticsd.sock"),
            "bundled launch.toml must own local dev stream binds",
        );
        // training_defaults + file caps moved to LaunchConfig during
        // the launch/user-pref split; pin the bundled values so a
        // future drift surfaces in CI rather than as a runtime change.
        assert_eq!(launch.training_defaults, TrainingDefaults::default());
        assert_eq!(launch.file, FileCfg::default());
    }

    /// A launch.toml without `[[backbone.candidates]]` (older /
    /// hand-edited file) parses to an empty catalogue, not a hard
    /// failure.  The daemon then runs without inference, exactly as
    /// it does today when head files are missing.
    #[test]
    fn launch_load_accepts_missing_backbone_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        // Write a valid catalogue but no `[backbone]` table at all.
        let mut launch = fresh_launch_for_load(dir.path());
        launch.backbone = BackboneCatalogue::default();
        let mut text = toml::to_string_pretty(&launch).expect("ser");
        // toml::to_string emits a `[backbone]` header even when the
        // candidate list is empty; strip it so the test mirrors a
        // hand-edited file that never had one in the first place.
        text = text
            .replace("[backbone]\n", "")
            .replace("\n[backbone]\n", "\n");
        std::fs::write(&path, text).expect("write");
        let parsed = LaunchConfig::load(&path).expect("load must accept missing backbone");
        assert!(parsed.backbone.is_empty());
    }

    /// Catalogue-level validation rejects malformed hashes at load
    /// time, surfaced via `ConfigError::Invalid` so operators see
    /// the diagnostic in systemd logs.
    #[test]
    fn launch_load_rejects_malformed_backbone_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load(dir.path());
        launch.backbone.candidates.push(BackboneRef {
            kind: BackboneKind::Burn,
            path: PathBuf::from("operator-supplied/backbone.mpk"),
            hash: None,
        });
        // 63-char hash -- fails the 64-hex validator.
        launch.backbone.candidates[0].hash = Some("a".repeat(63));
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("must reject");
        match err {
            ConfigError::Invalid { msg, .. } => {
                assert!(
                    msg.contains("backbone catalogue") && msg.contains("64"),
                    "diagnostic should name the catalogue + hash length: {msg}",
                );
            }
            other => panic!("expected ConfigError::Invalid, got {other:?}"),
        }
    }

    /// `validate_policy_against_catalogue` rejects a Fixed-id
    /// that's not in the catalogue.  This is the entry point the
    /// daemon uses at boot + on hot-reload.
    #[test]
    fn validate_policy_rejects_unknown_mic_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("dev.toml");
        let launch = LaunchConfig::default_for();
        let policy = MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("not-in-catalogue"),
            },
            channel: ChannelSelection::Auto,
        };
        let err = validate_policy_against_catalogue(&policy, &launch.mic, &path)
            .expect_err("must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("not-in-catalogue"),
                "diagnostic should name the missing id, got {msg}",
            );
        }
    }

    /// `LaunchConfig::load` gates the `[file]` admission caps at
    /// boot: zero on either field would brick uploads on a daemon
    /// that otherwise looks healthy.  The `[file]` block moved from
    /// the user-pref TOML to the launch manifest during the split.
    #[test]
    fn launch_load_rejects_invalid_file_cfg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load(dir.path());
        launch.file.max_upload_bytes = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero max_upload_bytes must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("file") && msg.contains("max_upload_bytes"),
                "diagnostic should name the offending field: {msg}",
            );
        }

        let mut launch = fresh_launch_for_load(dir.path());
        launch.file.max_concurrent_uploads = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero max_concurrent_uploads must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
    }

    /// Launch TOMLs without a `[file]` block load with the
    /// on-device defaults; `#[serde(default)]` on the field keeps
    /// pre-migration files booting.  Pinned values guard against a
    /// silent loosening of the upload cap.
    #[test]
    fn legacy_launch_without_file_block_loads_defaults() {
        let launch = LaunchConfig::default_for();
        let mut value = toml::Value::try_from(&launch).expect("launch to toml value");
        value
            .as_table_mut()
            .expect("launch is a table")
            .remove("file");
        let text = toml::to_string_pretty(&value).expect("serialize legacy launch");

        let parsed: LaunchConfig = toml::from_str(&text).expect("parse legacy launch");
        assert_eq!(parsed.file, FileCfg::default());
        assert_eq!(parsed.file.max_upload_bytes, 256 * 1024 * 1024);
        // Workspace-redesign §9 storage-table default; bumped
        // from 2 to 4 in the redesign for better bulk-load
        // throughput.  Pinned here so a future loosening surfaces
        // in CI.
        assert_eq!(parsed.file.max_concurrent_uploads, 4);
    }

    /// Launch TOMLs without a `[training_defaults]` block load
    /// with the on-device defaults; `#[serde(default)]` keeps the
    /// upgrade non-breaking.
    #[test]
    fn legacy_launch_without_training_defaults_loads_defaults() {
        let launch = LaunchConfig::default_for();
        let mut value = toml::Value::try_from(&launch).expect("launch to toml value");
        value
            .as_table_mut()
            .expect("launch is a table")
            .remove("training_defaults");
        let text = toml::to_string_pretty(&value).expect("serialize legacy launch");

        let parsed: LaunchConfig = toml::from_str(&text).expect("parse legacy launch");
        assert_eq!(parsed.training_defaults, TrainingDefaults::default());
    }

    /// `mutate` updates both memory and disk atomically.
    #[test]
    fn mutate_persists_and_swaps() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");

        h.mutate(|c| {
            c.inference.hop_samples = 22_050;
            c.inference.top_k = 5;
        })
        .expect("mutate");

        // In-memory.
        let snap = h.snapshot();
        assert_eq!(snap.inference.hop_samples, 22_050);
        assert_eq!(snap.inference.top_k, 5);

        // On disk.
        let on_disk = std::fs::read_to_string(&path).expect("read");
        let parsed: Config = toml::from_str(&on_disk).expect("parse");
        assert_eq!(parsed.inference.hop_samples, 22_050);
        assert_eq!(parsed.inference.top_k, 5);
    }

    /// Concurrent `mutate` calls must not lose updates.  Two threads
    /// alternate updating disjoint fields (mic vs inference); after
    /// they finish, both fields' final values must reflect their
    /// caller's last mutation, AND the on-disk file must agree with
    /// the in-memory snapshot.
    ///
    /// This test would FAIL: each mutator clones the OLD
    /// snapshot, so the second-to-write loses its write of the
    /// other field (the snapshot it cloned didn't include the
    /// first writer's change).  With the mutate_lock, the snapshot
    /// clone happens INSIDE the critical section, so the second
    /// writer's clone sees the first's modification.
    ///
    /// IMPORTANT: the writer's value range MUST NOT overlap with
    /// the field's pre-loop default.  Otherwise a lost-update bug
    /// could leave the field at the default and the assertion
    /// "value is in writer range" would still pass -- false negative.
    /// Defaults: `mic = Auto`, `inference.hop_samples = 11_025`.  We
    /// pick mic = `mic-N` (Fixed, N < 1000) and hop_samples in
    /// `[100_000, 100_200)` -- both clearly distinct from defaults.
    #[test]
    fn concurrent_mutate_preserves_disjoint_field_updates() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::thread;

        // Sanity-check the test's offset choice against the default
        // to prevent the false-negative described above.
        let default_hop = fresh_default().inference.hop_samples;
        assert!(
            !(100_000..=100_199).contains(&default_hop),
            "test offset overlaps default; would mask lost-update bug",
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("initial persist");
        let h = Arc::new(h);
        let mic_seq = Arc::new(AtomicU32::new(0));
        let inf_seq = Arc::new(AtomicU32::new(0));

        // 200 mutations per thread on disjoint fields.
        let h1 = h.clone();
        let mic_seq_w = mic_seq.clone();
        let writer_mic = thread::spawn(move || {
            for i in 0..200 {
                h1.mutate(|c| {
                    let id = mic_seq_w.fetch_add(1, Ordering::Relaxed);
                    c.mic.mic = MicSelection::Fixed {
                        id: MicId::parse(&format!("mic-{id}")).expect("test mic id"),
                    };
                    let _ = i;
                })
                .expect("mic mutate");
            }
        });
        let h2 = h.clone();
        let inf_seq_w = inf_seq.clone();
        let writer_inf = thread::spawn(move || {
            for i in 0..200 {
                h2.mutate(|c| {
                    let id = inf_seq_w.fetch_add(1, Ordering::Relaxed);
                    // Pick a base inside `hop_samples`'s
                    // `1..=MAX_HOP_SAMPLES` window so
                    // `mutate_then`'s validator passes;
                    // uniqueness is still guaranteed by the seq
                    // counter (id < 200, base + id <
                    // 22 200 << MAX_HOP_SAMPLES).
                    c.inference.hop_samples = 22_000 + id as usize;
                    let _ = i;
                })
                .expect("inf mutate");
            }
        });
        writer_mic.join().unwrap();
        writer_inf.join().unwrap();

        let final_mic_seq = mic_seq.load(Ordering::Relaxed);
        let final_inf_seq = inf_seq.load(Ordering::Relaxed);
        assert_eq!(final_mic_seq, 200);
        assert_eq!(final_inf_seq, 200);

        // The final on-disk + in-memory state must reflect SOME
        // valid (mic_id, hop_samples) pair where each came from one
        // of the writers -- but critically, BOTH writers' last writes
        // must be visible (no field reverted to a pre-loop default).
        let mem = h.snapshot();
        // hop_samples must come from the inf writer's range,
        // NOT the default 11_025.
        assert!(
            (22_000..=22_199).contains(&mem.inference.hop_samples),
            "hop_samples reverted to {} (lost-update bug)",
            mem.inference.hop_samples,
        );
        // policy.mic must be Fixed, NOT the default FirstAvailable.
        match &mem.mic.mic {
            MicSelection::Fixed { id } => {
                let s = id.as_str();
                assert!(s.starts_with("mic-"), "mic reverted to non-Fixed name: {s}",);
                let n: u32 = s["mic-".len()..].parse().expect("parse mic id");
                assert!(n < 200, "mic id out of range: {n}");
            }
            other => panic!("mic reverted to default Auto / lost-update bug: {other:?}"),
        }

        // The on-disk file must EXACTLY match the in-memory snapshot.
        // (The atomic write + ArcSwap store happen within the same
        // critical section, so this invariant holds even under
        // concurrent mutation.)
        let on_disk: Config = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk, *mem);
    }

    /// Many editors emit several FS events per save.  The watcher
    /// MUST coalesce them inside the 100 ms debounce window so that
    /// a burst of N writes triggers <= 2 reloads (one for the burst
    /// itself; up to one more if the platform's event ordering
    /// happens to release a lone event after the quiet window --
    /// rare in practice but allowed by the contract).
    ///
    /// We force a burst by emitting 8 distinct writes within ~50 ms
    /// and asserting `reload_count() <= 2` after the watcher has had
    /// time to process them.  Without debouncing, each FS event
    /// would trigger its own reload (count would be ~8 on Linux,
    /// platform-dependent elsewhere).
    #[tokio::test(flavor = "current_thread")]
    async fn watcher_debounces_burst_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let _guard = h.watch().expect("watch");

        // Let the FS-watcher backend settle.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // 8 rapid writes with monotonically increasing top_k.  Each
        // is a complete TOML doc so any reload picks up a
        // different value.
        for k in 2..=9 {
            let mut next = cfg.clone();
            next.inference.top_k = k;
            std::fs::write(&path, toml::to_string_pretty(&next).unwrap()).expect("rewrite");
            // No sleep -- pack the writes inside the 100 ms window.
        }

        // Wait long enough for the debounce window to fire, plus a
        // healthy margin for FSEvents/inotify to deliver.
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;

        // The final value MUST be observable.
        assert_eq!(h.snapshot().inference.top_k, 9);
        // And we must have done at most 2 reloads.  Most platforms
        // produce exactly 1; allow 2 for FSEvents' coalescing
        // edge cases.
        let n = h.reload_count();
        assert!(
            n <= 2,
            "burst of 8 writes triggered {n} reloads; debounce \
             expected <= 2"
        );
        // And at least 1 -- the debounce must still fire eventually.
        assert!(n >= 1, "burst of 8 writes triggered 0 reloads");
    }

    /// Watcher reloads the snapshot when an external editor rewrites
    /// the file.  The notify backend is platform-specific
    /// (kqueue/inotify/FSEvents); we wait up to 5 s with a tight poll
    /// loop to avoid timing flakiness.
    #[tokio::test(flavor = "current_thread")]
    async fn watcher_reloads_on_external_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let _guard = h.watch().expect("watch");

        // External edit: change top_k.
        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 7;
        let new_text = toml::to_string_pretty(&new_cfg).unwrap();

        // notify takes a beat to spin up the FSEvents/inotify session;
        // wait briefly so our write is observed.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        std::fs::write(&path, &new_text).expect("rewrite");

        // Poll up to 5 s for the watcher to fire.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if h.snapshot().inference.top_k == 7 {
                return;
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "watcher did not reload within timeout; current top_k = {}",
                    h.snapshot().inference.top_k
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// A panicking `watch_with` callback must not kill the
    /// debounce worker -- operators would silently lose hot-reload
    /// after a single buggy callback fires.  We exercise the
    /// catch_unwind path by intentionally panicking on the FIRST
    /// reload, then asserting that a SECOND reload still updates
    /// the snapshot.
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_callback_panic_does_not_kill_watcher() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        // Counter so the callback panics on the first call only;
        // subsequent calls succeed.  If catch_unwind weren't in
        // place, the worker would die on the first panic and the
        // second edit would never observe a callback.
        let calls: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let calls_clone = calls.clone();
        let _guard = h
            .watch_with(move |_c| {
                let n = {
                    let mut g = calls_clone.lock().unwrap();
                    *g += 1;
                    *g
                };
                if n == 1 {
                    panic!("intentional callback panic for test");
                }
                Ok(())
            })
            .expect("watch_with");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // First write -> first callback -> panic.
        let mut cfg2 = cfg.clone();
        cfg2.inference.top_k = 5;
        std::fs::write(&path, toml::to_string_pretty(&cfg2).unwrap()).expect("write 1");
        // Wait for the callback to fire and panic.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if *calls.lock().unwrap() >= 1 {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("first callback never fired");
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Second write -> if the watcher survived the panic, this
        // also fires the callback.  The callback no-ops the second
        // time (n == 2, not == 1).
        let mut cfg3 = cfg.clone();
        cfg3.inference.top_k = 9;
        std::fs::write(&path, toml::to_string_pretty(&cfg3).unwrap()).expect("write 2");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if *calls.lock().unwrap() >= 2 {
                return; // success -- watcher survived the panic
            }
            if std::time::Instant::now() > deadline {
                let n = *calls.lock().unwrap();
                panic!(
                    "second callback never fired (calls={n}) -- watcher died after \
                     panicking callback",
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// When the callback returns `Err(diagnostic)`, the worker
    /// MUST discard the reload -- `inner` keeps its prior value,
    /// `reload_count` does not increment, and the diagnostic
    /// surfaces in the warn log.  This is the daemon's hook for
    /// cross-validating the reloaded user config against state
    /// (the launch catalogue) the `config` crate doesn't know
    /// about.
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_callback_err_rejects_reload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        // Always-rejecting callback.
        let _guard = h
            .watch_with(|_| Err(ConfigValidationError::Callback("test rejection".into())))
            .expect("watch_with");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // External write to a different value.
        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 7;
        std::fs::write(&path, toml::to_string_pretty(&new_cfg).unwrap()).expect("rewrite");

        // Wait long enough for any reload attempt to settle.
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;

        // The in-memory snapshot MUST equal the prior value
        // (top_k from `fresh_default`, NOT 7) because the
        // callback rejected.
        let snap = h.snapshot();
        assert_ne!(
            snap.inference.top_k, 7,
            "rejected reload was committed anyway: top_k = {}",
            snap.inference.top_k,
        );
        assert_eq!(
            snap.inference.top_k, cfg.inference.top_k,
            "snapshot drifted from the prior valid value despite rejection",
        );
        // reload_count stays at 0 -- we never committed.
        assert_eq!(
            h.reload_count(),
            0,
            "reload_count incremented on rejected reload",
        );
    }

    /// `watch_with` invokes the user-supplied callback on each
    /// successful reload.  The daemon uses this to update live
    /// ArcSwaps (mic_settings, inference_cfg) that downstream
    /// subsystems read on the hot path; without it, file edits
    /// would only update `inner` and never reach the runtime.
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_invokes_callback_on_reload() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let observed: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let observed_clone = observed.clone();
        let _guard = h
            .watch_with(move |c| {
                observed_clone.lock().unwrap().push(c.inference.top_k);
                Ok(())
            })
            .expect("watch_with");

        // Let the FS-watcher backend settle.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // External edit.
        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 9;
        std::fs::write(&path, toml::to_string_pretty(&new_cfg).unwrap()).expect("rewrite");

        // Poll up to 5 s for the callback to fire with top_k=9.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if observed.lock().unwrap().contains(&9) {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "watch_with callback never observed top_k=9; saw {:?}",
                    observed.lock().unwrap(),
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}
