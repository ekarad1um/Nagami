//! Mic settings store + handle traits + the
//! `MicSettingsCell` impl.
//!
//! The trait surface lives in `config` (not `common`) because
//! its method signatures reference `crate::audio_io::mic_arbitrator::MicPolicy`
//! and `MicSettings` -- types that can't move into `common`
//! without violating Invariant 4 (common has zero workspace
//! deps; pulling in `audio_io` would transitively bring `alsa`,
//! `rubato`, etc.).  `config` already depends on `audio_io`, so
//! the trait + impl colocating here is the natural layering.
//!
//! ## Three actors share the live MicSettings
//!
//! 1. **Daemon** owns the cell at boot, watches config reloads.
//! 2. **Arbitrator** reads it per-iteration via
//!    `Arc<dyn MicSettingsStore>` (read-only).
//! 3. **API** mutates on `POST /mic/policy` and reads on
//!    `GET /mic` via `Arc<dyn MicSettingsHandle>` (read+write).
//!
//! ## Trait upcasting
//!
//! `MicSettingsHandle: MicSettingsStore` lets the API hold the
//! superset trait while passing the same concrete cell to the
//! arbitrator as the read-only subset (via separate trait-object
//! `Arc`s constructed at boot).
//!
//! ## Why concrete `try_set_policy`, not generic `try_mutate`
//!
//! `impl FnOnce` in argument position desugars to a generic,
//! making the trait non-object-safe.  The concrete form
//! `try_set_policy(MicPolicy)` matches the only mutation
//! today's API performs (POST /mic/policy carries a MicPolicy);
//! the catalogue is launch-immutable.

use crate::audio_io::mic_arbitrator::{MicCatalogue, MicPolicy, MicSettings, MicSettingsStore};
use crate::common::error::{Categorized, ErrorKind};
use crate::common::version::{ResourceVersion, SwapReceipt, VersionedSwap};
use crate::config::ConfigError;
use std::sync::Arc;

/// Errors `MicSettingsHandle::try_set_policy` can return.
#[derive(Debug, thiserror::Error)]
pub enum MicError {
    /// Cross-validation of the new policy against the launch
    /// catalogue failed (typically: `Fixed { id }` references a
    /// candidate that isn't in the catalogue).
    #[error("mic policy rejected: {0}")]
    Rejected(String),
    /// Persistence to the user-config TOML failed.  Wraps the
    /// underlying `ConfigError`.
    #[error("persist mic policy: {0}")]
    Persist(#[from] ConfigError),
}

impl Categorized for MicError {
    fn kind(&self) -> ErrorKind {
        match self {
            // Operator-supplied policy didn't satisfy a cross-
            // validation predicate -- request shape was valid but
            // content failed against runtime state.
            MicError::Rejected(_) => ErrorKind::UserInput,
            MicError::Persist(e) => e.kind(),
        }
    }
}

/// Read + atomic mutate-and-persist surface.  Implemented by
/// [`MicSettingsCell`]; held by the api as
/// `Arc<dyn MicSettingsHandle>`.
///
/// Extends [`crate::audio_io::mic_arbitrator::MicSettingsStore`] (the
/// read-only side) so a single concrete cell satisfies both
/// trait objects: the api gets `Arc<dyn MicSettingsHandle>` for
/// read+write, the arbitrator gets `Arc<dyn MicSettingsStore>`
/// for wait-free reads.  `MicSettingsStore` lives in `audio_io`
/// because its return type (`Arc<MicSettings>`) is defined there;
/// pulling it into `common` would force `common` to take
/// `audio_io`'s deps, violating Invariant 4.
pub trait MicSettingsHandle: MicSettingsStore {
    /// Atomic policy swap + persistence.  The new policy is
    /// validated against the launch catalogue under the
    /// underlying `VersionedSwap`'s writer mutex; on success the
    /// new value is published to all readers and persisted to
    /// the user-config TOML.  Returns the post-mutation
    /// `SwapReceipt` for read-your-write semantics.
    ///
    /// Blocking: persistence step writes a TOML file + fsyncs
    /// the parent directory (~us-ms).  Callers in async contexts
    /// should wrap in `tokio::task::spawn_blocking`.
    fn try_set_policy(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError>;

    /// Apply a policy update WITHOUT persisting it back to the
    /// user-config TOML.  Used by the `ConfigCell` watcher path:
    /// when the operator edits the on-disk TOML, the watcher loads,
    /// parses, and cross-validates the new policy (already on disk)
    /// and publishes it to the live cell.  Calling `try_set_policy`
    /// from there would re-enter `ConfigCell::mutate` and trip the
    /// `IN_MUTATE` reentrancy guard.  The on-disk value IS the
    /// source of truth in the watcher path, so the persist step is
    /// skipped.
    ///
    /// API code paths must NOT call this: they need persistence
    /// to keep the operator's TOML in sync with the runtime
    /// state.
    fn try_set_policy_no_persist(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError>;
}

/// Production impl.  Owns the `VersionedSwap<MicSettings>`
/// (the in-memory hot-swap cell) plus a back-handle to the
/// `ConfigHandle` for persistence.  The launch-immutable
/// `MicCatalogue` is stored separately and re-bound into every
/// new `MicSettings` on mutation -- the catalogue Arc is shared
/// across snapshots so its memory is amortized.
///
/// Constructed once at daemon boot.  `Arc<MicSettingsCell>` is
/// the natural shape; both `Arc<dyn MicSettingsStore>` and
/// `Arc<dyn MicSettingsHandle>` cast from the same underlying
/// `Arc<MicSettingsCell>` (cheap Arc clone).
#[derive(Debug)]
pub struct MicSettingsCell {
    /// In-memory hot-swap cell.  Reads are wait-free; writes
    /// serialise through the writer mutex (per `VersionedSwap`).
    inner: VersionedSwap<MicSettings>,
    /// Launch-immutable mic catalogue.  Reused (Arc clone) when
    /// constructing new `MicSettings` snapshots on mutation.
    catalogue: Arc<MicCatalogue>,
    /// Back-handle to the user-config persistence layer.
    /// Mutations write the new policy to disk via
    /// `ConfigHandle::mutate`.  The handle is wait-free-clone
    /// (Arc internally).
    config: Arc<crate::config::ConfigCell>,
}

impl MicSettingsCell {
    /// Construct from the launch-time catalogue + the initial
    /// user-pref policy.  The two are bundled into the first
    /// `MicSettings` snapshot at version 0.
    pub fn new(
        catalogue: Arc<MicCatalogue>,
        initial_policy: MicPolicy,
        config: Arc<crate::config::ConfigCell>,
    ) -> Self {
        let initial = MicSettings {
            catalogue: catalogue.clone(),
            policy: initial_policy,
        };
        Self {
            inner: VersionedSwap::new(initial),
            catalogue,
            config,
        }
    }
}

impl MicSettingsStore for MicSettingsCell {
    fn snapshot(&self) -> Arc<MicSettings> {
        self.inner.snapshot()
    }

    fn version(&self) -> ResourceVersion {
        self.inner.version()
    }

    fn snapshot_with_version(&self) -> (Arc<MicSettings>, ResourceVersion) {
        // True atomic pair: a single `VersionedSwap` load gives
        // both halves from the same snapshot guard, so a
        // concurrent `try_set_policy` can't slip between them
        // and pair the OLD policy with the NEW version.
        self.inner.snapshot_with_version()
    }
}

impl MicSettingsHandle for MicSettingsCell {
    fn try_set_policy(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError> {
        // Cross-validate the new policy against the launch-time
        // catalogue.  The catalogue is immutable for the daemon's
        // lifetime; this is the same predicate the existing
        // user-pref hot-reload path uses (see
        // `validate_policy_against_catalogue`).
        crate::config::launch::validate_policy_against_catalogue(
            &policy,
            &self.catalogue,
            self.config.path(),
        )
        .map_err(|e| MicError::Rejected(e.to_string()))?;

        // Persist + in-memory swap atomic under one
        // `mutate_lock` critical section.  Three invariants this
        // shape preserves:
        //
        // 1. **Failure-path honesty.** If
        //    `write_toml_atomically` fails inside `mutate_then`,
        //    the `after` callback never runs -> in-memory cell
        //    stays at N-1.  `Err` iff neither disk nor mem
        //    changed.  Swap-first ordering would leave mem at N
        //    while the API returned Err -- operator-confusing
        //    when eMMC write hiccups are rare but real.
        //
        // 2. **Concurrent-drift prevention.** `mutate_then`'s own
        //    doc calls out the failure mode this shape closes:
        //      `config.mutate(...)?; runtime.store(...)` lets two
        //      concurrent api handlers interleave their persist +
        //      runtime.store steps, leaving disk = T2 / mem = T1.
        //    Mic POSTs go through `spawn_blocking`, so concurrent
        //    calls are real.
        //    Putting the in-memory swap inside `mutate_then`'s
        //    `after` callback (which runs while `mutate_lock` is
        //    still held) collapses persist + swap into one
        //    critical section, eliminating the interleave window.
        //
        // 3. **No fsync under the VersionedSwap writer mutex.**
        //    `write_toml_atomically` runs inside `mutate_lock`
        //    (a separate parking_lot::Mutex on `ConfigCell`), NOT
        //    inside the closure passed to `inner.try_mutate`.  The
        //    original "no fsync under the in-memory writer mutex"
        //    invariant is preserved -- the in-memory writer mutex
        //    is only held for ~us during the after-callback's
        //    Arc::new + ArcSwap::store.
        //
        // Tradeoff: readers see the old policy for the persist
        // window (~ms on eMMC) before the in-memory swap completes.
        // The mic_arbitrator polls the cell each pump period
        // (~10 ms), so this lag is bounded to one period -- well
        // below human-perceptible latency for a UI-initiated
        // config change.
        //
        // The `after` callback can't return values, so we ferry
        // the SwapReceipt out via a borrow-cell.  The cell is
        // populated only on the success path (after persist +
        // swap completed); a missing receipt at the end would be
        // a contract violation in `mutate_then` itself.
        let receipt_slot: std::cell::RefCell<Option<SwapReceipt>> = std::cell::RefCell::new(None);
        self.config.mutate_then(
            |c| {
                c.mic = policy.clone();
            },
            |_committed| {
                // mutate_lock still held; persist already
                // succeeded.  Now publish to the in-memory cell --
                // atomic-from-any-other-handler's-perspective with
                // the persist that just landed.
                //
                // Validation ran above + persist succeeded, so
                // the inner closure is infallible by construction.
                let new_settings = MicSettings {
                    catalogue: self.catalogue.clone(),
                    policy: policy.clone(),
                };
                let (receipt, _) = self
                    .inner
                    .try_mutate::<(), MicError>(|_cur| Ok((Arc::new(new_settings), ())))
                    .expect("infallible mutator");
                *receipt_slot.borrow_mut() = Some(receipt);
            },
        )?;

        Ok(receipt_slot
            .into_inner()
            .expect("mutate_then's after callback must run on the Ok path"))
    }

    fn try_set_policy_no_persist(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError> {
        // Same validate-then-swap as `try_set_policy`, minus the
        // `self.config.mutate(...)` step.  The watcher path calls
        // this from inside `ConfigCell::mutate`'s callback;
        // re-entering `mutate` would trip the `IN_MUTATE` guard.
        // The on-disk TOML is already the source of truth here
        // (the watcher reacted to a disk edit), so persisting
        // would be redundant.
        crate::config::launch::validate_policy_against_catalogue(
            &policy,
            &self.catalogue,
            self.config.path(),
        )
        .map_err(|e| MicError::Rejected(e.to_string()))?;

        let new_settings = MicSettings {
            catalogue: self.catalogue.clone(),
            policy,
        };
        let (receipt, _) = self
            .inner
            .try_mutate::<(), MicError>(|_cur| Ok((Arc::new(new_settings), ())))
            .expect("infallible mutator");

        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::mic_arbitrator::{ChannelSelection, MicSelection};

    /// Failure-path contract: when the persist step fails
    /// (e.g. the config file's parent dir is unwritable),
    /// `try_set_policy` must return Err AND leave the in-memory
    /// snapshot unchanged.  The
    /// swap-first ordering had the OPPOSITE behavior: the
    /// in-memory cell took the change while the API returned Err, so
    /// the operator believed nothing happened while the live mic kept
    /// applying the new policy until the next daemon restart.
    ///
    /// Setup uses `chmod 0o555` on the config-file parent dir so
    /// `write_toml_atomically`'s `tempfile::NamedTempFile::new_in`
    /// fails with EACCES at staging time.  Restores 0o755 in the
    /// cleanup-on-Drop guard so the tempdir can be deleted.
    #[cfg(unix)]
    #[test]
    fn try_set_policy_persist_failure_leaves_in_memory_unchanged() {
        use std::os::unix::fs::PermissionsExt;

        // RAII guard: restore writable perms on drop so tempdir
        // cleanup can succeed even on test panic.
        struct RestorePerms(std::path::PathBuf);
        impl Drop for RestorePerms {
            fn drop(&mut self) {
                let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
            }
        }

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmpdir.path().join("config.toml");
        let mut initial_cfg = crate::config::Config::default_for(tmpdir.path().join("workspaces"));
        // `Config::default_for` builds
        // `<workspace_root>/var/run/acoustics_lab.sock`.
        // Production's `load_or_init_config` mkdirs `var/run/`
        // before `validate()`; this test calls `from_value`
        // directly, so patch to a tempdir-relative socket so
        // `validate_uds_path` accepts it.
        initial_cfg.stream.uds_path = tmpdir.path().join("test.sock");
        let initial_policy = initial_cfg.mic.clone();
        let cfg_cell = Arc::new(
            crate::config::ConfigCell::from_value(initial_cfg, cfg_path)
                .expect("valid initial config"),
        );

        let launch = crate::config::LaunchConfig::default_for();
        let mic_cell = MicSettingsCell::new(Arc::new(launch.mic), initial_policy.clone(), cfg_cell);

        // Sanity check: initial snapshot reflects the initial policy.
        let pre = mic_cell.snapshot();
        assert_eq!(pre.policy, initial_policy);

        // Make the parent dir read-only so write_toml_atomically
        // fails at the tempfile-staging step.
        std::fs::set_permissions(tmpdir.path(), std::fs::Permissions::from_mode(0o555))
            .expect("chmod 555");
        let _restore = RestorePerms(tmpdir.path().to_path_buf());

        // A different-but-valid policy: same default-mock candidate,
        // but switch channel from Auto -> Fixed{0} (channel 0 is in
        // the catalogue's whitelist, so catalogue-validation passes).
        let new_policy = MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Fixed { channel: 0 },
        };
        assert_ne!(
            new_policy, initial_policy,
            "test target must differ from initial"
        );

        let err = mic_cell
            .try_set_policy(new_policy.clone())
            .expect_err("persist must fail under chmod 555");
        assert!(
            matches!(err, MicError::Persist(_)),
            "expected MicError::Persist, got {err:?}",
        );

        // Critical invariant: in-memory snapshot must STILL
        // reflect the initial policy, NOT the failed-to-persist
        // new_policy. swap-first ordering would have
        // updated this to new_policy.
        let post = mic_cell.snapshot();
        assert_eq!(
            post.policy, initial_policy,
            "in-memory must stay at initial policy when persist fails; \
             got {:?}",
            post.policy,
        );
    }
}
