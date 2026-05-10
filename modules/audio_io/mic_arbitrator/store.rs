//! Read-only `MicSettingsStore` trait.
//!
//! The arbitrator (in this module) needs a wait-free read of
//! the live [`MicSettings`].  The production impl
//! (`config::MicSettingsCell`) lives in `config` so it can
//! also own the writer mutex + persistence handle that
//! `MicSettingsHandle::try_set_policy` (also in `config`)
//! needs.
//!
//! The trait belongs here rather than in
//! [`crate::common::traits`] because its return type
//! ([`Arc<MicSettings>`]) is defined in `audio_io`; lifting
//! [`MicSettings`] into `common` would force `common` to take
//! dependencies on this module's ALSA / mock /
//! candidate-source machinery, which the contract module
//! forbids.

use crate::audio_io::mic_arbitrator::MicSettings;
use crate::common::version::ResourceVersion;
use std::sync::Arc;

/// Wait-free read-side view of the live mic settings
/// (catalogue + policy).  Implemented by
/// `config::MicSettingsCell` in production and by the mock
/// cells used by `audio_io`'s tests.
///
/// `Send + Sync + 'static` so an `Arc<dyn
/// MicSettingsStore>` can sit on the
/// [`crate::audio_io::mic_arbitrator::MicArbitrator`]'s
/// spawned thread state.
pub trait MicSettingsStore: Send + Sync + 'static {
    /// Wait-free read.  Returns an `Arc<MicSettings>`
    /// aliasing the current snapshot; safe to hold across
    /// mutations.
    fn snapshot(&self) -> Arc<MicSettings>;

    /// Wait-free version read.  Bumps on each successful
    /// mutation through the matching `MicSettingsHandle`.
    fn version(&self) -> ResourceVersion;

    /// Atomic `(snapshot, version)` pair read.  Required by
    /// `?min_version=N` GET handlers that would otherwise
    /// risk pairing a stale value with a newer version (or
    /// vice versa) if a writer slipped between separate
    /// `snapshot()` + `version()` calls.  Default impl falls
    /// back to two reads for adapters without true atomicity
    /// (see [`ArcSwapStore`]); the production
    /// `config::MicSettingsCell` overrides with a single
    /// `VersionedSwap::snapshot_with_version` load.
    fn snapshot_with_version(&self) -> (Arc<MicSettings>, ResourceVersion) {
        (self.snapshot(), self.version())
    }
}

/// Transitional adapter.  Lets callers that hold an
/// `Arc<ArcSwap<MicSettings>>` produce an `Arc<dyn
/// MicSettingsStore>` without committing to the cell shape
/// yet.  The version always reads as
/// [`ResourceVersion::ZERO`] since the bare
/// [`arc_swap::ArcSwap`] has no version semantics; callers
/// that need real versioning must wire the production
/// `config::MicSettingsCell`.
#[derive(Debug)]
pub struct ArcSwapStore(pub Arc<arc_swap::ArcSwap<MicSettings>>);

impl MicSettingsStore for ArcSwapStore {
    fn snapshot(&self) -> Arc<MicSettings> {
        self.0.load_full()
    }
    fn version(&self) -> ResourceVersion {
        ResourceVersion::ZERO
    }
}
