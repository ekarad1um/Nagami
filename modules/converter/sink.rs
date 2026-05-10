//! `ArtifactSink` trait + the production [`MpkSink`]
//! impl.
//!
//! Today's only sink emits Burn `head.mpk` + `labels.txt` +
//! `metadata.json`; future sinks (e.g. on-device-only
//! single-file format) plug in here.
//!
//! Every artifact write routes through
//! [`crate::file_mgr::FsService::put_atomic`]: the [`MpkSink::publish`]
//! body builds the `head.mpk` payload (header + Burn-recorder
//! bytes) entirely in memory via [`burn::record::NamedMpkBytesRecorder`],
//! then drives one atomic-write per file.  No more `head-partial-*`
//! or `head-with-header-partial-*` siblings under the dst dir
//! between recorder return and final rename.

use crate::converter::source::LoadedSource;
use crate::converter::{ConvertError, HeadArtifacts};
use crate::file_mgr::FsService;
use std::path::Path;
use std::sync::Arc;

/// Publisher of one set of converted head artifacts to durable
/// storage.  Sinks own the on-disk layout decision (subdir
/// names, atomic-write protocol, metadata file emission); the
/// [`crate::converter::Pipeline`] composes a [`crate::converter::source::SourceModel`]
/// with one of these.
pub trait ArtifactSink: Send + Sync + std::fmt::Debug {
    /// Write all artifacts under `dst_dir` via `fs.put_atomic`
    /// (per-file tempfile + sync_all + rename + parent-dir
    /// fsync).  `metadata.json` is written LAST so its presence
    /// is the consistency marker -- a crash between `head.mpk`
    /// and `metadata.json` leaves a workspace without
    /// metadata.json, which downstream loaders treat as "not
    /// yet converted".
    ///
    /// `loaded.source_sha256` is forwarded into the emitted
    /// metadata so callers don't need to re-hash.
    ///
    /// `fs` is taken as `&Arc<dyn FsService>` rather than
    /// `&dyn FsService` so impls that want to stash the handle
    /// (or pass it onward to a sub-sink) can `Arc::clone` it
    /// without re-acquiring from the surrounding state.
    fn publish(
        &self,
        loaded: &LoadedSource,
        labels: &[String],
        dst_dir: &Path,
        source_kind: crate::converter::SourceKind,
        fs: &Arc<dyn FsService>,
    ) -> Result<HeadArtifacts, ConvertError>;
}

/// Burn-`.mpk` + labels.txt + metadata.json sink.  Stateless
/// unit struct; the on-disk layout + crash-safety protocol live
/// in [`crate::converter::write_head_artifacts`] which this delegates to.
#[derive(Clone, Copy, Debug, Default)]
pub struct MpkSink;

// Object-safety smoke for the converter trait pair.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn ArtifactSink>();
    assert_obj_safe::<dyn crate::converter::source::SourceModel>();
};

impl ArtifactSink for MpkSink {
    fn publish(
        &self,
        loaded: &LoadedSource,
        labels: &[String],
        dst_dir: &Path,
        source_kind: crate::converter::SourceKind,
        fs: &Arc<dyn FsService>,
    ) -> Result<HeadArtifacts, ConvertError> {
        crate::converter::write_head_artifacts(
            &loaded.weights,
            labels,
            dst_dir,
            source_kind,
            loaded.source_sha256.clone(),
            fs.as_ref(),
        )
    }
}
