//! `Pipeline<S, K>`: composes one
//! [`crate::converter::source::SourceModel`] with one
//! [`crate::converter::sink::ArtifactSink`].  The only current
//! instantiation is `Pipeline<TfjsSource, MpkSink>`; a future ONNX
//! source or single-file sink plugs in by adding a new (S, K) pair.
//!
//! The generic shape monomorphizes per (S, K); with one pair active
//! the binary-size impact is trivial.  Past ~4 sources x ~2 sinks the
//! trait-object form
//! `Pipeline { source: Box<dyn SourceModel>, sink: Box<dyn ArtifactSink> }`
//! becomes preferable.
//!
//! `Pipeline::run` forwards the `Arc<dyn FsService>` into
//! `ArtifactSink::publish` so the sink drives every atomic write
//! through one canonical primitive
//! ([`crate::file_mgr::fs_atomic::put_atomic`]).

use crate::converter::sink::ArtifactSink;
use crate::converter::source::SourceModel;
use crate::converter::{ConvertError, HeadArtifacts};
use std::path::Path;
use std::sync::Arc;

/// Compose a source-model loader + an artifact sink.  The
/// `Arc<dyn FsService>` is forwarded into `sink.publish` on every
/// `run` call so the sink can route its atomic writes through the
/// canonical [`crate::file_mgr::FsService::put_atomic`] primitive.
pub struct Pipeline<S: SourceModel, K: ArtifactSink> {
    source: S,
    sink: K,
    fs: Arc<dyn crate::file_mgr::FsService>,
}

impl<S: SourceModel, K: ArtifactSink> std::fmt::Debug for Pipeline<S, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("source", &self.source)
            .field("sink", &self.sink)
            .finish_non_exhaustive()
    }
}

impl<S: SourceModel, K: ArtifactSink> Pipeline<S, K> {
    pub fn new(source: S, sink: K, fs: Arc<dyn crate::file_mgr::FsService>) -> Self {
        Self { source, sink, fs }
    }

    /// Load the source from `src`, publish to `dst_dir`, return
    /// the on-disk artifacts.  Sync; api callers wrap in
    /// `tokio::task::spawn_blocking` per the
    /// `FsService::install_from_path` precedent.
    pub fn run(
        &self,
        src: &Path,
        labels: &[String],
        dst_dir: &Path,
    ) -> Result<HeadArtifacts, ConvertError> {
        let loaded = self.source.load(src, labels)?;
        self.sink
            .publish(&loaded, labels, dst_dir, self.source.kind(), &self.fs)
    }
}
