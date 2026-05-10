//! Classifier-head storage abstraction.
//!
//! `inference::HotHead` is the production impl: an `ArcSwap`-backed
//! [`crate::common::version::VersionedSwap`] of fp32 weights +
//! biases + labels + the head's stable identity.  The trait
//! surface lets `api`'s handlers (and tests) interact with the
//! head without depending on `inference` concretely; a
//! `MockHeadStore` in `api::tests` substitutes for the real one
//! without spinning up a Burn engine.
//!
//! The trait is intentionally narrow: a [`HeadStore::snapshot`]
//! for reads, a [`HeadStore::version`] for read-your-write
//! semantics (consumed by the `?min_version=N` query param), and
//! a [`HeadStore::try_swap`] for atomic load-and-install.  The
//! richer in-memory shape -- weights, biases, labels, n_classes
//! -- stays inside `inference::HeadInner`; the trait's
//! [`HeadView`] DTO exposes only what the API needs to render
//! `/inference` responses.

use crate::common::dims::BackboneFeatureDim;
use crate::common::ids::HeadId;
use crate::common::version::{ResourceVersion, SwapReceipt};
use std::path::PathBuf;
use std::sync::Arc;

/// Read-shape for the daemon's view of the active head.
///
/// Smaller than `inference::HeadInner` (which carries the
/// weight, bias, and label tensors): this DTO is what
/// `/inference` HTTP responses need, plus what the
/// [`HeadStore::try_swap`] success acknowledgment echoes.  The
/// fp32 weights stay in the engine's heap.  Operators wanting
/// the on-disk location of the active head read
/// `head_active.head_mpk` from `GET /config` -- the engine does
/// not retain the source path past load time, so plumbing it
/// onto this DTO would either lie (empty `PathBuf`) or duplicate
/// state already owned by `config`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeadView {
    /// Stable identity (UUID-v4 strict; see
    /// [`crate::common::ids::HeadId`]).
    pub head_id: HeadId,
    /// The backbone-feature dim the head consumes.  Cached via
    /// the [`BackboneFeatureDim`] newtype rather than a raw
    /// `usize` so any future change to the canonical value
    /// propagates here automatically.
    pub feature_dim: BackboneFeatureDim,
    /// Number of output classes (equal to `labels.len()` in the
    /// engine).
    pub num_classes: u32,
}

/// Write-shape: the inputs [`HeadStore::try_swap`] needs to load
/// and install a new head.
///
/// Carries paths (not in-memory weights) so callers don't have
/// to pre-load the head -- the impl runs the I/O internally and
/// the trait method bounds the work behind a single `try_swap`
/// call.  `head_id` is the operator-supplied identity stamped on
/// every emitted `InferenceFrame.head_id` after the swap.
#[derive(Clone, Debug)]
pub struct HeadCandidate {
    pub head_mpk: PathBuf,
    pub labels: PathBuf,
    pub head_id: HeadId,
}

/// Errors [`HeadStore::try_swap`] can return.
///
/// Four categories so the API layer can map a swap failure to
/// the right HTTP status without re-parsing the typed source:
/// missing file (404), content-validation failure (400),
/// non-NotFound I/O failure on operator-supplied paths (400),
/// and daemon-internal invariant violation (500).
#[derive(Debug, thiserror::Error)]
pub enum HeadStoreError {
    /// Referenced file is missing on disk.  The production impl
    /// surfaces this when [`std::io::ErrorKind::NotFound`] fires
    /// while reading `head_mpk` or `labels`.
    #[error("head not found: {path}")]
    NotFound { path: String },
    /// Bytes were readable but failed runtime validation:
    /// ACSTHEAD magic / CRC / `feature_dim` mismatch, schema
    /// too old, weight / bias / label shape mismatch, non-finite
    /// tensor entries, or an unrecognized class count.
    #[error("head content invalid: {source}")]
    InvalidContent {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// I/O failure other than `NotFound` -- typically permission
    /// denied or a transient filesystem error on an operator-
    /// supplied path.  The operator can fix the input and retry.
    #[error("head load failed: {source}")]
    LoadFailed {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Daemon-internal failure unrelated to caller input
    /// (e.g. Burn `TensorData::to_vec` returning an unexpected
    /// shape after a recorder upgrade).  Operator action cannot
    /// recover the request.
    #[error("head store internal: {source}")]
    Internal {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The trait surface does not expose the requested
    /// operation (e.g.
    /// [`HeadStore::install_prevalidated`] on an impl that did
    /// not override the default).  Maps to `Internal` at the
    /// HTTP layer; operator-facing only when a custom impl is
    /// wired in production by mistake.
    #[error("head store does not support this operation")]
    Unsupported,
}

impl crate::common::error::Categorized for HeadStoreError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            HeadStoreError::NotFound { .. } => NotFound,
            // Caller-supplied bytes (or the path they pointed at)
            // failed validation.  Both classify as UserInput so the
            // route returns 400 with the standard envelope.
            HeadStoreError::InvalidContent { .. } | HeadStoreError::LoadFailed { .. } => UserInput,
            HeadStoreError::Internal { .. } | HeadStoreError::Unsupported => Internal,
        }
    }
}

/// Read + swap surface for the active classifier head.
///
/// `Send + Sync + 'static` so an `Arc<dyn HeadStore>` can sit
/// on `api::AppState` and flow through axum handlers (which
/// require state to be `Clone + Send + Sync + 'static`).
///
/// All methods are `&self` -- the trait makes no claim about
/// interior mutability beyond what [`Self::try_swap`] does, and
/// that is serialised via the underlying
/// [`crate::common::version::VersionedSwap`]'s writer mutex.
pub trait HeadStore: Send + Sync + 'static {
    /// Read the current head view.  Wait-free; returns an
    /// `Arc<HeadView>` aliasing a snapshot.  The underlying
    /// weights stay in the engine's heap; this is just the
    /// read-shape DTO.
    fn snapshot(&self) -> Arc<HeadView>;

    /// Read the current resource version.  Used by `/inference`
    /// handlers to populate the response's `version` field and
    /// by `?min_version=N` query-param filters.
    fn version(&self) -> ResourceVersion;

    /// Atomic `(snapshot, version)`.  Default impl reads
    /// sequentially (non-atomic; a swap landing between the
    /// two reads can leave the returned pair inconsistent);
    /// concrete impls backed by a
    /// [`crate::common::version::VersionedSwap`] should override
    /// with the single-syscall primitive.
    fn snapshot_with_version(&self) -> (Arc<HeadView>, ResourceVersion) {
        (self.snapshot(), self.version())
    }

    /// Atomic load-and-install of a new head.
    ///
    /// Runs the I/O, parse, and validate inside the trait
    /// (callers don't pre-load); on success returns the
    /// post-mutation [`SwapReceipt`] so HTTP responses can echo
    /// the new version for read-your-write semantics.
    ///
    /// Blocking: typically ~5 ms (Burn `.mpk` parse +
    /// label-file read).  Async callers should wrap in
    /// `tokio::task::spawn_blocking`.
    fn try_swap(&self, candidate: HeadCandidate) -> Result<SwapReceipt, HeadStoreError>;

    /// Install a prevalidated runtime candidate.
    ///
    /// The activation flow per redesign §5 stages a new active
    /// generation, pre-loads the head on `spawn_blocking`, and
    /// installs the validated runtime candidate into `HotHead`
    /// only AFTER `current.json` is durable.  This method is
    /// the trait-surface install point: the `candidate` is a
    /// `Box<dyn Any + Send>` carrying the impl-specific
    /// pre-validated runtime state (production: an
    /// `inference::HeadInner`; tests: any
    /// implementation-defined token).  The default impl
    /// returns [`HeadStoreError::Unsupported`]; the production
    /// `HotHead` impl overrides.
    ///
    /// Layering note: the trait lives in `common` (frozen
    /// foundation crate); `HeadInner` lives in `inference`
    /// downstream.  The `Box<dyn Any>` indirection sidesteps
    /// the cycle without inverting the layer order.  The
    /// production impl downcasts to `HeadInner` and returns
    /// `Unsupported` on a type mismatch -- catches caller bugs
    /// without leaking impl details into the trait.
    fn install_prevalidated(
        &self,
        _candidate: Box<dyn std::any::Any + Send>,
    ) -> Result<SwapReceipt, HeadStoreError> {
        Err(HeadStoreError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::{Categorized, ErrorKind};

    #[derive(Debug, thiserror::Error)]
    #[error("synthetic")]
    struct Synthetic;

    #[test]
    fn categorized_maps_each_variant_to_intended_http_class() {
        let cases: [(HeadStoreError, ErrorKind, u16); 5] = [
            (
                HeadStoreError::NotFound {
                    path: "/missing".into(),
                },
                ErrorKind::NotFound,
                404,
            ),
            (
                HeadStoreError::InvalidContent {
                    source: Box::new(Synthetic),
                },
                ErrorKind::UserInput,
                400,
            ),
            (
                HeadStoreError::LoadFailed {
                    source: Box::new(Synthetic),
                },
                ErrorKind::UserInput,
                400,
            ),
            (
                HeadStoreError::Internal {
                    source: Box::new(Synthetic),
                },
                ErrorKind::Internal,
                500,
            ),
            (HeadStoreError::Unsupported, ErrorKind::Internal, 500),
        ];
        for (err, expected_kind, expected_status) in cases {
            assert_eq!(err.kind(), expected_kind, "variant kind mismatch for {err}",);
            assert_eq!(
                err.kind().http_status_code(),
                expected_status,
                "variant http status mismatch for {err}",
            );
        }
    }
}
