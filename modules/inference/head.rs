//! Hot-swappable classifier head.
//!
//! [`HotHead`] is the side-channel handle through which
//! the daemon's API layer atomically swaps weights without
//! restarting the inference engine.  Cheaply cloneable
//! (`Send + Sync + Clone`) so the engine, the API, and the
//! file manager can each hold an owned reference to the same
//! swap point.  Internally it wraps an
//! [`Arc<VersionedSwap<HeadInner>>`]:
//!
//! - [`crate::common::version::VersionedSwap`] adds a
//!   monotonic [`ResourceVersion`] (returned in
//!   [`SwapReceipt`] for read-your-write semantics) and a
//!   writer mutex so concurrent `try_swap`s linearise.
//! - Reads stay wait-free: [`HotHead::snapshot`] resolves
//!   to one [`arc_swap::ArcSwap::load_full`] under the hood
//!   (~5 ns, atomic).
//! - [`HeadInner`] bundles all swap-coupled fields (weight,
//!   bias, labels, head_id, `n_classes`) so a mid-flight
//!   swap can never publish a state where the engine reads
//!   new labels against the old weights.
//!
//! Loading parses the [`crate::common::head_header`]
//! (`ACSTHEAD`) prefix, then delegates the payload bytes to
//! [`Head::load_mpk_bytes`] (which wraps Burn's
//! `NamedMpkBytesRecorder`).  Blocking I/O + small CPU;
//! callers MUST invoke `load` / `swap` from
//! [`tokio::task::spawn_blocking`] or a non-async context.

use crate::model::Head;
use burn::backend::NdArray;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use crate::common::dims::BackboneFeatureDim;
use crate::common::ids::HeadId;
use crate::common::traits::head_store::{HeadCandidate, HeadStore, HeadStoreError, HeadView};
use crate::common::version::{ResourceVersion, SwapReceipt, VersionedSwap};

// File-local alias so thiserror's `#[error("...
// {BACKBONE_FEATURE_DIM} ...")]` format-string syntax
// (which can only interpolate bare idents, not
// `Type::CONST` paths) stays readable + sourced from the contract crate.
const BACKBONE_FEATURE_DIM: usize = BackboneFeatureDim::USIZE;

/// Backend used only on the cold path (head load + bias hoist).  The
/// engine bypasses Burn in steady state via `kernel::head_forward`.
type B = NdArray<f32>;

/// One self-consistent snapshot of head weights + metadata.
/// `Clone` is supported but DEEP -- it copies the weight + bias +
/// labels Vecs (~136 KB for a 17-class head).  Hot-path code uses
/// [`HotHead::snapshot`] which returns an `Arc<HeadInner>` shared
/// with all readers; `Clone` is intended for cold paths (e.g. the
/// activation pipeline staging a candidate before publish).
#[derive(Debug, Clone)]
pub struct HeadInner {
    /// Row-major `[BACKBONE_FEATURE_DIM x n_classes]` flat fp32.
    /// Burn stores Linear weights as `[in_features, out_features]`; we
    /// keep the same orientation here.  `kernel::head_forward` walks
    /// `weight.chunks_exact(n_classes)` so any drift would be caught
    /// by the `weight.len() == BACKBONE_FEATURE_DIM x n` invariant in
    /// `load_inner`.
    pub weight: Vec<f32>,

    /// Per-class bias (`len == n_classes`).  Heads always
    /// carry a bias today (Burn's `LinearConfig` defaults
    /// to `bias = true`); a bias-free head would have zeros
    /// synthesised at load so [`crate::inference::head_forward`]
    /// stays uniform.
    pub bias: Vec<f32>,

    /// Human-readable class labels (`len == n_classes`).
    /// Loaded from a `labels.txt`-style file, one label
    /// per line; blank lines are stripped before the count
    /// check, and a count mismatch is rejected with
    /// [`HeadError::LabelCountMismatch`] at load time so
    /// the hot path can index unconditionally.
    pub labels: Vec<String>,

    /// Stable identity of this head.  UUID-v4 strict
    /// (validated at construction by [`HeadId`]); `Copy`
    /// (16-byte Uuid wrapper) so per-frame
    /// `snapshot.head_id` reads are register-free.  Wire
    /// encode still allocates a `String` once per frame
    /// because `proto::InferenceFrame.head_id` is `String`.
    pub head_id: HeadId,

    /// Number of output classes.  Cached to avoid
    /// recomputing `weight.len() / BackboneFeatureDim::USIZE`
    /// at every snapshot.
    pub n_classes: usize,
}

/// Shareable head handle.  Cheaply cloned (one [`Arc`]
/// bump).  All clones observe each other's swaps; there is
/// exactly one underlying hot-state cell per logical head.
///
/// Storage is `Arc<VersionedSwap<HeadInner>>`.  The
/// [`crate::common::version::VersionedSwap`] wrapper adds:
///
/// - a monotonic [`ResourceVersion`] stamped on every
///   successful mutation, so the API layer can return a
///   [`SwapReceipt`] that supports `?min_version=N`
///   read-your-write semantics;
/// - a writer mutex so two concurrent `try_swap` calls
///   linearise rather than racing the load-then-store
///   cycle.
///
/// Reads ([`Self::snapshot`]) stay wait-free.
#[derive(Clone, Debug)]
pub struct HotHead {
    inner: Arc<VersionedSwap<HeadInner>>,
}

/// Failure shapes from [`HotHead::load`] and
/// [`HotHead::swap`].
#[derive(Debug, Error)]
pub enum HeadError {
    #[error("read head .mpk {path}: {message}")]
    LoadMpk {
        path: String,
        // Burn returns a domain error type that is `Display` but not
        // necessarily `'static` in older versions, and thiserror's
        // `#[source]` requires `std::error::Error`.  Render to String
        // at construction; we lose the typed source but keep the
        // operator-facing message.
        message: String,
    },
    #[error(
        "head .mpk weight has {got} elements, expected {expected} ({BACKBONE_FEATURE_DIM} x n_classes)"
    )]
    WeightShape { got: usize, expected: usize },
    #[error("head .mpk bias has {got} elements, expected {expected} (= n_classes)")]
    BiasShape { got: usize, expected: usize },
    #[error("head .mpk linear input dim is {got}, expected {BACKBONE_FEATURE_DIM}")]
    InputDim { got: usize },
    #[error("head .mpk produced n_classes = {got}; refusing (must be > 0 and <= {max})")]
    BadClassCount { got: usize, max: usize },
    #[error("read labels file {path}: {source}")]
    ReadLabels {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("read head .mpk {path}: {source}")]
    ReadHeadMpk {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "labels file {path} has {got} entries but head has {n_classes} classes \
         (mismatch -- labels are required to align)"
    )]
    LabelCountMismatch {
        path: String,
        got: usize,
        n_classes: usize,
    },
    /// `HeadInner::validate` rejection
    /// path.  Mirrors the file-side `LabelCountMismatch` but
    /// without a path because the caller has the inner already
    /// in memory (no .mpk on disk to point at).
    #[error("HeadInner labels has {got} entries but n_classes is {n_classes}")]
    LabelShape { got: usize, n_classes: usize },
    /// Head weight at flat index `idx` is non-finite (NaN /
    /// +Inf / -Inf).  Detected on the cold path by
    /// [`HeadInner::validate`]; rejection here keeps
    /// `head_forward` and `softmax_into` from propagating
    /// NaN into emitted `InferenceFrame`s.
    #[error("head weight[{idx}] = {value} is not finite")]
    NonFiniteWeight { idx: usize, value: f32 },
    /// Head bias at index `idx` is non-finite.  Same
    /// rationale as [`HeadError::NonFiniteWeight`].
    #[error("head bias[{idx}] = {value} is not finite")]
    NonFiniteBias { idx: usize, value: f32 },
    #[error("internal: tensor.into_data().to_vec failed: {0}")]
    TensorIntoVec(String),
    /// `.mpk` is missing the `ACSTHEAD` magic header
    /// (typically a headerless artifact).  Operator
    /// regenerates via the converter.
    #[error(
        "head .mpk {path} predates the ACSTHEAD persistence header -- \
         regenerate via the converter (POST /api/v1/converter)"
    )]
    SchemaTooOld { path: String },
    /// `.mpk` header parse error (bad CRC, schema too
    /// new, truncated).  Distinct from `SchemaTooOld` which is
    /// the "no header at all" path; here the header bytes are
    /// present but malformed.
    #[error("head .mpk {path} header corrupt: {reason}")]
    HeaderCorrupt { path: String, reason: String },
    /// `.mpk` header carries `feature_dim={got}` which
    /// disagrees with this build's backbone.  Either the head
    /// was extracted for a different topology or a corrupt
    /// header slipped through CRC validation.
    #[error("head .mpk {path} feature_dim {got} != build's {expected}")]
    HeaderFeatureDimMismatch {
        path: String,
        got: u32,
        expected: u32,
    },
}

impl crate::common::error::Categorized for HeadError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // The supplied .mpk / labels file is malformed or
            // doesn't match the contract; operator can re-export
            // and retry.
            HeadError::LoadMpk { .. }
            | HeadError::WeightShape { .. }
            | HeadError::BiasShape { .. }
            | HeadError::InputDim { .. }
            | HeadError::BadClassCount { .. }
            | HeadError::LabelCountMismatch { .. }
            | HeadError::LabelShape { .. }
            | HeadError::NonFiniteWeight { .. }
            | HeadError::NonFiniteBias { .. }
            | HeadError::SchemaTooOld { .. }
            | HeadError::HeaderCorrupt { .. }
            | HeadError::HeaderFeatureDimMismatch { .. } => UserInput,
            // Operator-facing IO; reading the labels file or head
            // .mpk failed for reasons we can't see through
            // (permission, missing file, etc.).
            HeadError::ReadLabels { .. } | HeadError::ReadHeadMpk { .. } => Internal,
            // Burn TensorData decode that succeeded shape-wise
            // but failed in a way that suggests Burn version drift;
            // not the operator's fault.
            HeadError::TensorIntoVec(_) => Internal,
        }
    }
}

/// Re-export of [`crate::common::dims::MAX_N_CLASSES`].  Kept as
/// a public alias here because `inference.rs` re-exports
/// `MAX_N_CLASSES` from this module's path; the canonical
/// definition (and the ~800 MB headroom derivation) lives in
/// `common::dims`.
pub use crate::common::dims::MAX_N_CLASSES;

impl HeadInner {
    /// Assert every cold-path invariant `kernel::head_forward`,
    /// `softmax_into`, and the broadcast encode rely on.  Used
    /// by [`HotHead::store_inner`] / [`HotHead::try_from_inner`]
    /// to refuse malformed inputs before they reach the hot path.
    ///
    /// Invariants:
    ///   1. `n_classes in [1, MAX_N_CLASSES]`
    ///   2. `weight.len() == n_classes * BACKBONE_FEATURE_DIM`
    ///   3. `bias.len() == n_classes`
    ///   4. `labels.len() == n_classes`
    ///   5. every `weight[i]` is finite (no NaN / +-Inf)
    ///   6. every `bias[i]`   is finite
    ///
    /// `O(n_classes * BACKBONE_FEATURE_DIM)` finite scan; cold
    /// path only.  Splits into [`Self::validate_finite`] so the
    /// shape and finite halves can be tested separately.
    pub fn validate(&self) -> Result<(), HeadError> {
        if self.n_classes == 0 || self.n_classes > MAX_N_CLASSES {
            return Err(HeadError::BadClassCount {
                got: self.n_classes,
                max: MAX_N_CLASSES,
            });
        }
        let expected_weight = self.n_classes * BACKBONE_FEATURE_DIM;
        if self.weight.len() != expected_weight {
            return Err(HeadError::WeightShape {
                got: self.weight.len(),
                expected: expected_weight,
            });
        }
        if self.bias.len() != self.n_classes {
            return Err(HeadError::BiasShape {
                got: self.bias.len(),
                expected: self.n_classes,
            });
        }
        if self.labels.len() != self.n_classes {
            return Err(HeadError::LabelShape {
                got: self.labels.len(),
                n_classes: self.n_classes,
            });
        }
        self.validate_finite()
    }

    /// Linear scan over `weight` then `bias`, returning the
    /// flat index + value of the first non-finite (NaN / +Inf /
    /// -Inf) entry.  `O(n)`; cold path only.  Split out so
    /// callers can exercise the numeric half independently of
    /// the shape half (e.g. fuzzers, regression tests).
    pub fn validate_finite(&self) -> Result<(), HeadError> {
        for (idx, &v) in self.weight.iter().enumerate() {
            if !v.is_finite() {
                return Err(HeadError::NonFiniteWeight { idx, value: v });
            }
        }
        for (idx, &v) in self.bias.iter().enumerate() {
            if !v.is_finite() {
                return Err(HeadError::NonFiniteBias { idx, value: v });
            }
        }
        Ok(())
    }
}

impl HotHead {
    /// Build a head from disk.  Blocking -- call from spawn_blocking.
    pub fn load(head_mpk: &Path, labels: &Path, head_id: HeadId) -> Result<Self, HeadError> {
        let inner = load_inner(head_mpk, labels, head_id)?;
        // Defence in depth: `load_inner` already validated; this
        // second pass guards a future revision losing one of the
        // invariants.
        Self::try_from_inner(inner)
    }

    /// Atomic in-place swap.  Loads + validates, then
    /// rotates the [`crate::common::version::VersionedSwap`].
    /// The engine sees the new [`HeadInner`] on its next
    /// [`Self::snapshot`] call.
    ///
    /// The old `Arc<HeadInner>` is dropped only after the
    /// last outstanding snapshot guard releases: no
    /// use-after-free even if the engine is mid-frame at
    /// swap time.
    ///
    /// Returns a [`SwapReceipt`] carrying the post-mutation
    /// version so the API layer can echo it for
    /// read-your-write semantics.
    pub fn swap(
        &self,
        head_mpk: &Path,
        labels: &Path,
        head_id: HeadId,
    ) -> Result<SwapReceipt, HeadError> {
        let new = load_inner(head_mpk, labels, head_id)?;
        let (receipt, _) = self
            .inner
            .try_mutate::<(), HeadError>(|_cur| Ok((Arc::new(new), ())))
            .expect("infallible mutator");
        Ok(receipt)
    }

    /// Validating constructor: runs [`HeadInner::validate`]
    /// (shape + finite checks) before publishing.  Use this for
    /// any caller that owns a `HeadInner` constructed
    /// out-of-band (converter, tests, daemon's synthetic-dev
    /// fallback) -- it surfaces the diagnostic on the cold path
    /// instead of letting a malformed inner reach `head_forward`
    /// and panic the engine's hot loop.
    ///
    /// The legacy [`Self::from_inner`] also runs the validator
    /// today (panics on failure) for source-compat with existing
    /// trusted callers.
    pub fn try_from_inner(inner: HeadInner) -> Result<Self, HeadError> {
        inner.validate()?;
        Ok(Self {
            inner: Arc::new(VersionedSwap::new(inner)),
        })
    }

    /// Build a HotHead from a pre-constructed inner.  Validates
    /// via [`HeadInner::validate`] and panics on failure; prefer
    /// [`Self::try_from_inner`] for any input the caller did not
    /// hand-construct.
    pub fn from_inner(inner: HeadInner) -> Self {
        Self::try_from_inner(inner)
            .unwrap_or_else(|e| panic!("HotHead::from_inner: {e}; use try_from_inner"))
    }

    /// Replace the inner directly.  Mirrors `swap` but skips file I/O.
    /// Returns the post-mutation version receipt.
    ///
    /// Validates `inner` via [`HeadInner::validate`] before
    /// publishing.  A shape-inconsistent inner (e.g.
    /// `weight.len() != n_classes * BackboneFeatureDim::USIZE`)
    /// would otherwise trigger a hot-path panic in
    /// [`crate::inference::head_forward`]; surfacing the diagnostic here at
    /// the cold path keeps the engine's frame loop
    /// panic-free.
    pub fn store_inner(&self, inner: HeadInner) -> Result<SwapReceipt, HeadError> {
        inner.validate()?;
        let (receipt, _) = self
            .inner
            .try_mutate::<(), ()>(|_cur| Ok((Arc::new(inner), ())))
            .expect("infallible mutator");
        Ok(receipt)
    }

    /// One ArcSwap::load_full -> an `Arc<HeadInner>` aliasing the
    /// current snapshot. ~5 ns.  Hold for one frame.
    pub fn snapshot(&self) -> Arc<HeadInner> {
        self.inner.snapshot()
    }

    /// Current resource version.  Bumps on every successful swap.
    pub fn version(&self) -> ResourceVersion {
        self.inner.version()
    }

    /// Atomic `(snapshot, version)`.  Reading them
    /// separately would race a swap landing between the two
    /// loads, leading to a frame whose logits came from version
    /// N but whose stamped version reads N+1.  Surfaces the
    /// underlying `VersionedSwap` primitive so the inference
    /// engine can stamp `InferenceFrame.head_version` truthfully.
    pub fn snapshot_with_version(&self) -> (Arc<HeadInner>, ResourceVersion) {
        self.inner.snapshot_with_version()
    }

    /// Convenience: number of classes in the current snapshot.
    pub fn n_classes(&self) -> usize {
        self.snapshot().n_classes
    }
}

/// `HeadStore` impl backing `api::AppState::head: Arc<dyn
/// HeadStore>`.  Delegates reads to `HotHead::snapshot` /
/// `version`; `try_swap` builds a `HeadView` DTO from the freshly-
/// loaded `HeadInner` and runs the load+install under the
/// `VersionedSwap` writer mutex via `HotHead::swap`.
impl HeadStore for HotHead {
    fn snapshot(&self) -> Arc<HeadView> {
        let inner = self.snapshot();
        Arc::new(HeadView {
            head_id: inner.head_id,
            feature_dim: BackboneFeatureDim::default(),
            num_classes: inner.n_classes as u32,
        })
    }

    fn version(&self) -> ResourceVersion {
        self.version()
    }

    /// Atomic override via the underlying
    /// `VersionedSwap`'s `snapshot_with_version` primitive,
    /// avoiding the race-prone default impl.
    fn snapshot_with_version(&self) -> (Arc<HeadView>, ResourceVersion) {
        let (inner, version) = self.snapshot_with_version();
        let view = Arc::new(HeadView {
            head_id: inner.head_id,
            feature_dim: BackboneFeatureDim::default(),
            num_classes: inner.n_classes as u32,
        });
        (view, version)
    }

    fn try_swap(&self, candidate: HeadCandidate) -> Result<SwapReceipt, HeadStoreError> {
        self.swap(&candidate.head_mpk, &candidate.labels, candidate.head_id)
            .map_err(classify_head_error)
    }

    /// Install a prevalidated [`HeadInner`] consumed by the
    /// activation flow.  The activation flow stages a new
    /// generation, pre-loads + validates the `HeadInner` on
    /// `spawn_blocking`, then hands it to the trait surface for
    /// the runtime install AFTER `current.json` is durable.
    /// Downcasting from `dyn Any` keeps the trait crate
    /// (`common`) layered above the inference crate; a type
    /// mismatch surfaces as `HeadStoreError::Unsupported` so
    /// caller bugs do not silently no-op.
    fn install_prevalidated(
        &self,
        candidate: Box<dyn std::any::Any + Send>,
    ) -> Result<SwapReceipt, HeadStoreError> {
        let inner = candidate
            .downcast::<HeadInner>()
            .map_err(|_| HeadStoreError::Unsupported)?;
        // `store_inner` re-runs `HeadInner::validate` defence
        // in depth.  The activation pre-load already validated;
        // this second pass guards a future caller that hands in
        // a partially-constructed inner.
        self.store_inner(*inner).map_err(classify_head_error)
    }
}

/// Map a domain `HeadError` to the categorized `HeadStoreError`
/// the trait exposes.  The classification is intentionally
/// shallow: `NotFound` is reserved for `io::ErrorKind::NotFound`
/// surfacing through the two read paths, `Internal` for Burn
/// recorder drift, everything else collapses to `InvalidContent`
/// (content failed validation) or `LoadFailed` (other I/O on
/// operator-supplied paths).  The route layer maps each variant
/// to the correct HTTP status via `Categorized`.
fn classify_head_error(err: HeadError) -> HeadStoreError {
    use std::io::ErrorKind as Io;
    let io_not_found_path: Option<String> = match &err {
        HeadError::ReadHeadMpk { path, source } | HeadError::ReadLabels { path, source } => {
            (source.kind() == Io::NotFound).then(|| path.clone())
        }
        _ => None,
    };
    if let Some(path) = io_not_found_path {
        return HeadStoreError::NotFound { path };
    }
    match err {
        e @ HeadError::ReadHeadMpk { .. } | e @ HeadError::ReadLabels { .. } => {
            HeadStoreError::LoadFailed {
                source: Box::new(e),
            }
        }
        e @ HeadError::TensorIntoVec(_) => HeadStoreError::Internal {
            source: Box::new(e),
        },
        e => HeadStoreError::InvalidContent {
            source: Box::new(e),
        },
    }
}

fn load_inner(
    head_mpk: &Path,
    labels_path: &Path,
    head_id: HeadId,
) -> Result<HeadInner, HeadError> {
    let device: burn::tensor::Device<B> = Default::default();

    // Parse the `ACSTHEAD` header before delegating to
    // Burn's recorder.  The header validates magic + CRC +
    // feature_dim; the in-memory prost payload then goes
    // straight to Burn's `NamedMpkBytesRecorder` via
    // `Head::load_mpk_bytes` -- no FS round-trip.
    let bytes = std::fs::read(head_mpk).map_err(|e| HeadError::ReadHeadMpk {
        path: head_mpk.display().to_string(),
        source: e,
    })?;
    if bytes.len() < crate::common::head_header::HEAD_HEADER_SIZE {
        return Err(HeadError::SchemaTooOld {
            path: head_mpk.display().to_string(),
        });
    }
    let header = crate::common::head_header::parse_header(
        &bytes[..crate::common::head_header::HEAD_HEADER_SIZE],
    )
    .map_err(|e| match e {
        crate::common::head_header::HeadHeaderError::BadMagic { .. } => HeadError::SchemaTooOld {
            path: head_mpk.display().to_string(),
        },
        other => HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!("{other}"),
        },
    })?;
    if header.feature_dim as usize != BACKBONE_FEATURE_DIM {
        return Err(HeadError::HeaderFeatureDimMismatch {
            path: head_mpk.display().to_string(),
            got: header.feature_dim,
            expected: BACKBONE_FEATURE_DIM as u32,
        });
    }
    if bytes.len() - crate::common::head_header::HEAD_HEADER_SIZE != header.payload_len as usize {
        return Err(HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!(
                "header.payload_len={}, file_len-header_size={}",
                header.payload_len,
                bytes.len() - crate::common::head_header::HEAD_HEADER_SIZE,
            ),
        });
    }

    // Burn's recorder needs to own the payload bytes (see
    // `Head::load_mpk_bytes` doc).  `drain` shifts the
    // payload tail forward in place: it keeps the original
    // allocation but pays an O(payload_len) memmove.  An
    // O(1) `split_off(HEAD_HEADER_SIZE)` would hand back a
    // tail-only `Vec` if a future revision wants to skip
    // the shift.
    let mut payload = bytes;
    payload.drain(..crate::common::head_header::HEAD_HEADER_SIZE);

    let head = Head::<B>::load_mpk_bytes(payload, &device).map_err(|e| HeadError::LoadMpk {
        path: head_mpk.display().to_string(),
        message: format!("{e}"),
    })?;
    let weight_dims = head.linear.weight.val().dims();
    if weight_dims[0] != BACKBONE_FEATURE_DIM {
        return Err(HeadError::InputDim {
            got: weight_dims[0],
        });
    }
    let n_classes = weight_dims[1];
    if n_classes == 0 || n_classes > MAX_N_CLASSES {
        return Err(HeadError::BadClassCount {
            got: n_classes,
            max: MAX_N_CLASSES,
        });
    }

    // Hoist weights + bias out of Burn into flat `Vec<f32>`
    // for the hot path.
    let weight: Vec<f32> = head
        .linear
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| HeadError::TensorIntoVec(format!("{e:?}")))?;
    if weight.len() != BACKBONE_FEATURE_DIM * n_classes {
        return Err(HeadError::WeightShape {
            got: weight.len(),
            expected: BACKBONE_FEATURE_DIM * n_classes,
        });
    }

    let bias: Vec<f32> = match head.linear.bias.as_ref() {
        Some(b) => b
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| HeadError::TensorIntoVec(format!("{e:?}")))?,
        None => vec![0.0; n_classes],
    };
    // Release-checked: a corrupt .mpk with bias.len() != n_classes would
    // otherwise survive load and panic at the first frame in
    // `head_forward`'s `logits.copy_from_slice(bias)`.  Surface the
    // diagnostic on the cold path.
    if bias.len() != n_classes {
        return Err(HeadError::BiasShape {
            got: bias.len(),
            expected: n_classes,
        });
    }

    // Labels file.  Required to align with n_classes -- we make this an
    // error rather than a warning because the API needs a stable
    // contract (every TopK has a label, never `idx=N`).
    let labels_text = std::fs::read_to_string(labels_path).map_err(|e| HeadError::ReadLabels {
        path: labels_path.display().to_string(),
        source: e,
    })?;
    let labels: Vec<String> = labels_text
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if labels.len() != n_classes {
        return Err(HeadError::LabelCountMismatch {
            path: labels_path.display().to_string(),
            got: labels.len(),
            n_classes,
        });
    }

    let inner = HeadInner {
        weight,
        bias,
        labels,
        head_id,
        n_classes,
    };
    // Shape invariants were enforced inline above; the finite
    // scan catches NaN / +/-Inf weights or biases that would
    // otherwise propagate through `head_forward` ->
    // `softmax_into` and surface as NaN `prob` values inside
    // emitted `InferenceFrame`s (where `top_k_indices_into`
    // defensively treats NaN comparisons as Equal, masking the
    // upstream corruption).  Cold path; ~us at n_classes=17.
    inner.validate_finite()?;
    Ok(inner)
}

#[cfg(test)]
mod tests {
    // Tests stage `.mpk` fixtures via `std::fs::write` + the
    // raw Burn recorder; the production discipline (writes
    // route through file_mgr's atomic writer) doesn't apply
    // to test scaffolding.
    #![allow(clippy::disallowed_methods)]
    use super::*;

    fn synth_inner(n_classes: usize, head_id: HeadId) -> HeadInner {
        HeadInner {
            weight: vec![0.01; BACKBONE_FEATURE_DIM * n_classes],
            bias: vec![0.0; n_classes],
            labels: (0..n_classes).map(|i| format!("class_{i}")).collect(),
            head_id,
            n_classes,
        }
    }

    #[test]
    fn from_inner_round_trip() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 3);
        assert_eq!(snap.weight.len(), BACKBONE_FEATURE_DIM * 3);
        assert_eq!(snap.labels.len(), 3);
        assert_eq!(snap.head_id, id);
        assert_eq!(h.n_classes(), 3);
    }

    /// `HotHead::store_inner` rejects
    /// shape-inconsistent inners that would otherwise panic the
    /// engine's hot path.  Exercises every branch of
    /// `HeadInner::validate`.
    #[test]
    fn store_inner_validates_shape() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));

        // 1. n_classes = 0 -> BadClassCount.
        let bad_zero = HeadInner {
            weight: vec![],
            bias: vec![],
            labels: vec![],
            head_id: id,
            n_classes: 0,
        };
        let err = h
            .store_inner(bad_zero)
            .expect_err("n_classes=0 must reject");
        assert!(
            matches!(err, HeadError::BadClassCount { got: 0, .. }),
            "expected BadClassCount, got {err:?}",
        );

        // 2. n_classes > MAX_N_CLASSES -> BadClassCount.
        // Construct only the n_classes field oversized; the
        // shape mismatch on weight would otherwise mask
        // BadClassCount, but BadClassCount is checked first.
        let bad_huge = HeadInner {
            weight: vec![],
            bias: vec![],
            labels: vec![],
            head_id: id,
            n_classes: MAX_N_CLASSES + 1,
        };
        let err = h
            .store_inner(bad_huge)
            .expect_err("n_classes > MAX must reject");
        assert!(
            matches!(err, HeadError::BadClassCount { got, max } if got == MAX_N_CLASSES + 1 && max == MAX_N_CLASSES),
            "expected BadClassCount, got {err:?}",
        );

        // 3. weight.len() != n_classes * BACKBONE_FEATURE_DIM
        //    -> WeightShape.
        let bad_weight = HeadInner {
            weight: vec![0.0; 99], // wrong: should be 3 * BACKBONE_FEATURE_DIM
            bias: vec![0.0; 3],
            labels: vec!["a".into(), "b".into(), "c".into()],
            head_id: id,
            n_classes: 3,
        };
        let err = h
            .store_inner(bad_weight)
            .expect_err("weight shape must reject");
        assert!(
            matches!(err, HeadError::WeightShape { got: 99, expected }
                     if expected == 3 * BACKBONE_FEATURE_DIM),
            "expected WeightShape, got {err:?}",
        );

        // 4. bias.len() != n_classes -> BiasShape.
        let bad_bias = HeadInner {
            weight: vec![0.0; 3 * BACKBONE_FEATURE_DIM],
            bias: vec![0.0; 5], // wrong: should be 3
            labels: vec!["a".into(), "b".into(), "c".into()],
            head_id: id,
            n_classes: 3,
        };
        let err = h.store_inner(bad_bias).expect_err("bias shape must reject");
        assert!(
            matches!(
                err,
                HeadError::BiasShape {
                    got: 5,
                    expected: 3
                }
            ),
            "expected BiasShape, got {err:?}",
        );

        // 5. labels.len() != n_classes -> LabelShape.
        let bad_labels = HeadInner {
            weight: vec![0.0; 3 * BACKBONE_FEATURE_DIM],
            bias: vec![0.0; 3],
            labels: vec!["only_one".into()], // wrong: should be 3
            head_id: id,
            n_classes: 3,
        };
        let err = h
            .store_inner(bad_labels)
            .expect_err("label shape must reject");
        assert!(
            matches!(
                err,
                HeadError::LabelShape {
                    got: 1,
                    n_classes: 3
                }
            ),
            "expected LabelShape, got {err:?}",
        );

        // 6. Well-shaped inner -> accepted, version bumps.
        let v_before = h.version();
        let receipt = h
            .store_inner(synth_inner(7, HeadId::new()))
            .expect("well-shaped store must succeed");
        assert!(receipt.version > v_before, "version did not advance");
        assert_eq!(h.snapshot().n_classes, 7);
    }

    /// `swap` is observed by readers immediately.  No snapshot is torn
    /// across a swap-storm: every snapshot has weight, bias, labels and
    /// head_id all aligned to a single (n_classes, head_id) pairing
    /// that the writer published as one bundle.
    ///
    /// The bundling check is encoded as: for any snapshot, all of the
    /// following hold simultaneously --
    ///   weight.len() == BACKBONE_FEATURE_DIM x n_classes
    ///   bias.len()   == n_classes
    ///   labels.len() == n_classes
    ///   labels[i]    starts with the version tag baked into head_id
    /// The fourth invariant (which is the real anti-tearing check --
    /// it would fail if the swap published new labels against old
    /// weights) requires that we tag labels with the same version
    /// string as head_id at synth time.
    #[test]
    fn store_inner_no_torn_snapshot() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        // Tagged synth: every label embeds the head_id's hyphenated
        // string form, so a torn snapshot (labels-vs-head_id
        // mismatch) is visible at read time. n_classes choice is
        // independent of head_id so we can sweep both axes.
        fn tagged(n_classes: usize, head_id: HeadId) -> HeadInner {
            let v = head_id.to_string();
            HeadInner {
                weight: vec![0.01; BACKBONE_FEATURE_DIM * n_classes],
                bias: vec![0.0; n_classes],
                labels: (0..n_classes).map(|i| format!("{v}/cls_{i}")).collect(),
                head_id,
                n_classes,
            }
        }

        // Two distinct UUIDs identify the alternation; a third
        // distinct one is used to seed the head before the writer
        // thread starts so the very-first-snapshot path also runs
        // against a stable head_id.
        //
        // The literals below are strict UUID-v4 (version nibble
        // '4' at byte 14, variant nibble '8' at byte 19); only the
        // trailing hex disambiguates them.  Naming the sentinels
        // separates "what the bytes are" (canonical zero-suffix
        // v4 fixtures) from "what role each plays in this test"
        // (init / even / odd).
        const TEST_V4_NIL: &str = "00000000-0000-4000-8000-000000000000";
        const TEST_V4_NIL_1: &str = "00000000-0000-4000-8000-000000000001";
        const TEST_V4_NIL_2: &str = "00000000-0000-4000-8000-000000000002";
        let v_init = HeadId::parse(TEST_V4_NIL).unwrap();
        let v_even = HeadId::parse(TEST_V4_NIL_1).unwrap();
        let v_odd = HeadId::parse(TEST_V4_NIL_2).unwrap();

        let h = HotHead::from_inner(tagged(3, v_init));
        let stop = Arc::new(AtomicBool::new(false));

        // Background swap storm: alternate between (n=3, even) and
        // (n=5, odd).  The (n_classes, head_id) bundle changes every
        // swap, so a torn snapshot would be statistically observable.
        let h_writer = h.clone();
        let stop_w = stop.clone();
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !stop_w.load(Ordering::Relaxed) {
                let (n, v) = if i.is_multiple_of(2) {
                    (3, v_even)
                } else {
                    (5, v_odd)
                };
                // Soak asserts on snapshot consistency; receipt
                // version is intentionally unobserved.
                let _ = h_writer
                    .store_inner(tagged(n, v))
                    .expect("tagged() always produces a valid HeadInner");
                i = i.wrapping_add(1);
            }
            i
        });

        // Reader: 10 k snapshots, each must be self-consistent.
        let h_reader = h.clone();
        let reader = thread::spawn(move || {
            for _ in 0..10_000 {
                let s = h_reader.snapshot();
                let v = s.head_id.to_string();
                assert_eq!(
                    s.weight.len(),
                    BACKBONE_FEATURE_DIM * s.n_classes,
                    "weight/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                assert_eq!(
                    s.bias.len(),
                    s.n_classes,
                    "bias/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                assert_eq!(
                    s.labels.len(),
                    s.n_classes,
                    "labels/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                for (i, lbl) in s.labels.iter().enumerate() {
                    assert!(
                        lbl.starts_with(&format!("{v}/")),
                        "labels/head_id torn: head_id={v}, labels[{i}]={lbl}",
                    );
                }
            }
        });

        reader.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        let swaps = writer.join().unwrap();
        // Sanity: the writer actually swapped, otherwise the bundling
        // check is degenerate (we'd just be re-reading the initial
        // value).  On any reasonable scheduler we'll see >>1 swap.
        assert!(
            swaps > 0,
            "writer thread completed 0 swaps; test is degenerate"
        );
    }

    /// `HotHead::load` rejects headerless
    /// `.mpk` files with `HeadError::SchemaTooOld`.  The error
    /// message points the operator at the converter for
    /// regeneration.
    #[test]
    fn load_rejects_headerless_mpk() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let mpk_stem = dir.path().join("test_head");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), mpk_stem)
            .expect("record head");
        // Note: headerless -- the recorder's raw output, no
        // ACSTHEAD prepend.
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");
        let err = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect_err("headerless mpk must reject");
        assert!(
            matches!(err, HeadError::SchemaTooOld { .. }),
            "expected SchemaTooOld, got {err:?}",
        );
    }

    /// `HotHead::load` round-trips a header-prepended
    /// `.mpk` end-to-end.  Constructs a tiny head, writes it
    /// via the converter's header-prepending shape (mirroring
    /// `MpkSink::publish`), then loads it back via
    /// `HotHead::load` and verifies the snapshot.
    #[test]
    fn load_round_trips_through_header_prepended_mpk() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record head");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        let mut composed = std::fs::File::create(&mpk_path).expect("create mpk");
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write head with header");
        drop(composed);
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");

        let h = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect("load header-prepended mpk");
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 2);
        assert_eq!(snap.labels, vec!["alpha".to_string(), "beta".to_string()]);
    }

    /// `head::load_inner` no longer
    /// stages the prost payload to a sibling
    /// `.head-load-{head_id}.mpk` tempfile.  The bytes-recorder
    /// path consumes the in-memory payload directly.  Swap a
    /// head 100x and assert the workspace dir contains no
    /// `.head-load-*.mpk` artifacts at any point.
    #[test]
    fn load_inner_does_not_create_sibling_tempfile() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record head");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        // Compose the header-prepended .mpk that production
        // ships (mirroring `MpkSink::publish`).
        let mut composed = std::fs::File::create(&mpk_path).expect("create mpk");
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write head with header");
        drop(composed);
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");

        // Pre-clean: confirm the dir starts with exactly the
        // 3 expected entries (raw mpk, prod mpk, labels).
        let count_sibling_tempfiles = || -> usize {
            std::fs::read_dir(dir.path())
                .expect("read tempdir")
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let s = name.to_string_lossy();
                    s.starts_with(".head-load-")
                })
                .count()
        };
        assert_eq!(count_sibling_tempfiles(), 0, "tempdir starts clean");

        // Swap 100x, asserting after each that no
        // `.head-load-*.mpk` sibling tempfile ever appears.
        // The bytes-recorder path keeps the load entirely
        // in memory; this regression-guards against a
        // future revision reverting to a tempfile-staging
        // shape.
        let h = HotHead::load(&mpk_path, &labels_path, HeadId::new()).expect("first load");
        for i in 0..100 {
            let new_id = HeadId::new();
            let receipt = h
                .swap(&mpk_path, &labels_path, new_id)
                .unwrap_or_else(|e| panic!("swap #{i} failed: {e:?}"));
            assert!(receipt.version.get() > 0, "receipt version is monotonic");
            // Mid-storm sibling-tempfile check.
            assert_eq!(
                count_sibling_tempfiles(),
                0,
                "swap #{i} left a `.head-load-*` artifact in the workspace dir",
            );
        }
        // Post-storm.
        assert_eq!(
            count_sibling_tempfiles(),
            0,
            "100 swaps left a `.head-load-*` artifact in the workspace dir",
        );
    }

    /// Corrupt-payload-via-CRC: flip a byte inside
    /// the header's `feature_dim` slot.  The CRC over bytes
    /// [0..28) catches the mismatch and `HotHead::load`
    /// returns `HeadError::HeaderCorrupt` (NOT `SchemaTooOld`,
    /// since the magic bytes are intact).
    #[test]
    fn load_rejects_corrupt_header_via_crc() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        let mut bytes: Vec<u8> = Vec::new();
        let mut composed = std::io::Cursor::new(&mut bytes);
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write");
        // Flip one byte in the feature_dim slot WITHOUT
        // updating the CRC.
        bytes[12] ^= 0xFF;
        std::fs::write(&mpk_path, &bytes).expect("write tampered");
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");
        let err = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect_err("corrupt header must reject");
        assert!(
            matches!(err, HeadError::HeaderCorrupt { .. }),
            "expected HeaderCorrupt, got {err:?}",
        );
    }

    /// Loading the bundled `00000000-default` head from
    /// `misc/heads/` gives a head whose class count matches the
    /// shipped `labels.txt`.  Marked `#[ignore]` because the test
    /// is path-fragile (depends on repo layout) and does file
    /// I/O against the bundled fixture.
    ///
    /// Run via:
    ///   cargo test --release -- --include-ignored reference_head_loads
    #[test]
    #[ignore = "depends on bundled fixture assets; --include-ignored"]
    fn reference_head_loads() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        let head_mpk = crate_root.join("misc/heads/00000000-default/head.mpk");
        let labels = crate_root.join("misc/heads/00000000-default/labels.txt");
        let h = HotHead::load(&head_mpk, &labels, HeadId::new()).expect("load reference head");
        let snap = h.snapshot();
        assert!(snap.n_classes >= 1);
        assert_eq!(snap.weight.len(), BACKBONE_FEATURE_DIM * snap.n_classes);
        assert_eq!(snap.bias.len(), snap.n_classes);
        assert_eq!(snap.labels.len(), snap.n_classes);
    }

    /// `HotHead::try_from_inner` rejects a non-finite weight
    /// before publishing.  Without this the bad value would
    /// reach the engine's hot path, propagate through
    /// `head_forward` -> `softmax_into` to NaN probs, and
    /// `top_k_indices_into`'s defensive NaN handling would
    /// emit them inside an `InferenceFrame`.
    #[test]
    fn try_from_inner_rejects_non_finite_weight() {
        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.weight[7] = f32::NAN;
        let err = HotHead::try_from_inner(inner).expect_err("NaN weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 7, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );

        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.weight[2] = f32::INFINITY;
        let err = HotHead::try_from_inner(inner).expect_err("+Inf weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 2, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );
    }

    /// `HotHead::try_from_inner` rejects a non-finite bias
    /// before publishing.  Mirror of the weight test.
    #[test]
    fn try_from_inner_rejects_non_finite_bias() {
        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.bias[1] = f32::NAN;
        let err = HotHead::try_from_inner(inner).expect_err("NaN bias must reject");
        assert!(
            matches!(err, HeadError::NonFiniteBias { idx: 1, .. }),
            "expected NonFiniteBias, got {err:?}",
        );

        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.bias[0] = f32::NEG_INFINITY;
        let err = HotHead::try_from_inner(inner).expect_err("-Inf bias must reject");
        assert!(
            matches!(err, HeadError::NonFiniteBias { idx: 0, .. }),
            "expected NonFiniteBias, got {err:?}",
        );
    }

    /// `HotHead::try_from_inner` accepts a well-formed
    /// inner and the resulting snapshot reflects the input.
    /// Regression-guards against `try_from_inner`'s validation
    /// gaining a false-positive that rejects valid heads.
    #[test]
    fn try_from_inner_accepts_well_shaped() {
        let id = HeadId::new();
        let h = HotHead::try_from_inner(synth_inner(5, id)).expect("well-shaped accepts");
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 5);
        assert_eq!(snap.head_id, id);
    }

    /// `HotHead::store_inner` rejects a non-finite weight
    /// (re-uses the same `validate` path as `try_from_inner`,
    /// just exercised via the swap-in-place entrypoint).
    #[test]
    fn store_inner_rejects_non_finite_weight() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));
        let mut bad = synth_inner(3, HeadId::new());
        bad.weight[0] = f32::NAN;
        let err = h.store_inner(bad).expect_err("NaN weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 0, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );
    }

    /// `classify_head_error` routes I/O `NotFound` through the
    /// `HeadStoreError::NotFound` variant, other I/O failures
    /// through `LoadFailed`, Burn drift through `Internal`, and
    /// the remaining content-validation variants through
    /// `InvalidContent`.  This is the contract the `HeadStore`
    /// trait surface exposes to the API layer for HTTP class
    /// mapping; a regression here would silently re-collapse
    /// every failure to a 400.
    #[test]
    fn classify_head_error_maps_each_category() {
        // I/O NotFound on either read path -> HeadStoreError::NotFound.
        let nf_mpk = HeadError::ReadHeadMpk {
            path: "/missing.mpk".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        match classify_head_error(nf_mpk) {
            HeadStoreError::NotFound { path } => assert_eq!(path, "/missing.mpk"),
            other => panic!("expected NotFound, got {other:?}"),
        }
        let nf_lbl = HeadError::ReadLabels {
            path: "/missing.txt".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        match classify_head_error(nf_lbl) {
            HeadStoreError::NotFound { path } => assert_eq!(path, "/missing.txt"),
            other => panic!("expected NotFound, got {other:?}"),
        }

        // Permission-denied (or any non-NotFound I/O) on either
        // read path -> HeadStoreError::LoadFailed.
        let denied = HeadError::ReadHeadMpk {
            path: "/locked.mpk".into(),
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        };
        assert!(
            matches!(
                classify_head_error(denied),
                HeadStoreError::LoadFailed { .. }
            ),
            "non-NotFound I/O must route to LoadFailed",
        );

        // Burn `into_data().to_vec()` drift -> Internal.
        let drift = HeadError::TensorIntoVec("shape mismatch".into());
        assert!(
            matches!(classify_head_error(drift), HeadStoreError::Internal { .. }),
            "TensorIntoVec must route to Internal",
        );

        // Header / shape / finiteness failures -> InvalidContent.
        let representative = [
            HeadError::SchemaTooOld {
                path: "/old.mpk".into(),
            },
            HeadError::HeaderCorrupt {
                path: "/c.mpk".into(),
                reason: "bad crc".into(),
            },
            HeadError::HeaderFeatureDimMismatch {
                path: "/d.mpk".into(),
                got: 7,
                expected: BACKBONE_FEATURE_DIM as u32,
            },
            HeadError::WeightShape {
                got: 1,
                expected: 2,
            },
            HeadError::BiasShape {
                got: 1,
                expected: 2,
            },
            HeadError::InputDim { got: 7 },
            HeadError::BadClassCount { got: 0, max: 100 },
            HeadError::LabelCountMismatch {
                path: "/l.txt".into(),
                got: 1,
                n_classes: 2,
            },
            HeadError::LabelShape {
                got: 1,
                n_classes: 2,
            },
            HeadError::NonFiniteWeight {
                idx: 0,
                value: f32::NAN,
            },
            HeadError::NonFiniteBias {
                idx: 0,
                value: f32::NAN,
            },
            HeadError::LoadMpk {
                path: "/x.mpk".into(),
                message: "burn parse failed".into(),
            },
        ];
        for err in representative {
            let formatted = format!("{err}");
            let mapped = classify_head_error(err);
            assert!(
                matches!(mapped, HeadStoreError::InvalidContent { .. }),
                "{formatted} must classify as InvalidContent, got {mapped:?}",
            );
        }
    }
}
