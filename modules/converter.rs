//! Head-weight extractor for the Acoustics Lab daemon.
//!
//! ## What this crate does
//!
//! Given an operator-uploaded classifier bundle in TFJS Layers-Model
//! (`model.json` + one or more weight shards) format, this crate
//! locates the head's `Linear` layer (Keras `Dense` kernel + bias),
//! extracts its weights in Burn's `[in_dim, n_classes]` orientation,
//! and writes a Burn `head.mpk` plus a sibling `labels.txt` and
//! `metadata.json`.
//!
//! TFJS is the only supported source format;
//! TFLite uploads are rejected at the HTTP boundary
//! with `405 Method Not Allowed`.
//!
//! ## What this crate does NOT do
//!
//! - It does NOT convert the backbone to RKNN.  The daemon ships with
//!   a pre-built `backbone.rknn` covering the canonical 9 conv +
//!   dense_1 stack of the Teachable Machine "audio classifier" model
//!   family.  Uploading a model only replaces the **head**; we trust
//!   the operator to ensure their uploaded model uses the same
//!   backbone topology + weights as our pre-built RKNN.
//! - It does NOT validate the backbone weights byte-for-byte against
//!   `misc/backbones/backbone.mpk`.  An incorrect backbone manifests as
//!   garbage classification at runtime, not as a daemon error --
//!   operators detect via the model's val accuracy on a known
//!   dataset.
//!
//! ## TFJS layout
//!
//! TFJS Layers-Model stores a `model.json` with a `weightsManifest`
//! that lists tensors in declaration order; the actual fp32 bytes
//! live in one or more `weights.bin` shards (paths relative to the
//! `model.json` directory).  Tensor offsets are implicit: each entry
//! starts where the previous one ended, with shards concatenated in
//! manifest order.
//!
//! Keras Dense kernel layout is `[in_dim, out_dim]` row-major --
//! identical to Burn's `Linear` weight, so no transpose is needed.
//!
//! Head identification, in priority:
//!   1. Names ending in `NewHeadDense/kernel` + `NewHeadDense/bias`
//!      (Teachable Machine's fine-tuned head naming convention).
//!   2. The unique 2-D weight with shape `[BACKBONE_FEATURE_DIM, N]`
//!      paired with the unique 1-D weight of shape `[N]`.
//!
//! If neither yields a unique match, conversion fails with a
//! diagnostic listing the manifest contents.

#![warn(missing_debug_implementations)]

// `pub(crate)` keeps the trait composition in-tree; the public
// surface flows through the parent re-exports below.
pub(crate) mod pipeline;
pub(crate) mod sink;
pub(crate) mod source;

pub use pipeline::Pipeline;
pub use sink::{ArtifactSink, MpkSink};
pub use source::{LoadedSource, SourceModel, TfjsSource, TfjsSourceLimited};

use std::path::{Path, PathBuf};

use crate::common::dims::BACKBONE_FEATURE_DIM;
use crate::common::hex::hex_lowercase;
use crate::common::ids::HeadId;
use crate::common::log_truncate::truncate_log_message;
use crate::file_mgr::FsService;
use crate::model::{self, Head};
use burn::backend::NdArray;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::TensorData;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use time::OffsetDateTime;

type B = NdArray<f32>;

/// Daemon-wide single-slot concurrency cap.  TFJS extraction holds
/// ~500 MB of transient state (parsed weights + Burn
/// `Param<Tensor>` bumps); a concurrent run would risk OOM.
static CONVERT_SEMAPHORE: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();

/// Try to acquire the convert permit; returns [`ConvertError::Busy`]
/// (HTTP 409) instead of blocking so operators get a progress signal.
pub fn acquire_convert_permit() -> Result<tokio::sync::OwnedSemaphorePermit, ConvertError> {
    CONVERT_SEMAPHORE
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(1)))
        .clone()
        .try_acquire_owned()
        .map_err(|_| ConvertError::Busy)
}

/// Resource caps applied at the TFJS ingestion boundary BEFORE
/// per-tensor allocation.  `model.json` is operator-supplied; an
/// adversarial manifest can declare enormous shapes long before any
/// shard data has proven legitimate, so each cap rejects manifests
/// before any allocation those declarations would justify.
///
/// `max_n_classes` is sized so
/// `max_kernel_bytes / (BACKBONE_FEATURE_DIM * 4)` lands on the same
/// boundary -- both limits fire consistently rather than one being
/// phantom.  `max_bias_bytes` is loose (n_classes already bounds it)
/// and exists as headroom for future schemas with multiple bias
/// tensors.
#[derive(Clone, Copy, Debug)]
pub struct ConvertLimits {
    pub max_n_classes: usize,
    pub max_kernel_bytes: usize,
    pub max_bias_bytes: usize,
    pub max_manifest_bytes: usize,
    pub max_shards: usize,
}

impl Default for ConvertLimits {
    fn default() -> Self {
        Self {
            // 80 MiB / (BACKBONE_FEATURE_DIM * 4) ~= 10_485 classes;
            // rounded down for headroom.
            max_n_classes: 10_000,
            // 2000 * 10_000 * 4 = 80 MiB.
            max_kernel_bytes: 80 * 1024 * 1024,
            // Loose; n_classes (40 KiB at the cap) fires first and
            // owns the operator-facing rejection.
            max_bias_bytes: 1024 * 1024,
            // Real `model.json` is O(10) KiB.
            max_manifest_bytes: 1024 * 1024,
            // Real exports use 1-5 shards.
            max_shards: 64,
        }
    }
}

/// Source format the operator uploaded.  TFJS is the only supported
/// variant; TFLite is no longer accepted.
///
/// `#[non_exhaustive]`: the singleton variant is a current-
/// acceptable snapshot, not a closed set (TFLite may return,
/// ONNX is on the roadmap).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum SourceKind {
    Tfjs,
}

/// Failures from the TFJS-to-Burn conversion pipeline.
/// Mapped to HTTP statuses by the API layer via the
/// [`crate::common::error::Categorized`] impl below.
#[derive(Debug, Error)]
pub enum ConvertError {
    #[error("read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("write {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("burn record: {0}")]
    Record(String),
    #[error("burn tensor: {0}")]
    Tensor(String),
    #[error("metadata serialize: {0}")]
    MetadataSerialize(#[from] serde_json::Error),
    #[error("labels file: {0}")]
    Labels(String),
    #[error("tfjs parse {what}: {msg}")]
    TfjsParse { what: &'static str, msg: String },
    #[error("tfjs head locator: {0}")]
    TfjsLocator(String),
    #[error("tfjs weights.bin too short: have {have} bytes, manifest needs {need}")]
    TfjsShortRead { have: usize, need: usize },
    #[error("tfjs unsupported dtype `{dtype}` on weight `{name}` (only float32 supported)")]
    TfjsDtype { name: String, dtype: String },
    #[error("tfjs unsafe shard path `{0}`: must be a relative path with no parent traversal")]
    TfjsUnsafePath(String),
    #[error(
        "tfjs manifest declares dimensions that overflow usize on weight `{name}` (shape {shape:?})"
    )]
    TfjsShapeOverflow { name: String, shape: Vec<usize> },
    #[error("tfjs blob length mismatch: have {have} bytes, manifest declares {declared}")]
    TfjsBlobLength { have: usize, declared: usize },
    /// Resource cap (`ConvertLimits`) tripped before
    /// allocation.  `what` is a stable identifier for the
    /// cap that fired (e.g. `"n_classes"`,
    /// `"kernel_bytes"`); operators map it to the
    /// corresponding `ConvertLimits` field.
    #[error("tfjs limit `{what}` exceeded: {value} > max {max}")]
    LimitExceeded {
        what: &'static str,
        value: u64,
        max: u64,
    },
    /// Manifest declared a zero dimension (kernel `[D, 0]`,
    /// bias `[0]`, etc.).  Burn's `LinearConfig::init`
    /// panics on zero out-features, so we reject at the
    /// boundary with a structured diagnostic.
    #[error("tfjs zero dimension on weight `{name}` (shape {shape:?})")]
    TfjsZeroDimension { name: String, shape: Vec<usize> },
    /// A decoded fp32 weight is non-finite (NaN, +/-Inf).
    /// Defense in depth: the inference cold path also
    /// refuses non-finite heads, but the converter is the
    /// right boundary for this check because the source
    /// model is operator-supplied.  `tensor` names the
    /// kernel/bias entry; `index` is the offset within
    /// that tensor's f32 sequence so operators can locate
    /// the bad element in the original export.
    #[error("tfjs non-finite weight in `{tensor}` at index {index}: {value}")]
    NonFiniteWeight {
        tensor: String,
        index: usize,
        value: f32,
    },
    /// Head construction rejected because `n_classes` is
    /// outside `1..=MAX_N_CLASSES`.  Mirrors
    /// [`crate::model::Error::BadClassCount`]; raised at
    /// the converter boundary so operators receive a
    /// `UserInput` (HTTP 400) rather than a daemon-internal
    /// failure forwarded from `model::Head::try_new`.
    #[error("invalid head: n_classes = {got} (must be in 1..={max})")]
    BadClassCount { got: usize, max: usize },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// Another converter job is already in flight.  Single-
    /// tenant by design: concurrent extracts risk OOM under the
    /// configured RAM budget.  Operators retry after the active
    /// job completes (typically 10-30 s).
    #[error("converter job already running")]
    Busy,
}

/// Shorthand for `ConvertError::Read { path: path.to_string(), source }`.
/// `path` is `impl Display` so call sites can pass `Path::display()`
/// adapters directly without per-site `to_string()` boilerplate.
pub(crate) fn convert_read_err(
    path: impl std::fmt::Display,
    source: std::io::Error,
) -> ConvertError {
    ConvertError::Read {
        path: path.to_string(),
        source,
    }
}

/// Shorthand for `ConvertError::Write { path: path.to_string(), source }`.
/// Companion to [`convert_read_err`]; same shape, different
/// variant.  Several call sites construct `source` via
/// `io::Error::other(format!("{e}"))` to wrap a `FileError` into
/// the converter's local `io::Error`-shaped variant; the helper
/// is signature-agnostic.
pub(crate) fn convert_write_err(
    path: impl std::fmt::Display,
    source: std::io::Error,
) -> ConvertError {
    ConvertError::Write {
        path: path.to_string(),
        source,
    }
}

impl crate::common::error::Categorized for ConvertError {
    /// Exhaustive `match` so adding a new variant forces a
    /// classification update; the API layer dispatches uniformly via
    /// `ErrorKind`.
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Source model malformed / unsupported (operator can
            // re-export the model and retry).
            ConvertError::Labels(_)
            | ConvertError::TfjsParse { .. }
            | ConvertError::TfjsLocator(_)
            | ConvertError::TfjsShortRead { .. }
            | ConvertError::TfjsDtype { .. }
            | ConvertError::TfjsUnsafePath(_)
            | ConvertError::TfjsShapeOverflow { .. }
            | ConvertError::TfjsBlobLength { .. }
            | ConvertError::LimitExceeded { .. }
            | ConvertError::TfjsZeroDimension { .. }
            | ConvertError::NonFiniteWeight { .. }
            | ConvertError::BadClassCount { .. } => UserInput,

            // Caller asked for a converter mode we haven't built.
            ConvertError::NotImplemented(_) => NotImplemented,

            // Concurrency cap rejected; same
            // 409 wire shape as the training-registry's
            // `Busy { AnotherJobRunning }`.  Operator retries
            // after the active job completes.
            ConvertError::Busy => Conflict,

            // Filesystem / serializer / Burn-record / tensor: daemon-
            // internal infrastructure failure, not the uploader's
            // fault.
            ConvertError::Read { .. }
            | ConvertError::Write { .. }
            | ConvertError::Record(_)
            | ConvertError::Tensor(_)
            | ConvertError::MetadataSerialize(_) => Internal,
        }
    }
}

/// Extracted head weights, ready to be saved as `head.mpk`.  Kernel is
/// already in Burn's `[in_dim, n_classes]` layout.
#[derive(Clone, Debug)]
pub struct HeadWeights {
    pub kernel: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_classes: usize,
    pub in_dim: usize,
}

/// On-disk artifacts produced by [`write_head_artifacts`].
#[derive(Clone, Debug, Serialize)]
pub struct HeadArtifacts {
    pub head_mpk: PathBuf,
    pub labels_txt: PathBuf,
    pub metadata_json: PathBuf,
    pub head_id: HeadId,
    pub n_classes: usize,
    /// SHA-256 of the source model bytes (lowercase hex).  Captured
    /// here so callers can surface it in their own metadata without
    /// re-reading the source file.
    pub source_sha256: String,
}

/// Per-conversion metadata persisted alongside the head.  The schema
/// number bumps on breaking change; v1 is the only version today.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversionMetadata {
    pub schema_version: u32,
    pub source_kind: SourceKind,
    pub source_sha256: String,
    pub n_classes: usize,
    pub labels: Vec<String>,
    pub created_at: String,
    pub head_id: HeadId,
}

/// Extract the head from a TFJS Layers-Model directory containing
/// `model.json` plus one or more weight shards (paths declared in
/// `weightsManifest[*].paths`, resolved relative to the directory
/// holding `model.json`).
///
/// On success, returns weights already in Burn's `[in_dim,
/// n_classes]` orientation (no transpose: TFJS/Keras Dense kernel
/// matches Burn).
pub fn extract_head_from_tfjs_dir(tfjs_dir: &Path) -> Result<HeadWeights, ConvertError> {
    let limits = ConvertLimits::default();
    let model_json = tfjs_dir.join("model.json");
    let json_bytes =
        std::fs::read(&model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // Stream-read each shard once and copy out only the kernel +
    // bias byte ranges.  No SHA contract on this entry point, so
    // the hasher is `None`.
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    let (kernel_bytes, bias_bytes) =
        source::read_head_bytes_streaming(tfjs_dir, &manifest, k_entry, b_entry, None)?;
    head_weights_from_head_byte_ranges(&manifest, k_entry, b_entry, &kernel_bytes, &bias_bytes)
}

/// Single-shard convenience: parse `model_json`, then read the head
/// out of the supplied `weights_bin`.  Errors if the manifest
/// references more than one shard path.
pub fn extract_head_from_tfjs(
    model_json: &Path,
    weights_bin: &Path,
) -> Result<HeadWeights, ConvertError> {
    let limits = ConvertLimits::default();
    let json_bytes =
        std::fs::read(model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;
    if manifest.shards.len() != 1 {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: format!(
                "expected exactly 1 shard path, got {}: {:?}; use \
                 extract_head_from_tfjs_dir for multi-shard models",
                manifest.shards.len(),
                manifest.shards
            ),
        });
    }
    let weights_blob =
        std::fs::read(weights_bin).map_err(|e| convert_read_err(weights_bin.display(), e))?;
    extract_head_from_tfjs_buffers(&manifest, &weights_blob)
}

/// Convenience: extract + persist a TFJS upload in one call.
///
/// `tfjs_dir` must contain `model.json` and the shard(s) referenced
/// from its manifest.  If `metadata.json` exists alongside
/// `model.json` and contains a `wordLabels` or `words` array,
/// callers may use [`read_tfjs_labels`] to source the labels list;
/// this function does not auto-discover labels (it takes them as
/// an argument so callers can normalize / validate).
///
/// `source_sha256` is computed as the SHA-256 of `model.json` bytes
/// concatenated with each shard's bytes in manifest order.  The
/// hash is deterministic for a given on-disk layout.
pub fn convert_tfjs(
    tfjs_dir: &Path,
    labels: &[String],
    dst_dir: &Path,
    fs: &dyn FsService,
) -> Result<HeadArtifacts, ConvertError> {
    let limits = ConvertLimits::default();
    let model_json = tfjs_dir.join("model.json");
    let json_bytes =
        std::fs::read(&model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // Stream each shard once: feed every byte through the SHA-256
    // hasher AND copy out only the kernel + bias byte ranges.
    // Peak heap is one shard's `Vec<u8>` (~50 MB on the canonical
    // Teachable-Machine layout) rather than the concatenated payload.
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    let mut hasher = Sha256::new();
    hasher.update(&json_bytes);
    let (kernel_bytes, bias_bytes) = source::read_head_bytes_streaming(
        tfjs_dir,
        &manifest,
        k_entry,
        b_entry,
        Some(&mut hasher),
    )?;
    let source_sha256 = hex_lowercase(&hasher.finalize());

    let weights = head_weights_from_head_byte_ranges(
        &manifest,
        k_entry,
        b_entry,
        &kernel_bytes,
        &bias_bytes,
    )?;
    write_head_artifacts(
        &weights,
        labels,
        dst_dir,
        SourceKind::Tfjs,
        source_sha256,
        fs,
    )
}

/// Parse the labels array from a TFJS `metadata.json`, accepting
/// either `wordLabels: [String]` (Teachable Machine fine-tuned
/// export) or `words: [String]` (upstream Google Speech-Commands
/// bundle).  `wordLabels` wins when both keys are present, a
/// defensive tiebreak since exporters in practice emit only
/// one.  Empty array, non-string element, or missing both keys
/// is an error.
pub fn read_tfjs_labels(metadata_json: &Path) -> Result<Vec<String>, ConvertError> {
    let bytes =
        std::fs::read(metadata_json).map_err(|e| convert_read_err(metadata_json.display(), e))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        ConvertError::Labels(format!("{}: invalid JSON: {e}", metadata_json.display()))
    })?;
    let (key, arr) = if let Some(arr) = v.get("wordLabels").and_then(serde_json::Value::as_array) {
        ("wordLabels", arr)
    } else if let Some(arr) = v.get("words").and_then(serde_json::Value::as_array) {
        ("words", arr)
    } else {
        return Err(ConvertError::Labels(format!(
            "{}: missing `wordLabels` or `words` array",
            metadata_json.display()
        )));
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, e) in arr.iter().enumerate() {
        let s = e.as_str().ok_or_else(|| {
            ConvertError::Labels(format!(
                "{}: {key}[{i}] is not a string",
                metadata_json.display(),
            ))
        })?;
        out.push(s.to_string());
    }
    if out.is_empty() {
        return Err(ConvertError::Labels(format!(
            "{}: {key} is empty",
            metadata_json.display()
        )));
    }
    Ok(out)
}

/// Materialize a Burn `Head<NdArray>` from extracted weights and
/// write `head.mpk`, `labels.txt`, `metadata.json` into `dst_dir`.
///
/// Returns the on-disk paths + the freshly-generated UUID head_id.
///
/// `labels.len()` must equal `weights.n_classes`.  Caller is
/// responsible for stripping leading numeric prefixes ("0 Background")
/// or normalizing labels -- we write them verbatim.
///
/// ## Crash safety
///
/// All three artifacts ship through [`FsService::put_atomic`]
/// (tempfile + sync_all + rename + parent-dir fsync).  The
/// `head.mpk` payload is built ENTIRELY in memory via
/// [`burn::record::NamedMpkBytesRecorder`] and a single
/// `serialize_header` + `extend_from_slice` -- no
/// `head-partial-*` or `head-with-header-partial-*` ever
/// appears under `dst_dir`.  `metadata.json` is written LAST so
/// its presence is the consistency marker -- a crash between
/// `head.mpk` and `metadata.json` leaves a workspace without
/// metadata.json, which downstream loaders treat as "not yet
/// converted" rather than "converted with unknown n_classes".
///
/// Each `put_atomic` call already fsyncs the parent directory
/// after rename, so post-crash readers see either zero or all
/// of {head.mpk, labels.txt} and either presence or absence of
/// metadata.json.
pub fn write_head_artifacts(
    weights: &HeadWeights,
    labels: &[String],
    dst_dir: &Path,
    source_kind: SourceKind,
    source_sha256: String,
    fs: &dyn FsService,
) -> Result<HeadArtifacts, ConvertError> {
    // Defensive: this is a `pub` entry point; re-check
    // n_classes so the converter owns rejection on every code
    // path (a hand-built `HeadWeights { n_classes: 0, .. }`
    // would otherwise panic inside `Head::new`).
    validate_head_class_count(weights.n_classes, &ConvertLimits::default())?;

    if labels.len() != weights.n_classes {
        return Err(ConvertError::Labels(format!(
            "labels.len() = {}, but n_classes = {}",
            labels.len(),
            weights.n_classes
        )));
    }
    if weights.in_dim != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::Tensor(format!(
            "kernel in_dim {} != BACKBONE_FEATURE_DIM {BACKBONE_FEATURE_DIM}",
            weights.in_dim
        )));
    }

    std::fs::create_dir_all(dst_dir).map_err(|e| convert_write_err(dst_dir.display(), e))?;

    let head_id = HeadId::new();
    let head_mpk = dst_dir.join("head.mpk");
    let labels_txt = dst_dir.join("labels.txt");
    let metadata_json = dst_dir.join("metadata.json");

    // 1. Build the ACSTHEAD-wrapped .mpk blob in memory.  One
    //    `put_atomic` call below publishes the whole blob -- no
    //    `head-partial-*` / `head-with-header-partial-*` siblings
    //    ever appear under `dst_dir`, which closes the
    //    "recorder return != data on disk" gap (the prior
    //    `NamedMpkFileRecorder` returned without `sync_all`, so a
    //    hard-crash between recorder return and the
    //    read-back-and-prepend dance could SHA in zeroes / stale
    //    pagecache).
    let head_blob = build_head_mpk_blob(weights)?;
    fs.put_atomic(&head_mpk, &head_blob).map_err(|e| {
        convert_write_err(head_mpk.display(), std::io::Error::other(format!("{e}")))
    })?;

    // 2. labels.txt -- one label per line, no trailing whitespace.
    let labels_blob = labels.join("\n") + "\n";
    fs.put_atomic(&labels_txt, labels_blob.as_bytes())
        .map_err(|e| {
            convert_write_err(labels_txt.display(), std::io::Error::other(format!("{e}")))
        })?;

    // 3. metadata.json (last; presence = converter run committed)
    let meta = ConversionMetadata {
        schema_version: 1,
        source_kind,
        source_sha256: source_sha256.clone(),
        n_classes: weights.n_classes,
        labels: labels.to_vec(),
        created_at: OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .map_err(|e| ConvertError::Labels(format!("RFC3339: {e}")))?,
        head_id,
    };
    let bytes = serde_json::to_vec_pretty(&meta)?;
    fs.put_atomic(&metadata_json, &bytes).map_err(|e| {
        convert_write_err(
            metadata_json.display(),
            std::io::Error::other(format!("{e}")),
        )
    })?;

    Ok(HeadArtifacts {
        head_mpk,
        labels_txt,
        metadata_json,
        head_id,
        n_classes: weights.n_classes,
        source_sha256,
    })
}

/// Parse a TFJS `model.json` byte buffer and return the ordered
/// list of validated shard relative paths declared in its
/// `weightsManifest`.  Used by API surfaces that need to enumerate
/// the shard set up front (e.g., to stage user-uploaded assets into
/// a tempdir matching the manifest layout) without committing to
/// reading any shard bytes yet.
///
/// The returned paths are guaranteed to satisfy the same
/// safety predicate enforced internally by [`extract_head_from_tfjs_dir`]:
/// strictly relative, no parent traversal, no NUL bytes, no
/// backslashes.  Order matches the manifest's declared concatenation
/// order.
pub fn list_tfjs_shard_paths(model_json_bytes: &[u8]) -> Result<Vec<String>, ConvertError> {
    Ok(parse_tfjs_manifest_with_limits(model_json_bytes, &ConvertLimits::default())?.shards)
}

// MARK: TFJS manifest helpers

/// One entry in the TFJS weights manifest.  `offset_bytes` is the
/// absolute offset into the *concatenated* shard buffer (shards
/// joined in manifest order).
///
/// Fields are `pub(crate)` so the streaming reader in
/// `source.rs` can compute per-shard byte-range
/// overlaps without an accessor maze.
#[derive(Clone, Debug)]
pub(crate) struct TfjsManifestEntry {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) offset_bytes: usize,
    pub(crate) len_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TfjsManifest {
    pub(crate) entries: Vec<TfjsManifestEntry>,
    /// Shard paths (relative to the model.json directory) in the
    /// order their bytes appear in the concatenated buffer.
    pub(crate) shards: Vec<String>,
}

/// Backwards-compat wrapper that uses
/// [`ConvertLimits::default`].  Internal call sites and the
/// public API surface route through
/// [`parse_tfjs_manifest_with_limits`] so converter callers
/// can override the production caps; this thin wrapper
/// preserves the long-standing `parse_tfjs_manifest(bytes)`
/// shape used by adjacent tests in the workspace (e.g.
/// `inference::backbone::tests`, the converter tests below).
#[allow(dead_code)] // Used by sibling-crate tests; lib build sees no callers.
pub(crate) fn parse_tfjs_manifest(model_json_bytes: &[u8]) -> Result<TfjsManifest, ConvertError> {
    parse_tfjs_manifest_with_limits(model_json_bytes, &ConvertLimits::default())
}

pub(crate) fn parse_tfjs_manifest_with_limits(
    model_json_bytes: &[u8],
    limits: &ConvertLimits,
) -> Result<TfjsManifest, ConvertError> {
    // Cap the raw payload size before paying for JSON parsing.
    // A multi-megabyte adversarial `model.json` would otherwise
    // OOM the parser itself.
    if model_json_bytes.len() > limits.max_manifest_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "manifest_bytes",
            value: model_json_bytes.len() as u64,
            max: limits.max_manifest_bytes as u64,
        });
    }
    let v: serde_json::Value =
        serde_json::from_slice(model_json_bytes).map_err(|e| ConvertError::TfjsParse {
            what: "model.json",
            msg: format!("invalid JSON: {e}"),
        })?;
    let manifest = v
        .get("weightsManifest")
        .and_then(serde_json::Value::as_array)
        .ok_or(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "missing or not an array".to_string(),
        })?;
    if manifest.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "empty array".to_string(),
        });
    }

    let mut entries: Vec<TfjsManifestEntry> = Vec::new();
    let mut shards: Vec<String> = Vec::new();
    let mut offset: usize = 0;

    for (gi, group) in manifest.iter().enumerate() {
        let paths = group
            .get("paths")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].paths",
                msg: format!("group {gi}: missing or not an array"),
            })?;
        for p in paths {
            let s = p.as_str().ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].paths[j]",
                msg: format!("group {gi}: path is not a string"),
            })?;
            // Reject anything that isn't a plain relative path within
            // the model directory: absolute paths, parent traversal,
            // backslashes (Windows separator), drive letters, and
            // empty strings.  The TFJS exporter writes flat names
            // like `weights.bin` or `group1-shard1of2`, so this is
            // strict-by-default.
            validate_shard_path(s)?;
            // Cap shard count BEFORE pushing -- an adversarial
            // manifest could otherwise declare millions of paths
            // (each `String` is 24 B + content) and balloon the
            // `shards` Vec.
            if shards.len() >= limits.max_shards {
                return Err(ConvertError::LimitExceeded {
                    what: "shards",
                    value: (shards.len() as u64).saturating_add(1),
                    max: limits.max_shards as u64,
                });
            }
            shards.push(s.to_string());
        }
        let weights = group
            .get("weights")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].weights",
                msg: format!("group {gi}: missing or not an array"),
            })?;
        for (wi, w) in weights.iter().enumerate() {
            let name = w
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.name",
                    msg: format!("group {gi} entry {wi}: missing or non-string"),
                })?
                .to_string();
            let dtype = w
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("float32");
            if dtype != "float32" {
                return Err(ConvertError::TfjsDtype {
                    name,
                    dtype: dtype.to_string(),
                });
            }
            let shape_arr = w
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.shape",
                    msg: format!("`{name}`: missing or not an array"),
                })?;
            let mut shape: Vec<usize> = Vec::with_capacity(shape_arr.len());
            // Cloned on each call so the error carries the partial
            // shape accumulated up to the failing dim.  Takes args
            // (rather than capturing) so the loop below can keep
            // mutating `shape` between calls.
            let overflow = |name: &str, shape: &[usize]| ConvertError::TfjsShapeOverflow {
                name: name.to_string(),
                shape: shape.to_vec(),
            };
            for d in shape_arr {
                let n = d.as_u64().ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.shape[]",
                    msg: format!("`{name}`: shape dim is not a non-negative integer"),
                })?;
                // On 32-bit hosts (ARMv7, etc.) the manifest may
                // declare a u64 dim that doesn't fit in usize.  Carry
                // the partial shape through the error so the
                // operator sees which dim blew up, not just the name.
                let n_us = usize::try_from(n).map_err(|_| overflow(&name, &shape))?;
                shape.push(n_us);
            }
            // Reject zero dimensions BEFORE the product-checked
            // arithmetic below.  A `[D, 0]` kernel or `[0]` bias
            // would parse as a zero-byte tensor and pass the
            // overflow check trivially, then crash inside Burn's
            // `LinearConfig::init` (zero out-features) or load a
            // head with `n_classes = 0` that inference cannot
            // dispatch.  Structured rejection here gives the
            // operator a precise diagnostic instead of a downstream
            // panic.
            if shape.contains(&0) {
                return Err(ConvertError::TfjsZeroDimension {
                    name: name.clone(),
                    shape: shape.clone(),
                });
            }
            // Overflow-safe count = product(shape); len_bytes = count * 4.
            let mut count: usize = 1;
            for &d in &shape {
                count = count
                    .checked_mul(d)
                    .ok_or_else(|| overflow(&name, &shape))?;
            }
            let len_bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| overflow(&name, &shape))?;
            // For potential head kernels and biases, gate on
            // the declared class-count dimension FIRST -- it's
            // the more specific diagnostic and operators care
            // about the class count, not the byte product.
            // A 2-D weight with `shape[0] == BACKBONE_FEATURE_DIM`
            // is the shape-based head kernel; `shape[1]` is the
            // prospective `n_classes`.  A 1-D weight is the
            // prospective head bias; `shape[0]` is its declared
            // length.  Catching these at parse time keeps
            // pathological N out of the locator and downstream
            // allocators.
            if shape.len() == 2
                && shape[0] == BACKBONE_FEATURE_DIM
                && shape[1] > limits.max_n_classes
            {
                return Err(ConvertError::LimitExceeded {
                    what: "n_classes",
                    value: shape[1] as u64,
                    max: limits.max_n_classes as u64,
                });
            }
            if shape.len() == 1 && shape[0] > limits.max_n_classes {
                return Err(ConvertError::LimitExceeded {
                    what: "n_classes",
                    value: shape[0] as u64,
                    max: limits.max_n_classes as u64,
                });
            }
            // Per-tensor byte cap.  We apply `max_kernel_bytes`
            // here as the upper envelope across ALL tensors
            // (backbone weights are smaller still); the smaller
            // `max_bias_bytes` is enforced once the head bias is
            // identified.  This gates manifest-declared
            // allocations long before any shard data is read.
            if len_bytes > limits.max_kernel_bytes {
                return Err(ConvertError::LimitExceeded {
                    what: "tensor_bytes",
                    value: len_bytes as u64,
                    max: limits.max_kernel_bytes as u64,
                });
            }
            entries.push(TfjsManifestEntry {
                name,
                shape,
                offset_bytes: offset,
                len_bytes,
            });
            offset =
                offset
                    .checked_add(len_bytes)
                    .ok_or_else(|| ConvertError::TfjsShapeOverflow {
                        name: "<cumulative offset>".to_string(),
                        shape: vec![],
                    })?;
        }
    }

    if entries.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "no weight entries declared".to_string(),
        });
    }
    if shards.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "no shard paths declared".to_string(),
        });
    }
    Ok(TfjsManifest { entries, shards })
}

/// Locate the head's kernel + bias entries in a parsed TFJS manifest.
/// Mirrors the upstream binary's two-stage strategy.
///
/// `pub(crate)` so the streaming reader in `source.rs` can call it
/// before any shard bytes are touched.
pub(crate) fn pick_tfjs_head_entries(
    manifest: &TfjsManifest,
) -> Result<(&TfjsManifestEntry, &TfjsManifestEntry), ConvertError> {
    let kernel = manifest
        .entries
        .iter()
        .find(|e| e.name.ends_with("NewHeadDense/kernel"));
    let bias = manifest
        .entries
        .iter()
        .find(|e| e.name.ends_with("NewHeadDense/bias"));
    if let (Some(k), Some(b)) = (kernel, bias) {
        return Ok((k, b));
    }

    // Shape-based fallback.  Disambiguate by requiring shape[0] ==
    // BACKBONE_FEATURE_DIM.  `dense_1` is `[704, 2000]` so it won't
    // collide unless someone trains a 2000->2000 layer -- at which
    // point we error and ask the user to use the canonical naming.
    let kernel_candidates: Vec<&TfjsManifestEntry> = manifest
        .entries
        .iter()
        .filter(|e| e.shape.len() == 2 && e.shape[0] == BACKBONE_FEATURE_DIM)
        .collect();
    if kernel_candidates.len() != 1 {
        return Err(ConvertError::TfjsLocator(format!(
            "expected exactly one 2-D weight with shape [{BACKBONE_FEATURE_DIM}, N], \
             got {} candidates. manifest:\n{}",
            kernel_candidates.len(),
            format_manifest(manifest)
        )));
    }
    let k = kernel_candidates[0];
    let n = k.shape[1];
    let bias_candidates: Vec<&TfjsManifestEntry> =
        manifest.entries.iter().filter(|e| e.shape == [n]).collect();
    if bias_candidates.len() != 1 {
        return Err(ConvertError::TfjsLocator(format!(
            "expected exactly one 1-D weight with shape [{n}] to pair with `{}`, got {}. manifest:\n{}",
            k.name,
            bias_candidates.len(),
            format_manifest(manifest)
        )));
    }
    Ok((k, bias_candidates[0]))
}

fn format_manifest(manifest: &TfjsManifest) -> String {
    manifest
        .entries
        .iter()
        .map(|e| format!("  {} {:?}", e.name, e.shape))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn extract_head_from_tfjs_buffers(
    manifest: &TfjsManifest,
    weights_blob: &[u8],
) -> Result<HeadWeights, ConvertError> {
    // Total declared payload size = last entry's offset + len.  Already
    // overflow-checked at parse time, so a plain sum is safe.
    let declared: usize = manifest
        .entries
        .last()
        .map(|e| e.offset_bytes + e.len_bytes)
        .unwrap_or(0);
    if weights_blob.len() != declared {
        return Err(ConvertError::TfjsBlobLength {
            have: weights_blob.len(),
            declared,
        });
    }

    let (k_entry, b_entry) = pick_tfjs_head_entries(manifest)?;

    if k_entry.shape.len() != 2 || k_entry.shape[0] != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::TfjsLocator(format!(
            "head kernel `{}` has shape {:?}; expected 2-D [{BACKBONE_FEATURE_DIM}, N]",
            k_entry.name, k_entry.shape
        )));
    }
    let n_classes = k_entry.shape[1];
    if b_entry.shape != [n_classes] {
        return Err(ConvertError::TfjsLocator(format!(
            "head bias `{}` has shape {:?}; expected [{n_classes}]",
            b_entry.name, b_entry.shape
        )));
    }
    validate_head_class_count(n_classes, &ConvertLimits::default())?;

    let need = k_entry
        .offset_bytes
        .saturating_add(k_entry.len_bytes)
        .max(b_entry.offset_bytes.saturating_add(b_entry.len_bytes));
    // Strictly speaking redundant given the total-blob-length check
    // above (`need <= declared == weights_blob.len()`), but kept as a
    // belt-and-suspenders gate in case the strict-equality blob check
    // is ever loosened (e.g. to allow trailing padding bytes).
    if weights_blob.len() < need {
        return Err(ConvertError::TfjsShortRead {
            have: weights_blob.len(),
            need,
        });
    }

    // Keras Dense kernel layout `[in, out]` matches Burn's
    // `Linear.weight` -- copy verbatim, no transpose.  Use the
    // checked decoder so NaN / +-Inf weights are rejected with a
    // structured `NonFiniteWeight` naming the source tensor and
    // element index instead of being published into `head.mpk`.
    let kernel_count = BACKBONE_FEATURE_DIM.checked_mul(n_classes).ok_or_else(|| {
        ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        }
    })?;
    let kernel = read_f32_at_checked(
        weights_blob,
        k_entry.offset_bytes,
        kernel_count,
        &k_entry.name,
    )?;
    let bias = read_f32_at_checked(weights_blob, b_entry.offset_bytes, n_classes, &b_entry.name)?;

    Ok(HeadWeights {
        kernel,
        bias,
        n_classes,
        in_dim: BACKBONE_FEATURE_DIM,
    })
}

/// Decode `count` little-endian f32 values starting at
/// `offset` in `bytes`, rejecting non-finite values (NaN,
/// +/-Inf).  Used at the converter boundary because TFJS
/// source models are operator-supplied; a non-finite
/// weight published into `head.mpk` would later trip the
/// inference cold path's finite validator and fail the
/// workspace activation.  Catching it here gives the
/// operator a precise diagnostic naming the source tensor
/// and offending element index.
///
/// Caller has already validated that `offset + count*4 <=
/// bytes.len()` (via the per-shape overflow check at parse
/// time plus the blob-length / kernel-bytes checks in the
/// caller).
fn read_f32_at_checked(
    bytes: &[u8],
    offset: usize,
    count: usize,
    tensor: &str,
) -> Result<Vec<f32>, ConvertError> {
    let end = offset + count * std::mem::size_of::<f32>();
    let slice = &bytes[offset..end];
    let mut out = vec![0.0f32; count];
    for (i, chunk) in slice.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !v.is_finite() {
            return Err(ConvertError::NonFiniteWeight {
                tensor: tensor.to_string(),
                index: i,
                value: v,
            });
        }
        out[i] = v;
    }
    Ok(out)
}

/// Sibling of [`extract_head_from_tfjs_buffers`] for
/// the streaming-read path.  Takes the kernel + bias byte ranges
/// already extracted (one slice each, length-matching
/// `entry.len_bytes`) and validates the same shape invariants
/// before constructing [`HeadWeights`].  The streaming reader in
/// `source.rs::read_head_bytes_streaming` produces these slices
/// without materializing the full concatenated shard set.
pub(crate) fn head_weights_from_head_byte_ranges(
    _manifest: &TfjsManifest,
    k_entry: &TfjsManifestEntry,
    b_entry: &TfjsManifestEntry,
    kernel_bytes: &[u8],
    bias_bytes: &[u8],
) -> Result<HeadWeights, ConvertError> {
    if k_entry.shape.len() != 2 || k_entry.shape[0] != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::TfjsLocator(format!(
            "head kernel `{}` has shape {:?}; expected 2-D [{BACKBONE_FEATURE_DIM}, N]",
            k_entry.name, k_entry.shape
        )));
    }
    let n_classes = k_entry.shape[1];
    if b_entry.shape != [n_classes] {
        return Err(ConvertError::TfjsLocator(format!(
            "head bias `{}` has shape {:?}; expected [{n_classes}]",
            b_entry.name, b_entry.shape
        )));
    }
    let limits = ConvertLimits::default();
    validate_head_class_count(n_classes, &limits)?;
    let kernel_count = BACKBONE_FEATURE_DIM.checked_mul(n_classes).ok_or_else(|| {
        ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        }
    })?;
    let kernel_need = kernel_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        })?;
    // Belt-and-suspenders byte cap.  parse_tfjs_manifest already
    // checked this against `max_kernel_bytes`, but
    // `head_weights_from_head_byte_ranges` is called from a
    // different path (the streaming reader) so re-checking
    // closes any future loosening of the parse-time cap.
    if kernel_need > limits.max_kernel_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "kernel_bytes",
            value: kernel_need as u64,
            max: limits.max_kernel_bytes as u64,
        });
    }
    if kernel_bytes.len() != kernel_need {
        return Err(ConvertError::TfjsShortRead {
            have: kernel_bytes.len(),
            need: kernel_need,
        });
    }
    let bias_need = n_classes
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: b_entry.name.clone(),
            shape: b_entry.shape.clone(),
        })?;
    if bias_need > limits.max_bias_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "bias_bytes",
            value: bias_need as u64,
            max: limits.max_bias_bytes as u64,
        });
    }
    if bias_bytes.len() != bias_need {
        return Err(ConvertError::TfjsShortRead {
            have: bias_bytes.len(),
            need: bias_need,
        });
    }
    // Keras Dense kernel layout `[in, out]` matches Burn's
    // `Linear.weight` -- copy verbatim, no transpose.  The
    // checked decoder rejects NaN / +-Inf with a structured
    // diagnostic naming the source tensor + element index, so
    // bad weights never reach `Head::try_new` / Burn's
    // `from_data` (which would otherwise propagate them into
    // `head.mpk`).
    let kernel = read_f32_at_checked(kernel_bytes, 0, kernel_count, &k_entry.name)?;
    let bias = read_f32_at_checked(bias_bytes, 0, n_classes, &b_entry.name)?;
    Ok(HeadWeights {
        kernel,
        bias,
        n_classes,
        in_dim: BACKBONE_FEATURE_DIM,
    })
}

// MARK: convert pipeline

/// Inputs to the convert worker.  Built by the api producer from a
/// validated [`crate::file_mgr::ConvertRequest`] + snapshotted state;
/// all paths are pre-resolved under `<workspace_dir>/converters/` so
/// the worker reads them directly without re-validating.
#[derive(Clone, Debug)]
pub struct ConvertJob {
    /// Drives the JSONL log filename
    /// `<workspace_dir>/converter_logs/<job_id>.jsonl`.
    pub job_id: crate::common::ids::JobId,
    pub workspace_id: crate::common::ids::WorkspaceId,
    /// Pre-allocated; published verbatim on success.
    pub head_id: crate::common::ids::HeadId,
    /// Producer-snapshotted; recorded in the per-head manifest for
    /// stale detection.
    pub workspace_revision: crate::common::workspace::WorkspaceRevision,
    pub model_json_path: PathBuf,
    /// In manifest-declared order; per-shard bytes are cross-checked
    /// against declared offsets inside [`run_convert_job`].
    pub shard_paths: Vec<PathBuf>,
    pub labels_path: PathBuf,
    pub labels_format: crate::file_mgr::LabelsFormat,
}

/// Run a convert job end-to-end: open the JSONL log, parse the
/// manifest, stream shards into staging, build [`HeadWeights`],
/// resolve labels, stage the ACSTHEAD-wrapped `.mpk`, and call
/// `FsService::publish_trained_head` to land the head into the
/// workspace's 2-slot rotation.  On failure no head record is
/// committed; the staged `.mpk` is best-effort cleaned up and the
/// JSONL log records the terminal error.  The caller holds the
/// convert permit + workspace job-reference until this returns.
pub fn run_convert_job(
    files: std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: ConvertJob,
) -> Result<ConvertOutcome, ConvertError> {
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &job.workspace_id);
    let mut log = ConvertJobLog::open(&workspace_dir, job.job_id)?;
    log.event("started", None, None)?;
    let result = run_convert_job_inner(&files, &job, &workspace_dir, &mut log);

    // Unconditional sweep of per-job tempfiles; both removes are
    // NotFound-tolerant since on success the `.mpk` was renamed
    // out by `publish_trained_head` and only the staging dir
    // remains.
    let staging_dir = convert_staging_dir(&workspace_dir, job.job_id);
    let mpk_tempfile = convert_mpk_tempfile(&workspace_dir, job.job_id, job.head_id);
    if let Err(e) = std::fs::remove_dir_all(&staging_dir)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            target: "converter",
            err = %e,
            path = %staging_dir.display(),
            "convert: failed to remove staging dir; boot recovery should sweep",
        );
    }
    if let Err(e) = std::fs::remove_file(&mpk_tempfile)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            target: "converter",
            err = %e,
            path = %mpk_tempfile.display(),
            "convert: failed to remove .mpk tempfile; boot recovery should sweep",
        );
    }

    // Terminal log write must not promote a log-IO error over the
    // inner result: a failed `completed` write after a successful
    // publish would otherwise make the caller see Err while the
    // head is on disk.  Surface log failures via tracing instead.
    match &result {
        Ok(_) => {
            if let Err(log_err) = log.event("completed", None, None) {
                tracing::warn!(
                    target: "converter",
                    err = %log_err,
                    "convert succeeded but terminal `completed` log write failed",
                );
            }
        }
        Err(e) => {
            if let Err(log_err) = log.event("failed", None, Some(&e.to_string())) {
                tracing::warn!(
                    target: "converter",
                    err = %log_err,
                    original = %e,
                    "convert failed and terminal `failed` log write also failed",
                );
            }
        }
    }
    result
}

/// Path of the per-job staging directory under the workspace's
/// `.tmp/`.  Centralized so the inner pipeline and the outer
/// cleanup wrapper agree on the location.
fn convert_staging_dir(workspace_dir: &Path, job_id: crate::common::ids::JobId) -> PathBuf {
    workspace_dir.join(".tmp").join(format!("convert-{job_id}"))
}

/// Path of the per-job `.mpk` tempfile under the workspace's
/// `.tmp/`.  On success the file is renamed out by
/// `publish_trained_head`; on failure it remains for the wrapper
/// to clean up.
fn convert_mpk_tempfile(
    workspace_dir: &Path,
    job_id: crate::common::ids::JobId,
    head_id: crate::common::ids::HeadId,
) -> PathBuf {
    workspace_dir
        .join(".tmp")
        .join(format!("convert-{job_id}-{head_id}.mpk"))
}

/// Terminal outcome of a successful convert job.  Surfaced on
/// `JobResult::Convert` so `GET /jobs/{job_id}` carries the produced
/// head's identity + integrity hash + class count without requiring
/// the operator to read the JSONL log.
#[derive(Clone, Debug)]
pub struct ConvertOutcome {
    /// Matches the input `ConvertJob.head_id` verbatim.
    pub head_id: crate::common::ids::HeadId,
    /// Lowercase-hex SHA-256 of the published `.mpk` bytes.
    pub sha256: String,
    pub n_classes: u32,
}

/// Inner body of [`run_convert_job`]; the wrapper opens / closes the
/// JSONL log so terminal events are always recorded.
fn run_convert_job_inner(
    files: &std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: &ConvertJob,
    workspace_dir: &Path,
    log: &mut ConvertJobLog,
) -> Result<ConvertOutcome, ConvertError> {
    log.event("read_model_json", None, None)?;
    let json_bytes = std::fs::read(&job.model_json_path)
        .map_err(|e| convert_read_err(job.model_json_path.display(), e))?;
    let limits = ConvertLimits::default();
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // The converter reads via `job.shard_paths` (not the manifest
    // names), but cardinality must match so the streaming reader
    // walks the right number of files.
    if manifest.shards.len() != job.shard_paths.len() {
        return Err(ConvertError::TfjsParse {
            what: "shards",
            msg: format!(
                "request declared {} shards but manifest declares {}",
                job.shard_paths.len(),
                manifest.shards.len(),
            ),
        });
    }

    // Stage the shard set into a per-job dir under `.tmp/` so the
    // streaming reader can resolve the manifest's declared paths
    // (which are sibling-relative to `model.json`).
    let tmp_root = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_root).map_err(|e| convert_write_err(tmp_root.display(), e))?;
    let staging_dir = convert_staging_dir(workspace_dir, job.job_id);
    std::fs::create_dir_all(&staging_dir)
        .map_err(|e| convert_write_err(staging_dir.display(), e))?;
    // Stage via `put_atomic` so writes share the daemon-wide
    // atomic-write discipline.  Shard bytes are bounded by
    // `ConvertLimits::max_kernel_bytes` (~80 MiB envelope) -- well
    // within the staging FS budget.
    let model_json_staged = staging_dir.join("model.json");
    files
        .put_atomic(&model_json_staged, &json_bytes)
        .map_err(|e| {
            convert_write_err(
                model_json_staged.display(),
                std::io::Error::other(e.to_string()),
            )
        })?;
    for (declared, src) in manifest.shards.iter().zip(job.shard_paths.iter()) {
        let dst = staging_dir.join(declared);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| convert_write_err(parent.display(), e))?;
        }
        // Hard-link intra-FS to skip the ~80 MiB heap copy per
        // shard; fall back to `fs::copy` if the FS rejects links
        // (EXDEV / unsupported).  NotFound on the source is a
        // real Read error and surfaces as such.
        if let Err(e) = std::fs::hard_link(src, &dst) {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Err(convert_read_err(src.display(), e));
            }
            std::fs::copy(src, &dst).map_err(|source| convert_read_err(src.display(), source))?;
        }
    }

    log.event("extract_weights", None, None)?;
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    // The source-bundle sha is not persisted on the head manifest
    // schema; pass `None` so the streaming reader skips the hasher
    // work over the (potentially ~80 MiB) shard set.
    let (kernel_bytes, bias_bytes) =
        source::read_head_bytes_streaming(&staging_dir, &manifest, k_entry, b_entry, None)?;
    let weights = head_weights_from_head_byte_ranges(
        &manifest,
        k_entry,
        b_entry,
        &kernel_bytes,
        &bias_bytes,
    )?;

    log.event("read_labels", None, None)?;
    let labels = read_labels_from_path(&job.labels_path, job.labels_format, weights.n_classes)?;

    log.event("stage_head_mpk", None, None)?;
    // Build the ACSTHEAD-wrapped `.mpk` in memory and stage it under
    // `.tmp/` so the rotation primitive can rename it intra-FS into
    // `heads/<head_id>.mpk` atomically.
    let head_blob = build_head_mpk_blob(&weights)?;
    let head_sha256 = hex_lowercase(&Sha256::digest(&head_blob));
    let head_size = head_blob.len() as u64;
    let mpk_tempfile = convert_mpk_tempfile(workspace_dir, job.job_id, job.head_id);
    files.put_atomic(&mpk_tempfile, &head_blob).map_err(|e| {
        convert_write_err(mpk_tempfile.display(), std::io::Error::other(e.to_string()))
    })?;

    let n_classes_u32 =
        u32::try_from(weights.n_classes).map_err(|_| ConvertError::BadClassCount {
            got: weights.n_classes,
            max: ConvertLimits::default().max_n_classes,
        })?;

    // Heads carry only sha256 / n_classes / size_bytes / labels /
    // workspace_revision; the JSONL convert log is the durable
    // record of which inputs produced this head.
    let manifest_struct = crate::common::workspace::HeadManifest {
        head_id: job.head_id,
        workspace_id: job.workspace_id,
        workspace_revision: job.workspace_revision.clone(),
        sha256: head_sha256.clone(),
        n_classes: n_classes_u32,
        size_bytes: head_size,
        created_at: crate::file_mgr::now_rfc3339(),
        labels: labels.clone(),
    };
    let pending = crate::file_mgr::PendingHead {
        head_id: job.head_id,
        mpk_tempfile: mpk_tempfile.clone(),
        manifest: manifest_struct,
    };

    log.event("publish_head", None, None)?;
    files
        .publish_trained_head(&job.workspace_id, pending)
        .map_err(|e| {
            convert_write_err(
                workspace_dir.display(),
                std::io::Error::other(e.to_string()),
            )
        })?;

    // Tempfile cleanup happens unconditionally in `run_convert_job`
    // after the inner returns -- this covers the failure paths
    // above too without each `?` site needing its own scrub.
    Ok(ConvertOutcome {
        head_id: job.head_id,
        sha256: head_sha256,
        n_classes: n_classes_u32,
    })
}

/// Read labels from `labels_path` per `labels_format`, then
/// cross-validate the count against `expected_n_classes` from
/// the head kernel's shape.
fn read_labels_from_path(
    labels_path: &Path,
    labels_format: crate::file_mgr::LabelsFormat,
    expected_n_classes: usize,
) -> Result<Vec<String>, ConvertError> {
    let labels = match labels_format {
        crate::file_mgr::LabelsFormat::Lines => {
            let text = std::fs::read_to_string(labels_path)
                .map_err(|e| convert_read_err(labels_path.display(), e))?;
            text.lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        }
        crate::file_mgr::LabelsFormat::TfjsMetadata => read_tfjs_labels(labels_path)?,
    };
    if labels.len() != expected_n_classes {
        return Err(ConvertError::Labels(format!(
            "labels source declares {} entries; head kernel declares {} classes",
            labels.len(),
            expected_n_classes,
        )));
    }
    Ok(labels)
}

/// Build the ACSTHEAD-wrapped `.mpk` payload as one in-memory blob.
/// Callers pair this with [`FsService::put_atomic`] -- either
/// directly into `<dst_dir>/head.mpk` (the
/// [`write_head_artifacts`] path) or after staging under `.tmp/`
/// for publication through the head rotation primitive (the
/// convert-pipeline path).
fn build_head_mpk_blob(weights: &HeadWeights) -> Result<Vec<u8>, ConvertError> {
    validate_head_class_count(weights.n_classes, &ConvertLimits::default())?;
    if weights.in_dim != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::Tensor(format!(
            "kernel in_dim {} != BACKBONE_FEATURE_DIM {BACKBONE_FEATURE_DIM}",
            weights.in_dim
        )));
    }
    let device: burn::tensor::Device<B> = Default::default();
    let mut head = Head::<B>::try_new(weights.n_classes, &device).map_err(|e| match e {
        model::Error::BadClassCount { got, max } => ConvertError::BadClassCount { got, max },
        other => ConvertError::Tensor(format!("head construct: {other}")),
    })?;
    let kernel_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(weights.kernel.clone(), [weights.in_dim, weights.n_classes]),
        &device,
    );
    let bias_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(weights.bias.clone(), [weights.n_classes]),
        &device,
    );
    head.linear.weight = Param::from_tensor(kernel_tensor);
    head.linear.bias = Some(Param::from_tensor(bias_tensor));
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
    let payload = recorder
        .record(head.into_record(), ())
        .map_err(|e| ConvertError::Record(format!("{e}")))?;
    let header = crate::common::head_header::serialize_header(
        weights.in_dim as u32,
        weights.n_classes as u32,
        payload.len() as u32,
    );
    let mut blob: Vec<u8> = Vec::with_capacity(header.len() + payload.len());
    blob.extend_from_slice(&header);
    blob.extend_from_slice(&payload);
    Ok(blob)
}

/// Bounded JSONL writer for
/// `<workspace_dir>/converter_logs/<job_id>.jsonl`.  One event per
/// line; flushes once per terminal event (open / publish / close).
/// `message` is capped at 8 KiB; structural fields stay uncapped
/// because they are fixed-shape.
struct ConvertJobLog {
    file: std::fs::File,
    seq: u64,
}

impl ConvertJobLog {
    fn open(workspace_dir: &Path, job_id: crate::common::ids::JobId) -> Result<Self, ConvertError> {
        let dir = workspace_dir.join("converter_logs");
        std::fs::create_dir_all(&dir).map_err(|e| convert_write_err(dir.display(), e))?;
        let path = dir.join(format!("{job_id}.jsonl"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| convert_write_err(path.display(), e))?;
        Ok(Self { file, seq: 0 })
    }

    /// Append one JSONL event line.  `state` carries the
    /// lifecycle phase (`started`, `read_model_json`, ...,
    /// `completed`, `failed`); `message` is an optional bounded
    /// diagnostic payload.
    fn event(
        &mut self,
        state: &str,
        progress: Option<u64>,
        message: Option<&str>,
    ) -> Result<(), ConvertError> {
        use std::io::Write as _;
        self.seq = self.seq.saturating_add(1);
        let now = crate::file_mgr::now_rfc3339();
        let truncated_msg = message.map(truncate_log_message);
        let line = serde_json::json!({
            "seq": self.seq,
            "at": now,
            "state": state,
            "progress": progress,
            "message": truncated_msg,
        });
        let mut bytes = serde_json::to_vec(&line).map_err(ConvertError::MetadataSerialize)?;
        bytes.push(b'\n');
        self.file
            .write_all(&bytes)
            .map_err(|e| convert_write_err("<converter_logs>", e))?;
        // Best-effort flush.  fsync per-line would 10x the cost
        // for negligible recovery benefit (a crash mid-job loses
        // at most the trailing event; the workspace state is
        // recovered via boot recovery).
        let _ = self.file.flush();
        Ok(())
    }
}

/// Validate `n_classes` against `1..=ConvertLimits::max_n_classes`.
/// Mirrors `model::Head::try_new`'s rejection so a structured
/// `ConvertError` reaches the operator instead of a downstream
/// `model::Error` surfaced as Internal.
fn validate_head_class_count(n_classes: usize, limits: &ConvertLimits) -> Result<(), ConvertError> {
    if n_classes == 0 || n_classes > limits.max_n_classes {
        return Err(ConvertError::BadClassCount {
            got: n_classes,
            max: limits.max_n_classes,
        });
    }
    Ok(())
}

/// Reject TFJS shard `paths` entries that could escape the model
/// directory or are otherwise non-portable.  We only allow strictly
/// relative paths whose components are all `Normal` (no `..`, `.`,
/// no root, no prefix).  Backslashes are rejected to keep behavior
/// identical across platforms (a literal backslash file name on
/// Linux is allowed by the OS but never produced by the TFJS
/// exporter -- disallowing avoids ambiguity).  NUL bytes are also
/// rejected: `std::fs::read` would error on them anyway, but an
/// explicit reject is a cheaper, clearer diagnostic and prevents
/// any future code path that uses the path string as a key.
fn validate_shard_path(s: &str) -> Result<(), ConvertError> {
    if s.is_empty() || s.contains('\\') || s.contains('\0') {
        return Err(ConvertError::TfjsUnsafePath(s.to_string()));
    }
    let p = std::path::Path::new(s);
    if p.is_absolute() {
        return Err(ConvertError::TfjsUnsafePath(s.to_string()));
    }
    for comp in p.components() {
        match comp {
            std::path::Component::Normal(_) => {}
            // CurDir (`.`), ParentDir (`..`), RootDir, Prefix all rejected.
            _ => return Err(ConvertError::TfjsUnsafePath(s.to_string())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Test helpers stage `.mpk` payloads via
    // `std::fs::write` (transient stripped-payload
    // tempfiles); the production constraint in `clippy.toml`
    // (writes go through file_mgr's atomic writer) doesn't
    // apply to test scaffolding.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use crate::model::HeadRecord;
    use burn::record::NamedMpkFileRecorder;

    /// Resolves the crate root.
    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    /// Test helper: strip the 32-byte ACSTHEAD header
    /// from a `.mpk` file and write the prost-payload portion
    /// to a sibling tempfile.  Production code goes through
    /// `inference::HotHead::load` which does the same dance
    /// atomically; the converter crate doesn't depend on
    /// inference, so tests carry the helper inline.
    fn strip_head_header(mpk: &Path, dir: &Path) -> PathBuf {
        let bytes = std::fs::read(mpk).expect("read mpk");
        let payload = &bytes[crate::common::head_header::HEAD_HEADER_SIZE..];
        let stripped = dir.join("payload_only.mpk");
        std::fs::write(&stripped, payload).expect("write stripped");
        stripped
    }

    #[test]
    fn convert_tfjs_writes_artifacts() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            return;
        }
        let labels = read_tfjs_labels(&dir.join("metadata.json")).expect("labels");
        let dst = tempfile::tempdir().expect("tempdir");
        // `convert_tfjs` now drives writes through
        // an `FsService` so its atomic-write discipline is
        // unified with the rest of the daemon.  Tests construct
        // a throwaway `FsServiceImpl` rooted at any tempdir
        // (the converter only uses `put_atomic`, which is
        // workspace-agnostic).
        let fs_root = tempfile::tempdir().expect("fs root");
        let fs = crate::file_mgr::FsServiceImpl::new(fs_root.path().to_path_buf());
        let arts = convert_tfjs(&dir, &labels, dst.path(), &fs).expect("convert");
        assert!(arts.head_mpk.exists());
        assert!(arts.labels_txt.exists());
        assert!(arts.metadata_json.exists());
        assert_eq!(arts.n_classes, labels.len());
        assert_eq!(arts.source_sha256.len(), 64);

        // Verify no `head-partial-*` /
        // `head-with-header-partial-*` siblings linger after
        // a successful convert.  The new in-memory build means
        // dst_dir contains EXACTLY the published triple.
        let mut entries: Vec<String> = std::fs::read_dir(dst.path())
            .expect("read dst")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert_eq!(
            entries,
            vec![
                "head.mpk".to_string(),
                "labels.txt".to_string(),
                "metadata.json".to_string(),
            ],
            "dst dir should contain exactly the published triple",
        );

        let meta_bytes = std::fs::read(&arts.metadata_json).unwrap();
        let meta: ConversionMetadata = serde_json::from_slice(&meta_bytes).unwrap();
        assert_eq!(meta.source_kind, SourceKind::Tfjs);
        assert_eq!(meta.n_classes, labels.len());
        assert_eq!(meta.labels, labels);

        // Re-load and verify shape. strip the
        // ACSTHEAD header before handing to Burn's recorder
        // (production decode goes through `inference::HotHead::load`).
        let payload_only = strip_head_header(&arts.head_mpk, dst.path());
        let device: burn::tensor::Device<B> = Default::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let head_rec: HeadRecord<B> = recorder
            .load(payload_only.clone(), &device)
            .expect("load head.mpk");
        let head = Head::<B>::new(1, &device).load_record(head_rec);
        assert_eq!(
            head.linear.weight.val().dims(),
            [BACKBONE_FEATURE_DIM, labels.len()]
        );
    }

    // Fixture-mutating regen helpers live in
    // `examples/regen_fixtures.rs` so `cargo test
    // --include-ignored` does not rewrite tracked fixtures.

    /// `read_tfjs_labels` and `extract_head_from_tfjs_dir` agree
    /// on class count for the upstream Google Speech-Commands
    /// bundle in `misc/models/`, exercising the multi-shard read
    /// path and the `words`-schema fallback.  Silently skipped
    /// when the gitignored bundle is absent.
    #[test]
    fn tfjs_labels_match_head_models_dir() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            return;
        }
        let labels = read_tfjs_labels(&dir.join("metadata.json")).expect("labels");
        let weights = extract_head_from_tfjs_dir(&dir).expect("weights");
        assert_eq!(labels.len(), weights.n_classes);
    }

    /// `read_tfjs_labels` accepts both `wordLabels` (Teachable
    /// Machine) and `words` (upstream Speech-Commands), with
    /// `wordLabels` winning when both are present.  Uses synthetic
    /// inputs so the test runs without the gitignored upstream
    /// bundle.
    #[test]
    fn read_tfjs_labels_accepts_words_and_word_labels() {
        let dir = tempfile::tempdir().expect("tempdir");

        let tm_path = dir.path().join("tm.json");
        std::fs::write(
            &tm_path,
            br#"{"wordLabels":["a","b","c"],"modelName":"tm"}"#,
        )
        .unwrap();
        assert_eq!(read_tfjs_labels(&tm_path).unwrap(), ["a", "b", "c"]);

        let sc_path = dir.path().join("sc.json");
        std::fs::write(&sc_path, br#"{"words":["go","stop"],"frameSize":232}"#).unwrap();
        assert_eq!(read_tfjs_labels(&sc_path).unwrap(), ["go", "stop"]);

        // Both keys present: `wordLabels` wins (defensive tiebreak).
        let both_path = dir.path().join("both.json");
        std::fs::write(&both_path, br#"{"wordLabels":["x"],"words":["y","z"]}"#).unwrap();
        assert_eq!(read_tfjs_labels(&both_path).unwrap(), ["x"]);
    }

    /// Neither key present surfaces the structured `Labels` error
    /// naming both accepted schemas, not a confusing single-key
    /// diagnostic.
    #[test]
    fn read_tfjs_labels_errors_when_neither_key_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("missing.json");
        std::fs::write(&p, br#"{"otherField":42}"#).unwrap();
        let err = read_tfjs_labels(&p).unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, ConvertError::Labels(_)), "{err:?}");
        assert!(
            msg.contains("wordLabels") && msg.contains("words"),
            "diagnostic must list both schemas: {msg}",
        );
    }

    /// Empty array surfaces a structured error naming the key
    /// that was found (so operators know which schema was
    /// detected).
    #[test]
    fn read_tfjs_labels_errors_on_empty_array() {
        let dir = tempfile::tempdir().expect("tempdir");

        let tm_path = dir.path().join("tm_empty.json");
        std::fs::write(&tm_path, br#"{"wordLabels":[]}"#).unwrap();
        let err = read_tfjs_labels(&tm_path).unwrap_err();
        assert!(format!("{err}").contains("wordLabels"));

        let sc_path = dir.path().join("sc_empty.json");
        std::fs::write(&sc_path, br#"{"words":[]}"#).unwrap();
        let err = read_tfjs_labels(&sc_path).unwrap_err();
        assert!(format!("{err}").contains("words"));
    }

    /// Extraction smoke test for the upstream Google Speech-Commands
    /// bundle in `misc/models/`: exercises the multi-shard read
    /// path (`group1-shard{1,2}of2`) and shape-based head matching
    /// (the unique 2-D weight whose first dim equals
    /// `BACKBONE_FEATURE_DIM`).  Silently skipped when the
    /// gitignored bundle is absent.
    #[test]
    fn extract_from_shipped_tfjs_models_dir() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            eprintln!("skipping: {} not present", dir.display());
            return;
        }
        let weights = extract_head_from_tfjs_dir(&dir).expect("extract");
        assert_eq!(weights.in_dim, BACKBONE_FEATURE_DIM);
        assert_eq!(weights.n_classes, 20);
        assert_eq!(weights.kernel.len(), BACKBONE_FEATURE_DIM * 20);
        assert_eq!(weights.bias.len(), 20);
        let nonzero = weights.kernel.iter().filter(|x| **x != 0.0).count();
        assert!(
            nonzero > weights.kernel.len() / 100,
            "implausibly sparse kernel: {nonzero}/{}",
            weights.kernel.len()
        );
    }

    /// Garbage manifest -> structured error, not panic.
    #[test]
    fn tfjs_invalid_json_errors_cleanly() {
        let err = parse_tfjs_manifest(b"{not json").unwrap_err();
        assert!(matches!(err, ConvertError::TfjsParse { .. }));
    }

    /// Path-traversal guard rejects `..`, absolute paths, backslashes,
    /// empty strings, NUL bytes -- and accepts ordinary shard names.
    #[test]
    fn tfjs_validate_shard_path_rejects_unsafe() {
        assert!(validate_shard_path("weights.bin").is_ok());
        assert!(validate_shard_path("group1-shard1of2.bin").is_ok());
        assert!(validate_shard_path("sub/dir/file.bin").is_ok());

        for bad in [
            "",
            "../escape",
            "../../etc/passwd",
            "/abs/path",
            "a\\b",
            ".",
            "weights\0.bin",
            "\0",
        ] {
            let r = validate_shard_path(bad);
            assert!(
                matches!(r, Err(ConvertError::TfjsUnsafePath(_))),
                "expected unsafe-path rejection for {bad:?}, got {r:?}",
            );
        }
    }

    /// Manifest declaring `..` shard path fails parse with TfjsUnsafePath.
    #[test]
    fn tfjs_manifest_path_traversal_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["../etc/passwd"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "float32"},
              {"name": "b", "shape": [3], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(matches!(err, ConvertError::TfjsUnsafePath(_)), "{err:?}");
    }

    /// Adversarial shape that would overflow `usize::product()` is
    /// caught with a structured error instead of UB / panic.
    #[test]
    fn tfjs_manifest_shape_overflow_rejected() {
        let huge = u64::MAX / 2; // product of two will overflow usize
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "huge", "shape": [{huge}, {huge}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsShapeOverflow { .. }),
            "{err:?}"
        );
    }

    /// `ConvertLimits` rejects a manifest whose head kernel
    /// declares `n_classes > max_n_classes`, BEFORE any
    /// allocation derived from that count is attempted.
    /// Uses a class count that fits in `usize` (so the
    /// existing `TfjsShapeOverflow` arm can't fire) but
    /// exceeds the default `max_n_classes` cap; the exact
    /// kernel byte product is `BACKBONE_FEATURE_DIM *
    /// MAX_N_CLASSES * 4 = 800 MB`, which would risk OOM if it
    /// ever reached the allocator.  The `LimitExceeded` variant
    /// carries the offending dimension so operators see the cap
    /// fired.
    #[test]
    fn tfjs_huge_n_classes_rejected_before_allocation() {
        // shape = [BACKBONE_FEATURE_DIM, max+1] -- the parser
        // identifies it as a head-kernel-shape and trips the
        // n_classes cap before len_bytes is evaluated.
        let huge_n = ConvertLimits::default().max_n_classes + 1;
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "k", "shape": [{BACKBONE_FEATURE_DIM}, {huge_n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(what, "n_classes", "{what:?}");
                assert_eq!(value, huge_n as u64);
                assert_eq!(max, ConvertLimits::default().max_n_classes as u64);
            }
            other => panic!("expected LimitExceeded n_classes, got {other:?}"),
        }
    }

    /// Per-tensor byte cap fires when a single weight's
    /// declared `len_bytes` exceeds `max_kernel_bytes` --
    /// even when `n_classes` would have passed the
    /// per-dimension cap.  Constructs a 1-D weight whose
    /// length stays under `max_n_classes` (so it is not
    /// rejected as a candidate bias) but whose 4-byte stride
    /// pushes total bytes over a tightened
    /// `max_kernel_bytes`.
    #[test]
    fn tfjs_per_tensor_byte_cap_rejected() {
        // Tighten kernel cap to 1 KiB so a 600-element f32
        // tensor (2400 bytes) trips it.
        let limits = ConvertLimits {
            max_kernel_bytes: 1024,
            ..ConvertLimits::default()
        };
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "k", "shape": [600], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest_with_limits(json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, .. } => {
                assert_eq!(what, "tensor_bytes", "{what:?}");
            }
            other => panic!("expected LimitExceeded tensor_bytes, got {other:?}"),
        }
    }

    /// Manifest payload size cap is enforced before the
    /// JSON parser is invoked -- a multi-megabyte
    /// `model.json` must not be allowed to OOM the parser.
    #[test]
    fn tfjs_manifest_payload_cap_rejected() {
        let limits = ConvertLimits {
            max_manifest_bytes: 64,
            ..ConvertLimits::default()
        };
        // 128 bytes of valid JSON; no need to be a real
        // manifest because the cap fires before parsing.
        let json = vec![b' '; 128];
        let err = parse_tfjs_manifest_with_limits(&json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(what, "manifest_bytes", "{what:?}");
                assert_eq!(value, 128);
                assert_eq!(max, 64);
            }
            other => panic!("expected LimitExceeded manifest_bytes, got {other:?}"),
        }
    }

    /// Shard-count cap fires when the manifest declares
    /// more `paths` entries than `max_shards` allows.
    #[test]
    fn tfjs_shard_count_cap_rejected() {
        let limits = ConvertLimits {
            max_shards: 2,
            ..ConvertLimits::default()
        };
        // Three valid paths trips a max_shards = 2 cap.
        let json = br#"{
          "weightsManifest": [{
            "paths": ["a.bin", "b.bin", "c.bin"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest_with_limits(json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, .. } => {
                assert_eq!(what, "shards", "{what:?}");
            }
            other => panic!("expected LimitExceeded shards, got {other:?}"),
        }
    }

    /// Zero-dimensional shape (e.g. `[BACKBONE_FEATURE_DIM,
    /// 0]` head kernel or `[0]` bias) is rejected at the
    /// manifest layer with a structured `TfjsZeroDimension`,
    /// preventing a `Head::new(0, ...)` panic downstream.
    #[test]
    fn tfjs_zero_dimension_kernel_rejected() {
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, 0], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        match err {
            ConvertError::TfjsZeroDimension { name, shape } => {
                assert_eq!(name, "head/kernel");
                assert_eq!(shape, vec![BACKBONE_FEATURE_DIM, 0]);
            }
            other => panic!("expected TfjsZeroDimension, got {other:?}"),
        }
    }

    /// Zero-dim bias `[0]` is similarly rejected.
    #[test]
    fn tfjs_zero_dimension_bias_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "head/bias", "shape": [0], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsZeroDimension { .. }),
            "{err:?}"
        );
    }

    /// `write_head_artifacts` rejects `n_classes = 0` even
    /// when called via the public boundary with a
    /// hand-built `HeadWeights`.  Defense in depth so the
    /// converter is the boundary that owns this rejection
    /// for every direct caller, not just the manifest path.
    #[test]
    fn write_head_artifacts_rejects_zero_classes() {
        let weights = HeadWeights {
            kernel: vec![],
            bias: vec![],
            n_classes: 0,
            in_dim: BACKBONE_FEATURE_DIM,
        };
        let dst = tempfile::tempdir().expect("tempdir");
        let fs_root = tempfile::tempdir().expect("fs root");
        let fs = crate::file_mgr::FsServiceImpl::new(fs_root.path().to_path_buf());
        let err = write_head_artifacts(
            &weights,
            &[],
            dst.path(),
            SourceKind::Tfjs,
            "deadbeef".to_string(),
            &fs,
        )
        .unwrap_err();
        match err {
            ConvertError::BadClassCount { got, max } => {
                assert_eq!(got, 0);
                assert_eq!(max, ConvertLimits::default().max_n_classes);
            }
            other => panic!("expected BadClassCount, got {other:?}"),
        }
    }

    /// NaN / +-Inf weights in the TFJS source are rejected
    /// at the converter boundary with a structured
    /// `NonFiniteWeight` naming the offending tensor and
    /// element index.  Without this the bad value would be
    /// published into `head.mpk` and only fail later in
    /// `inference::head::HeadInner::validate_finite`.
    #[test]
    fn tfjs_non_finite_kernel_rejected() {
        // 3-class head: kernel = [BACKBONE_FEATURE_DIM, 3] f32,
        // bias = [3] f32.  Total = (BACKBONE_FEATURE_DIM*3 + 3) * 4 bytes.
        let n = 3usize;
        let kernel_count = BACKBONE_FEATURE_DIM * n;
        let bias_count = n;
        let mut blob = Vec::<u8>::with_capacity((kernel_count + bias_count) * 4);
        // Fill kernel with finite small values, then poison
        // index 17 with NaN.
        for i in 0..kernel_count {
            let v = if i == 17 { f32::NAN } else { 0.5_f32 };
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..bias_count {
            blob.extend_from_slice(&0.25_f32.to_le_bytes());
        }
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n}], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [{n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let err = extract_head_from_tfjs_buffers(&manifest, &blob).unwrap_err();
        match err {
            ConvertError::NonFiniteWeight {
                tensor,
                index,
                value,
            } => {
                assert_eq!(tensor, "head/kernel");
                assert_eq!(index, 17);
                assert!(value.is_nan(), "value should be NaN, got {value}");
            }
            other => panic!("expected NonFiniteWeight, got {other:?}"),
        }
    }

    /// +Inf in the bias is similarly caught at the
    /// converter boundary with the bias tensor name +
    /// element index in the structured error.
    #[test]
    fn tfjs_non_finite_bias_rejected() {
        let n = 4usize;
        let kernel_count = BACKBONE_FEATURE_DIM * n;
        let mut blob = Vec::<u8>::with_capacity((kernel_count + n) * 4);
        for _ in 0..kernel_count {
            blob.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        // Poison bias index 2 with +Inf.
        for i in 0..n {
            let v = if i == 2 { f32::INFINITY } else { 0.0_f32 };
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n}], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [{n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let err = extract_head_from_tfjs_buffers(&manifest, &blob).unwrap_err();
        match err {
            ConvertError::NonFiniteWeight {
                tensor,
                index,
                value,
            } => {
                assert_eq!(tensor, "head/bias");
                assert_eq!(index, 2);
                assert!(value.is_infinite() && value.is_sign_positive());
            }
            other => panic!("expected NonFiniteWeight on bias, got {other:?}"),
        }
    }

    /// Truncated weights blob fails with `TfjsBlobLength`, not a
    /// panic.  The synthetic manifest keeps the test fixture-free;
    /// its shape values are arbitrary because `weights_blob.len()
    /// != declared` fires before the head locator runs.
    #[test]
    fn tfjs_truncated_blob_rejected() {
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, 3], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [3], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let truncated = vec![0u8; 32];
        let err = extract_head_from_tfjs_buffers(&manifest, &truncated).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsBlobLength { .. }),
            "{err:?}"
        );
    }

    /// Manifest declaring an unsupported dtype is rejected.
    #[test]
    fn tfjs_unsupported_dtype_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "int32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(matches!(err, ConvertError::TfjsDtype { .. }), "{err:?}");
    }

    /// `Categorized::kind` correctly classifies every variant.  Adding
    /// a new variant without updating the impl fails to compile,
    /// giving us compile-time exhaustiveness on the api-crate's HTTP
    /// error mapping.
    #[test]
    fn convert_error_classification() {
        use crate::common::error::{Categorized, ErrorKind};
        use std::io;

        fn assert_kind(err: ConvertError, expected: ErrorKind) {
            assert_eq!(err.kind(), expected, "{err:?}");
        }

        // User-input -- operator-supplied source model malformed.
        assert_kind(ConvertError::Labels("x".into()), ErrorKind::UserInput);
        assert_kind(
            ConvertError::TfjsParse {
                what: "x",
                msg: "y".into(),
            },
            ErrorKind::UserInput,
        );
        assert_kind(ConvertError::TfjsLocator("x".into()), ErrorKind::UserInput);
        assert_kind(
            ConvertError::TfjsShortRead { have: 0, need: 1 },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsDtype {
                name: "x".into(),
                dtype: "y".into(),
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsUnsafePath("x".into()),
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsShapeOverflow {
                name: "x".into(),
                shape: vec![],
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsBlobLength {
                have: 0,
                declared: 1,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::LimitExceeded {
                what: "n_classes",
                value: 1_000_000,
                max: 100_000,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsZeroDimension {
                name: "x".into(),
                shape: vec![0],
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::NonFiniteWeight {
                tensor: "x".into(),
                index: 0,
                value: f32::NAN,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::BadClassCount { got: 0, max: 100 },
            ErrorKind::UserInput,
        );

        // Internal -- daemon-side IO / serializer / Burn record / tensor.
        assert_kind(
            ConvertError::Read {
                path: "x".into(),
                source: io::Error::other("y"),
            },
            ErrorKind::Internal,
        );
        assert_kind(
            ConvertError::Write {
                path: "x".into(),
                source: io::Error::other("y"),
            },
            ErrorKind::Internal,
        );
        assert_kind(ConvertError::Record("x".into()), ErrorKind::Internal);
        assert_kind(ConvertError::Tensor("x".into()), ErrorKind::Internal);

        // Not-implemented -- distinct category, maps to 501.
        assert_kind(ConvertError::NotImplemented("x"), ErrorKind::NotImplemented);

        // `#[from]`-only variants -- construct via their conversions
        // so this test documents the full classification contract.
        let serde_err: serde_json::Error =
            serde_json::from_slice::<serde_json::Value>(b"{").unwrap_err();
        assert_kind(ConvertError::from(serde_err), ErrorKind::Internal);
    }

    /// Cross-test serialization gate for every test in this
    /// module that touches the global [`CONVERT_SEMAPHORE`].
    /// The semaphore is shared across the test binary; without
    /// this gate two `#[test]` cases acquiring the permit
    /// concurrently would step on each other under the default
    /// `cargo test` parallel runner.  `cargo test --lib --
    /// --test-threads=1` already serializes; this gate makes
    /// the unit tests robust under either invocation.
    static CONVERT_TEST_SERIALIZER: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn serialize_convert_test() -> parking_lot::MutexGuard<'static, ()> {
        CONVERT_TEST_SERIALIZER.lock()
    }

    /// Converter concurrency cap.  The first
    /// `acquire_convert_permit` call succeeds; while that permit
    /// is held a second call returns `ConvertError::Busy`
    /// (mapped to 409 Conflict by the api boundary).  Dropping
    /// the permit makes the next acquire succeed.
    ///
    /// `try_acquire_owned` doesn't require a runtime context, so
    /// this test runs as plain `#[test]` rather than `#[tokio::test]`.
    #[test]
    fn convert_permit_caps_at_one_in_flight() {
        let _gate = serialize_convert_test();
        let p1 = acquire_convert_permit().expect("first acquire");
        // Second acquire while p1 alive must reject as Busy.
        let err = acquire_convert_permit()
            .expect_err("second acquire must reject while first permit held");
        assert!(
            matches!(err, ConvertError::Busy),
            "expected ConvertError::Busy, got {err:?}",
        );

        // Categorise -> 409 Conflict (matches the wire shape of
        // training-registry's busy rejection).
        use crate::common::error::{Categorized, ErrorKind};
        assert_eq!(err.kind(), ErrorKind::Conflict);

        // Drop the first permit; the next acquire succeeds.
        drop(p1);
        let p2 = acquire_convert_permit().expect("acquire after drop must succeed");
        drop(p2);
    }

    // MARK: convert pipeline -- unit tests

    use crate::common::ids::{HeadId, JobId, WorkspaceId};
    use crate::common::workspace::WorkspaceRevision;

    /// Stage a workspace under `<root>/workspaces/<id>/` so the
    /// publish primitive can land a head.
    fn stage_test_workspace(
        root: &Path,
    ) -> (WorkspaceId, std::sync::Arc<dyn crate::file_mgr::FsService>) {
        let fs = std::sync::Arc::new(crate::file_mgr::FsServiceImpl::new(root.to_path_buf()));
        fs.ensure_root_layout().expect("layout");
        let id = fs.create("convert-test").expect("create workspace");
        (id, fs as std::sync::Arc<dyn crate::file_mgr::FsService>)
    }

    /// Build a minimal TFJS bundle on disk containing one
    /// `model.json` + one `weights.bin` shard with the head
    /// kernel + bias the converter will pick up.  Returns the
    /// staged on-disk paths for the convert job inputs.
    fn stage_minimal_tfjs(
        workspace_dir: &Path,
        n_classes: usize,
    ) -> (PathBuf, Vec<PathBuf>, PathBuf) {
        let datasets_dir = workspace_dir.join("converters/tfjs");
        std::fs::create_dir_all(&datasets_dir).unwrap();

        // model.json: head kernel `[BACKBONE_FEATURE_DIM, n_classes]`
        // + head bias `[n_classes]`.
        let model_json = datasets_dir.join("model.json");
        let manifest_json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "NewHeadDense/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n_classes}], "dtype": "float32"}},
                  {{"name": "NewHeadDense/bias", "shape": [{n_classes}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        std::fs::write(&model_json, manifest_json.as_bytes()).unwrap();

        // weights.bin: kernel + bias as f32 LE.  Use 0.5 / 0.25
        // so the resulting head decodes finite.
        let kernel_count = BACKBONE_FEATURE_DIM * n_classes;
        let mut weights_blob = Vec::with_capacity((kernel_count + n_classes) * 4);
        for _ in 0..kernel_count {
            weights_blob.extend_from_slice(&0.5_f32.to_le_bytes());
        }
        for _ in 0..n_classes {
            weights_blob.extend_from_slice(&0.25_f32.to_le_bytes());
        }
        let shard = datasets_dir.join("weights.bin");
        std::fs::write(&shard, &weights_blob).unwrap();

        // labels.txt: n_classes labels, one per line.
        let labels = datasets_dir.join("labels.txt");
        let labels_text = (0..n_classes)
            .map(|i| format!("class_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&labels, &labels_text).unwrap();

        (model_json, vec![shard], labels)
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    /// `run_convert_job` on a minimal synthetic TFJS bundle
    /// publishes the head into the workspace's 2-slot rotation
    /// and the `heads.json` index reflects it.  Single-class
    /// head keeps the bundle small (~24 KB vs the multi-MB
    /// upstream Speech-Commands fixture).
    #[test]
    fn convert_with_minimal_tfjs_fixture_publishes_head() {
        // Cross-test serialization: the convert semaphore is a
        // module-level static; the serializer mutex above
        // gates every test in this module.
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 3);

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            model_json_path: model_json,
            shard_paths: shards,
            labels_path: labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };

        // The convert semaphore is held by `_gate` for the test's
        // body so cross-test races stay clean.  `run_convert_job`
        // does not re-acquire (the api producer owns the permit).
        run_convert_job(fs.clone(), job).expect("convert publishes");

        // Head landed in the workspace's index.
        let summary = fs.summary(&ws).expect("summary");
        assert_eq!(summary.heads.heads.len(), 1, "exactly one head");
        let h = &summary.heads.heads[0];
        assert_eq!(h.head_id, head_id);
        assert_eq!(h.n_classes, 3);
        // The head bytes exist on disk.
        let mpk = crate::file_mgr::schema::head_artifact_path(&workspace_dir, head_id);
        assert!(mpk.is_file(), "head .mpk missing at {}", mpk.display());
        let manifest = crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
            .expect("read manifest");
        assert_eq!(manifest.head_id, head_id);
        assert_eq!(manifest.workspace_id, ws);
        // Manifest carries only the workspace-revision snapshot;
        // convert provenance lives in the JSONL log.
        assert_eq!(manifest.workspace_revision.id, 0);
        assert_eq!(manifest.labels.len(), 3);

        // Wrapper sweep clears both the staging dir and the `.mpk`
        // tempfile on success.
        let staging_dir = workspace_dir.join(".tmp").join(format!("convert-{job_id}"));
        let mpk_tempfile = workspace_dir
            .join(".tmp")
            .join(format!("convert-{job_id}-{head_id}.mpk"));
        assert!(
            !staging_dir.exists(),
            "staging dir {} must be swept on success",
            staging_dir.display(),
        );
        assert!(
            !mpk_tempfile.exists(),
            ".mpk tempfile {} must be swept on success",
            mpk_tempfile.display(),
        );
    }

    /// A failed `run_convert_job` does NOT commit a head record.
    /// Force failure by giving `model.json` an invalid manifest;
    /// the workspace's `heads.json` stays empty and a `failed`
    /// log line lands in the JSONL.
    #[test]
    fn convert_failure_releases_references_and_no_head_committed() {
        // Cross-test serialization (see above).
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let datasets_dir = workspace_dir.join("datasets");
        std::fs::create_dir_all(&datasets_dir).unwrap();

        // Bad model.json (not even valid JSON); causes
        // `parse_tfjs_manifest_with_limits` to fail.
        let bad_model = datasets_dir.join("model.json");
        std::fs::write(&bad_model, b"not json").unwrap();
        let bad_shard = datasets_dir.join("weights.bin");
        std::fs::write(&bad_shard, b"any").unwrap();
        let bad_labels = datasets_dir.join("labels.txt");
        std::fs::write(&bad_labels, b"x\n").unwrap();

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            model_json_path: bad_model,
            shard_paths: vec![bad_shard],
            labels_path: bad_labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };
        let err = run_convert_job(fs.clone(), job).expect_err("invalid manifest must fail");
        assert!(
            matches!(err, ConvertError::TfjsParse { .. }),
            "expected TfjsParse, got {err:?}",
        );

        // No record committed and the wrapper sweep cleared
        // both the staging dir and the `.mpk` tempfile.
        let summary = fs.summary(&ws).expect("summary");
        assert!(summary.heads.heads.is_empty(), "no head record committed");
        assert_eq!(summary.core.head_count, 0);
        let staging_dir = workspace_dir.join(".tmp").join(format!("convert-{job_id}"));
        let mpk_tempfile = workspace_dir
            .join(".tmp")
            .join(format!("convert-{job_id}-{head_id}.mpk"));
        assert!(
            !staging_dir.exists(),
            "staging dir {} must be swept on failure",
            staging_dir.display(),
        );
        assert!(
            !mpk_tempfile.exists(),
            ".mpk tempfile {} must be swept on failure",
            mpk_tempfile.display(),
        );

        // The JSONL log carries a `failed` line.
        let log_path = workspace_dir
            .join("converter_logs")
            .join(format!("{job_id}.jsonl"));
        assert!(log_path.is_file(), "log missing");
        let log = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            log.lines().any(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|v| v["state"].as_str().map(str::to_string))
                    .as_deref()
                    == Some("failed")
            }),
            "log must contain a `failed` event line; got:\n{log}",
        );
    }

    /// Each line in `<workspace>/converter_logs/<job_id>.jsonl`
    /// is well-formed JSON and carries the documented fixed-shape
    /// fields.  Pins the JSONL contract so a future writer change
    /// can't silently emit non-JSON garbage.
    #[test]
    fn convert_log_events_are_jsonl() {
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            model_json_path: model_json,
            shard_paths: shards,
            labels_path: labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };
        run_convert_job(fs.clone(), job).expect("convert publishes");

        let log_path = workspace_dir
            .join("converter_logs")
            .join(format!("{job_id}.jsonl"));
        let log = std::fs::read_to_string(&log_path).expect("read log");
        let mut prev_seq: u64 = 0;
        let mut saw_started = false;
        let mut saw_completed = false;
        for line in log.lines() {
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("non-JSON log line {line:?}: {e}"));
            // Fixed-shape fields.
            assert!(v["seq"].is_u64(), "seq must be u64; line={line}");
            assert!(v["at"].is_string(), "at must be string");
            assert!(v["state"].is_string(), "state must be string");
            // Monotonic seq.
            let seq = v["seq"].as_u64().unwrap();
            assert!(seq > prev_seq, "seq not monotonic: {prev_seq} -> {seq}");
            prev_seq = seq;
            match v["state"].as_str().unwrap() {
                "started" => saw_started = true,
                "completed" => saw_completed = true,
                _ => {}
            }
        }
        assert!(saw_started, "log must contain a `started` event");
        assert!(saw_completed, "log must contain a `completed` event");
    }

    // MARK: convert-pipeline manifest-shape pins
    //
    // The wire shape, lease shape, and head-publish primitives are
    // pinned elsewhere; these tests pin the convert worker's
    // construction of the on-disk `HeadManifest` so a quiet
    // re-addition of a legacy field surfaces here first.  The pins
    // use the synthetic minimal TFJS bundle so they run on every
    // CI host.

    /// The convert producer publishes a `<head_id>.json` whose
    /// on-disk field set is exactly the minimized contract.
    #[test]
    fn convert_publishes_minimized_manifest_field_set() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(7),
            model_json_path: model_json,
            shard_paths: shards,
            labels_path: labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };
        run_convert_job(fs.clone(), job).expect("convert publishes");

        let manifest_bytes = std::fs::read(crate::file_mgr::schema::head_manifest_path(
            &workspace_dir,
            head_id,
        ))
        .expect("read on-disk manifest");
        let v: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
        let obj = v.as_object().expect("manifest is a JSON object");
        let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
        let expected: std::collections::BTreeSet<&str> = [
            "head_id",
            "workspace_id",
            "workspace_revision",
            "sha256",
            "n_classes",
            "size_bytes",
            "created_at",
            "labels",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            actual, expected,
            "convert-published manifest must carry exactly the minimized field set; got {actual:?}",
        );
        // Legacy convert provenance fields must not appear.
        for forbidden in [
            "dataset_path",
            "training_cfg",
            "training_cfg_sha256",
            "dataset_revision",
            "dataset_revision_at_train",
            "convert_provenance",
            "input_paths",
        ] {
            assert!(
                !obj.contains_key(forbidden),
                "legacy field {forbidden:?} must not appear in published manifest",
            );
        }
    }

    /// The convert producer's published `heads.json` `HeadRecord`
    /// carries exactly the minimized contract.
    #[test]
    fn convert_publishes_minimized_head_record_field_set() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(7),
            model_json_path: model_json,
            shard_paths: shards,
            labels_path: labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };
        run_convert_job(fs.clone(), job).expect("convert publishes");

        let index_bytes = std::fs::read(crate::file_mgr::schema::head_index_path(&workspace_dir))
            .expect("read on-disk heads.json");
        let v: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
        let entries = v["heads"].as_array().expect("heads is an array");
        assert_eq!(entries.len(), 1, "exactly one published head");
        let rec = entries[0].as_object().expect("HeadRecord is a JSON object");
        let actual: std::collections::BTreeSet<&str> = rec.keys().map(String::as_str).collect();
        let expected: std::collections::BTreeSet<&str> = [
            "head_id",
            "workspace_revision",
            "sha256",
            "n_classes",
            "size_bytes",
            "created_at",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            actual, expected,
            "convert-published HeadRecord must carry exactly the minimized field set; got {actual:?}",
        );
        for forbidden in [
            "dataset_path",
            "training_cfg_sha256",
            "dataset_revision_at_train",
            "labels",
            "workspace_id",
        ] {
            assert!(
                !rec.contains_key(forbidden),
                "legacy / non-record field {forbidden:?} must not appear in HeadRecord",
            );
        }
    }

    /// The converter publishes with the producer-snapshotted
    /// `workspace_revision`, never re-fetching at publish time.  A
    /// re-fetch would break stale-head detection.
    #[test]
    fn convert_manifest_workspace_revision_matches_producer_snapshot() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        // Snapshot a deliberately-distinct revision id so the
        // assertion is load-bearing (an accidental re-read at
        // publish time would surface revision id = 0 because the
        // synthetic workspace's current revision is fresh).
        let snapshot = rev(42);
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: snapshot.clone(),
            model_json_path: model_json,
            shard_paths: shards,
            labels_path: labels,
            labels_format: crate::file_mgr::LabelsFormat::Lines,
        };
        run_convert_job(fs.clone(), job).expect("convert publishes");

        let manifest = crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
            .expect("read manifest");
        assert_eq!(manifest.workspace_revision.id, snapshot.id);
        assert_eq!(manifest.workspace_revision.at, snapshot.at);
        // The heads.json record carries the same snapshot.
        let summary = fs.summary(&ws).expect("summary");
        assert_eq!(summary.heads.heads[0].workspace_revision.id, snapshot.id);
    }

    /// Defence in depth: the typed in-memory `HeadManifest` the
    /// converter constructs has no slot for the legacy
    /// `dataset_path` / `training_cfg` / `training_cfg_sha256`
    /// fields, so a future cascade that re-added them would surface
    /// here as well as at the schema-level test in `common::workspace`.
    #[test]
    fn convert_constructed_manifest_has_no_legacy_provenance() {
        let manifest = crate::common::workspace::HeadManifest {
            head_id: HeadId::new(),
            workspace_id: WorkspaceId::new(),
            workspace_revision: rev(3),
            sha256: "abc".into(),
            n_classes: 2,
            size_bytes: 1024,
            created_at: "2026-05-08T12:00:00Z".into(),
            labels: vec!["a".into(), "b".into()],
        };
        let v = serde_json::to_value(&manifest).unwrap();
        let obj = v.as_object().expect("manifest serializes as JSON object");
        for forbidden in [
            "dataset_path",
            "training_cfg",
            "training_cfg_sha256",
            "convert_provenance",
            "input_paths",
        ] {
            assert!(
                !obj.contains_key(forbidden),
                "HeadManifest must not carry legacy field {forbidden:?}",
            );
        }
    }
}
