//! Workspace + asset error type.  Re-exported from the
//! parent module via `pub use`.

use crate::file_mgr::metadata::AssetKind;
use thiserror::Error;

/// Failure shapes from the workspace + asset manager.
/// Mapped to HTTP statuses via the
/// [`crate::common::error::Categorized`] impl below.
#[derive(Debug, Error)]
pub enum FileError {
    #[error("workspace not found: {0}")]
    NotFound(String),
    #[error("asset not found in workspace {ws}: {kind:?} {name}")]
    AssetNotFound {
        ws: String,
        kind: AssetKind,
        name: String,
    },
    #[error("invalid identifier: {0}")]
    Id(#[from] crate::common::ids::IdError),
    #[error("invalid asset name: {0}")]
    InvalidName(String),
    #[error("invalid asset extension: got {got}, expected one of {expected:?}")]
    InvalidExtension {
        got: String,
        expected: Vec<&'static str>,
    },
    #[error("workspace name conflict: {0}")]
    NameConflict(String),
    /// Workspace metadata declares a schema version this
    /// daemon build doesn't understand yet.  Refuses the load
    /// so an older daemon can't write its older shape over a
    /// newer-shape file and silently lose fields.  Bump
    /// [`crate::file_mgr::WorkspaceMetadata::CURRENT`] when
    /// adding fields that an older daemon can't serialize.
    #[error(
        "workspace {path} schema version {found} is newer than this build (max {max}); upgrade the daemon"
    )]
    SchemaTooNew { path: String, found: u32, max: u32 },
    /// Workspace metadata is too old for this daemon to load.
    /// Activates when a future revision raises
    /// [`crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE`]
    /// above 1; today only a deliberately hand-edited `0` can
    /// trigger it.
    #[error(
        "workspace {path} schema version {found} is older than this build supports (min {min}); migrate or recreate the workspace"
    )]
    SchemaTooOld { path: String, found: u32, min: u32 },
    #[error("io {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("metadata parse {path}: {source}")]
    MetadataParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("metadata serialize: {0}")]
    MetadataSerialize(#[from] serde_json::Error),
    #[error("persist tempfile: {0}")]
    Persist(#[from] tempfile::PersistError),
    /// Upload exceeded
    /// [`crate::file_mgr::AdmissionCfg::max_upload_bytes`].
    /// Detected mid-stream so the tempfile is dropped (no
    /// partial commit) and no metadata row is written.  Maps
    /// to 400 -- the closest taxonomy fit; the canonical 413
    /// is folded into `UserInput` per
    /// [`crate::common::error`]'s "when in doubt, classify
    /// into the nearest existing category" rule.
    #[error("upload exceeded max_upload_bytes: {observed} > {max}")]
    PayloadTooLarge { observed: u64, max: u64 },
    /// Concurrent upload count is at
    /// [`crate::file_mgr::AdmissionCfg::max_concurrent_uploads`].
    /// Operator can retry once an in-flight upload finishes.
    /// Maps to 409 (similar shape to `NameConflict`: request
    /// well-formed, state isn't ready to accept it).
    #[error("too many concurrent uploads: {active}/{max}")]
    TooManyConcurrentUploads { active: u32, max: u32 },
    /// On-disk metadata file (e.g. `workspace.json`) exceeded
    /// the size cap.  Indicates daemon-internal corruption or
    /// operator tampering with a daemon-owned file -- both
    /// surface as `Internal` to the operator.  Today: enforced
    /// for `workspace.json`
    /// (`crate::file_mgr::schema::MAX_WORKSPACE_CORE_BYTES`,
    /// 64 KiB per the workspace-redesign §9 storage table).
    #[error("metadata at {path} too large: {observed} bytes > {max}")]
    MetadataTooLarge {
        path: String,
        observed: u64,
        max: u64,
    },
    /// Another running job already references the target
    /// workspace or an ancestor / descendant dataset path.
    /// Returned by upload, dataset delete, and workspace delete
    /// when conflict detection (`DatasetRefRegistry` for asset
    /// scope; `JobRegistry` for job scope) fires.  Maps to HTTP
    /// 409 via the `Conflict` taxonomy.
    #[error("job conflict: {message}")]
    JobConflict { message: String },
    /// A train job is already running.  Per redesign §9
    /// `max_train_jobs = 1`: at most one unfinished train job
    /// daemon-wide.  Distinct from `JobConflict` so the api
    /// layer renders the dedicated `another_train_running`
    /// discriminator code.  Maps to HTTP 409.
    #[error("another train job is already running daemon-wide (max_train_jobs = 1)")]
    AnotherTrainRunning,
}

/// Shorthand for `FileError::Io { path: path.to_string(), source }`.
/// `path` is `impl Display` so call sites can pass a
/// `Path::display()` adapter directly without the per-site
/// `to_string()` boilerplate.  Used at every site in `file_mgr`
/// and its api-route callers that surfaces a `std::io::Error`
/// against a known path.
pub(crate) fn io_err(path: impl std::fmt::Display, source: std::io::Error) -> FileError {
    FileError::Io {
        path: path.to_string(),
        source,
    }
}

/// Shorthand for
/// `FileError::MetadataParse { path: path.to_string(), source }`.
/// Same `impl Display` shape as [`io_err`]; the pattern recurs in
/// ~20 sites across `schema.rs` (workspace + heads + active
/// manifest readers) and the small parsers in `metadata.rs`,
/// `staging.rs`, `uploader.rs`, `request_payload.rs`.
pub(crate) fn metadata_parse_err(
    path: impl std::fmt::Display,
    source: serde_json::Error,
) -> FileError {
    FileError::MetadataParse {
        path: path.to_string(),
        source,
    }
}

impl crate::common::error::Categorized for FileError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Operator referenced a workspace / asset by id;
            // the request shape was valid, the resource just
            // doesn't exist.
            FileError::NotFound(_) | FileError::AssetNotFound { .. } => NotFound,
            // Identifier validation already classified by IdError.
            FileError::Id(e) => e.kind(),
            // Operator-supplied name / extension shape is wrong.
            FileError::InvalidName(_) | FileError::InvalidExtension { .. } => UserInput,
            // A workspace with this name already exists; the
            // operator can pick a different name.
            FileError::NameConflict(_) => Conflict,
            // Operator-supplied request is too large.  Closest
            // taxonomy fit is UserInput (400); the canonical
            // 413 isn't a separate `ErrorKind` variant per
            // common::error's guidance to avoid premature
            // taxonomy expansion.
            FileError::PayloadTooLarge { .. } => UserInput,
            // Semaphore exhausted; the request is well-formed
            // but the daemon's concurrent-upload budget is
            // full.  Operator can retry once an in-flight
            // upload releases its slot.
            FileError::TooManyConcurrentUploads { .. } => Conflict,
            // Schema version mismatch is also a conflict
            // shape: the request is well-formed but the
            // resource's version doesn't match what this
            // build can serve.  Operator action (upgrade /
            // migrate the daemon) is required.
            FileError::SchemaTooNew { .. } | FileError::SchemaTooOld { .. } => Conflict,
            // An overlapping running job already holds a
            // reference on this workspace / dataset path.
            // 409 Conflict matches the redesign §6 contract.
            FileError::JobConflict { .. } => Conflict,
            // Single-train-job invariant (`max_train_jobs = 1`)
            // already at capacity.  409 Conflict matches the
            // redesign §6 / §9 wire contract.
            FileError::AnotherTrainRunning => Conflict,
            // Daemon-internal: filesystem failures mid-write,
            // malformed metadata.json a previous boot wrote,
            // tempfile-persist failures, daemon-owned metadata
            // size cap violation (corruption or tampering).
            FileError::Io { .. }
            | FileError::MetadataParse { .. }
            | FileError::MetadataSerialize(_)
            | FileError::MetadataTooLarge { .. }
            | FileError::Persist(_) => Internal,
        }
    }
}
