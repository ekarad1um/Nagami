//! Workspace + asset file manager.
//!
//! # Layout
//!
//! ```text
//! $WORKSPACE_ROOT/
//!     active/
//!         current.json
//!         generations/<activation_id>/{head.mpk,labels.txt,manifest.json}
//!     workspaces/<workspace_id>/
//!         workspace.json             -- core (id, name, tags,
//!                                       created_at, workspace_revision,
//!                                       head_count)
//!         heads.json                 -- 2-slot sliding head index
//!         heads/<head_id>.{mpk,json} -- bytes + per-head manifest
//!         datasets/<class>/<...>.wav -- per-class samples (recursive)
//!         converters/<sub>/...       -- operator-uploaded convert inputs
//!         training_logs/<job>.jsonl
//!         converter_logs/<job>.jsonl
//!         .tmp/                      -- staging for atomic uploads
//!     .tmp/                          -- root-level staging
//! ```
//!
//! # Write protocol
//!
//! All on-disk writes go through `fs_atomic::put_atomic` (tempfile,
//! fsync, rename, then parent-dir fsync) so a partial write is never
//! visible.  The schema layer pairs an artifact write
//! (`<head_id>.mpk` + `<head_id>.json`) with an index commit
//! (`heads.json` + `workspace.json`) under the per-workspace mutation
//! mutex; a crash between artifact and index leaves orphan bytes
//! that boot recovery sweeps.
//!
//! # Recovery
//!
//! Reconciliation lives in [`recovery::recover_all`].  On boot it
//! drains pending workspace-delete + asset-delete tombstones,
//! sweeps daemon-owned head orphans, repairs `head_count`, and
//! verifies the active generation (with previous-generation +
//! bundled-default fallback).  Runs once before the api goes live.

#![warn(missing_debug_implementations)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(test)]
use crate::common::ids::AssetId;
use crate::common::ids::WorkspaceId;
use crate::common::workspace::{HeadIndex, HeadStatus, WorkspaceCore};
use serde::Serialize;
#[cfg(test)]
use sha2::{Digest, Sha256};

mod error;
// Canonical atomic-write primitive shared across the daemon
// (file_mgr, converter, training) so every on-disk write uses
// the same durability path.
pub mod fs_atomic;
// [`FsService`] trait + `FsServiceImpl` facade.  Colocated with
// its DTO types so `common` does not need to depend on `time`.
pub mod fs_service;
mod metadata;
// On-disk schema persistence: `workspace.json`, `heads.json`,
// per-head `<head_id>.json` plus path layout + the 64 KiB core
// cap.
pub mod schema;
// Per-workspace eager `ArcSwap`-backed cache cells holding the
// hot core + head index.  Wait-free reads, mutex-serialised
// writes.
pub mod cache;
// Atomic delete-staging primitives: tombstone + stage + drain +
// finalize state machine for workspace and asset deletes.
pub mod staging;
// RFC3339 wall-clock helper centralized so every write stamps
// timestamps in one format.
pub mod time_util;
mod validate;

// Forward-only JSONL paging for the daemon's per-job JSONL
// backstop (`<workspace>/{training,converter}_logs/<job>.jsonl`).
// Canonical reader for the `?after_seq=&limit=` shape on the
// unified `/assets/{*path}` surface.
pub mod log_page;

mod asset;
// AssetPath-shaped dataset surface: paginated child listing,
// mutex-release-before-stream file open, atomic single-file
// upload, async tombstoned delete.
pub mod dataset;
// Cross-cutting in-process job registry: admission gate
// (per-`JobType` concurrency caps + reference-overlap detection),
// per-job event ring for SSE replay, bounded recent-history
// surface backing `GET /jobs`.
pub mod job_registry;
// Extension-only MIME-type table for asset GET responses; no
// byte sniffing.
pub mod mime;
mod registry;
// Shared request-payload contracts for `POST .../train` +
// `POST .../convert`: typed `TrainingCfg`, `ConvertRequest`,
// bounded numeric validators, canonical SHA-256 helper.
pub mod request_payload;
// Trained-head 2-slot rotation primitive: index-atomic publish
// of `<head_id>.{mpk,json}` + `heads.json` +
// `workspace.json.head_count` under the per-workspace mutation
// mutex held by the caller.
pub mod head_rotation;
// Active-head activation pipeline shared between the daemon's
// first-boot path and the `POST /active` route: stage, validate
// hashes, build + validate the active manifest, publish, prune.
pub mod active_head_writer;
// Idempotent boot-time recovery: drain pending tombstones, sweep
// daemon-owned orphans, repair `head_count`, verify the active
// generation.  Runs once before the api goes live.
pub mod recovery;
// Runtime storage hygiene: periodic `.tmp/` orphan sweep +
// per-workspace log retention.  Wired into the daemon's
// background-task registry alongside the training reaper.
pub mod storage_reaper;
mod uploader;

// Process-wide hook registry letting the daemon publish into
// `status::WorkspaceMetrics` without `file_mgr` referencing the
// `status` crate (preserves the `file_mgr -> common` edge guard).
pub mod metrics_hooks;

pub use active_head_writer::{
    ActivationError, ActivationOriginInput, ActivationResult, HeadInnerLoader, PendingActivation,
    prune_old_generations, publish_active_generation, stage_and_validate_activation,
    staging_path_for,
};
pub use cache::WorkspaceCacheCell;
pub use dataset::{
    CONVERTER_LOGS_DIR_NAME, DATASETS_DIR_NAME, DEFAULT_DATASET_LIST_LIMIT, DatasetEntry,
    DatasetListing, DatasetUploadReceipt, EntryKind, JobRefHandle, MAX_DATASET_LIST_LIMIT,
    TRAINING_LOGS_DIR_NAME,
};
pub use error::FileError;
// `pub(crate)` so any in-crate caller that constructs
// `FileError::Io` from outside the `file_mgr` submodules
// (today: `api/routes`) shares the same `io_err` shorthand.
pub(crate) use error::io_err;
pub use fs_atomic::put_atomic;
pub use fs_service::{
    FileMetadataGuard, FsError, FsService, FsServiceImpl, MetadataGuard, UploadPermit,
};
pub use head_rotation::{HeadRotationResult, PendingHead, publish_trained_head};
pub use job_registry::{
    EventGap, EventStream, EventStreamError, JobEvent, JobHandle, JobProgress, JobRegistry,
    JobRegistryCfg, JobRegistryCounters, JobResult as RegistryJobResult, JobSnapshot,
    JobState as RegistryJobState, LeaseGuard as JobRegistryLeaseGuard, RegistryConflict,
    RegistryEvent,
};
pub use metadata::{AssetKind, AssetRecord, WorkspaceMetadata};
pub use mime::content_type_from_path;
pub use recovery::{
    RecoveryActiveResult, RecoveryError, RecoveryReport, RecoveryRootReport,
    RecoveryWorkspaceReport, recover_active_head, recover_all, recover_root_staging,
    recover_workspaces,
};
pub use request_payload::{
    ConvertRequest, ConverterPath, ConverterPathError, LabelsFormat, MAX_BATCH_SIZE,
    MAX_CONVERT_SHARDS, MAX_EPOCHS, MAX_LEARNING_RATE, MIN_BATCH_SIZE, MIN_EPOCHS,
    TfjsConvertParams, TrainRequest, TrainingCfg, ValidationError, canonical_training_cfg_sha256,
    from_manifest_value, to_manifest_value, validate_convert_request, validate_training_cfg,
};
pub use schema::{
    ACTIVE_CURRENT_FILENAME, ACTIVE_DIR_NAME, ACTIVE_GENERATIONS_DIR_NAME, ACTIVE_HEAD_FILENAME,
    ACTIVE_LABELS_FILENAME, ACTIVE_MANIFEST_FILENAME, ACTIVE_TMP_DIR_NAME, ActiveCurrentPointer,
    HEAD_ARTIFACT_EXTENSION, HEAD_INDEX_FILENAME, HEAD_MANIFEST_EXTENSION, HEADS_DIR_NAME,
    MAX_WORKSPACE_CORE_BYTES, ROOT_TMP_DIR_NAME, WORKSPACE_CORE_FILENAME, WORKSPACES_DIR_NAME,
    active_current_path, active_dir, active_generation_dir, active_generations_dir,
    active_staging_dir, head_artifact_path, head_index_path, head_manifest_path, heads_dir,
    read_active_current, read_active_manifest, read_head_index, read_head_manifest,
    read_workspace_core, root_tmp_dir, workspace_core_path, workspace_dir_for, workspaces_dir,
    write_active_current, write_active_manifest, write_head_index, write_head_manifest,
    write_workspace_core,
};
pub use staging::{
    DATASET_TOMBSTONE_PREFIX, DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult,
    STAGED_PAYLOAD_NAME, StagedDelete, WORKSPACE_TOMBSTONE_PREFIX, drain_staged_payload,
    finalize_staged_delete, read_tombstone, stage_payload, write_tombstone,
};
pub use storage_reaper::{SweepConfig, SweepReport, sweep_once};
pub use time_util::now_rfc3339;
pub use uploader::AdmissionCfg;
pub use validate::validate_asset_name;
// `pub(crate)` so the training-side per-head sha256 stamp reuses
// the same streaming hasher every other on-disk digest goes
// through.  Not exposed publicly: the signature returns
// `FileError`, an internal taxonomy.
pub(crate) use validate::sha256_file_streaming;
// `pub(crate)` so the api-route upload digest can reuse the
// nibble-lookup hex encoder rather than carry its own copy.
pub(crate) use validate::hex_lowercase;

#[cfg(test)]
use validate::validate_extension;

/// Workspace + asset manager.  Thread-safe (`&self` everywhere),
/// cheap to clone (`Arc` bumps), and external consumers reach it
/// through the [`fs_service::FsService`] trait surface.
///
/// # Concurrency
///
/// File writes go through tempfile + atomic rename so readers see
/// either old or new bytes, never partial.  Index-atomic publishes
/// (`heads.json` + `workspace.json`) serialize per workspace via a
/// `parking_lot::Mutex<()>` sharded through a `DashMap`; concurrent
/// mutations on different workspaces don't contend.  The lock map
/// is unbounded -- a daemon that creates and deletes many
/// workspaces accumulates ~64 B per stale id.  Production cleanup
/// is deferred (single-workspace deployments are the dominant
/// shape today).
///
/// Method bodies are split across sibling modules:
/// `registry.rs` (create/delete/list), `metadata.rs` (legacy
/// metadata.json store), `asset.rs` (legacy asset reads),
/// `uploader.rs` (admission caps), `dataset.rs` (asset surface),
/// `head_rotation.rs` (publish primitive), `recovery.rs` (boot
/// reconciliation).
#[derive(Clone, Debug)]
pub struct WorkspaceMgr {
    pub(crate) root: PathBuf,
    pub(crate) metadata_locks: Arc<dashmap::DashMap<WorkspaceId, Arc<parking_lot::Mutex<()>>>>,
    /// Serializes [`Self::create`]'s name-uniqueness check +
    /// commit so two concurrent `create("main")` calls on the
    /// same root can't both observe an empty registry and
    /// both succeed with distinct UUIDs.  In-process only --
    /// multi-process safety would need a `flock`-style
    /// directory lock on `root`, deferred until the daemon
    /// supports a multi-instance topology.
    pub(crate) registry_lock: Arc<parking_lot::Mutex<()>>,
    /// `None` -- no admission control (the legacy
    /// [`Self::new`] path; preserves backward compatibility
    /// for tests and first-boot daemon construction).
    /// `Some(_)` -- enforced caps + concurrency gate.  Use
    /// [`Self::with_admission`] to configure.
    pub(crate) admission: Option<Arc<uploader::AdmissionState>>,
    /// Per-workspace eager cache of `workspace.json` +
    /// `heads.json`.  Populated on `create` and lazily on
    /// `summary` for already-existing workspaces.  Hot-path
    /// reads (workspace list / summary) hit the cache only;
    /// disk is consulted only on cold load.
    pub(crate) caches: Arc<dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>>,
    /// Cross-cutting in-process job registry.  Conflicting
    /// requests surface as 409 (`FileError::JobConflict` /
    /// `FileError::AnotherTrainRunning`).
    pub(crate) jobs: Arc<job_registry::JobRegistry>,
}

impl WorkspaceMgr {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: None,
            caches: Arc::new(dashmap::DashMap::new()),
            jobs: Arc::new(job_registry::JobRegistry::new(
                job_registry::JobRegistryCfg::default(),
            )),
        }
    }

    /// Construct with admission caps enforced.  Daemon code
    /// should prefer this over [`Self::new`]; tests and
    /// bootstrap paths that exercise unbounded upload sizes
    /// can keep using the un-admitted form.
    pub fn with_admission(root: PathBuf, cfg: AdmissionCfg) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: Some(Arc::new(uploader::AdmissionState::new(cfg))),
            caches: Arc::new(dashmap::DashMap::new()),
            jobs: Arc::new(job_registry::JobRegistry::new(
                job_registry::JobRegistryCfg::default(),
            )),
        }
    }

    /// Construct with a shared, pre-configured [`JobRegistry`].
    /// Daemon boot uses this so the api `AppState`'s job registry
    /// (the one `GET /jobs` reads) is the same instance the
    /// `WorkspaceMgr` admission paths register against.
    pub fn with_admission_and_jobs(
        root: PathBuf,
        cfg: AdmissionCfg,
        jobs: Arc<job_registry::JobRegistry>,
    ) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: Some(Arc::new(uploader::AdmissionState::new(cfg))),
            caches: Arc::new(dashmap::DashMap::new()),
            jobs,
        }
    }

    /// Shared handle to the in-process job registry.  Cheap to
    /// clone (`Arc` bump).  Daemon callers pass this to
    /// `AppState::jobs` so the api crate sees the same registry
    /// the workspace-side admission paths register against.
    pub fn jobs(&self) -> Arc<job_registry::JobRegistry> {
        self.jobs.clone()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Returned by [`WorkspaceMgr::upload`] and
/// [`WorkspaceMgr::install_from_path`].  The wire shape
/// echoed to the client; carries the on-disk path,
/// recorded sha256, and observed size.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct AssetReceipt {
    pub kind: AssetKind,
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub path: PathBuf,
}

/// Returned by [`WorkspaceMgr::validate`].  `ok` is `true`
/// iff `missing` and `corrupt` are both empty; `extra`
/// holds files present on disk but unknown to the
/// metadata.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceReport {
    pub ok: bool,
    pub missing: Vec<(AssetKind, String)>,
    pub corrupt: Vec<(AssetKind, String)>,
    pub extra: Vec<(PathBuf, String)>,
}

/// Hot summary returned by [`WorkspaceMgr::summary`]: cached
/// `workspace.json` core, cached `heads.json` snapshot, and one
/// [`HeadStatus`] per head record paired by index.  `head_statuses`
/// is computed at summary time from the workspace's current
/// `workspace_revision.id`; never persisted.
#[derive(Debug, Clone)]
pub struct WorkspaceSummary {
    pub core: Arc<WorkspaceCore>,
    pub heads: Arc<HeadIndex>,
    pub head_statuses: Vec<HeadStatus>,
}

#[cfg(test)]
mod tests;
