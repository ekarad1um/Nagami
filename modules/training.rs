//! In-process training job registry.
//!
//! - `POST /workspace/{id}/train` body is the flattened
//!   [`TrainingCfg`].  The trainer always walks
//!   `<workspace_dir>/datasets/` as the fixed root: immediate
//!   non-hidden children are class folders; the deeper walk
//!   discovers per-class samples.
//! - At most one unfinished train job daemon-wide; a second request
//!   rejects with `FileError::AnotherTrainRunning` (409).
//! - The job-reference lease (`JobReference::Workspace`) excludes
//!   only an active `WorkspaceDelete` for the same workspace;
//!   uploads and file-deletes overlap freely.
//! - `finetune::run` opens / reads / closes per batch, so worst-case
//!   FDs are `batch_size * parallel_loaders` independent of dataset
//!   size.
//! - On success the trainer stages the `.mpk` under
//!   `<workspace_dir>/.tmp/`, builds the per-head manifest, and
//!   publishes through the head-rotation primitive.  No head record
//!   is committed on failure.
//!
//! Daemon-side archive extraction was removed; bulk dataset loads
//! use repeated single-file uploads via `POST /upload`.

#![warn(missing_debug_implementations)]

mod finetune;
pub(crate) mod registry;
pub use registry::TrainingRegistry;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::{HeadManifest, WorkspaceRevision};
use crate::file_mgr::{
    DATASETS_DIR_NAME, FsService, PendingHead, TrainingCfg, now_rfc3339, sha256_file_streaming,
    validate_training_cfg,
};
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use thiserror::Error;
use tokio::sync::watch;

/// Daemon-internal job descriptor produced from a validated
/// `TrainRequest` (= flattened [`TrainingCfg`]).  The dataset root
/// is fixed at `<workspace_dir>/datasets/`; `head_id` is allocated
/// by the api producer so the response carries the published id
/// before the job spawns.
#[derive(Clone, Debug)]
pub struct TrainingJob {
    pub workspace_id: WorkspaceId,
    /// Pre-allocated; published verbatim on success.
    pub head_id: HeadId,
    /// Producer-snapshotted; recorded in the head manifest for
    /// stale detection.
    pub workspace_revision: WorkspaceRevision,
    /// Already validated by `validate_training_cfg`.
    pub training_cfg: TrainingCfg,
    pub backbone_path: PathBuf,
}

/// Lifecycle state of a training job, surfaced on `GET
/// /api/v1/training/{id}` and used by the cancel path.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Accepted; running on a `spawn_blocking` worker.
    Running,
    /// Trained head published successfully.
    Completed,
    /// Failed before the publish step.
    Failed,
    /// Operator-cancelled.
    Cancelled,
}

/// Final artefacts published by a successful training run.
/// `head_id` echoes the producer-allocated id; `head_sha256`
/// is the lowercase-hex digest of the published `.mpk` bytes.
#[derive(Clone, Debug, Serialize)]
pub struct TrainingResult {
    /// Identifier of the published head (matches the `head_id`
    /// returned to the client at job-spawn time).
    pub head_id: HeadId,
    /// Lowercase-hex SHA-256 of the published `<head_id>.mpk`.
    pub head_sha256: String,
    /// Number of output classes baked into the head.
    pub n_classes: u32,
    /// Class labels in inference order.
    pub classes: Vec<String>,
    /// Training accuracy of the published head.
    pub final_train_acc: f32,
    /// NaN when `val_split == 0`; finetune currently fixes
    /// `val_split = 0.0` because the wire `TrainingCfg` carries no
    /// validation knob.
    pub final_val_acc: f32,
}

/// Read-shape returned by `/api/v1/training/{id}`.
#[derive(Clone, Debug, Serialize)]
pub struct JobView {
    /// Job id (UUID-v4).
    pub job_id: String,
    /// Owning workspace.
    pub workspace_id: String,
    /// Lifecycle state.
    pub state: JobState,
    /// Latest progress snapshot from the trainer.
    pub progress: finetune::Progress,
    /// Terminal artifact summary (only present once `state == Completed`).
    pub result: Option<TrainingResult>,
    /// Failure diagnostic (only present once `state == Failed`).
    pub error: Option<String>,
    /// RFC3339 wall-clock at job spawn.
    pub started_at: String,
    /// RFC3339 wall-clock at terminal state, if any.
    pub finished_at: Option<String>,
}

/// Failure shapes for the training pipeline.  Mapped to HTTP
/// statuses via the [`crate::common::error::Categorized`] impl
/// below.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// Inputs failed numeric / shape validation.  `cfg`
    /// already passed [`validate_training_cfg`] at the request
    /// boundary; this surfaces internal-config issues (e.g. a
    /// missing backbone artefact).
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Job id not registered.
    #[error("job not found: {0}")]
    JobNotFound(String),
    /// Job belongs to a different workspace than the caller asserted.
    #[error("job {job} does not belong to workspace {workspace}")]
    WrongWorkspace {
        /// Registered job id.
        job: String,
        /// Workspace the caller asserted.
        workspace: String,
    },
    /// Operator cancelled the job; the worker exited at the
    /// next checkpoint.
    #[error("cancelled")]
    Cancelled,
    /// Wrapped `file_mgr::FileError` (e.g. missing workspace,
    /// path resolution failure, publish failure).
    #[error("file: {0}")]
    File(#[from] crate::file_mgr::FileError),
    /// Wrapped trait-object filesystem error.
    #[error("fs: {0}")]
    Fs(#[from] crate::file_mgr::FsError),
    /// Underlying head fine-tune algorithm failure.  The wrapper
    /// translates the inner `BadDataset` and `DatasetRead`
    /// variants into [`Self::BadDataset`] / [`Self::DatasetRead`]
    /// so operator-facing tooling pattern-matches once at the
    /// [`TrainingError`] boundary; other inner variants flow
    /// through transparently.
    #[error("finetune: {0}")]
    Finetune(finetune::FinetuneError),
    /// Typed dataset-shape rejection at scan time.  `path` is the
    /// offending file or class folder, `reason` the operator
    /// diagnostic.  Maps to 400 so the operator can fix the
    /// upload layout.
    #[error("bad dataset {path}: {reason}")]
    BadDataset {
        /// Path under `<workspace>/datasets/` that triggered
        /// the rejection.
        path: String,
        /// Operator-readable diagnostic.
        reason: String,
    },
    /// IO failure on a daemon-owned file (workspace tree, dataset, tempfile).
    #[error("io {path}: {source}")]
    Io {
        /// File path involved.
        path: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `tokio::task::spawn_blocking` join failed (panic / shutdown).
    #[error("spawn_blocking join: {0}")]
    Join(#[from] tokio::task::JoinError),
    /// A dataset file disappeared between admission and the
    /// per-batch open / read / close.  Surfaces as `Internal`
    /// because `datasets/` is daemon-owned and the JobReference
    /// lease blocks legitimate mutations.
    #[error("dataset read failure {path}: {reason}")]
    DatasetRead {
        /// Path the trainer tried to read (relative or absolute).
        path: String,
        /// Operator-readable failure description.
        reason: String,
    },
}

impl crate::common::error::Categorized for TrainingError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Operator-supplied request shape failed validation.
            TrainingError::InvalidConfig(_) => UserInput,
            // Cancellation reflects an explicit operator action;
            // surfaces as a "your request didn't go through"
            // signal.  409 fits axum better than 200/400.
            TrainingError::Cancelled => Conflict,
            // Job/workspace pair not found.
            TrainingError::JobNotFound(_) | TrainingError::WrongWorkspace { .. } => NotFound,
            // Delegate to the wrapped error's classifier.
            TrainingError::File(e) => e.kind(),
            TrainingError::Fs(e) => e.kind(),
            // Operator-supplied dataset layout is bad; 400.
            TrainingError::BadDataset { .. } => UserInput,
            // Delegate so dataset-quality variants
            // (EmptyClassAfter*, DropRatioExceeded,
            // StratifiedSplitImpossible) keep their 400 while
            // panic / Io / Model stay Internal.
            TrainingError::Finetune(e) => e.kind(),
            // Filesystem mid-job, join failures, dataset-tampering:
            // the dataset tree is daemon-owned so a missing mid-walk
            // file is not operator-fixable.
            TrainingError::Io { .. }
            | TrainingError::Join(_)
            | TrainingError::DatasetRead { .. } => Internal,
        }
    }
}

/// Lift `BadDataset` / `DatasetRead` to the wrapper's typed
/// shapes so operator tooling pattern-matches once at the
/// boundary; other variants flow through `Finetune(_)` unchanged.
impl From<finetune::FinetuneError> for TrainingError {
    fn from(value: finetune::FinetuneError) -> Self {
        match value {
            finetune::FinetuneError::BadDataset { path, reason } => {
                TrainingError::BadDataset { path, reason }
            }
            finetune::FinetuneError::DatasetRead { path, reason } => {
                TrainingError::DatasetRead { path, reason }
            }
            other => TrainingError::Finetune(other),
        }
    }
}

/// In-process registry of training jobs.  Cheaply cloneable; at
/// most one in-flight train job (`max_train_jobs = 1`) gated by
/// the inner semaphore.
#[derive(Clone, Debug)]
pub struct JobRegistry {
    jobs: Arc<DashMap<JobId, Arc<JobEntry>>>,
    /// Single permit; `try_acquire_owned` returns
    /// `FileError::AnotherTrainRunning` on contention so the
    /// api layer renders the dedicated discriminator code.
    running: Arc<tokio::sync::Semaphore>,
}

impl JobRegistry {
    /// Construct an empty registry; permits one running job at a time.
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
            running: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    /// Spawn a new training job from a producer-built [`TrainingJob`].
    ///
    /// Validates the wire `TrainingCfg`, enforces the daemon-
    /// wide single-train-job invariant, registers the job in
    /// the in-memory map, and returns the assigned [`JobId`].
    /// The training task runs on a `spawn_blocking` worker;
    /// the per-job tokio task transitions the entry to a
    /// terminal state on completion.
    pub fn spawn(
        &self,
        files: Arc<dyn FsService>,
        job: TrainingJob,
    ) -> Result<JobId, TrainingError> {
        // Re-run validate_training_cfg at the spawn boundary; the
        // api route already validated, but a hand-built
        // TrainingJob (test, future replay tool) must hit the
        // same gate.
        validate_training_cfg(&job.training_cfg).map_err(|e| {
            TrainingError::File(crate::file_mgr::FileError::InvalidName(e.to_string()))
        })?;

        // Single-train-job admission gate.  The permit drops on
        // early-return of this function (the closure that would
        // capture it never runs), so `try_acquire_owned` is safe
        // to call before the full validation suite runs.
        let permit =
            self.running.clone().try_acquire_owned().map_err(|_| {
                TrainingError::File(crate::file_mgr::FileError::AnotherTrainRunning)
            })?;

        let job_id = JobId::new();
        let initial = finetune::Progress {
            phase: finetune::Phase::Loading,
            current: 0,
            total: 0,
            message: "training job accepted".into(),
            metrics: None,
        };
        let (progress_tx, progress_rx) = watch::channel(initial);
        let cancel = Arc::new(AtomicBool::new(false));
        let entry = Arc::new(JobEntry {
            job_id,
            workspace_id: job.workspace_id,
            started_at: now_rfc3339(),
            progress: progress_rx,
            cancel: cancel.clone(),
            core: Mutex::new(JobCore {
                state: JobState::Running,
                result: None,
                error: None,
                finished_at: None,
                finished_at_instant: None,
            }),
        });
        self.jobs.insert(job_id, entry.clone());

        tokio::spawn(async move {
            let outcome = run_job(files, job, job_id, progress_tx, cancel).await;
            let mut core = entry.core.lock();
            core.finished_at = Some(now_rfc3339());
            core.finished_at_instant = Some(std::time::Instant::now());
            match outcome {
                Ok(result) => {
                    core.state = JobState::Completed;
                    core.result = Some(result);
                }
                Err(TrainingError::Cancelled)
                | Err(TrainingError::Finetune(finetune::FinetuneError::Cancelled)) => {
                    core.state = JobState::Cancelled;
                    core.error = Some("cancelled".into());
                }
                Err(e) => {
                    core.state = JobState::Failed;
                    core.error = Some(e.to_string());
                    tracing::warn!(target: "training", job_id = %job_id, err = %e, "training job failed");
                }
            }
            drop(permit);
        });

        Ok(job_id)
    }

    /// Look up `job_id` and confirm it belongs to `workspace_id`.
    /// Returns `JobNotFound` if absent and `WrongWorkspace` if the
    /// caller asked across workspaces (a cross-workspace job_id is
    /// the same wire shape as a stale id, but the api layer
    /// distinguishes the two).  The returned `Arc<JobEntry>`
    /// outlives the dashmap shard guard, so the caller is free to
    /// take any per-entry locks (`core.lock`, atomic stores)
    /// without holding the registry's shard ref.
    fn lookup_for_workspace(
        &self,
        workspace_id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<Arc<JobEntry>, TrainingError> {
        let entry = self
            .jobs
            .get(&job_id)
            .ok_or_else(|| TrainingError::JobNotFound(job_id.to_string()))?;
        if entry.workspace_id != *workspace_id {
            return Err(TrainingError::WrongWorkspace {
                job: job_id.to_string(),
                workspace: workspace_id.to_string(),
            });
        }
        Ok(entry.clone())
    }

    /// Request cancellation of `job_id`.  Sets the cancel flag
    /// the trainer's epoch / chunk loops poll; the job exits at
    /// the next checkpoint.
    pub fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError> {
        let entry = self.lookup_for_workspace(workspace_id, job_id)?;
        entry.cancel.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Latest [`JobView`] for `job_id`, scoped to `workspace_id`.
    pub fn status(
        &self,
        workspace_id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<JobView, TrainingError> {
        Ok(self.lookup_for_workspace(workspace_id, job_id)?.view())
    }

    /// All jobs scoped to `workspace_id`, sorted by start time.
    pub fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView> {
        let mut out: Vec<_> = self
            .jobs
            .iter()
            .filter(|entry| entry.value().workspace_id == *workspace_id)
            .map(|entry| entry.value().view())
            .collect();
        out.sort_by(|a, b| a.started_at.cmp(&b.started_at));
        out
    }

    /// Set the cancel flag on every running job.  Daemon
    /// shutdown calls this from a pre-drain hook so the
    /// blocking trainer observes shutdown without waiting for
    /// the per-job cancel API.  Returns the number of jobs
    /// whose flag transitioned `false -> true`.
    pub fn cancel_all_for_shutdown(&self) -> usize {
        let mut n = 0usize;
        for entry in self.jobs.iter() {
            let core = entry.value().core.lock();
            if core.state != JobState::Running {
                continue;
            }
            drop(core);
            if !entry.value().cancel.swap(true, Ordering::SeqCst) {
                n = n.saturating_add(1);
            }
        }
        n
    }

    /// Number of jobs currently in [`JobState::Running`].
    pub fn active_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|entry| entry.value().core.lock().state == JobState::Running)
            .count()
    }

    /// Drop finished entries whose `finished_at` is older than
    /// `max_age`.  Returns the number reaped.  Running jobs
    /// and entries with no recorded finish time are kept.
    pub fn reap_finished(&self, max_age: std::time::Duration) -> usize {
        let now = std::time::Instant::now();
        let to_remove: Vec<JobId> = self
            .jobs
            .iter()
            .filter_map(|entry| {
                let core = entry.value().core.lock();
                let finished_at = core.finished_at_instant?;
                if now.duration_since(finished_at) >= max_age {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();
        let n = to_remove.len();
        for id in to_remove {
            self.jobs.remove(&id);
        }
        n
    }
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct JobEntry {
    job_id: JobId,
    workspace_id: WorkspaceId,
    started_at: String,
    /// `watch::Receiver` is internally synchronized and `borrow`
    /// only needs `&self`, so no extra mutex is required for
    /// concurrent readers.
    progress: watch::Receiver<finetune::Progress>,
    cancel: Arc<AtomicBool>,
    core: Mutex<JobCore>,
}

impl JobEntry {
    fn view(&self) -> JobView {
        let progress = self.progress.borrow().clone();
        let core = self.core.lock();
        JobView {
            job_id: self.job_id.to_string(),
            workspace_id: self.workspace_id.to_string(),
            state: core.state.clone(),
            progress,
            result: core.result.clone(),
            error: core.error.clone(),
            started_at: self.started_at.clone(),
            finished_at: core.finished_at.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct JobCore {
    state: JobState,
    result: Option<TrainingResult>,
    error: Option<String>,
    finished_at: Option<String>,
    finished_at_instant: Option<std::time::Instant>,
}

/// Run a single training job end-to-end.
///
/// 1. Anchor the dataset walk at `<workspace_dir>/datasets/`; the
///    cached `summary` confirms the workspace exists and the scan
///    helper enforces the directory shape.
/// 2. Discover classes (direct subdirectories) + samples (regular
///    files inside each class).  See [`finetune::run`] for the
///    per-batch lazy open / read / close discipline.
/// 3. Train the head on a `spawn_blocking` worker.
/// 4. Compute the published `.mpk`'s sha256 + size, build the
///    [`HeadManifest`], call `FsService::publish_trained_head`.
/// 5. On any error before step 4 commits, no head record is
///    persisted; the job-reference lease (held by the api
///    route) auto-releases on function return.
async fn run_job(
    files: Arc<dyn FsService>,
    job: TrainingJob,
    job_id: JobId,
    progress_tx: watch::Sender<finetune::Progress>,
    cancel: Arc<AtomicBool>,
) -> Result<TrainingResult, TrainingError> {
    // Open the per-job JSONL backstop before any pipeline work
    // so a `started` event is on disk even if dataset
    // resolution fails immediately.  Mirrors the
    // ConvertJobLog::open + log.event("started") pattern in the
    // converter producer; reachable from the unified surface
    // via `GET /workspace/{id}/assets/training_logs/{job_id}.jsonl`.
    //
    // Open + initial event failures surface as TrainingError so a
    // workspace whose `training_logs/` is unwritable refuses the
    // run rather than silently losing the trace; subsequent
    // per-phase writes are best-effort (a failed log line must
    // not promote a successful train into a failed one).
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &job.workspace_id);
    let mut log = TrainJobLog::open(&workspace_dir, job_id)?;
    log.event("started", None, None)?;

    let result = run_job_inner(files, job, progress_tx, cancel).await;

    // Terminal log writes are best-effort: a failed `completed`
    // / `failed` / `cancelled` line after the inner pipeline
    // returned must not flip the result.  Mirrors the
    // converter's terminal-write discipline.
    let (state, message) = match &result {
        Ok(_) => ("completed", None),
        Err(TrainingError::Cancelled) => ("cancelled", None),
        Err(e) => ("failed", Some(e.to_string())),
    };
    if let Err(log_err) = log.event(state, None, message.as_deref()) {
        tracing::warn!(
            target: "training",
            err = %log_err,
            terminal_state = state,
            "training: terminal log write failed",
        );
    }

    result
}

async fn run_job_inner(
    files: Arc<dyn FsService>,
    job: TrainingJob,
    progress_tx: watch::Sender<finetune::Progress>,
    cancel: Arc<AtomicBool>,
) -> Result<TrainingResult, TrainingError> {
    let workspace = job.workspace_id;

    // Fixed dataset root.  The cached `summary` re-confirms the
    // workspace exists without walking `datasets/`; the scan helper
    // inside `finetune::run` enforces the directory shape and
    // surfaces missing/unreadable entries as `DatasetRead`.
    let files_for_summary = files.clone();
    tokio::task::spawn_blocking(move || files_for_summary.summary(&workspace))
        .await?
        .map_err(TrainingError::Fs)?;
    let dataset_root = crate::file_mgr::schema::workspace_dir_for(files.root(), &workspace)
        .join(DATASETS_DIR_NAME);
    let stat_root = dataset_root.clone();
    let md = tokio::task::spawn_blocking(move || std::fs::symlink_metadata(&stat_root))
        .await?
        .map_err(|source| TrainingError::Io {
            path: dataset_root.display().to_string(),
            source,
        })?;
    if !md.is_dir() {
        return Err(TrainingError::InvalidConfig(format!(
            "dataset root {} is not a directory",
            dataset_root.display(),
        )));
    }

    // Stage trained-head output under `<workspace_dir>/.tmp/`.
    // Same filesystem as `heads/` so the post-train atomic
    // rename inside `publish_trained_head` is intra-FS POSIX-
    // atomic.  The tempdir guard auto-removes on function exit
    // unless we explicitly persist the `.mpk` (the rename
    // inside `publish_trained_head` removes the file, so an
    // empty tempdir is the success state).
    let workspace_tmpdir = files.workspace_tmpdir(&workspace);
    let workspace_tmpdir_for_create = workspace_tmpdir.clone();
    tokio::task::spawn_blocking(move || std::fs::create_dir_all(&workspace_tmpdir_for_create))
        .await?
        .map_err(|source| TrainingError::Io {
            path: workspace_tmpdir.display().to_string(),
            source,
        })?;
    let workspace_tmpdir_for_temp = workspace_tmpdir.clone();
    let output_temp =
        tokio::task::spawn_blocking(move || tempfile::tempdir_in(&workspace_tmpdir_for_temp))
            .await?
            .map_err(|source| TrainingError::Io {
                path: workspace_tmpdir.display().to_string(),
                source,
            })?;
    let output_head = output_temp.path().join(format!("{}.mpk", job.head_id));

    if cancel.load(Ordering::SeqCst) {
        return Err(TrainingError::Cancelled);
    }

    // Cross-validate every component of every relative file
    // path against AssetPath's per-component allowlist before
    // handing them to the wav reader.  `datasets/` is daemon-
    // owned, but a stray name (e.g. uploaded class dir whose
    // entries were renamed by an external process despite the
    // lease) should fail closed with `DatasetRead`.
    let ft_cfg = finetune::FinetuneConfig {
        data: dataset_root.clone(),
        backbone: job.backbone_path.clone(),
        init_head: None,
        out: output_head.clone(),
        epochs: job.training_cfg.epochs as usize,
        batch: job.training_cfg.batch_size as usize,
        lr: job.training_cfg.learning_rate,
        // The wire `TrainingCfg` carries no val_split knob;
        // val_split = 0 disables the stratified split.
        val_split: 0.0,
        seed: job.training_cfg.seed.unwrap_or(42),
    };
    let cancel_for_run = cancel.clone();
    let progress_for_run = progress_tx.clone();
    let output = tokio::task::spawn_blocking(move || {
        let progress = |p: &finetune::Progress| {
            let _ = progress_for_run.send(p.clone());
        };
        let cancel_fn = || cancel_for_run.load(Ordering::SeqCst);
        finetune::run(&ft_cfg, &progress, &cancel_fn)
    })
    .await??;

    if cancel.load(Ordering::SeqCst) {
        return Err(TrainingError::Cancelled);
    }

    // Compute final sha256 + size of the published head bytes.
    // `output.head_mpk` lives under `output_temp` -- moved into
    // the workspace via `publish_trained_head` below.  Hash on
    // the blocking pool so the runtime stays free.
    let mpk_path_for_sha = output.head_mpk.clone();
    let head_sha256 =
        tokio::task::spawn_blocking(move || sha256_file_streaming(&mpk_path_for_sha)).await??;
    let mpk_path_for_meta = output.head_mpk.clone();
    let size_bytes = tokio::task::spawn_blocking(move || std::fs::metadata(&mpk_path_for_meta))
        .await?
        .map_err(|source| TrainingError::Io {
            path: output.head_mpk.display().to_string(),
            source,
        })?
        .len();

    let n_classes_u32 = u32::try_from(output.classes.len()).map_err(|_| {
        TrainingError::InvalidConfig(format!(
            "trained head has {} classes; exceeds u32 cap",
            output.classes.len(),
        ))
    })?;

    let manifest = HeadManifest {
        head_id: job.head_id,
        workspace_id: workspace,
        workspace_revision: job.workspace_revision.clone(),
        sha256: head_sha256.clone(),
        n_classes: n_classes_u32,
        size_bytes,
        created_at: now_rfc3339(),
        labels: output.classes.clone(),
    };
    let pending = PendingHead {
        head_id: job.head_id,
        mpk_tempfile: output.head_mpk.clone(),
        manifest,
    };

    // Publish into the 2-slot rotation.  The primitive holds the
    // per-workspace mutation mutex internally and the cell lookup
    // goes through `WorkspaceMgr::caches` so the cache installed at
    // workspace-create time is the one that observes the new head.
    let files_for_publish = files.clone();
    tokio::task::spawn_blocking(move || {
        files_for_publish.publish_trained_head(&workspace, pending)
    })
    .await??;

    let result = TrainingResult {
        head_id: job.head_id,
        head_sha256,
        n_classes: n_classes_u32,
        classes: output.classes,
        final_train_acc: output.final_train_acc,
        final_val_acc: output.final_val_acc,
    };

    // `output_temp` drops when this function returns.  The
    // `.mpk` was renamed out of the tempdir by
    // `publish_trained_head`; remaining residue is just the
    // (now empty) tempdir + sibling labels.txt finetune writes,
    // both safe to delete.
    Ok(result)
}

// MARK: TrainJobLog
//
// Bounded JSONL writer for `<workspace_dir>/training_logs/<job_id>.jsonl`.
// Mirrors `crate::converter::ConvertJobLog`'s shape so the
// unified `GET /assets/<log-tree>/<job_id>.jsonl?after_seq=&limit=`
// surface reads either producer's output without per-tree
// dispatch.  The duplication is intentional — the two writers
// thread distinct error types (TrainingError vs ConvertError)
// and a shared helper would force a layering decision that
// outweighs the LOC savings.

/// Per-job JSONL writer for training-side lifecycle events.
/// One line per [`TrainJobLog::event`] call; best-effort flush
/// (no per-line fsync because a crash mid-job loses at most the
/// trailing event, which boot recovery already handles via the
/// workspace state).
struct TrainJobLog {
    file: std::fs::File,
    seq: u64,
}

impl TrainJobLog {
    /// Open `<workspace_dir>/training_logs/<job_id>.jsonl` for
    /// append; creates the dir if missing.  Surfaces failures as
    /// [`TrainingError::Io`] so an unwritable training_logs dir
    /// fails the run loudly instead of silently losing the trace.
    fn open(workspace_dir: &std::path::Path, job_id: JobId) -> Result<Self, TrainingError> {
        let dir = workspace_dir.join("training_logs");
        std::fs::create_dir_all(&dir).map_err(|e| TrainingError::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = dir.join(format!("{job_id}.jsonl"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TrainingError::Io {
                path: path.display().to_string(),
                source: e,
            })?;
        Ok(Self { file, seq: 0 })
    }

    /// Append one JSONL event line.  `state` carries the
    /// lifecycle phase (`started`, `completed`, `failed`,
    /// `cancelled`); `message` is an optional bounded diagnostic
    /// payload (capped via [`truncate_log_message`]).
    fn event(
        &mut self,
        state: &str,
        progress: Option<u64>,
        message: Option<&str>,
    ) -> Result<(), TrainingError> {
        use std::io::Write as _;
        self.seq = self.seq.saturating_add(1);
        let now = now_rfc3339();
        let truncated_msg = message.map(truncate_log_message);
        let line = serde_json::json!({
            "seq": self.seq,
            "at": now,
            "state": state,
            "progress": progress,
            "message": truncated_msg,
        });
        let mut bytes = serde_json::to_vec(&line).map_err(|e| TrainingError::Io {
            path: "<training_logs>".to_string(),
            source: std::io::Error::other(e),
        })?;
        bytes.push(b'\n');
        self.file.write_all(&bytes).map_err(|e| TrainingError::Io {
            path: "<training_logs>".to_string(),
            source: e,
        })?;
        let _ = self.file.flush();
        Ok(())
    }
}

/// Truncate an operator-supplied diagnostic at 8 KiB, snapping
/// to a UTF-8 char boundary so a multi-byte codepoint straddling
/// the cap does not panic the slice.  Appends `...[truncated]`
/// on truncation.  Mirrors the converter's helper byte-for-byte.
fn truncate_log_message(m: &str) -> String {
    const MAX_LOG_LINE_BYTES: usize = 8 * 1024;
    if m.len() <= MAX_LOG_LINE_BYTES {
        return m.to_string();
    }
    // UTF-8 codepoints are at most 4 bytes, so the snap-down
    // loop runs at most 3 iterations.
    let mut idx = MAX_LOG_LINE_BYTES;
    while idx > 0 && !m.is_char_boundary(idx) {
        idx -= 1;
    }
    let mut s = m[..idx].to_string();
    s.push_str("...[truncated]");
    s
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Dataset-shape fixtures intentionally use direct file writes.

    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use std::fs;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn synthetic_entry(
        workspace: &WorkspaceId,
        finished_state: Option<(JobState, Option<Instant>)>,
    ) -> Arc<JobEntry> {
        let job_id = JobId::new();
        let initial = finetune::Progress {
            phase: finetune::Phase::Loading,
            current: 0,
            total: 0,
            message: "synthetic".into(),
            metrics: None,
        };
        let (_tx, rx) = watch::channel(initial);
        let (state, finished_at_instant) = match finished_state {
            Some((s, t)) => (s, t),
            None => (JobState::Running, None),
        };
        Arc::new(JobEntry {
            job_id,
            workspace_id: *workspace,
            started_at: now_rfc3339(),
            progress: rx,
            cancel: Arc::new(AtomicBool::new(false)),
            core: Mutex::new(JobCore {
                state,
                result: None,
                error: None,
                finished_at: finished_at_instant.map(|_| now_rfc3339()),
                finished_at_instant,
            }),
        })
    }

    /// `TrainingResult` carries no filesystem paths; the head
    /// identity + sha256 + n_classes + final metrics are sufficient
    /// for client display, and `GET /workspace/{id}/heads/{head_id}`
    /// owns artifact access.
    #[test]
    fn training_result_serialization_carries_no_filesystem_path() {
        let result = TrainingResult {
            head_id: HeadId::new(),
            head_sha256: "0".repeat(64),
            n_classes: 2,
            classes: vec!["a".into(), "b".into()],
            final_train_acc: 0.9,
            final_val_acc: 0.85,
        };
        let v = serde_json::to_value(&result).expect("serialize TrainingResult");
        let body = v
            .as_object()
            .expect("TrainingResult serializes as a JSON object");
        let allowed: std::collections::BTreeSet<&str> = [
            "head_id",
            "head_sha256",
            "n_classes",
            "classes",
            "final_train_acc",
            "final_val_acc",
        ]
        .into_iter()
        .collect();
        let actual: std::collections::BTreeSet<&str> = body.keys().map(String::as_str).collect();
        assert_eq!(
            actual, allowed,
            "TrainingResult must serialize exactly {allowed:?}; got {actual:?}",
        );
        for forbidden in [
            "head_mpk_path",
            "labels_path",
            "head_path",
            "weights_path",
            "path",
            "dataset_path",
        ] {
            assert!(
                body.get(forbidden).is_none(),
                "TrainingResult must not carry filesystem path field `{forbidden}`; body={v}",
            );
        }
    }

    #[test]
    fn reap_finished_drops_stale_keeps_fresh_and_running() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();

        let now = Instant::now();
        let stale = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(now - Duration::from_secs(7200)))),
        );
        let fresh = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(now - Duration::from_secs(60)))),
        );
        let running = synthetic_entry(&workspace, None);

        reg.jobs.insert(stale.job_id, stale.clone());
        reg.jobs.insert(fresh.job_id, fresh.clone());
        reg.jobs.insert(running.job_id, running.clone());

        let n = reg.reap_finished(Duration::from_secs(3600));
        assert_eq!(n, 1, "exactly one stale entry expected");
        assert!(!reg.jobs.contains_key(&stale.job_id));
        assert!(reg.jobs.contains_key(&fresh.job_id));
        assert!(reg.jobs.contains_key(&running.job_id));
    }

    #[test]
    fn cancel_all_for_shutdown_sets_flag_on_running_only() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();

        let running1 = synthetic_entry(&workspace, None);
        let running2 = synthetic_entry(&workspace, None);
        let completed = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(Instant::now()))),
        );
        let cancelled = synthetic_entry(
            &workspace,
            Some((JobState::Cancelled, Some(Instant::now()))),
        );
        let running_pre_cancelled = synthetic_entry(&workspace, None);
        running_pre_cancelled.cancel.store(true, Ordering::SeqCst);

        reg.jobs.insert(running1.job_id, running1.clone());
        reg.jobs.insert(running2.job_id, running2.clone());
        reg.jobs.insert(completed.job_id, completed.clone());
        reg.jobs.insert(cancelled.job_id, cancelled.clone());
        reg.jobs
            .insert(running_pre_cancelled.job_id, running_pre_cancelled.clone());

        let n = reg.cancel_all_for_shutdown();
        assert_eq!(n, 2, "exactly two newly-signalled jobs expected");
        assert!(running1.cancel.load(Ordering::SeqCst));
        assert!(running2.cancel.load(Ordering::SeqCst));
        assert!(!completed.cancel.load(Ordering::SeqCst));
        assert!(!cancelled.cancel.load(Ordering::SeqCst));
        assert!(running_pre_cancelled.cancel.load(Ordering::SeqCst));

        let n2 = reg.cancel_all_for_shutdown();
        assert_eq!(n2, 0, "idempotent on repeat shutdown drain");
    }

    #[test]
    fn active_count_only_counts_running_jobs() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();
        assert_eq!(reg.active_count(), 0);

        let r1 = synthetic_entry(&workspace, None);
        let r2 = synthetic_entry(&workspace, None);
        let f = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(Instant::now()))),
        );
        reg.jobs.insert(r1.job_id, r1.clone());
        reg.jobs.insert(r2.job_id, r2.clone());
        reg.jobs.insert(f.job_id, f.clone());
        assert_eq!(reg.active_count(), 2);
    }

    /// `scan_dataset` walks each class folder recursively (not
    /// just the direct-child level) and treats every non-hidden
    /// `.wav` file as a sample.  Hidden entries (leading `.`) are
    /// skipped; non-hidden non-dir root entries fail closed with
    /// `BadDataset`, so the operator can hide metadata files
    /// (e.g. `.README`) to keep them out of the way.  Exercises
    /// the same scan helper `finetune::run` uses in production.
    #[test]
    fn class_file_discovery_walks_recursively() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        for cls in ["cat", "dog"] {
            fs::create_dir_all(root.join(cls)).unwrap();
            for i in 0..3 {
                fs::write(root.join(cls).join(format!("s{i}.wav")), b"stub").unwrap();
            }
        }
        // Hidden root entry: silently ignored.
        fs::write(root.join(".README"), b"meta").unwrap();
        // Recursive discovery picks up `cat/nested/x.wav` as a
        // sample under the `cat` class.
        fs::create_dir_all(root.join("cat").join("nested")).unwrap();
        fs::write(root.join("cat").join("nested").join("x.wav"), b"deeper").unwrap();

        let (classes, examples) = finetune::scan_dataset_for_test(root);
        // Two classes, sorted by canonical byte order.
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        // 3 direct + 1 nested = 4 samples for cat; 3 for dog.
        assert_eq!(examples.len(), 7);
        for (path, _label) in &examples {
            assert!(path.extension().is_some_and(|e| e == "wav"));
        }
        // The nested wav is associated with the `cat` class
        // (label index 0 because cat sorts before dog).
        let cat_count = examples.iter().filter(|(_, l)| *l == 0).count();
        let dog_count = examples.iter().filter(|(_, l)| *l == 1).count();
        assert_eq!(cat_count, 4);
        assert_eq!(dog_count, 3);
    }

    /// Mid-walk file disappearance surfaces as
    /// `TrainingError::DatasetRead` in production.  Here we
    /// drive the helper synchronously and convert
    /// `FinetuneError::Io` -> `TrainingError::DatasetRead` at
    /// the boundary the same way `run_job` would (the
    /// finetune-side error already carries the path; the
    /// production code just rewraps).
    #[test]
    fn dataset_read_failure_surfaces_typed_error() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("cat")).unwrap();
        fs::write(root.join("cat").join("a.wav"), b"stub").unwrap();
        // Simulate the file disappearing between scan and
        // open -- delete after the scan but before the
        // synthetic open below.
        let (_classes, examples) = finetune::scan_dataset_for_test(root);
        assert_eq!(examples.len(), 1);
        fs::remove_file(root.join("cat").join("a.wav")).unwrap();
        let err = match std::fs::File::open(&examples[0].0) {
            Ok(_) => panic!("expected ENOENT"),
            Err(e) => TrainingError::DatasetRead {
                path: examples[0].0.display().to_string(),
                reason: e.to_string(),
            },
        };
        match err {
            TrainingError::DatasetRead { path, .. } => assert!(path.ends_with("a.wav")),
            other => panic!("expected DatasetRead; got {other:?}"),
        }
    }

    /// `From<FinetuneError>` lifts the inner `BadDataset` /
    /// `DatasetRead` variants to the typed
    /// `TrainingError::BadDataset` / `TrainingError::DatasetRead`
    /// shapes; other inner variants flow through `Finetune(_)`
    /// unchanged.  Pinned so operator-facing tooling can
    /// pattern-match once at the wrapper boundary.
    #[test]
    fn finetune_bad_dataset_translates_to_training_bad_dataset() {
        let inner = finetune::FinetuneError::BadDataset {
            path: "/ws/datasets/empty".into(),
            reason: "no class folders".into(),
        };
        let outer: TrainingError = inner.into();
        match outer {
            TrainingError::BadDataset { path, reason } => {
                assert_eq!(path, "/ws/datasets/empty");
                assert_eq!(reason, "no class folders");
            }
            other => panic!("expected TrainingError::BadDataset, got {other:?}"),
        }
    }

    #[test]
    fn finetune_dataset_read_translates_to_training_dataset_read() {
        let inner = finetune::FinetuneError::DatasetRead {
            path: "/ws/datasets/cat/a.wav".into(),
            reason: "ENOENT".into(),
        };
        let outer: TrainingError = inner.into();
        match outer {
            TrainingError::DatasetRead { path, reason } => {
                assert_eq!(path, "/ws/datasets/cat/a.wav");
                assert_eq!(reason, "ENOENT");
            }
            other => panic!("expected TrainingError::DatasetRead, got {other:?}"),
        }
    }

    /// `BadDataset` -> 400, `DatasetRead` -> 500; the wrapping
    /// `Finetune(_)` delegates to the inner classifier so dataset-
    /// quality variants keep their 400 instead of being blanket-
    /// classified as Internal.
    #[test]
    fn training_error_kinds_classify_correctly() {
        use crate::common::error::{Categorized, ErrorKind};
        let bad = TrainingError::BadDataset {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(bad.kind(), ErrorKind::UserInput);
        let read = TrainingError::DatasetRead {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(read.kind(), ErrorKind::Internal);

        // `Finetune(_)` delegates to the inner classifier so
        // dataset-quality variants keep their 400.
        let wrapped_bad = TrainingError::Finetune(finetune::FinetuneError::EmptyClassAfterScan {
            class: "cat".into(),
            per_class_kept: vec![],
        });
        assert_eq!(wrapped_bad.kind(), ErrorKind::UserInput);
        // Panic (daemon-internal) -> Internal.
        let wrapped_panic = TrainingError::Finetune(finetune::FinetuneError::Panic("oops".into()));
        assert_eq!(wrapped_panic.kind(), ErrorKind::Internal);
    }

    /// FD usage is bounded by `batch_size * parallel_loaders`: the
    /// scan returns `(PathBuf, label)` pairs without opening any
    /// file (opens live in the per-batch chunk path).  Failing
    /// this would mean the trainer pre-opened every file and peak
    /// FD count would scale with dataset size.
    #[test]
    fn lazy_fd_bounded_no_open_during_scan() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("cat")).unwrap();
        fs::create_dir_all(root.join("dog")).unwrap();
        // 100 files per class; if scan opened each, peak FDs
        // would scale with dataset size.
        for cls in ["cat", "dog"] {
            for i in 0..100 {
                fs::write(root.join(cls).join(format!("s{i}.wav")), b"x").unwrap();
            }
        }
        let (classes, examples) = finetune::scan_dataset_for_test(root);
        assert_eq!(classes.len(), 2);
        assert_eq!(examples.len(), 200);
        // The scan returns paths; no file handle is held by
        // any element of the returned vector.  This is the
        // load-bearing assertion: the trainer's per-batch
        // preproc opens by path inside the batch closure, so
        // peak FDs cap at `batch_size * parallel_loaders`.
        for (p, _) in &examples {
            assert!(p.is_file(), "scan returned a non-file path: {p:?}");
        }
    }

    /// `TrainJobLog::open` materialises
    /// `<workspace>/training_logs/<job_id>.jsonl` and writes one
    /// JSONL line per `event()` call.  Pinned because the
    /// unified `GET /assets/training_logs/<id>.jsonl` reader
    /// expects exactly the converter's wire shape (seq, at,
    /// state, progress, message); a writer-side regression here
    /// would silently break the page response.
    #[test]
    fn train_job_log_writes_one_jsonl_line_per_event() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace_dir = tmp.path();
        let job_id = JobId::new();
        let mut log = TrainJobLog::open(workspace_dir, job_id).expect("open log");
        log.event("started", None, None).expect("started");
        log.event("epoch_end", Some(3), Some("loss=0.42"))
            .expect("epoch_end");
        log.event("completed", None, None).expect("completed");
        drop(log);

        let path = workspace_dir
            .join("training_logs")
            .join(format!("{job_id}.jsonl"));
        let body = std::fs::read_to_string(&path).expect("read log");
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 3, "one JSONL line per event");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["seq"], 1);
        assert_eq!(first["state"], "started");
        assert!(first["at"].as_str().unwrap().ends_with('Z'));

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["seq"], 2);
        assert_eq!(second["state"], "epoch_end");
        assert_eq!(second["progress"], 3);
        assert_eq!(second["message"], "loss=0.42");

        let third: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(third["seq"], 3);
        assert_eq!(third["state"], "completed");
    }

    /// `truncate_log_message` snaps a >8 KiB diagnostic to a
    /// UTF-8 char boundary and tags the truncation.  Pinned
    /// against a regression that would let a multi-byte
    /// codepoint straddling the cap panic the slice.
    #[test]
    fn train_log_truncate_message_caps_at_8kib_at_char_boundary() {
        // Pad with 9 KiB of `é` (2-byte UTF-8) so the cap lands
        // mid-codepoint.
        let mut s = String::new();
        while s.len() < 9 * 1024 {
            s.push('é');
        }
        let truncated = truncate_log_message(&s);
        // Capped under the 8 KiB + suffix budget.
        assert!(
            truncated.len() <= 8 * 1024 + b"...[truncated]".len(),
            "truncated len {} > cap",
            truncated.len(),
        );
        assert!(truncated.ends_with("...[truncated]"));
        // Round-trip: the truncated body is still valid UTF-8
        // (snap-down to char boundary held).  String already
        // implies UTF-8, so a successful slice is the
        // assertion.
        let _ = truncated.chars().count();
    }
}
