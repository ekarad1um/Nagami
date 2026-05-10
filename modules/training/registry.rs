//! `TrainingRegistry` trait + production
//! [`JobRegistry`] impl.  Held by the api crate as
//! `Arc<dyn TrainingRegistry>`; tests substitute mocks
//! without touching the in-process job machinery.
//!
//! ## Trait location
//!
//! Colocates with [`JobRegistry`] and the training DTOs so the
//! trait sits next to its DTO references.  Lifting into
//! `crate::common::traits` would force `common` to re-export
//! `TrainingJob`, `JobView`, `TrainingError`, etc. -- wide
//! surface for zero architectural benefit; the api crate already
//! imports `training` for those DTOs.
//!
//! ## Object safety
//!
//! Every method takes concrete params + returns concrete
//! `Result<_, TrainingError>` -- no generics, no `Self:
//! Sized`.  `Arc<dyn TrainingRegistry>` is constructible.

use crate::common::ids::{JobId, WorkspaceId};
use crate::file_mgr::FsService;
use crate::training::{JobRegistry, JobView, TrainingError, TrainingJob};
use std::sync::Arc;

/// Submit + observe + cancel in-process training jobs.
/// Production impl: [`JobRegistry`] (DashMap-backed; one
/// in-flight job at a time per redesign §9 `max_train_jobs = 1`).
pub trait TrainingRegistry: Send + Sync + std::fmt::Debug {
    /// Submit a training job.  Validates the wire `TrainingCfg`
    /// and runs the admission gate (one running train job
    /// daemon-wide) before returning a [`JobId`]; the actual
    /// training runs on a `spawn_blocking` worker.  Returns the
    /// wrapped `FileError::AnotherTrainRunning` if another job
    /// is already in flight, mapped to HTTP 409 with the
    /// `another_train_running` discriminator code.
    fn spawn(&self, files: Arc<dyn FsService>, job: TrainingJob) -> Result<JobId, TrainingError>;

    /// Cancel an in-flight job.  The training task observes
    /// the cancel flag at its next progress emit and exits;
    /// the result is reported as
    /// [`crate::training::JobState::Cancelled`].
    fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError>;

    /// Read one job's view by `(workspace, job_id)`.
    fn status(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<JobView, TrainingError>;

    /// All jobs registered against `workspace_id`.
    fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView>;

    /// Set the cancel flag on every running job.  Daemon's
    /// drain registry uses this as a pre-drain hook so blocking
    /// trainers observe shutdown immediately.  Returns the
    /// number of jobs whose flag was newly set.
    fn cancel_all_for_shutdown(&self) -> usize;

    /// Number of jobs currently running.  Surfaced through the
    /// `training` heartbeat so `/api/v1/status` distinguishes
    /// "idle" from "running" from "cancelling N jobs".
    fn active_count(&self) -> usize;
}

impl TrainingRegistry for JobRegistry {
    fn spawn(&self, files: Arc<dyn FsService>, job: TrainingJob) -> Result<JobId, TrainingError> {
        JobRegistry::spawn(self, files, job)
    }
    fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError> {
        JobRegistry::cancel(self, workspace_id, job_id)
    }
    fn status(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<JobView, TrainingError> {
        JobRegistry::status(self, workspace_id, job_id)
    }
    fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView> {
        JobRegistry::list_for_workspace(self, workspace_id)
    }
    fn cancel_all_for_shutdown(&self) -> usize {
        JobRegistry::cancel_all_for_shutdown(self)
    }
    fn active_count(&self) -> usize {
        JobRegistry::active_count(self)
    }
}

// Object-safety smoke.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn TrainingRegistry>();
};
