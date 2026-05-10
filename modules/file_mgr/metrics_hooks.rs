//! Tier-respecting metrics publication for `file_mgr` hot paths.
//!
//! `file_mgr` is restricted to depend only on `common`, so
//! `crate::status::WorkspaceMetrics` cannot be referenced
//! directly from here.  Instead, this module exposes typed `Fn`
//! slots that the daemon (which depends on both `file_mgr` and
//! `status`) installs at boot.  Hot paths invoke the slot if
//! installed and treat absence as a no-op.
//!
//! Each closure slot is a `OnceLock<Box<dyn Fn(...) + Send +
//! Sync>>` so installation is single-shot and lock-free on
//! the read path.  Tests that care about the counters install
//! their own closures; everything else (the unit-test sweep
//! against `WorkspaceMgr` directly) sees a no-op slot and
//! pays one atomic load per hot-path event.

use std::sync::OnceLock;
use std::time::Duration;

/// Callback signature for `record_workspace_core_write` /
/// `record_head_index_write`.  The duration argument is the
/// wall-clock time spent inside the atomic rewrite (cap
/// check + serialize + tempfile write + fsync + rename +
/// parent fsync).
pub type WriteDurationHook = dyn Fn(Duration) + Send + Sync + 'static;

/// Callback signature for `record_upload`.  The argument is
/// the observed upload size in bytes (the receipt's
/// `size_bytes`).  Implementations bump `assets_uploaded_total`
/// by 1 and `bytes_uploaded_total` by `bytes`.
pub type UploadHook = dyn Fn(u64) + Send + Sync + 'static;

/// Callback signature for parameterless counters
/// (`record_dataset_mutation_rejected`,
/// `record_converter_mutation_rejected`).
pub type IncrementHook = dyn Fn() + Send + Sync + 'static;

/// Callback signature for `record_job_events_dropped`.
/// The argument is the number of broadcast events that
/// failed to deliver because every receiver had lagged out.
pub type EventsDroppedHook = dyn Fn(u64) + Send + Sync + 'static;

static WORKSPACE_CORE_WRITE: OnceLock<Box<WriteDurationHook>> = OnceLock::new();
static HEAD_INDEX_WRITE: OnceLock<Box<WriteDurationHook>> = OnceLock::new();
static UPLOAD: OnceLock<Box<UploadHook>> = OnceLock::new();
static DATASET_MUTATION_REJECTED: OnceLock<Box<IncrementHook>> = OnceLock::new();
static CONVERTER_MUTATION_REJECTED: OnceLock<Box<IncrementHook>> = OnceLock::new();
static JOB_EVENTS_DROPPED: OnceLock<Box<EventsDroppedHook>> = OnceLock::new();

/// Install the `workspace.json` write-duration hook.  Daemon
/// boot calls this once with a closure that forwards into
/// `crate::status::WorkspaceMetrics::record_workspace_core_write`.
/// Subsequent calls are silently dropped (single-shot
/// `OnceLock`).
pub fn install_workspace_core_write_hook<F>(f: F)
where
    F: Fn(Duration) + Send + Sync + 'static,
{
    let _ = WORKSPACE_CORE_WRITE.set(Box::new(f));
}

/// Install the `heads.json` write-duration hook.
pub fn install_head_index_write_hook<F>(f: F)
where
    F: Fn(Duration) + Send + Sync + 'static,
{
    let _ = HEAD_INDEX_WRITE.set(Box::new(f));
}

/// Install the `record_upload(bytes)` hook.
pub fn install_upload_hook<F>(f: F)
where
    F: Fn(u64) + Send + Sync + 'static,
{
    let _ = UPLOAD.set(Box::new(f));
}

/// Install the `record_dataset_mutation_rejected` hook.
pub fn install_dataset_mutation_rejected_hook<F>(f: F)
where
    F: Fn() + Send + Sync + 'static,
{
    let _ = DATASET_MUTATION_REJECTED.set(Box::new(f));
}

/// Install the converter-side mirror of
/// `install_dataset_mutation_rejected_hook`.  The dataset surface
/// dispatches by `AssetTree` and emits one or the other on
/// admission rejection.
pub fn install_converter_mutation_rejected_hook<F>(f: F)
where
    F: Fn() + Send + Sync + 'static,
{
    let _ = CONVERTER_MUTATION_REJECTED.set(Box::new(f));
}

/// Install the `record_job_events_dropped(n)` hook.
pub fn install_job_events_dropped_hook<F>(f: F)
where
    F: Fn(u64) + Send + Sync + 'static,
{
    let _ = JOB_EVENTS_DROPPED.set(Box::new(f));
}

/// Invoke the workspace-core write hook, if installed.
/// `pub(crate)` because external callers go through the
/// `record_*_write` shape on
/// `crate::status::WorkspaceMetrics`.
pub(crate) fn emit_workspace_core_write(d: Duration) {
    if let Some(h) = WORKSPACE_CORE_WRITE.get() {
        h(d);
    }
}

/// Invoke the head-index write hook, if installed.
pub(crate) fn emit_head_index_write(d: Duration) {
    if let Some(h) = HEAD_INDEX_WRITE.get() {
        h(d);
    }
}

/// Invoke the upload hook, if installed.  `bytes` is the
/// observed body size on the Ok arm of
/// `WorkspaceMgr::upload_dataset_file`.
pub(crate) fn emit_upload(bytes: u64) {
    if let Some(h) = UPLOAD.get() {
        h(bytes);
    }
}

/// Invoke the dataset-mutation-rejected hook, if installed.
pub(crate) fn emit_dataset_mutation_rejected() {
    if let Some(h) = DATASET_MUTATION_REJECTED.get() {
        h();
    }
}

/// Invoke the converter-mutation-rejected hook, if installed.
pub(crate) fn emit_converter_mutation_rejected() {
    if let Some(h) = CONVERTER_MUTATION_REJECTED.get() {
        h();
    }
}

/// Invoke the job-events-dropped hook with `n` as the number
/// of dropped events, if installed.
pub(crate) fn emit_job_events_dropped(n: u64) {
    if let Some(h) = JOB_EVENTS_DROPPED.get() {
        h(n);
    }
}
