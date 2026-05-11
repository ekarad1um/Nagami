//! In-process job registry: unified admission gate +
//! recent-history + per-job event ring + SSE broadcast channel.
//!
//! 1. enforces per-`JobType` concurrency caps
//!    (`max_train_jobs = max_convert_jobs = max_delete_jobs = 1`
//!    by default),
//! 2. enforces `WorkspaceDelete` exclusion on [`JobReference`]s
//!    (the only exclusive admission shape; everything else
//!    overlaps freely under its per-`JobType` cap),
//! 3. retains a bounded history of recent terminal jobs
//!    (memory-only, capped by `max_recent_jobs`),
//! 4. owns a per-job ring buffer (`max_job_event_ring = 1024`)
//!    the SSE stream replays from before following live events,
//! 5. exposes a `tokio::sync::broadcast` channel so SSE
//!    subscribers fan out without blocking workers, and
//! 6. line-caps log messages to `max_log_line_bytes = 8 KiB` so
//!    a noisy trainer cannot turn diagnostics into a memory
//!    benchmark.
//!
//! # Conflict detection
//!
//! Only [`JobReference::Workspace`] exists, so conflict reduces to:
//!
//! - A new `WorkspaceDelete` conflicts with any active job or
//!   bare lease in the same workspace.
//! - A new non-`WorkspaceDelete` conflicts only with an active
//!   `WorkspaceDelete` in the same workspace.
//! - Bare leases (uploads, head-delete) coexist with one another
//!   and with train/convert/file-delete; they block only an
//!   incoming `WorkspaceDelete`.
//!
//! # Lifecycle
//!
//! ```text
//! try_acquire(job_type, refs) -> JobHandle
//!     |
//!     | (worker holds the handle; drops it on completion)
//!     v
//! handle.update_progress(...)        -- rate-limited to 4 Hz per job
//! handle.append_log(...)             -- line-capped to 8 KiB
//! handle.terminate(JobState::...)    -- emits Terminal event, flips state
//!     |
//!     v
//! handle: Drop                       -- releases references; if no
//!                                       explicit terminate, recorded as
//!                                       Failed (worker abandoned the job)
//! ```
//!
//! Terminal entries are retained in memory until the recent-job
//! window slides them out (newest-first, capped by
//! `max_recent_jobs`).
//!
//! # Thread-safety
//!
//! All public methods take `&self`; the inner state is wrapped
//! in a `parking_lot::Mutex` (`!Send`-safe; never held across
//! `.await`).  The broadcast channel is itself `Send + Sync`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::common::asset_path::AssetPath;
use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType};
use crate::file_mgr::error::FileError;
use crate::file_mgr::recovery::RecoveryReport;
use crate::file_mgr::time_util::now_rfc3339;

use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::broadcast;

// MARK: Configuration

/// Tuning knobs for the [`JobRegistry`].
#[derive(Debug, Clone, Copy)]
pub struct JobRegistryCfg {
    /// `max_train_jobs`: at most one unfinished train job
    /// daemon-wide.  Producers acquire a [`JobHandle`] of
    /// [`JobType::Train`] up-front; a second concurrent
    /// request fails with [`RegistryConflict::AnotherTrainRunning`].
    pub max_train_jobs: usize,
    /// `max_convert_jobs`: bounded concurrent convert jobs.
    pub max_convert_jobs: usize,
    /// `max_delete_jobs`: bounded concurrent delete jobs (one
    /// shared slot across all five delete subtypes -- dataset,
    /// converter, training-logs, converter-logs, workspace).
    pub max_delete_jobs: usize,
    /// `max_recent_jobs = max_running_jobs + 1`.  Bounds the
    /// memory-only `GET /jobs` history.
    pub max_recent_jobs: usize,
    /// `max_job_event_ring`: bounded per-job ring buffer.
    /// Overflow drops the oldest events and increments the
    /// `events_dropped_total` counter.
    pub max_job_event_ring: usize,
    /// `max_log_line_bytes`: hard cap on a single log line's
    /// UTF-8 byte length.  Lines longer than this are truncated
    /// with a `... [truncated]` suffix.
    pub max_log_line_bytes: usize,
    /// Progress event rate limit (per job, in Hz).  Updates
    /// faster than this are coalesced (only the latest per
    /// throttle window is emitted).  Terminal-state events are
    /// never throttled.
    pub progress_throttle_hz: f32,
}

impl Default for JobRegistryCfg {
    fn default() -> Self {
        Self {
            max_train_jobs: 1,
            max_convert_jobs: 1,
            max_delete_jobs: 1,
            // max_running_jobs (1+1+1) + 1 spare for the
            // most-recent terminal entry.
            max_recent_jobs: 4,
            max_job_event_ring: 1024,
            max_log_line_bytes: 8 * 1024,
            progress_throttle_hz: 4.0,
        }
    }
}

// MARK: Public types

/// Terminal job result surfaced through [`JobSnapshot::result`].
/// Lives in `file_mgr` rather than `common::workspace` because
/// `common` is intentionally I/O-free.
///
/// `#[non_exhaustive]`: the wire `kind` discriminator is open by
/// design (operator tooling pattern-matches on it) and future
/// producers will add variants -- external matches must handle
/// the unknown case.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobResult {
    WorkspaceDelete {
        /// Whether the deleted workspace was the source of the
        /// current active generation.  (False today; the registry
        /// does not yet tee into the active manifest provenance.)
        active_source_deleted: bool,
    },
    /// Convert producer published a head; the typed result lets
    /// `GET /jobs/{job_id}` surface the head id directly so a
    /// follow-up `POST /active` can chain without reading the
    /// JSONL log.
    Convert {
        head_id: HeadId,
        /// Lowercase-hex SHA-256 of the published `head.mpk`.
        sha256: String,
        n_classes: u32,
    },
    /// Train producer published a head.  Structurally identical
    /// to `Convert`; the discriminator lets operator tooling
    /// pattern-match on producer type.
    Train {
        head_id: HeadId,
        sha256: String,
        n_classes: u32,
    },
}

/// Registry-side job lifecycle states surfaced through
/// [`JobSnapshot`].  Distinct from the per-domain state enums
/// (`training::JobState`, ...) because those carry domain
/// payloads (`TrainingResult`, ...) that don't belong in a
/// memory-only cross-cutting registry.
///
/// `#[non_exhaustive]`: `Queued` is documented as "reserved for
/// future scheduling work"; the variant set is expected to grow
/// (`Suspended`, etc.).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobState {
    /// Admission gate cleared, worker has not yet started.
    /// Producers immediately transition to [`Self::Running`]
    /// when they begin work; the queued state is reserved for
    /// future scheduling work.
    Queued,
    /// Worker is executing; references held.
    Running,
    /// Worker reported success.
    Succeeded,
    /// Worker reported failure (or was dropped without an
    /// explicit terminate, which is treated as failure).
    Failed,
    /// Operator cancelled the job before it reached a terminal
    /// state.
    Cancelled,
}

impl JobState {
    /// Whether this state means the job is still in flight.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

/// Unitless progress payload.  `total = None` renders as a
/// running counter (e.g. "423 events"); `Some(_)` renders as a
/// percent.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct JobProgress {
    /// Items completed.
    pub done: u64,
    /// Total items, when known.
    pub total: Option<u64>,
}

/// Memory-only job snapshot returned by `GET /jobs` /
/// `GET /jobs/{job_id}`.  Built from one registry entry's
/// current state under the registry mutex; no log file I/O.
#[derive(Clone, Debug, Serialize)]
pub struct JobSnapshot {
    /// Job identifier.
    pub job_id: JobId,
    /// Job type discriminator.
    pub job_type: JobType,
    /// Display workspace (first reference's workspace).
    pub workspace_id: Option<WorkspaceId>,
    /// Display primary file / delete target.  Operator-facing
    /// only; not used for conflict detection.  Populated by the
    /// producer at admission via [`JobRegistry::try_acquire`].
    /// Currently meaningful only for `dataset_delete` and
    /// `converter_delete`; train, convert, and workspace-delete
    /// jobs have no single primary path so the field is absent
    /// from the wire (skipped on None) rather than serialised as
    /// `null`.  Clients must treat absent and `null` as
    /// equivalent — both mean "no target path".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<AssetPath>,
    /// Lifecycle state.
    pub state: JobState,
    /// Latest progress snapshot.
    pub progress: Option<JobProgress>,
    /// Terminal result payload (only populated for jobs that
    /// produce one, e.g. `WorkspaceDelete`).
    pub result: Option<JobResult>,
    /// Last event sequence emitted for this job.  Clients pass
    /// this back via `?after_seq=` to resume an SSE stream
    /// without missing intermediate events.
    pub last_seq: u64,
    /// RFC3339 wall-clock at last state change.
    pub updated_at: String,
}

/// Per-job event surfaced over SSE and persisted (for
/// train/convert) as JSONL.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct JobEvent {
    /// Monotonic per-job sequence.  Strictly increasing;
    /// SSE clients use this as their reconnect cursor.
    pub seq: u64,
    /// RFC3339 wall-clock at event emission.
    pub at: String,
    /// State transition this event records, if any.  `None`
    /// for a pure-progress / pure-log line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<JobState>,
    /// Progress payload, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<JobProgress>,
    /// Free-form log message, if any.  Already line-capped to
    /// `max_log_line_bytes` at emission time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// 409 `EventGap` body returned when a `?after_seq=N` is older
/// than the registry's in-memory ring for a job.  Clients page
/// the durable JSONL log via the matching
/// `/{training,converter}_logs/{job_id}` endpoint to backfill,
/// then reconnect to the SSE stream with a fresher `after_seq`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EventGap {
    /// Oldest event seq still present in the ring.
    pub oldest_seq: u64,
    /// Most recent event seq emitted.
    pub latest_seq: u64,
}

/// Counter snapshot for `GET /api/v1/status`.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct JobRegistryCounters {
    /// Cumulative count of admission attempts that returned
    /// [`RegistryConflict`].
    pub admission_conflicts_total: u64,
    /// Cumulative count of events dropped from the in-memory
    /// ring (capacity exceeded).
    pub events_dropped_total: u64,
    /// Cumulative count of progress events suppressed by the
    /// rate limiter.
    pub progress_throttled_total: u64,
    /// Cumulative count of log lines truncated to
    /// `max_log_line_bytes`.
    pub log_lines_truncated_total: u64,
    /// Number of jobs currently in [`JobState::Running`].
    pub jobs_running: u64,
    /// Number of jobs retained in the recent-history window.
    pub jobs_retained: u64,
}

/// Why a [`JobRegistry::try_acquire`] failed.  Maps to HTTP
/// 409 via the `FileError` carrier; see [`From<RegistryConflict>
/// for FileError`].
#[derive(Debug, Clone)]
pub enum RegistryConflict {
    /// Train concurrency cap (`max_train_jobs = 1`) reached.
    /// Distinct from [`Self::JobConflict`] so the api layer
    /// renders the dedicated `another_train_running`
    /// discriminator code.
    AnotherTrainRunning,
    /// Generic conflict: the new job's references overlap an
    /// existing in-flight job, or the per-`JobType` concurrency
    /// cap is reached for convert / delete.
    JobConflict {
        /// The existing job whose reference / cap conflicts.
        job_id: JobId,
        /// The existing job's type (helps operators
        /// disambiguate "race with my own retry" from "race
        /// with a sibling subsystem").
        job_type: JobType,
    },
}

impl From<RegistryConflict> for FileError {
    fn from(c: RegistryConflict) -> Self {
        match c {
            RegistryConflict::AnotherTrainRunning => FileError::AnotherTrainRunning,
            RegistryConflict::JobConflict { job_id, job_type } => FileError::JobConflict {
                message: format!("conflicts with running job {job_id} ({job_type:?})"),
            },
        }
    }
}

// MARK: Internal entry shape

/// One entry in the registry's memory store.  Holds the
/// per-job ring + the references being released on terminate.
#[derive(Debug)]
struct JobEntry {
    job_id: JobId,
    job_type: JobType,
    references: Vec<JobReference>,
    /// Operator-facing display target (e.g. the dataset path
    /// being deleted).  Stored separately from `references`
    /// because references are workspace-only and carry no path;
    /// producers pass this in at admission for snapshot
    /// rendering only.
    target_path: Option<AssetPath>,
    state: JobState,
    progress: Option<JobProgress>,
    result: Option<JobResult>,
    last_seq: u64,
    updated_at: String,
    /// Bounded ring of events for SSE replay.  Capacity is
    /// `cfg.max_job_event_ring`.
    ring: VecDeque<JobEvent>,
    /// Progress rate-limit timestamp.  `None` until the first
    /// progress event lands; subsequent events compare against
    /// this and the configured throttle interval.
    last_progress_at: Option<Instant>,
}

impl JobEntry {
    fn snapshot(&self) -> JobSnapshot {
        // Display workspace = first reference's workspace; target
        // path is stored explicitly when the producer admitted it.
        let workspace_id = self.references.first().map(|r| r.workspace_id());
        JobSnapshot {
            job_id: self.job_id,
            job_type: self.job_type,
            workspace_id,
            target_path: self.target_path.clone(),
            state: self.state,
            progress: self.progress,
            result: self.result.clone(),
            last_seq: self.last_seq,
            updated_at: self.updated_at.clone(),
        }
    }

    fn push_event(&mut self, event: JobEvent, max_ring: usize, events_dropped: &AtomicU64) {
        if self.ring.len() >= max_ring {
            self.ring.pop_front();
            events_dropped.fetch_add(1, Ordering::Relaxed);
        }
        self.ring.push_back(event);
    }
}

// MARK: Registry

/// In-process job registry.  Held by `AppState` as
/// `Arc<JobRegistry>`.  See module-level docs for the lifecycle
/// contract.
#[derive(Debug)]
pub struct JobRegistry {
    inner: Arc<RegistryInner>,
}

#[derive(Debug)]
struct RegistryInner {
    /// `(retained jobs)`, including active + recent terminal,
    /// keyed by [`JobId`].
    jobs: Mutex<JobsState>,
    /// Live event broadcast.  Each subscriber receives every
    /// event after their subscribe point; replay of older
    /// events comes from the per-job ring.
    broadcast: broadcast::Sender<RegistryEvent>,
    counters: Counters,
    /// Boot-recovery report stash; set once at daemon boot.  `Arc`
    /// because [`RecoveryReport`] is not `Clone` (transitive
    /// `ActiveHeadManifest` clones are too expensive per status
    /// poll); status reads are `Arc` bumps only.
    boot_recovery: Mutex<Option<Arc<RecoveryReport>>>,
    cfg: JobRegistryCfg,
}

#[derive(Debug)]
struct JobsState {
    /// Newest-first list of retained job ids; the
    /// `entries` map is the source of truth, this vec is the
    /// ordering for `recent()`.
    order: VecDeque<JobId>,
    /// Per-job entries.
    entries: std::collections::HashMap<JobId, JobEntry>,
    /// Bare reference leases (upload short-lived gates).
    /// Keyed by a monotonic counter so [`LeaseGuard`] can
    /// release its slot on drop without a content-based search.
    leases: std::collections::HashMap<u64, Vec<JobReference>>,
    next_lease_id: u64,
}

impl JobsState {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            entries: std::collections::HashMap::new(),
            leases: std::collections::HashMap::new(),
            next_lease_id: 1,
        }
    }
}

#[derive(Debug, Default)]
struct Counters {
    admission_conflicts_total: AtomicU64,
    events_dropped_total: AtomicU64,
    progress_throttled_total: AtomicU64,
    log_lines_truncated_total: AtomicU64,
}

/// Live event published on the broadcast channel.  Subscribers
/// filter by `job_id`.
#[derive(Clone, Debug)]
pub struct RegistryEvent {
    /// The job this event is scoped to.
    pub job_id: JobId,
    /// The event payload.
    pub event: JobEvent,
}

/// Broadcast capacity (per-channel slot count).  A slow
/// subscriber gets `RecvError::Lagged` and clients reconnect
/// with the matching `after_seq` from their last recorded
/// event.  The number is generous because the registry-side
/// ring is the durable replay surface; the broadcast is just
/// the wakeup mechanism.
const BROADCAST_CAPACITY: usize = 256;

/// Send `event` on the registry's broadcast channel.  When
/// `send` returns `Err(SendError(_))` (every receiver dropped or
/// lagged out), increments the `record_job_events_dropped(1)`
/// metrics hook so operators can distinguish dropped-because-
/// no-listener from successful broadcasts and from in-ring
/// overflow (which the registry's own `events_dropped_total`
/// already counts).
fn send_or_count_drop(
    broadcast: &broadcast::Sender<RegistryEvent>,
    job_id: JobId,
    event: JobEvent,
) {
    // No-subscriber sends are the dominant case (jobs run
    // without an active SSE listener for most of their
    // lifetime); skip the broadcast call entirely so
    // `job_events_dropped_total` reflects ONLY genuine
    // overflow from a lagged subscriber, not the steady-state
    // "no listener" baseline that would dwarf real lag drops
    // on operator dashboards.
    if broadcast.receiver_count() == 0 {
        return;
    }
    if broadcast.send(RegistryEvent { job_id, event }).is_err() {
        crate::file_mgr::metrics_hooks::emit_job_events_dropped(1);
    }
}

impl JobRegistry {
    /// Construct an empty registry with the given configuration.
    pub fn new(cfg: JobRegistryCfg) -> Self {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            inner: Arc::new(RegistryInner {
                jobs: Mutex::new(JobsState::new()),
                broadcast: tx,
                counters: Counters::default(),
                boot_recovery: Mutex::new(None),
                cfg,
            }),
        }
    }

    /// Configuration in force.
    pub fn cfg(&self) -> &JobRegistryCfg {
        &self.inner.cfg
    }

    /// Atomic admission gate.  Allocates a fresh [`JobId`],
    /// validates the per-`JobType` concurrency cap, validates
    /// reference overlap against every active job, then
    /// registers the new entry in the memory store.  The
    /// returned [`JobHandle`] guards the entry; drop releases
    /// references and (if not explicitly terminated) records
    /// a [`JobState::Failed`] terminal event.
    pub fn try_acquire(
        self: &Arc<Self>,
        job_type: JobType,
        references: Vec<JobReference>,
        target_path: Option<AssetPath>,
    ) -> Result<JobHandle, RegistryConflict> {
        let mut state = self.inner.jobs.lock();

        // Per-type concurrency cap.  Counts only RUNNING /
        // QUEUED entries; terminal entries do not occupy a
        // slot.  The async-delete family (Dataset / Converter /
        // TrainingLogs / ConverterLogs / Workspace) shares one
        // `max_delete_jobs` slot, so the count must include any
        // in-flight delete subtype.
        let (cap, slot_predicate): (usize, fn(JobType) -> bool) = match job_type {
            JobType::Train => (self.inner.cfg.max_train_jobs, |t| t == JobType::Train),
            JobType::Convert => (self.inner.cfg.max_convert_jobs, |t| t == JobType::Convert),
            JobType::DatasetDelete
            | JobType::ConverterDelete
            | JobType::TrainingLogsDelete
            | JobType::ConverterLogsDelete
            | JobType::WorkspaceDelete => (self.inner.cfg.max_delete_jobs, |t| {
                matches!(
                    t,
                    JobType::DatasetDelete
                        | JobType::ConverterDelete
                        | JobType::TrainingLogsDelete
                        | JobType::ConverterLogsDelete
                        | JobType::WorkspaceDelete
                )
            }),
        };
        let active_same_type = state
            .entries
            .values()
            .filter(|e| e.state.is_active() && slot_predicate(e.job_type))
            .count();
        if active_same_type >= cap {
            self.inner
                .counters
                .admission_conflicts_total
                .fetch_add(1, Ordering::Relaxed);
            // Pick the first conflicting entry to surface in the
            // error (operators want to see a job_id they can
            // inspect).
            let conflicting = state
                .entries
                .values()
                .find(|e| e.state.is_active() && slot_predicate(e.job_type))
                .map(|e| (e.job_id, e.job_type));
            return Err(match (job_type, conflicting) {
                (JobType::Train, _) => RegistryConflict::AnotherTrainRunning,
                (_, Some((job_id, jt))) => RegistryConflict::JobConflict {
                    job_id,
                    job_type: jt,
                },
                (_, None) => RegistryConflict::JobConflict {
                    job_id: JobId::new(),
                    job_type,
                },
            });
        }

        // `WorkspaceDelete` is the only exclusive admission shape;
        // everything else overlaps freely under its per-`JobType`
        // cap.  Conflict iff the new ref's workspace matches an
        // existing ref's workspace AND at least one side is a
        // `WorkspaceDelete`.  `WorkspaceDelete` additionally
        // blocks active bare leases (uploads, head-delete).
        let new_is_workspace_delete = job_type == JobType::WorkspaceDelete;
        for new_ref in &references {
            let new_ws = new_ref.workspace_id();
            for existing in state.entries.values() {
                if !existing.state.is_active() {
                    continue;
                }
                if job_conflicts_with_existing_job(
                    new_is_workspace_delete,
                    new_ws,
                    existing.job_type,
                    &existing.references,
                ) {
                    self.inner
                        .counters
                        .admission_conflicts_total
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(RegistryConflict::JobConflict {
                        job_id: existing.job_id,
                        job_type: existing.job_type,
                    });
                }
            }
            if new_is_workspace_delete {
                for lease_refs in state.leases.values() {
                    if job_conflicts_with_existing_lease(
                        new_is_workspace_delete,
                        new_ws,
                        lease_refs,
                    ) {
                        self.inner
                            .counters
                            .admission_conflicts_total
                            .fetch_add(1, Ordering::Relaxed);
                        return Err(RegistryConflict::JobConflict {
                            // Bare leases have no real job id;
                            // surface a synthetic id so the wire
                            // shape stays uniform.  Operators
                            // looking via `GET /jobs` will see no
                            // matching entry, signalling "an
                            // upload was in flight".
                            job_id: JobId::new(),
                            job_type: JobType::WorkspaceDelete,
                        });
                    }
                }
            }
        }

        // Admission cleared; insert the entry.
        let job_id = JobId::new();
        let entry = JobEntry {
            job_id,
            job_type,
            references: references.clone(),
            target_path,
            state: JobState::Running,
            progress: None,
            result: None,
            last_seq: 0,
            updated_at: now_rfc3339(),
            ring: VecDeque::with_capacity(self.inner.cfg.max_job_event_ring.min(64)),
            last_progress_at: None,
        };
        state.order.push_front(job_id);
        state.entries.insert(job_id, entry);
        // Slide the recent window if we exceeded the cap.  We
        // only evict TERMINAL entries (active jobs are never
        // displaced).
        prune_recent(&mut state, self.inner.cfg.max_recent_jobs);

        Ok(JobHandle {
            registry: Arc::clone(self),
            job_id,
            terminated: false,
        })
    }

    /// Update the progress payload for `job_id`.  Rate-limited
    /// to `cfg.progress_throttle_hz`.  Updates faster than the
    /// throttle window are coalesced (the latest payload wins;
    /// no event is emitted until the window elapses).
    fn update_progress(&self, job_id: JobId, progress: JobProgress) {
        let now = Instant::now();
        let throttle =
            Duration::from_secs_f32(1.0 / self.inner.cfg.progress_throttle_hz.max(0.001));
        let mut state = self.inner.jobs.lock();
        let entry = match state.entries.get_mut(&job_id) {
            Some(e) => e,
            None => return,
        };
        // Latest payload always wins, even when throttled.
        entry.progress = Some(progress);
        if let Some(prev) = entry.last_progress_at
            && now.duration_since(prev) < throttle
        {
            self.inner
                .counters
                .progress_throttled_total
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        entry.last_progress_at = Some(now);
        entry.last_seq = entry.last_seq.saturating_add(1);
        entry.updated_at = now_rfc3339();
        let event = JobEvent {
            seq: entry.last_seq,
            at: entry.updated_at.clone(),
            state: None,
            progress: Some(progress),
            message: None,
        };
        entry.push_event(
            event.clone(),
            self.inner.cfg.max_job_event_ring,
            &self.inner.counters.events_dropped_total,
        );
        // Best-effort broadcast; subscribers are SSE clients
        // and may have lagged.  No subscriber yields a
        // `SendError`, which the helper counts on
        // `job_events_dropped_total`.
        send_or_count_drop(&self.inner.broadcast, job_id, event);
    }

    /// Append a free-form log line.  Truncates lines longer
    /// than `cfg.max_log_line_bytes` to fit, marking the
    /// truncation with a `... [truncated]` suffix.
    fn append_log(&self, job_id: JobId, message: String) {
        let mut msg = message;
        let max = self.inner.cfg.max_log_line_bytes;
        if msg.len() > max {
            self.inner
                .counters
                .log_lines_truncated_total
                .fetch_add(1, Ordering::Relaxed);
            // Truncate at a UTF-8 char boundary: take the
            // largest prefix that fits inside the cap minus the
            // suffix marker length.
            let suffix = " ... [truncated]";
            let target = max.saturating_sub(suffix.len()).max(1);
            let mut cut = target;
            while cut > 0 && !msg.is_char_boundary(cut) {
                cut -= 1;
            }
            msg.truncate(cut);
            msg.push_str(suffix);
        }
        let mut state = self.inner.jobs.lock();
        let entry = match state.entries.get_mut(&job_id) {
            Some(e) => e,
            None => return,
        };
        entry.last_seq = entry.last_seq.saturating_add(1);
        entry.updated_at = now_rfc3339();
        let event = JobEvent {
            seq: entry.last_seq,
            at: entry.updated_at.clone(),
            state: None,
            progress: None,
            message: Some(msg),
        };
        entry.push_event(
            event.clone(),
            self.inner.cfg.max_job_event_ring,
            &self.inner.counters.events_dropped_total,
        );
        send_or_count_drop(&self.inner.broadcast, job_id, event);
    }

    /// Terminate the job with the given final state.  Idempotent
    /// (subsequent calls are no-ops).  Releases the entry's
    /// references and emits the terminal event.
    fn terminate(&self, job_id: JobId, state: JobState, result: Option<JobResult>) {
        debug_assert!(
            !state.is_active(),
            "terminate called with non-terminal state"
        );
        let mut s = self.inner.jobs.lock();
        let entry = match s.entries.get_mut(&job_id) {
            Some(e) => e,
            None => return,
        };
        if !entry.state.is_active() {
            return; // already terminated
        }
        entry.state = state;
        if result.is_some() {
            entry.result = result;
        }
        entry.last_seq = entry.last_seq.saturating_add(1);
        entry.updated_at = now_rfc3339();
        // Release references by clearing the vec; the entry
        // remains in the recent history but no longer counts as
        // an active reference holder for conflict detection.
        entry.references.clear();
        let event = JobEvent {
            seq: entry.last_seq,
            at: entry.updated_at.clone(),
            state: Some(state),
            progress: entry.progress,
            message: None,
        };
        entry.push_event(
            event.clone(),
            self.inner.cfg.max_job_event_ring,
            &self.inner.counters.events_dropped_total,
        );
        prune_recent(&mut s, self.inner.cfg.max_recent_jobs);
        send_or_count_drop(&self.inner.broadcast, job_id, event);
    }

    /// Memory-only snapshot for `GET /jobs/{job_id}`.
    pub fn snapshot(&self, job_id: JobId) -> Option<JobSnapshot> {
        self.inner
            .jobs
            .lock()
            .entries
            .get(&job_id)
            .map(JobEntry::snapshot)
    }

    /// Most-recent-first list of retained job snapshots, capped
    /// at `min(limit, cfg.max_recent_jobs)`.
    pub fn recent(&self, limit: usize) -> Vec<JobSnapshot> {
        let state = self.inner.jobs.lock();
        let cap = limit.min(self.inner.cfg.max_recent_jobs);
        state
            .order
            .iter()
            .take(cap)
            .filter_map(|id| state.entries.get(id).map(JobEntry::snapshot))
            .collect()
    }

    /// Whether `workspace_id` has a [`JobType::Train`] job
    /// currently in [`JobState::Running`].  Used by the
    /// `DELETE /workspace/{id}/assets/training_logs[/<id>.jsonl]`
    /// dispatch (in
    /// [`crate::file_mgr::WorkspaceMgr::start_workspace_asset_delete`])
    /// to refuse with 409 while a producer is active and would
    /// otherwise race the wipe.
    pub fn has_active_train_for(&self, workspace_id: WorkspaceId) -> bool {
        self.has_active_for(workspace_id, JobType::Train)
    }

    /// Whether `workspace_id` has a [`JobType::Convert`] job
    /// currently in [`JobState::Running`].  Same role as
    /// [`Self::has_active_train_for`] for the converter side.
    pub fn has_active_convert_for(&self, workspace_id: WorkspaceId) -> bool {
        self.has_active_for(workspace_id, JobType::Convert)
    }

    fn has_active_for(&self, workspace_id: WorkspaceId, job_type: JobType) -> bool {
        let state = self.inner.jobs.lock();
        state.entries.values().any(|e| {
            e.state.is_active()
                && e.job_type == job_type
                && e.references
                    .iter()
                    .any(|r| r.workspace_id() == workspace_id)
        })
    }

    /// Acquire a bare reference lease against the registry's
    /// active references.  Used by short-lived operations (e.g.
    /// dataset upload) that need conflict gating but no job
    /// snapshot / event ring.  The returned [`LeaseGuard`]
    /// releases on drop; it does NOT count against any per-
    /// `JobType` concurrency cap and is not visible via
    /// [`Self::recent`] / [`Self::snapshot`].
    pub fn try_acquire_lease(
        self: &Arc<Self>,
        references: Vec<JobReference>,
    ) -> Result<LeaseGuard, RegistryConflict> {
        let mut state = self.inner.jobs.lock();
        // Bare leases (uploads, head-delete) are gated only by
        // `WorkspaceDelete` jobs in the same workspace.  They do
        // NOT conflict with train/convert/dataset-delete/
        // converter-delete or with other bare leases -- multiple
        // uploads to the same workspace coexist.
        for new_ref in &references {
            let new_ws = new_ref.workspace_id();
            for existing in state.entries.values() {
                if !existing.state.is_active() {
                    continue;
                }
                if lease_conflicts_with_existing_job(
                    new_ws,
                    existing.job_type,
                    &existing.references,
                ) {
                    self.inner
                        .counters
                        .admission_conflicts_total
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(RegistryConflict::JobConflict {
                        job_id: existing.job_id,
                        job_type: existing.job_type,
                    });
                }
            }
        }
        let lease_id = state.next_lease_id;
        state.next_lease_id = state.next_lease_id.wrapping_add(1);
        state.leases.insert(lease_id, references);
        Ok(LeaseGuard {
            registry: Arc::clone(self),
            lease_id,
            released: false,
        })
    }

    fn release_lease(&self, lease_id: u64) {
        let mut state = self.inner.jobs.lock();
        state.leases.remove(&lease_id);
    }

    /// Subscribe to the live event stream for `job_id`,
    /// optionally replaying ring events strictly newer than
    /// `after_seq`.  Returns an [`EventStream`] that yields
    /// replay events (oldest-first) followed by live events.
    ///
    /// If `after_seq` is older than the ring's oldest seq, the
    /// caller receives an [`EventGap`] error so it can backfill
    /// from the durable JSONL log before reconnecting with a
    /// fresher cursor.
    pub fn subscribe_events(&self, job_id: JobId, after_seq: u64) -> Result<EventStream, EventGap> {
        let state = self.inner.jobs.lock();
        let entry = match state.entries.get(&job_id) {
            Some(e) => e,
            None => {
                // Unknown job id: treat as no replay + live
                // subscription.  The api layer surfaces 404 by
                // checking `snapshot()` separately.
                return Ok(EventStream {
                    replay: std::collections::VecDeque::new(),
                    live: self.inner.broadcast.subscribe(),
                    job_id,
                    terminal_seen: false,
                });
            }
        };
        // Empty ring + after_seq=0 is "fresh subscribe": no gap.
        let oldest = entry.ring.front().map(|e| e.seq).unwrap_or(0);
        let latest = entry.last_seq;
        // Two gap shapes:
        //  - `after_seq < oldest_seq - 1` -- ring has dropped
        //    events the cursor referenced.
        //  - `after_seq > latest_seq` -- cursor came from a
        //    different stream (or a previous registry instance);
        //    the events the cursor references don't exist here.
        // Both surface as 409 with the same body so clients can
        // unambiguously backfill via the JSONL log endpoint.
        if after_seq > 0 && (after_seq < oldest.saturating_sub(1) || after_seq > latest) {
            return Err(EventGap {
                oldest_seq: oldest,
                latest_seq: latest,
            });
        }
        let replay: std::collections::VecDeque<JobEvent> = entry
            .ring
            .iter()
            .filter(|e| e.seq > after_seq)
            .cloned()
            .collect();
        let terminal_seen = !entry.state.is_active();
        Ok(EventStream {
            replay,
            live: self.inner.broadcast.subscribe(),
            job_id,
            terminal_seen,
        })
    }

    /// Counter snapshot for `GET /api/v1/status`.
    pub fn counters(&self) -> JobRegistryCounters {
        let state = self.inner.jobs.lock();
        let jobs_running = state
            .entries
            .values()
            .filter(|e| e.state.is_active())
            .count() as u64;
        let jobs_retained = state.entries.len() as u64;
        let c = &self.inner.counters;
        JobRegistryCounters {
            admission_conflicts_total: c.admission_conflicts_total.load(Ordering::Relaxed),
            events_dropped_total: c.events_dropped_total.load(Ordering::Relaxed),
            progress_throttled_total: c.progress_throttled_total.load(Ordering::Relaxed),
            log_lines_truncated_total: c.log_lines_truncated_total.load(Ordering::Relaxed),
            jobs_running,
            jobs_retained,
        }
    }

    /// Stash the boot-recovery report so the status surface can
    /// publish it; no-op if called more than once (boot is
    /// single-shot).
    pub fn record_boot_recovery(&self, report: RecoveryReport) {
        let mut slot = self.inner.boot_recovery.lock();
        if slot.is_none() {
            *slot = Some(Arc::new(report));
        }
    }

    /// Latest stashed boot-recovery report, if any.  Returns the
    /// `Arc` (cheap bump) so the status emitter can publish
    /// without holding the registry mutex and without deep-cloning
    /// the report.
    pub fn boot_recovery(&self) -> Option<Arc<RecoveryReport>> {
        self.inner.boot_recovery.lock().clone()
    }
}

// MARK: JobHandle

/// RAII guard for one admitted job.  Workers call
/// [`Self::update_progress`] / [`Self::append_log`] /
/// [`Self::succeed`] / [`Self::fail`] / [`Self::cancel`]
/// through this handle; `Drop` releases the references and
/// (if the worker abandoned the job without an explicit
/// terminate) records [`JobState::Failed`].
#[derive(Debug)]
pub struct JobHandle {
    registry: Arc<JobRegistry>,
    job_id: JobId,
    terminated: bool,
}

impl JobHandle {
    /// The admitted job's id.  Producers return this to the
    /// caller so the api response can carry it.
    pub fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Update the latest progress snapshot.  Rate-limited to
    /// the registry's configured throttle.
    pub fn update_progress(&self, progress: JobProgress) {
        self.registry.update_progress(self.job_id, progress);
    }

    /// Append a free-form log line.  Truncates to
    /// `max_log_line_bytes` if necessary.
    pub fn append_log<S: Into<String>>(&self, message: S) {
        self.registry.append_log(self.job_id, message.into());
    }

    /// Mark the job as `Succeeded` and release references.
    pub fn succeed(mut self, result: Option<JobResult>) {
        self.registry
            .terminate(self.job_id, JobState::Succeeded, result);
        self.terminated = true;
    }

    /// Mark the job as `Failed` and release references.
    pub fn fail<S: Into<String>>(mut self, reason: S) {
        // Emit a final log line so operators see the failure
        // reason without having to open the JSONL.
        self.registry.append_log(self.job_id, reason.into());
        self.registry.terminate(self.job_id, JobState::Failed, None);
        self.terminated = true;
    }

    /// Mark the job as `Cancelled` and release references.
    pub fn cancel(mut self) {
        self.registry
            .terminate(self.job_id, JobState::Cancelled, None);
        self.terminated = true;
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        if !self.terminated {
            // Abandon path: worker dropped without calling
            // succeed/fail/cancel.  Record as Failed so the
            // recent-history entry reflects the abnormal exit.
            self.registry.terminate(self.job_id, JobState::Failed, None);
        }
    }
}

// MARK: LeaseGuard

/// Short-lived reference-only lease guard.  Returned by
/// [`JobRegistry::try_acquire_lease`] for operations that need
/// conflict gating but don't surface as full jobs (today:
/// dataset upload).  Drop releases the lease.
#[derive(Debug)]
pub struct LeaseGuard {
    registry: Arc<JobRegistry>,
    lease_id: u64,
    released: bool,
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if !self.released {
            self.registry.release_lease(self.lease_id);
            self.released = true;
        }
    }
}

// MARK: EventStream

/// Stream of replay-then-live job events for one subscription.
/// Yields ring events (oldest-first) followed by live broadcast
/// events filtered to the subscribed job.  Lag from a slow
/// consumer surfaces as [`EventStreamError::Lagged`]; the
/// client reconnects with the matching `after_seq` from its
/// last recorded event.
#[derive(Debug)]
pub struct EventStream {
    replay: std::collections::VecDeque<JobEvent>,
    live: broadcast::Receiver<RegistryEvent>,
    job_id: JobId,
    terminal_seen: bool,
}

impl EventStream {
    /// Pop the next replay event, if any, before returning to
    /// the live channel.  O(1) via `VecDeque::pop_front`; the
    /// api layer interleaves replay drain + `recv()` on the
    /// live channel.
    pub fn next_replay(&mut self) -> Option<JobEvent> {
        self.replay.pop_front()
    }

    /// Whether all replay events have been delivered.
    pub fn replay_drained(&self) -> bool {
        self.replay.is_empty()
    }

    /// Whether a terminal event has already been observed.
    /// SSE streams close when this flips true after a recv.
    pub fn terminal_seen(&self) -> bool {
        self.terminal_seen
    }

    /// Receive the next live event scoped to the subscribed
    /// job.  Skips events for other jobs.
    pub async fn recv(&mut self) -> Result<JobEvent, EventStreamError> {
        loop {
            match self.live.recv().await {
                Ok(re) => {
                    if re.job_id != self.job_id {
                        continue;
                    }
                    if re.event.state.is_some_and(|s| !s.is_active()) {
                        self.terminal_seen = true;
                    }
                    return Ok(re.event);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(EventStreamError::Closed);
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    return Err(EventStreamError::Lagged);
                }
            }
        }
    }
}

/// Receive-side error from [`EventStream::recv`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EventStreamError {
    /// Subscriber lagged the broadcast channel; the client
    /// reconnects with `?after_seq=` set to its last recorded
    /// event seq.
    #[error("subscriber lagged the broadcast channel")]
    Lagged,
    /// Sender was dropped (registry shutdown).
    #[error("broadcast channel closed")]
    Closed,
}

// MARK: Helpers

/// Conflict predicate between a new job/lease request and an
/// existing in-flight job entry.  `WorkspaceDelete` is the only
/// exclusive admission shape; everything else overlaps freely
/// under its per-`JobType` cap.
///
/// Returns true iff the references target the same workspace AND
/// at least one side is a `WorkspaceDelete` job.  When neither
/// side is a `WorkspaceDelete`, sibling jobs (e.g. a train + a
/// dataset-delete in the same workspace) coexist.
fn job_conflicts_with_existing_job(
    new_is_workspace_delete: bool,
    new_workspace: WorkspaceId,
    existing_job_type: JobType,
    existing_refs: &[JobReference],
) -> bool {
    if !existing_refs
        .iter()
        .any(|r| r.workspace_id() == new_workspace)
    {
        return false;
    }
    new_is_workspace_delete || existing_job_type == JobType::WorkspaceDelete
}

/// Conflict predicate between a new admission and an existing
/// bare lease.  A bare lease is not a `JobType`-typed job; it
/// blocks only `WorkspaceDelete` admissions targeting the same
/// workspace.  Two bare leases coexist (e.g. concurrent uploads).
fn job_conflicts_with_existing_lease(
    new_is_workspace_delete: bool,
    new_workspace: WorkspaceId,
    lease_refs: &[JobReference],
) -> bool {
    if !new_is_workspace_delete {
        return false;
    }
    lease_refs.iter().any(|r| r.workspace_id() == new_workspace)
}

/// Conflict predicate for a new bare lease against an existing
/// in-flight job.  Bare leases (uploads, head-delete) only
/// conflict with `WorkspaceDelete`; train/convert/file-delete
/// in the same workspace coexist with the lease.
fn lease_conflicts_with_existing_job(
    new_workspace: WorkspaceId,
    existing_job_type: JobType,
    existing_refs: &[JobReference],
) -> bool {
    if existing_job_type != JobType::WorkspaceDelete {
        return false;
    }
    existing_refs
        .iter()
        .any(|r| r.workspace_id() == new_workspace)
}

/// Slide the recent window: drop the oldest TERMINAL entries
/// (active jobs are never displaced) until `entries.len() <=
/// max_recent`.
fn prune_recent(state: &mut JobsState, max_recent: usize) {
    if state.entries.len() <= max_recent {
        return;
    }
    // Walk `order` from the back (oldest) and remove terminal
    // entries until we fit.  Since we only displace terminal
    // jobs, an oversubscribed registry of all-active jobs would
    // fail admission earlier; this is a safety net.
    let mut i = state.order.len();
    while state.entries.len() > max_recent && i > 0 {
        i -= 1;
        let id = state.order[i];
        let drop_it = matches!(state.entries.get(&id), Some(e) if !e.state.is_active());
        if drop_it {
            state.order.remove(i);
            state.entries.remove(&id);
        }
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::asset_path::AssetPath;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }
    fn ws_id_b() -> WorkspaceId {
        WorkspaceId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn fresh() -> Arc<JobRegistry> {
        Arc::new(JobRegistry::new(JobRegistryCfg::default()))
    }

    fn ws_ref(id: WorkspaceId) -> Vec<JobReference> {
        vec![JobReference::Workspace { workspace_id: id }]
    }

    // MARK: Conflict semantics

    /// Per-`JobType` cap fires before any reference check: a
    /// second train job daemon-wide returns
    /// `AnotherTrainRunning` regardless of workspace.
    #[test]
    fn second_train_returns_another_train_running() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::Train, ws_ref(ws_id_b()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::AnotherTrainRunning));
    }

    /// `WorkspaceDelete` blocks any other job in the same
    /// workspace -- even a train (which has its own type-cap).
    /// The workspace-delete path is the only exclusive shape.
    #[test]
    fn workspace_delete_blocks_train_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    /// Reverse direction: an in-flight train blocks a
    /// `WorkspaceDelete` for the same workspace.
    #[test]
    fn train_blocks_workspace_delete_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    /// Train + dataset-delete in the SAME workspace are allowed
    /// to overlap; the legacy ancestor/descendant path-overlap
    /// check is gone.
    #[test]
    fn train_and_dataset_delete_coexist_in_same_workspace() {
        let r = fresh();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        // Dataset-delete acquires under its own delete-slot cap.
        let _h_del = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(AssetPath::parse("audio/cat").unwrap()),
            )
            .unwrap();
    }

    /// Train + convert can overlap (different per-type caps);
    /// same workspace is fine because neither is
    /// `WorkspaceDelete`.
    #[test]
    fn train_and_convert_coexist_in_same_workspace() {
        let r = fresh();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _h_conv = r
            .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
            .unwrap();
    }

    /// Two workspaces are independent: a train in one workspace
    /// does not block a `WorkspaceDelete` in another (the type
    /// caps are global, but the workspace-delete exclusion is
    /// workspace-scoped).
    #[test]
    fn workspace_delete_isolated_per_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        // WorkspaceDelete on a different workspace must succeed.
        let _h_del = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id_b()), None)
            .unwrap();
    }

    /// Bare leases (uploads, head-delete) are gated only by an
    /// in-flight `WorkspaceDelete` in the same workspace.
    #[test]
    fn upload_lease_blocked_by_active_workspace_delete() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap();
        let err = r.try_acquire_lease(ws_ref(ws_id())).unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    /// Bare leases coexist with train / convert / dataset-delete
    /// in the same workspace -- uploads are non-blocking
    /// (uploads and file deletes are allowed while training or
    /// conversion is running).
    #[test]
    fn upload_lease_coexists_with_train_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _lease = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
    }

    /// Two bare leases in the same workspace coexist; multiple
    /// concurrent uploads are explicitly allowed under
    /// `max_concurrent_uploads`.
    #[test]
    fn two_upload_leases_coexist_in_same_workspace() {
        let r = fresh();
        let _l1 = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
        let _l2 = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
    }

    /// `WorkspaceDelete` admission is blocked by any active bare
    /// lease in the workspace -- the in-flight upload must drain
    /// first or the operator must retry.
    #[test]
    fn workspace_delete_blocked_by_in_flight_upload_lease() {
        let r = fresh();
        let _lease = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
        let err = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    /// Dataset-delete + converter-delete share one delete slot;
    /// two simultaneous deletes hit the cap regardless of subtype.
    #[test]
    fn dataset_and_converter_deletes_share_one_slot() {
        let r = fresh();
        let _h = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(AssetPath::parse("a").unwrap()),
            )
            .unwrap();
        let err = r
            .try_acquire(
                JobType::ConverterDelete,
                ws_ref(ws_id_b()),
                Some(AssetPath::parse("tfjs").unwrap()),
            )
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    // MARK: Snapshot target_path

    /// `JobSnapshot::target_path` carries the producer-supplied
    /// display target: dataset-delete passes the path being
    /// deleted; train / convert / workspace-delete pass `None`.
    #[test]
    fn snapshot_target_path_round_trips_through_acquire() {
        let r = fresh();
        let target = AssetPath::parse("audio/cat").unwrap();
        let h = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(target.clone()),
            )
            .unwrap();
        let snap = r.snapshot(h.job_id()).unwrap();
        assert_eq!(snap.target_path.as_ref(), Some(&target));
        assert_eq!(snap.workspace_id, Some(ws_id()));
    }

    #[test]
    fn snapshot_target_path_none_for_train() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(h.job_id()).unwrap();
        assert!(snap.target_path.is_none());
    }

    // MARK: Lifecycle / events / counters

    #[tokio::test(flavor = "current_thread")]
    async fn update_progress_rate_limited_to_4hz() {
        // Use a very low throttle (1 Hz) so the test is stable.
        let cfg = JobRegistryCfg {
            progress_throttle_hz: 1.0,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        // Burst 5 progress updates back-to-back.
        for i in 0..5u64 {
            h.update_progress(JobProgress {
                done: i,
                total: Some(100),
            });
        }
        let snap = r.snapshot(h.job_id()).unwrap();
        // Only the FIRST progress (last_seq == 1) should have
        // emitted; the rest were throttled.  The latest payload
        // is still recorded on the entry, so progress.done == 4.
        assert_eq!(snap.last_seq, 1);
        assert_eq!(snap.progress.unwrap().done, 4);
        let counters = r.counters();
        assert!(counters.progress_throttled_total >= 4);
    }

    #[test]
    fn append_log_caps_line_length() {
        let cfg = JobRegistryCfg {
            max_log_line_bytes: 32,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
            .unwrap();
        let big = "x".repeat(1024);
        h.append_log(big);
        let counters = r.counters();
        assert_eq!(counters.log_lines_truncated_total, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscribe_events_replays_from_ring() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        h.append_log("hello");
        h.append_log("world");
        // Subscribe with after_seq=0 -> replay both.
        let mut stream = r.subscribe_events(h.job_id(), 0).unwrap();
        let mut got = Vec::new();
        while let Some(e) = stream.next_replay() {
            got.push(e.message.unwrap_or_default());
        }
        assert_eq!(got, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn subscribe_events_returns_event_gap_when_after_seq_too_old() {
        // Force a tiny ring so we can produce a gap with few writes.
        let cfg = JobRegistryCfg {
            max_job_event_ring: 2,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        // Push 5 events; ring drops the first 3.
        for _ in 0..5 {
            h.append_log("evt");
        }
        // Reconnect "from before everything" -- after_seq=0 is
        // a special-case "fresh subscribe" so we ask explicitly
        // for an old seq.  Use after_seq=1 (oldest_seq is 4).
        let res = r.subscribe_events(h.job_id(), 1);
        assert!(matches!(res, Err(EventGap { .. })));
    }

    #[test]
    fn terminate_releases_references() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let job_id = h.job_id();
        h.succeed(None);
        // Same workspace should now succeed (slot freed).
        let _h2 = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(job_id).unwrap();
        assert_eq!(snap.state, JobState::Succeeded);
    }

    #[test]
    fn drop_handle_releases_references_and_records_failed() {
        let r = fresh();
        let job_id = {
            let h = r
                .try_acquire(JobType::Train, ws_ref(ws_id()), None)
                .unwrap();
            h.job_id()
            // h drops here without explicit terminate.
        };
        // New acquire should succeed (the abandoned entry's
        // type-slot was released on Drop).
        let _h2 = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(job_id).unwrap();
        assert_eq!(snap.state, JobState::Failed);
    }

    #[test]
    fn recent_bounded_to_max_recent_jobs() {
        let cfg = JobRegistryCfg {
            max_recent_jobs: 2,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        for _ in 0..5 {
            // Each acquire creates a new job, then we
            // immediately terminate so the slot becomes
            // available for the next acquire.
            let h = r
                .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
                .unwrap();
            h.succeed(None);
        }
        let snaps = r.recent(100);
        assert!(snaps.len() <= 2, "got {} > 2 retained", snaps.len());
    }

    #[test]
    fn counters_track_admission_conflicts() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        // Second train globally -> AnotherTrainRunning -> counter++.
        let _ = r.try_acquire(JobType::Train, ws_ref(ws_id_b()), None);
        assert!(r.counters().admission_conflicts_total >= 1);
    }

    #[test]
    fn has_active_train_for_filters_by_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        assert!(r.has_active_train_for(ws_id()));
        assert!(!r.has_active_train_for(ws_id_b()));
        assert!(!r.has_active_convert_for(ws_id()));
    }

    #[test]
    fn registry_conflict_maps_to_file_error() {
        let err: FileError = RegistryConflict::AnotherTrainRunning.into();
        assert!(matches!(err, FileError::AnotherTrainRunning));
        let err: FileError = RegistryConflict::JobConflict {
            job_id: JobId::new(),
            job_type: JobType::Convert,
        }
        .into();
        assert!(matches!(err, FileError::JobConflict { .. }));
    }
}
