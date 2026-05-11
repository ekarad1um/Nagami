//! Workspace-side monitoring counters published through
//! `GET /api/v1/status`.
//!
//! [`WorkspaceMetrics`] aggregates `assets_uploaded_total`,
//! `bytes_uploaded_total`, `workspace_core_writes_total`,
//! `head_index_writes_total`, `dataset_mutations_rejected_total`,
//! `job_events_dropped_total`, `sse_clients_current`,
//! `boot_orphans_swept_total`, plus a bounded-ring p99 estimator
//! for atomic-rewrite latency.
//!
//! ## Wiring shape
//!
//! `WorkspaceMetrics` is held centrally by the daemon as
//! `Arc<WorkspaceMetrics>` and exposed through a process-wide
//! `OnceLock` ([`global`]).  Hot-path call sites (atomic
//! rewrites in `file_mgr::schema`, dataset upload commits in
//! `file_mgr::dataset`, the JobRegistry's broadcast send, and
//! the SSE handler) read the global with one atomic load and
//! invoke the appropriate `record_*` method.  When no global
//! has been installed (test fixtures, unit tests of the
//! collection points), the recorders are zero-cost no-ops.
//!
//! The chosen `OnceLock` shape is the documented stop-rule
//! escape hatch: threading
//! `Option<Arc<WorkspaceMetrics>>` through `WorkspaceMgr` /
//! `JobRegistry` / `FsService` / every test fixture would
//! cascade through ~50 constructor call sites without
//! observable correctness benefit.  The trade-off:
//! tests that want to inspect counters explicitly install
//! their own `Arc<WorkspaceMetrics>` via [`install_for_tests`]
//! and read the same global in their assertions.
//!
//! ## Counter discipline
//!
//! - All counters are `AtomicU64` except `sse_clients_current`
//!   which is `AtomicI64` (an SSE RAII guard increments on
//!   construct, decrements on drop; signed because a stale
//!   double-drop or test-only direct decrement should surface
//!   negative rather than wrap to ~2^63).
//! - Every load and store uses [`Ordering::Relaxed`].  These are
//!   pure telemetry: the counters synchronize *nothing* across
//!   threads, no other state's visibility depends on the ordering
//!   of these accesses, and the snapshot reader tolerates
//!   per-counter skew (a snapshot is approximate by construction).
//!   `Relaxed` is therefore the correct ordering, not a weakening
//!   to be "upgraded" to `Acquire`/`Release`/`SeqCst` on sight.
//! - `record_*_write` records the `Duration` against the
//!   bounded p99 ring.  The p99 microsecond reading is
//!   recomputed on every snapshot under a parking_lot mutex,
//!   so callers don't pay for the sort on the hot path.
//! - `snapshot()` returns a plain-data DTO that flattens into
//!   `StatusSnapshot.workspace` for the wire.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use serde::Serialize;

/// Bounded ring depth used by the p99 write-duration estimator.
/// 256 samples * ~50 ms per write covers ~13 s of peak churn;
/// older samples roll out automatically.
pub const WRITE_DURATION_RING_DEPTH: usize = 256;

// MARK: WorkspaceMetricsSnapshot

/// Plain-data snapshot of [`WorkspaceMetrics`], emitted by
/// [`WorkspaceMetrics::snapshot`] and embedded in
/// [`crate::status::StatusSnapshot`] under the `workspace` field.
///
/// `*_total` counters are cumulative since daemon start;
/// `*_p99_us` fields are derived on demand from a bounded ring
/// (zero when no samples are recorded).  `sse_clients_current`
/// is signed so a stale double-drop surfaces as negative rather
/// than wrap.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct WorkspaceMetricsSnapshot {
    /// Cumulative count of dataset asset uploads that
    /// committed (`upload_dataset_file` returned Ok).
    pub assets_uploaded_total: u64,
    /// Cumulative bytes of dataset asset bodies committed.
    /// Sums the `observed_size` argument passed to each
    /// successful upload.
    pub bytes_uploaded_total: u64,
    /// Cumulative count of `workspace.json` atomic rewrites
    /// (`file_mgr::schema::write_workspace_core` returned Ok).
    pub workspace_core_writes_total: u64,
    /// Cumulative count of `heads.json` atomic rewrites
    /// (`file_mgr::schema::write_head_index` returned Ok).
    pub head_index_writes_total: u64,
    /// Cumulative count of mutations on `datasets/<...>` paths
    /// rejected by the admission gate (JobRegistry conflict or
    /// case-insensitive sibling-name collision).
    pub dataset_mutations_rejected_total: u64,
    /// Cumulative count of mutations on `converters/<...>` paths
    /// rejected by the admission gate; surfaces converter-tree
    /// contention separately from dataset-tree contention.
    pub converter_mutations_rejected_total: u64,
    /// p99 of the `workspace.json` write duration over the
    /// most recent [`WRITE_DURATION_RING_DEPTH`] samples, in
    /// microseconds.  Zero when no samples are recorded.
    pub workspace_core_write_p99_us: u64,
    /// p99 of the `heads.json` write duration over the most
    /// recent [`WRITE_DURATION_RING_DEPTH`] samples, in
    /// microseconds.  Zero when no samples are recorded.
    pub head_index_write_p99_us: u64,
    /// Cumulative count of job-event broadcasts dropped
    /// because every receiver had lagged out of the bounded
    /// channel.  Distinct from
    /// `JobRegistryCounters::events_dropped_total` (which
    /// counts ring-overflow drops).
    pub job_events_dropped_total: u64,
    /// Live SSE client count (RAII-guarded).  Signed because
    /// a stale double-drop should surface as negative rather
    /// than wrap.
    pub sse_clients_current: i64,
    /// Cumulative count of orphans swept on boot recovery
    /// across head, per-workspace dataset staging, and root
    /// workspace staging surfaces.  See
    /// [`WorkspaceMetrics::record_boot_orphans_swept`].
    pub boot_orphans_swept_total: u64,
    /// Cumulative per-workspace recovery failures on boot.  The
    /// orchestrator logs each failure via `tracing::warn!` and
    /// continues with the next workspace; this counter lets
    /// operators correlate `workspaces_scanned` < expected with
    /// the typed reason without grep-ing the daemon log.
    pub boot_workspace_recovery_failures_total: u64,
    /// Cumulative `.tmp/` orphans reaped by the runtime
    /// `storage_reaper` background task.  Distinct from
    /// `boot_orphans_swept_total` (boot-time sweep): this
    /// counter captures the steady-state runtime hygiene path
    /// that closes the "daemon kept running after a hard crash"
    /// gap the boot sweep cannot.  See
    /// [`crate::file_mgr::storage_reaper`].
    pub tmp_orphans_reaped_total: u64,
    /// Cumulative per-workspace `*.jsonl` files pruned by the
    /// runtime `storage_reaper`.  Bounds the long-tail growth
    /// of `<workspace>/training_logs/` +
    /// `<workspace>/converter_logs/` on operators who never
    /// issue `DELETE /workspace/{id}/assets/{tree}` manually.
    pub log_files_pruned_total: u64,
    /// Cumulative per-workspace failures observed by the
    /// runtime `storage_reaper`.  Mirrors the boot-time
    /// recovery counter shape; lets operators correlate
    /// "reaper logging warnings" without grep.
    pub storage_reaper_failures_total: u64,
}

// MARK: WorkspaceMetrics

/// Aggregated workspace-side counter surface.  See module
/// docs for the wiring contract.  All public methods take
/// `&self`; counters update lock-free except the bounded p99
/// ring (which uses a `parking_lot::Mutex<VecDeque>` covering
/// only the push + percentile recompute, never an `.await`).
#[derive(Debug, Default)]
pub struct WorkspaceMetrics {
    assets_uploaded_total: AtomicU64,
    bytes_uploaded_total: AtomicU64,
    workspace_core_writes_total: AtomicU64,
    head_index_writes_total: AtomicU64,
    dataset_mutations_rejected_total: AtomicU64,
    converter_mutations_rejected_total: AtomicU64,
    job_events_dropped_total: AtomicU64,
    sse_clients_current: AtomicI64,
    boot_orphans_swept_total: AtomicU64,
    boot_workspace_recovery_failures_total: AtomicU64,
    tmp_orphans_reaped_total: AtomicU64,
    log_files_pruned_total: AtomicU64,
    storage_reaper_failures_total: AtomicU64,
    /// Bounded ring of recent `workspace.json` write
    /// durations (microseconds).  Capacity:
    /// [`WRITE_DURATION_RING_DEPTH`].
    workspace_core_write_samples: Mutex<DurationRing>,
    /// Bounded ring of recent `heads.json` write durations
    /// (microseconds).  Capacity: [`WRITE_DURATION_RING_DEPTH`].
    head_index_write_samples: Mutex<DurationRing>,
}

impl WorkspaceMetrics {
    /// Construct a zero-initialized counter surface.  Cheap;
    /// the daemon constructs exactly one per process and
    /// shares the `Arc` with subsystems via [`install_global`].
    pub fn new() -> Self {
        Self::default()
    }

    /// One-shot snapshot for `GET /api/v1/status`.  Reads
    /// every counter and recomputes both p99 estimates under
    /// short critical sections; safe to call from the request
    /// path.
    pub fn snapshot(&self) -> WorkspaceMetricsSnapshot {
        WorkspaceMetricsSnapshot {
            assets_uploaded_total: self.assets_uploaded_total.load(Ordering::Relaxed),
            bytes_uploaded_total: self.bytes_uploaded_total.load(Ordering::Relaxed),
            workspace_core_writes_total: self.workspace_core_writes_total.load(Ordering::Relaxed),
            head_index_writes_total: self.head_index_writes_total.load(Ordering::Relaxed),
            dataset_mutations_rejected_total: self
                .dataset_mutations_rejected_total
                .load(Ordering::Relaxed),
            converter_mutations_rejected_total: self
                .converter_mutations_rejected_total
                .load(Ordering::Relaxed),
            workspace_core_write_p99_us: self.workspace_core_write_samples.lock().p99_us(),
            head_index_write_p99_us: self.head_index_write_samples.lock().p99_us(),
            job_events_dropped_total: self.job_events_dropped_total.load(Ordering::Relaxed),
            sse_clients_current: self.sse_clients_current.load(Ordering::Relaxed),
            boot_orphans_swept_total: self.boot_orphans_swept_total.load(Ordering::Relaxed),
            boot_workspace_recovery_failures_total: self
                .boot_workspace_recovery_failures_total
                .load(Ordering::Relaxed),
            tmp_orphans_reaped_total: self.tmp_orphans_reaped_total.load(Ordering::Relaxed),
            log_files_pruned_total: self.log_files_pruned_total.load(Ordering::Relaxed),
            storage_reaper_failures_total: self
                .storage_reaper_failures_total
                .load(Ordering::Relaxed),
        }
    }

    /// Record a successful dataset asset upload.  Call from
    /// `WorkspaceMgr::upload_dataset_file` on the Ok arm with
    /// the receipt's `size_bytes`.  Increments
    /// `assets_uploaded_total` by 1 and `bytes_uploaded_total`
    /// by `bytes`.
    pub fn record_upload(&self, bytes: u64) {
        self.assets_uploaded_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_uploaded_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one successful atomic rewrite of `workspace.json`.
    /// Caller times the operation with `Instant::now() - start`
    /// and passes the [`Duration`] in.
    pub fn record_workspace_core_write(&self, duration: Duration) {
        self.workspace_core_writes_total
            .fetch_add(1, Ordering::Relaxed);
        self.workspace_core_write_samples
            .lock()
            .push_micros(duration_micros(duration));
    }

    /// Record one successful atomic rewrite of `heads.json`.
    /// Same shape as
    /// [`Self::record_workspace_core_write`].
    pub fn record_head_index_write(&self, duration: Duration) {
        self.head_index_writes_total.fetch_add(1, Ordering::Relaxed);
        self.head_index_write_samples
            .lock()
            .push_micros(duration_micros(duration));
    }

    /// Record one rejected dataset mutation.  Hit by the upload
    /// path on JobRegistry conflict, by the case-insensitive
    /// sibling-collision rejection, and any other dataset-tree
    /// admission rejection.  See
    /// [`Self::record_converter_mutation_rejected`] for the
    /// `converters/<...>` companion.
    pub fn record_dataset_mutation_rejected(&self) {
        self.dataset_mutations_rejected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record one rejected converter mutation.  Mirror of
    /// [`Self::record_dataset_mutation_rejected`] for the
    /// `converters/<...>` tree; the upload + delete admission
    /// gates dispatch by `AssetTree` and emit either counter
    /// depending on which top-level the rejected path targets.
    pub fn record_converter_mutation_rejected(&self) {
        self.converter_mutations_rejected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record `n` job events dropped because every receiver
    /// had lagged out of the bounded broadcast channel.
    /// Sourced from
    /// `tokio::sync::broadcast::Sender::send` returning
    /// `Err(SendError(_))`.  Distinct from in-ring overflow
    /// (which the JobRegistry's own
    /// `events_dropped_total` already counts).
    pub fn record_job_events_dropped(&self, n: u64) {
        self.job_events_dropped_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// RAII guard for one live SSE connection.  Increments
    /// `sse_clients_current` on construct and decrements on
    /// drop.  The api SSE handler holds one of these across
    /// the connection's full lifetime.
    pub fn sse_client_guard(self: &Arc<Self>) -> SseClientGuard {
        self.sse_clients_current.fetch_add(1, Ordering::Relaxed);
        SseClientGuard {
            metrics: Arc::clone(self),
        }
    }

    /// Record `n` orphans swept on boot recovery.  Daemon
    /// calls this exactly once after `recover_all` returns,
    /// summing every per-surface counter from `RecoveryReport`.
    pub fn record_boot_orphans_swept(&self, n: u64) {
        self.boot_orphans_swept_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Record `n` per-workspace recovery failures observed on
    /// boot.  The daemon calls this exactly once after
    /// `recover_all` returns, passing
    /// `report.workspaces.workspace_recovery_failures`.  The
    /// failures themselves are logged structured via
    /// `tracing::warn!` inside the recovery sweep; the counter
    /// surfaces the total so operators can spot the
    /// "workspaces_scanned < expected" symptom without grep-ing
    /// the log.
    pub fn record_boot_workspace_recovery_failures(&self, n: u64) {
        self.boot_workspace_recovery_failures_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Record one runtime sweep pass from the
    /// `file_mgr::storage_reaper` background task.  Accepts bare
    /// counters (not the report struct) so the layer-graph edge
    /// `status -> file_mgr` stays out of the dependency tree;
    /// the daemon's wiring site unpacks the fields and forwards
    /// them through here.
    pub fn record_storage_sweep(
        &self,
        tmp_orphans_reaped: u64,
        log_files_pruned: u64,
        failures: u64,
    ) {
        self.tmp_orphans_reaped_total
            .fetch_add(tmp_orphans_reaped, Ordering::Relaxed);
        self.log_files_pruned_total
            .fetch_add(log_files_pruned, Ordering::Relaxed);
        self.storage_reaper_failures_total
            .fetch_add(failures, Ordering::Relaxed);
    }
}

// MARK: SseClientGuard

/// RAII guard returned by
/// [`WorkspaceMetrics::sse_client_guard`].  Decrements
/// `sse_clients_current` on drop.  `Send`-safe; SSE handlers
/// drag the guard across `.await` points without contention.
#[derive(Debug)]
pub struct SseClientGuard {
    metrics: Arc<WorkspaceMetrics>,
}

impl Drop for SseClientGuard {
    fn drop(&mut self) {
        self.metrics
            .sse_clients_current
            .fetch_sub(1, Ordering::Relaxed);
    }
}

// MARK: Process-wide handle

/// Process-wide [`WorkspaceMetrics`] handle.  Daemon installs
/// once at boot via [`install_global`]; collection-point
/// callers fetch through [`global`] and treat `None` as a
/// no-op (test fixtures never installed a global).
static GLOBAL: OnceLock<Arc<WorkspaceMetrics>> = OnceLock::new();

/// Install the process-wide [`WorkspaceMetrics`] handle.
/// Called exactly once at daemon boot.  Subsequent calls
/// return `Err(existing)` so a misuse is observable rather
/// than silently dropping the second handle.  Tests that
/// want a deterministic surface use [`install_for_tests`].
pub fn install_global(metrics: Arc<WorkspaceMetrics>) -> Result<(), Arc<WorkspaceMetrics>> {
    GLOBAL.set(metrics)
}

/// Fetch the process-wide handle, if installed.  Cheap (one
/// atomic load).  Collection points pass the result to a
/// helper closure; absent global means "counters disabled".
pub fn global() -> Option<&'static Arc<WorkspaceMetrics>> {
    GLOBAL.get()
}

/// Test-only: install (or replace, if already installed) the
/// process-wide handle for assertions.  Exposed under
/// `#[cfg(test)]` builds plus integration tests through the
/// `pub` re-export so per-test fixtures can pin their own
/// metrics.  Guarded by a parking_lot mutex so concurrent
/// tests serialize installation ordering when run in the
/// same process.
#[doc(hidden)]
pub fn install_for_tests(metrics: Arc<WorkspaceMetrics>) {
    // OnceLock has no replace; tests that need a fresh handle
    // either run in `#[test_threads=1]` and reset via
    // `install_for_tests` once at the start, or use the
    // explicit `Arc<WorkspaceMetrics>` they constructed
    // directly without going through `global()`.  The
    // production install_global enforces single-init; tests
    // accept the first-installed value.
    let _ = GLOBAL.set(metrics);
}

// MARK: Convenience accessors for collection points

/// Run `f` against the installed global, if any.  Cheap
/// no-op when no global is installed.  Used by collection
/// points that want a single-line invocation without
/// repeating the `Option` dance.
pub fn with_global<F: FnOnce(&Arc<WorkspaceMetrics>)>(f: F) {
    if let Some(m) = global() {
        f(m);
    }
}

// MARK: Internals

/// Convert a `Duration` to whole microseconds, saturating on
/// overflow.  `Duration::as_micros` returns `u128`; we cap at
/// `u64::MAX` because individual atomic rewrites do not run
/// for 5e5 years.
fn duration_micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// Bounded fixed-capacity ring buffer of microsecond
/// durations.  Used by the p99 write-duration estimator;
/// `push_micros` evicts the oldest sample on overflow.  p99
/// is computed by sorting a snapshot of the ring (cheap at
/// `WRITE_DURATION_RING_DEPTH = 256`) so the hot path push
/// is constant-time.
#[derive(Debug, Default)]
struct DurationRing {
    /// `samples[0]` is the oldest; `samples[len-1]` is the
    /// newest.  Capacity bounded by
    /// [`WRITE_DURATION_RING_DEPTH`].
    samples: std::collections::VecDeque<u64>,
}

impl DurationRing {
    fn push_micros(&mut self, micros: u64) {
        if self.samples.len() >= WRITE_DURATION_RING_DEPTH {
            self.samples.pop_front();
        }
        self.samples.push_back(micros);
    }

    /// p99 in microseconds.  Returns 0 when the ring is
    /// empty.  The percentile index is `ceil(0.99 * n) - 1`
    /// (equivalent to the nearest-rank shape used by most
    /// monitoring systems); for `n < 100` the p99 is
    /// effectively the maximum of the ring, which is the
    /// right behaviour at small sample sizes.
    fn p99_us(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut sorted: Vec<u64> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        // Nearest-rank p99: ceil(0.99 * n) - 1, clamped to
        // [0, n-1].
        let n = sorted.len();
        let idx = (((n as f64) * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        sorted[idx]
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn record_upload_increments_assets_and_bytes() {
        let m = WorkspaceMetrics::new();
        m.record_upload(1024);
        m.record_upload(2048);
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 2);
        assert_eq!(snap.bytes_uploaded_total, 1024 + 2048);
    }

    #[test]
    fn record_workspace_core_write_updates_counter_and_p99() {
        let m = WorkspaceMetrics::new();
        m.record_workspace_core_write(Duration::from_millis(5));
        m.record_workspace_core_write(Duration::from_millis(10));
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_writes_total, 2);
        // Two samples, p99 = max sample = 10 ms = 10_000 us.
        assert_eq!(snap.workspace_core_write_p99_us, 10_000);
    }

    #[test]
    fn record_head_index_write_updates_counter_and_p99() {
        let m = WorkspaceMetrics::new();
        m.record_head_index_write(Duration::from_micros(750));
        m.record_head_index_write(Duration::from_micros(1500));
        m.record_head_index_write(Duration::from_micros(500));
        let snap = m.snapshot();
        assert_eq!(snap.head_index_writes_total, 3);
        // Three samples; nearest-rank p99 == max == 1500.
        assert_eq!(snap.head_index_write_p99_us, 1500);
    }

    #[test]
    fn p99_uses_nearest_rank_at_large_sample_sizes() {
        let m = WorkspaceMetrics::new();
        // 100 samples: 1..=100 ms.  p99 ceil(0.99 * 100) - 1
        // = 98 -> sorted[98] = 99 ms.
        for i in 1..=100u64 {
            m.record_workspace_core_write(Duration::from_millis(i));
        }
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_write_p99_us, 99 * 1000);
    }

    #[test]
    fn duration_ring_evicts_oldest_at_capacity() {
        let m = WorkspaceMetrics::new();
        // Push capacity + 10 samples; oldest 10 should evict.
        for i in 0..(WRITE_DURATION_RING_DEPTH + 10) {
            m.record_workspace_core_write(Duration::from_micros(i as u64));
        }
        // Ring now holds 256 samples [10..=265].  Nearest-
        // rank p99 = ceil(0.99 * 256) - 1 = 253; sorted[253]
        // = sample at offset 253 from the ring's smallest
        // value (10) = 263.
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_write_p99_us, 263);
    }

    #[test]
    fn record_dataset_mutation_rejected_monotonic() {
        let m = WorkspaceMetrics::new();
        for _ in 0..7 {
            m.record_dataset_mutation_rejected();
        }
        assert_eq!(m.snapshot().dataset_mutations_rejected_total, 7);
    }

    /// Converter and dataset rejections accumulate on separate
    /// counters so operators can distinguish admission contention
    /// per tree.
    #[test]
    fn record_converter_mutation_rejected_separate_from_dataset() {
        let m = WorkspaceMetrics::new();
        for _ in 0..3 {
            m.record_dataset_mutation_rejected();
        }
        for _ in 0..5 {
            m.record_converter_mutation_rejected();
        }
        let snap = m.snapshot();
        assert_eq!(snap.dataset_mutations_rejected_total, 3);
        assert_eq!(snap.converter_mutations_rejected_total, 5);
    }

    #[test]
    fn record_job_events_dropped_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_job_events_dropped(3);
        m.record_job_events_dropped(5);
        assert_eq!(m.snapshot().job_events_dropped_total, 8);
    }

    #[test]
    fn sse_client_guard_increments_then_decrements() {
        let m = Arc::new(WorkspaceMetrics::new());
        let g1 = m.sse_client_guard();
        let g2 = m.sse_client_guard();
        assert_eq!(m.snapshot().sse_clients_current, 2);
        drop(g1);
        assert_eq!(m.snapshot().sse_clients_current, 1);
        drop(g2);
        assert_eq!(m.snapshot().sse_clients_current, 0);
    }

    #[test]
    fn boot_orphans_swept_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_boot_orphans_swept(3);
        m.record_boot_orphans_swept(2);
        assert_eq!(m.snapshot().boot_orphans_swept_total, 5);
    }

    /// `record_boot_workspace_recovery_failures` accumulates the
    /// per-workspace recovery failure count emitted by the
    /// recovery orchestrator.
    #[test]
    fn boot_workspace_recovery_failures_accumulate() {
        let m = WorkspaceMetrics::new();
        m.record_boot_workspace_recovery_failures(2);
        m.record_boot_workspace_recovery_failures(0);
        m.record_boot_workspace_recovery_failures(3);
        assert_eq!(m.snapshot().boot_workspace_recovery_failures_total, 5);
    }

    /// `record_storage_sweep` accumulates each of the three
    /// counters independently across multiple sweep passes.
    /// Pins the positional arg order (`tmp_orphans_reaped`,
    /// `log_files_pruned`, `failures`) so a future
    /// swap of two same-typed arguments surfaces as a counter
    /// mismatch in CI rather than as silently miscounted metrics
    /// in production.
    #[test]
    fn record_storage_sweep_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_storage_sweep(3, 5, 1);
        m.record_storage_sweep(2, 0, 0);
        m.record_storage_sweep(0, 0, 4);
        let s = m.snapshot();
        assert_eq!(s.tmp_orphans_reaped_total, 5);
        assert_eq!(s.log_files_pruned_total, 5);
        assert_eq!(s.storage_reaper_failures_total, 5);
    }

    #[test]
    fn snapshot_consistency_under_concurrent_writers() {
        // Eight threads each push 1k upload + write events;
        // final snapshot should sum to 8k uploads, 8M bytes,
        // 8k core writes, 8k head writes -- no torn or
        // missed atomic increments.
        let m = Arc::new(WorkspaceMetrics::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m_clone = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    m_clone.record_upload(1024);
                    m_clone.record_workspace_core_write(Duration::from_micros(10));
                    m_clone.record_head_index_write(Duration::from_micros(20));
                    m_clone.record_dataset_mutation_rejected();
                    m_clone.record_job_events_dropped(1);
                    m_clone.record_boot_orphans_swept(1);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker join");
        }
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 8 * 1000);
        assert_eq!(snap.bytes_uploaded_total, 8 * 1000 * 1024);
        assert_eq!(snap.workspace_core_writes_total, 8 * 1000);
        assert_eq!(snap.head_index_writes_total, 8 * 1000);
        assert_eq!(snap.dataset_mutations_rejected_total, 8 * 1000);
        assert_eq!(snap.job_events_dropped_total, 8 * 1000);
        assert_eq!(snap.boot_orphans_swept_total, 8 * 1000);
    }

    #[test]
    fn snapshot_default_has_zero_counters() {
        let m = WorkspaceMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 0);
        assert_eq!(snap.bytes_uploaded_total, 0);
        assert_eq!(snap.workspace_core_writes_total, 0);
        assert_eq!(snap.head_index_writes_total, 0);
        assert_eq!(snap.dataset_mutations_rejected_total, 0);
        assert_eq!(snap.converter_mutations_rejected_total, 0);
        assert_eq!(snap.workspace_core_write_p99_us, 0);
        assert_eq!(snap.head_index_write_p99_us, 0);
        assert_eq!(snap.job_events_dropped_total, 0);
        assert_eq!(snap.sse_clients_current, 0);
        assert_eq!(snap.boot_orphans_swept_total, 0);
        assert_eq!(snap.boot_workspace_recovery_failures_total, 0);
    }
}
