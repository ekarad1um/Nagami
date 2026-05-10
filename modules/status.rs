//! Daemon-wide status aggregator.
//!
//! Each long-running subsystem (alsa_capture, mic_arbitrator, opus_stream,
//! inference, stream_io, training, converter) calls
//! [`StatusMonitor::register`] at start to get a `watch::Sender<Heartbeat>`,
//! and pushes a heartbeat once per outer-loop tick.  The status endpoint
//! aggregates the latest values into a [`StatusSnapshot`].
//!
//! ## Health model
//!
//! A subsystem is "healthy" if both:
//!   * Its most recent heartbeat is < `HEALTH_STALE_AFTER` ago, AND
//!   * The most recent heartbeat had `healthy = true`.
//!
//! Subsystems that haven't been registered are considered unknown
//! (not present in the snapshot's `subsystems` map).
//!
//! ## sysinfo refresh policy
//!
//! `sysinfo::System::new_all()` is heavy (5 % CPU on a 1 Hz tick on a
//! Pi 5).  We construct with `RefreshKind` listing only what we need
//! (system-wide CPU usage; per-process RSS for the daemon itself; the
//! disk list is fetched on demand).
//!
//! Process-wide metrics (CPU/RSS/disk-free) are
//! sampled by a background tokio task at 500 ms cadence, NOT on the
//! request path.  The design refreshed sysinfo under a
//! `Mutex<System>` per `/api/v1/status` request; 4 dashboard tabs at
//! 1 Hz serialised through one mutex and burned a tokio
//! `spawn_blocking` slot per request.  The new design publishes via
//! `ArcSwap<MetricsSnapshot>` -- every request is one wait-free
//! `load_full()`, no mutex, no syscall, no blocking-pool slot.  Cold
//! `StatusMonitor::new()` starts with a zeroed snapshot (the
//! sampler's first tick fires within 500 ms of `start_sampler()`
//! and populates real values).
//!
//! ## Memory accounting
//!
//! We report the daemon process's **resident set size** (RSS, the
//! amount of physical RAM currently mapped to the process) -- not
//! system-wide used/total.  This is what `top`/`htop` show in the
//! RES column and is the only memory metric an operator needs to
//! diagnose "why is acoustics_lab eating my Pi's RAM?".  RSS is
//! refreshed via `refresh_processes_specifics(Some(&[pid]), ..)`
//! which targets a single PID rather than walking the full process
//! table -- much cheaper than the full-system memory refresh that
//! came before.

#![warn(missing_debug_implementations)]

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use sysinfo::{Disks, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::AbortHandle;

/// Register-time failures.  Today the only variant is
/// [`StatusError::AlreadyRegistered`]; `register` previously
/// silently replaced the prior receiver, which masked a class of
/// daemon-boot bugs (two subsystems wired to the same name; the
/// second one's receiver overwrote the first; the first sent
/// heartbeats into a discarded watch slot, looking healthy in
/// internal logs but invisible on `/api/v1/status`).
#[derive(Debug, Error)]
pub enum StatusError {
    /// `register(name)` was called when `name` was already in the
    /// monitor's subsystem map.  Categorised as `Internal` -- the
    /// daemon's wiring is the only register caller, so a
    /// collision means a programmer error in the daemon, not an
    /// operator-correctable input.
    #[error("subsystem already registered: {name}")]
    AlreadyRegistered { name: String },
}

impl crate::common::error::Categorized for StatusError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// A heartbeat is considered stale (subsystem unhealthy) after this
/// long without an update.  5 seconds covers the slowest reasonable
/// outer-loop period (opus paused: 1 Hz; inference: >= 4 Hz; capture:
/// >= 50 Hz).
pub const HEALTH_STALE_AFTER: Duration = Duration::from_secs(5);

/// A host-metrics sample is considered stale after this long
/// without a fresh sampler tick.  5 seconds = 10x the
/// production sampler period (500 ms), so a single missed
/// tick is fresh, two missed ticks are still fresh, and the
/// stale flag fires only when the sampler is genuinely
/// wedged (or has not yet produced its first sample).
/// Operators reading `/api/v1/status` distinguish "real zero
/// metrics" from "stalled sampler" via `metrics_stale`.
pub const METRICS_STALE_AFTER: Duration = Duration::from_secs(5);

/// Floor enforced on `StatusMonitor::start_sampler`'s `period`
/// argument.  `tokio::time::interval(Duration::ZERO)` panics; values
/// well below 50 ms are also unhelpful (sysinfo's
/// `MINIMUM_CPU_UPDATE_INTERVAL` is ~200 ms, so faster ticks publish
/// repeated CPU samples and just burn syscalls).  Production callers
/// pass 500 ms; this floor only matters for misuse / config drift.
pub const MIN_SAMPLER_PERIOD: Duration = Duration::from_millis(50);

/// Per-subsystem heartbeat.  Cheap to construct + clone: both
/// `detail` and `degraded_reason` are `Cow<'static, str>`, so a
/// `&'static str` call site is a pointer copy and `String` from
/// a `format!` lands as `Cow::Owned`.
#[derive(Clone, Debug, Serialize)]
pub struct Heartbeat {
    /// Monotonic-clock tick the heartbeat was emitted at.  Used by
    /// the snapshot path to compute `age_ms` + the `stale` flag.
    /// `Instant` is not serializable, so `#[serde(skip)]`; the
    /// API surface receives the derived `HeartbeatView { age_ms,
    /// stale, .. }`, not this `last_tick` directly.
    #[serde(skip)]
    pub last_tick: Instant,
    /// Subsystem-reported health flag.  Goes false on internal errors
    /// the subsystem can recover from but wants visibility on.
    pub healthy: bool,
    /// Free-form.  Intended to fit on one line of a status
    /// dashboard.  `Cow<'static, str>` so static literals are a
    /// pointer copy and `format!` outputs land as `Cow::Owned`.
    pub detail: Cow<'static, str>,
    /// Non-fatal degradation reason.  When `Some`,
    /// the subsystem is reporting itself as still functional but
    /// in a degraded state -- e.g. inference falling behind real-time,
    /// RKNN throttling under thermal pressure, restart attempts
    /// exhausted but the daemon still serving.  `healthy` is
    /// orthogonal: a degraded subsystem can stay `healthy: true`
    /// (still serving but with a caveat) or go `healthy: false`
    /// (degraded AND a hard fault).  The wire DTO surfaces this
    /// as an optional `degraded_reason` field; old clients that
    /// ignore unknown fields keep working byte-for-byte.
    ///
    /// **Stale interaction.** When a producer stops sending and
    /// its heartbeat ages past `HEALTH_STALE_AFTER`, the snapshot
    /// path preserves `degraded_reason` AS-IS -- the operator
    /// dashboard sees `{stale: true, degraded_reason: ...}`
    /// rather than `{stale: true, degraded_reason: null}`.  The
    /// `stale` flag carries the temporal context (these aren't
    /// fresh assertions); the prior degradation reason still
    /// helps the operator diagnose what the subsystem last
    /// reported before it went silent.  The
    /// `stale_degraded_heartbeat_preserves_reason_on_wire` test
    /// pins this contract.
    ///
    /// Producers wire `Heartbeat::degraded` for non-fatal
    /// concerns (e.g. inference's stale-window detection); `ok`
    /// and `unhealthy` remain the binary case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<Cow<'static, str>>,
}

impl Heartbeat {
    /// Healthy heartbeat, now.
    pub fn ok(detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: true,
            detail: detail.into(),
            degraded_reason: None,
        }
    }

    /// Unhealthy, with reason.
    pub fn unhealthy(detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: false,
            detail: detail.into(),
            degraded_reason: None,
        }
    }

    /// Healthy-but-degraded.  The subsystem is still functional
    /// (so `healthy: true`) but reports a non-fatal concern in
    /// `reason`.
    pub fn degraded(
        detail: impl Into<Cow<'static, str>>,
        reason: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: true,
            detail: detail.into(),
            degraded_reason: Some(reason.into()),
        }
    }
}

impl Default for Heartbeat {
    fn default() -> Self {
        Heartbeat::unhealthy("not started")
    }
}

/// Process-wide sample published by the background
/// sampler task.  Reads on the request path are wait-free
/// `ArcSwap::load_full`.  Default value (zeros) is what callers see
/// before [`StatusMonitor::start_sampler`] runs (or in tests where
/// the sampler isn't started).
///
/// `captured_at` deliberately uses `std::time::Instant` (monotonic;
/// not Serialize) rather than `SystemTime` because the only consumer
/// today is in-process staleness detection (`now - captured_at`);
/// future wire-facing exposures should add a sibling DTO with a
/// `Duration`-since-capture field instead of leaking the Instant.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// System-wide CPU usage percentage (0..=100.N where N is the
    /// CPU count, per `sysinfo::System::global_cpu_usage`'s
    /// historical contract).
    pub cpu_pct: f32,
    /// Daemon process resident set size, in KiB.
    pub mem_rss_kb: u64,
    /// Free space available on the filesystem holding
    /// `workspace_root` (the path passed to `start_sampler`), in
    /// KiB.  `0` when the workspace path doesn't resolve to a
    /// known mount or no path was supplied.
    pub disk_free_kb: u64,
    /// Monotonic instant the sample was published.  `None` in the
    /// default (no-sample-yet) snapshot.  Reads can compare against
    /// `Instant::now()` to detect a wedged sampler task.
    pub captured_at: Option<Instant>,
}

/// Aggregator.  `Clone` is cheap (Arc bump).  All clones share state.
#[derive(Clone)]
pub struct StatusMonitor {
    inner: Arc<Inner>,
}

struct Inner {
    /// Wait-free metrics snapshot.  Updated by the background
    /// sampler task started via [`StatusMonitor::start_sampler`];
    /// read on the request path with one atomic `load_full`.
    /// `Arc` (not bare ArcSwap) so the sampler task can hold a
    /// shared handle without owning `Arc<Inner>` (which would
    /// create a Drop cycle: Inner owns the AbortHandle that ends
    /// the loop, the loop owns Inner, neither ever drops).
    metrics: Arc<ArcSwap<MetricsSnapshot>>,
    /// Keyed by `Cow<'static, str>` so static names register
    /// zero-alloc via `Cow::Borrowed` and runtime-derived names
    /// go through `Cow::Owned(String)` without `Box::leak`.
    heartbeats: DashMap<Cow<'static, str>, watch::Receiver<Heartbeat>>,
    started_at: Instant,
    /// `AbortHandle` for the background sampler.  `Mutex` so any
    /// thread can call `start_sampler` while `Inner` is shared
    /// via `Arc`; `Option` so `new()` constructs without
    /// spawning.  Aborted on replacement and in `Drop for Inner`
    /// (a `JoinHandle` drop only detaches; we need an explicit
    /// abort to stop the loop).
    sampler: Mutex<Option<AbortHandle>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Abort any running sampler when the last `StatusMonitor`
        // clone drops.  Without this, an infinite
        // `tokio::spawn`-ed loop would keep running on the runtime
        // (Tokio detaches tasks whose `JoinHandle` is dropped) and
        // hold an `Arc<ArcSwap<MetricsSnapshot>>` clone alive
        // until the runtime itself shut down.  Idempotent: calling
        // `abort()` on an already-finished or already-aborted task
        // is a no-op.
        if let Some(handle) = self.sampler.lock().take() {
            handle.abort();
        }
    }
}

impl std::fmt::Debug for StatusMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatusMonitor")
            .field("subsystems", &self.inner.heartbeats.len())
            .field("uptime_s", &self.inner.started_at.elapsed().as_secs())
            .finish_non_exhaustive()
    }
}

impl StatusMonitor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                metrics: Arc::new(ArcSwap::from_pointee(MetricsSnapshot::default())),
                heartbeats: DashMap::new(),
                started_at: Instant::now(),
                sampler: Mutex::new(None),
            }),
        }
    }

    /// Start the background sampler task.  Spawns a
    /// tokio task on the current runtime that refreshes
    /// process-wide CPU + RSS + workspace disk-free at `period`
    /// cadence, publishing each sample into the wait-free
    /// `metrics` `ArcSwap`.  Idempotent in practice: a second call
    /// aborts the prior sampler and starts a fresh one (so a
    /// daemon-config reload that changes `workspace_root` can
    /// re-point the disk-free probe; rare).
    ///
    /// Must be called from inside a tokio runtime context.  The
    /// spawned task is owned via an `AbortHandle` stored in
    /// `Inner::sampler`; it is aborted explicitly on replacement
    /// and in `Drop for Inner` -- merely dropping a Tokio
    /// `JoinHandle` would *detach* the task, not stop it.
    ///
    /// `workspace_root` is the path whose mount's free-space the
    /// sampler reports; `None` keeps `disk_free_kb` at 0 (useful
    /// for tests).  The path is re-canonicalized on every tick so
    /// a directory created *after* the sampler starts (the cold-
    /// boot daemon-wiring case where workspace creation is
    /// sequenced after `start_sampler`) is picked up automatically
    /// within one tick of the directory becoming visible.
    ///
    /// `period` of zero would panic inside `tokio::time::interval`;
    /// values below `MIN_SAMPLER_PERIOD` are clamped to that
    /// floor.  Public callers should pass a meaningful period
    /// (typical: 500 ms).
    pub fn start_sampler(&self, workspace_root: Option<PathBuf>, period: Duration) {
        // Guard against zero (or sub-millisecond) periods that
        // would either panic `tokio::time::interval` or burn the
        // runtime in a tight refresh loop.  Clamping rather than
        // returning an error keeps the call site infallible (the
        // daemon-wiring shape today is a fire-and-forget call);
        // if a config reload pushes a degenerate period through,
        // we fall back to the floor instead of taking the daemon
        // down.
        let period = period.max(MIN_SAMPLER_PERIOD);
        // Build the System inside the spawned task so neither
        // the sysinfo state nor the refresh-kind values cross
        // the spawn boundary as captured locals.  Clone the
        // `Arc<ArcSwap<_>>` so the task has shared (but not
        // owning-of-Inner) access -- see `Inner::metrics` doc.
        let metrics_for_task = Arc::clone(&self.inner.metrics);
        let handle = tokio::spawn(async move {
            let refresh = RefreshKind::nothing()
                .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage());
            let process_refresh = ProcessRefreshKind::nothing().with_memory();
            let pid = get_current_pid().ok();
            let mut sys = System::new_with_specifics(refresh);
            // First refresh primes the CPU baseline.  CPU usage is
            // computed as a delta between two refreshes; sysinfo
            // enforces `MINIMUM_CPU_UPDATE_INTERVAL` (~200 ms)
            // between refreshes -- calls before that yield the
            // prior delta or zero on first read.
            sys.refresh_specifics(refresh);
            if let Some(pid) = pid {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    process_refresh,
                );
            }

            // The `Disks` instance is constructed lazily on the
            // first tick where canonicalization succeeds.  On
            // macOS `Disks::new_with_refreshed_list()` walks the
            // full mount table (>100 ms with many APFS volumes),
            // so eager construction in the no-path or
            // not-yet-canonicalizable case (e.g. tests with
            // `workspace_root: None`, or the cold-boot daemon
            // shape where the workspace dir is created *after*
            // the sampler starts) would block the sampler past
            // its first tick and starve callers reading the
            // ArcSwap before any real publish.
            //
            // Once built, the same `Disks` value is reused; only
            // the `canonical` path is re-resolved each tick (cheap
            // syscall) so a workspace directory that springs into
            // existence post-startup is still picked up.
            let mut disks: Option<Disks> = None;

            let mut interval = tokio::time::interval(period);
            // If a tick is missed (the runtime was busy), skip the
            // backlog rather than firing repeatedly back-to-back.
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // First `tick()` returns immediately -- the loop's first
            // iteration publishes the priming sample we just took.
            loop {
                interval.tick().await;
                sys.refresh_specifics(refresh);
                let mem_rss_kb = match pid {
                    Some(pid) => {
                        sys.refresh_processes_specifics(
                            ProcessesToUpdate::Some(&[pid]),
                            false,
                            process_refresh,
                        );
                        sys.process(pid).map(|p| p.memory() / 1024).unwrap_or(0)
                    }
                    None => 0,
                };
                let cpu_pct = sys.global_cpu_usage();
                // Re-canonicalize per tick.  On cold boot the
                // daemon may have called `start_sampler` before
                // `create_dir_all(workspace_root)`; the prior
                // implementation cached the canonicalize result
                // exactly once and so left `disk_free_kb` at 0
                // forever in that ordering.  A per-tick
                // canonicalize is a single `lstat`-class syscall
                // and recovers cleanly the moment the directory
                // exists.
                let disk_free_kb = match workspace_root
                    .as_deref()
                    .and_then(|p| std::fs::canonicalize(p).ok())
                {
                    Some(canonical) => {
                        // Lazily build the `Disks` view the first
                        // time we have a real path -- defers the
                        // expensive macOS mount-table walk until
                        // it is actually needed.
                        let disks = disks.get_or_insert_with(Disks::new_with_refreshed_list);
                        // `refresh(true)` re-queries free space on
                        // each known mount; `false` would only
                        // refresh the list itself.
                        disks.refresh(true);
                        disks
                            .list()
                            .iter()
                            .filter(|d| canonical.starts_with(d.mount_point()))
                            .max_by_key(|d| d.mount_point().as_os_str().len())
                            .map(|d| d.available_space() / 1024)
                            .unwrap_or(0)
                    }
                    None => 0,
                };
                metrics_for_task.store(Arc::new(MetricsSnapshot {
                    cpu_pct,
                    mem_rss_kb,
                    disk_free_kb,
                    captured_at: Some(Instant::now()),
                }));
            }
        });
        // Replace any prior sampler.  Explicitly abort the
        // predecessor: dropping its `AbortHandle` alone would not
        // stop it (Tokio aborts on `abort()`, not on handle drop).
        let mut slot = self.inner.sampler.lock();
        if let Some(prior) = slot.replace(handle.abort_handle()) {
            prior.abort();
        }
        // `handle` itself is intentionally dropped here -- the
        // `AbortHandle` we kept above is the sole owner.  Dropping
        // a `JoinHandle` does NOT abort; it merely relinquishes
        // the join-side, which is what we want.
        drop(handle);
    }

    /// Register a subsystem for status tracking.  Returns the sender
    /// the subsystem should use to push heartbeats.
    ///
    /// Fail-fast on collision: a second `register(name)` for
    /// the same name returns `Err(StatusError::AlreadyRegistered)`
    /// so a daemon-wiring bug (two subsystems wired to the same
    /// name) surfaces at boot rather than masquerading as one
    /// healthy subsystem.
    ///
    /// Accepts any `Into<Cow<'static, str>>` so both `&'static
    /// str` (zero-alloc; the typical daemon-wiring shape) and
    /// runtime `String`s pass without `Box::leak`.
    pub fn register(
        &self,
        name: impl Into<Cow<'static, str>>,
    ) -> Result<watch::Sender<Heartbeat>, StatusError> {
        let name = name.into();
        let (tx, rx) = watch::channel(Heartbeat::default());
        // dashmap's Entry API is the atomic "insert iff absent"
        // primitive; without it a check-then-insert race could
        // double-register under contention.  The `name.into_owned()`
        // here is needed because Entry::Vacant gives a borrowed
        // key; the closure receives the owned name back if we go
        // ahead.
        match self.inner.heartbeats.entry(name) {
            dashmap::Entry::Occupied(occ) => Err(StatusError::AlreadyRegistered {
                name: occ.key().to_string(),
            }),
            dashmap::Entry::Vacant(vac) => {
                vac.insert(rx);
                Ok(tx)
            }
        }
    }

    /// One snapshot of the daemon state.
    ///
    /// `broadcast_lags` carries cumulative WS-broadcast lag counts
    /// for the audio + inference fan-out channels.  They live in
    /// `stream_io` (counted per WS-receiver lag event); the daemon
    /// reads them when assembling the snapshot.  Pass `Default`
    /// values when `stream_io` isn't wired (e.g. unit tests).
    ///
    /// Wait-free.  Process-wide metrics
    /// (`cpu_pct`, `mem_rss_kb`, `disk_free_kb`) come from the
    /// `ArcSwap<MetricsSnapshot>` published by the background
    /// sampler task.  If `start_sampler` was never called (e.g. in
    /// unit tests), those fields read as 0.  The DashMap walk over
    /// heartbeats is the only non-trivial work on the request
    /// path, and that's contention-free per-entry.
    pub fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot {
        let metrics = self.inner.metrics.load_full();
        let cpu_pct = metrics.cpu_pct;
        let mem_rss_kb = metrics.mem_rss_kb;
        let disk_free_kb = metrics.disk_free_kb;

        let uptime_s = self.inner.started_at.elapsed().as_secs();

        // Metrics-freshness: surface the age + stale flag so
        // operators reading `/api/v1/status` can distinguish a
        // wedged sampler from real zero metrics.  `captured_at`
        // is `None` before the sampler's first tick (or when no
        // sampler is started); we report `metrics_age_ms = 0`
        // and `metrics_stale = true` in that case so the
        // operator-visible signal is "no sampler has produced
        // a sample yet" rather than a misleading age of zero
        // for fresh data.  The `saturating_*` guards a
        // captured-in-the-future timestamp (e.g. cross-core
        // clock skew); saturate to zero -- a zero age reads as
        // fresh, which is the right answer for "future-tick"
        // data.
        let now = Instant::now();
        let (metrics_age_ms, metrics_stale) = match metrics.captured_at {
            Some(ts) => {
                let age = now.saturating_duration_since(ts);
                (age.as_millis() as u64, age > METRICS_STALE_AFTER)
            }
            None => (0, true),
        };

        // Snapshot all heartbeats.  `borrow()` returns the latest value
        // without changing internal state; we clone since the snapshot
        // outlives the receiver guard.  Reuses the `now` captured above
        // for the metrics-freshness math so heartbeat ages and metrics
        // age are computed against the same instant.
        let mut subsystems = std::collections::BTreeMap::new();
        for entry in self.inner.heartbeats.iter() {
            let rx = entry.value();
            let hb = rx.borrow().clone();
            // Single `duration_since` per heartbeat -- `stale` and
            // `age_ms` derive from the same `Duration`, avoiding a
            // redundant `Instant` subtraction.  `saturating_*`
            // guards against a heartbeat sender that constructed
            // `Heartbeat { last_tick: Instant::now() + offset }`
            // (or a clock that ran backwards across cores), which
            // would otherwise panic in `duration_since`.  Saturate
            // to zero -- the heartbeat reads as fresh + healthy,
            // which is the right answer for "future-tick" data.
            let age = now.saturating_duration_since(hb.last_tick);
            let stale = age > HEALTH_STALE_AFTER;
            let view = HeartbeatView {
                healthy: hb.healthy && !stale,
                // Cow -> String at the wire boundary (the DTO is
                // `Serialize`-only, no `'static` constraint).
                detail: hb.detail.into_owned(),
                age_ms: age.as_millis() as u64,
                stale,
                // Stale heartbeats carry the producer's last
                // reason forward -- operator sees `stale: true`
                // AND the prior degradation reason.
                degraded_reason: hb.degraded_reason.map(Cow::into_owned),
            };
            subsystems.insert(entry.key().to_string(), view);
        }

        // Read the shared [`WorkspaceMetrics`] global if the
        // daemon installed one at boot; otherwise the
        // default-zero snapshot is wire-equivalent to "surface
        // inactive" and old clients see it as a no-op (default
        // field via `serde(default)`).
        let workspace = workspace_metrics::global()
            .map(|m| m.snapshot())
            .unwrap_or_default();

        StatusSnapshot {
            cpu_pct,
            mem_rss_kb,
            disk_free_kb,
            metrics_age_ms,
            metrics_stale,
            uptime_s,
            subsystems,
            broadcast_audio_messages_dropped: broadcast_lags.audio_messages_dropped,
            broadcast_inference_messages_dropped: broadcast_lags.inference_messages_dropped,
            workspace,
        }
    }
}

impl Default for StatusMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialization-friendly heartbeat view (translates Instant into
/// millis-since-tick + a stale flag).
#[derive(Clone, Debug, Serialize)]
pub struct HeartbeatView {
    pub healthy: bool,
    pub detail: String,
    /// Millis since the last heartbeat was sent.
    pub age_ms: u64,
    /// True if `age_ms > HEALTH_STALE_AFTER`.  Independent of `healthy`
    /// (a stale-but-was-healthy entry shows healthy=false here).
    pub stale: bool,
    /// Non-fatal degradation reason mirrored from
    /// `Heartbeat::degraded_reason`.  Omitted from the serialized
    /// JSON when `None` (`#[serde(skip_serializing_if)]`), so the
    /// pre-B.3 wire shape is byte-identical for any subsystem that
    /// hasn't opted into reporting degradation.  Old web clients that
    /// don't know this field continue to deserialize the rest of the
    /// view without change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
}

/// Aggregated daemon health snapshot returned by `/api/v1/status`.
/// Per-subsystem [`Heartbeat`]s plus sampled host metrics
/// (`cpu_pct`, `mem_rss_kb`, `disk_free_kb`).  Built on demand by
/// [`StatusMonitor::snapshot`]; the metrics half is read wait-free
/// from the [`MetricsSnapshot`] published by the background sampler.
#[derive(Clone, Debug, Serialize)]
pub struct StatusSnapshot {
    pub cpu_pct: f32,
    /// Resident set size of the daemon process, in KiB: the amount
    /// of physical RAM currently mapped to the process (what
    /// `top`/`htop` show in the RES column; equivalent to
    /// VmRSS on Linux, resident_size on macOS, WorkingSetSize on
    /// Windows).  System-wide totals are intentionally not
    /// reported -- the operator's question is "is acoustics_lab
    /// using too much RAM", not "what is system memory pressure".
    /// Reads as 0 if the daemon's PID could not be determined.
    pub mem_rss_kb: u64,
    pub disk_free_kb: u64,
    /// Milliseconds since the host-metrics sampler last
    /// published a snapshot.  Reads as `0` when no sample has
    /// been published yet (e.g. before the sampler's first tick
    /// or when no sampler has been started); pair with
    /// `metrics_stale` to distinguish "no sample yet" from
    /// "fresh sample".
    pub metrics_age_ms: u64,
    /// `true` when the host-metrics sample is older than
    /// [`METRICS_STALE_AFTER`] OR no sample has been published
    /// yet.  Operators reading `cpu_pct` / `mem_rss_kb` /
    /// `disk_free_kb` MUST check this flag before treating
    /// the values as authoritative -- a wedged sampler reports
    /// the last good values + `metrics_stale: true`, not
    /// zeroes.
    pub metrics_stale: bool,
    pub uptime_s: u64,
    pub subsystems: std::collections::BTreeMap<String, HeartbeatView>,
    /// Cumulative WS audio-broadcast messages dropped since
    /// daemon start.  The producer increments by the
    /// `RecvError::Lagged(n)` count (= number of skipped
    /// messages), so the field measures TOTAL DROPPED
    /// MESSAGES, not lag-event count -- a single lag event
    /// where a receiver missed 100 packets contributes 100,
    /// not 1.  Complement to per-subsystem
    /// `frames_dropped_lag` counters (which track
    /// audio-buffer reader lags, a different surface).
    #[serde(default)]
    pub broadcast_audio_messages_dropped: u64,
    /// Cumulative WS inference-broadcast messages dropped.
    /// Same dropped-messages-not-events semantics as
    /// `broadcast_audio_messages_dropped`.
    #[serde(default)]
    pub broadcast_inference_messages_dropped: u64,
    /// Workspace-side counters (`assets_uploaded_total`,
    /// `bytes_uploaded_total`, `workspace_core_writes_total`,
    /// `head_index_writes_total`,
    /// `dataset_mutations_rejected_total`, the bounded-ring
    /// p99 write-duration estimates, `job_events_dropped_total`,
    /// `sse_clients_current`, `boot_orphans_swept_total`).
    /// Daemon installs the shared [`WorkspaceMetrics`] global at
    /// boot; the field reads as zero when no global was
    /// installed (test fixtures, unit tests of the bare snapshot
    /// path).  `serde(default)` keeps older client deserializers
    /// from breaking byte-for-byte.
    #[serde(default)]
    pub workspace: WorkspaceMetricsSnapshot,
}

// `BroadcastLagSnapshot` graduated to
// `crate::common::traits::lag_source` (the trait surface that
// `stream_io::BroadcastLagCounters` implements).  Re-exported here
// so the existing `status::BroadcastLagSnapshot` import path stays
// working for `daemon` / `api` / tests.
pub use crate::common::traits::lag_source::BroadcastLagSnapshot;

// `StatusReporter` trait so api holds
// `Arc<dyn StatusReporter>` instead of the concrete monitor.
pub mod reporter;
pub use reporter::StatusReporter;

// Typed workspace-side counter surface
// (`assets_uploaded_total`, `bytes_uploaded_total`,
// `workspace_core_writes_total`, ...).  Embedded in
// [`StatusSnapshot::workspace`] so `GET /api/v1/status` reads
// the full set in one wire response.
pub mod workspace_metrics;
pub use workspace_metrics::{SseClientGuard, WorkspaceMetrics, WorkspaceMetricsSnapshot};

#[cfg(test)]
mod tests {
    use super::*;

    /// Register + snapshot now exercises the
    /// wait-free read path; the per-process RSS sample comes
    /// from the background sampler task (started here on a tight
    /// 50 ms cadence), not from a per-snapshot sysinfo refresh.
    #[tokio::test]
    async fn register_and_snapshot_round_trip() {
        let mon = StatusMonitor::new();
        mon.start_sampler(None, Duration::from_millis(50));
        let tx = mon.register("test_subsystem").expect("register");
        tx.send(Heartbeat::ok("running")).expect("send");
        // Allow the sampler at least one tick to publish a real RSS
        // value.  The interval's first tick fires immediately on
        // `interval(period)`, so the tiny extra slack here is
        // belt-and-braces against scheduler jitter on busy CI.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let view = snap
            .subsystems
            .get("test_subsystem")
            .expect("registered subsystem");
        assert!(view.healthy);
        assert_eq!(view.detail, "running");
        // `view.age_ms` reflects time since `tx.send`; the sleep
        // above bumps it up to ~150 ms -- keep the bound generous
        // so a slightly slow CI machine doesn't flap.
        assert!(
            view.age_ms < 1000,
            "fresh heartbeat ages quickly: {}",
            view.age_ms
        );
        assert!(!view.stale);

        // The daemon's RSS is plausibly populated (the test
        // process always has *some* resident memory). 0 would
        // mean either sysinfo couldn't determine our PID or the
        // sampler task didn't publish.
        assert!(snap.mem_rss_kb > 0, "mem_rss_kb=0 -- sampler broken?");
    }

    #[tokio::test(start_paused = true)]
    async fn stale_heartbeat_is_unhealthy() {
        let mon = StatusMonitor::new();
        let tx = mon.register("stalled").expect("register");
        tx.send(Heartbeat::ok("ok at t=0")).expect("send");

        // Advance virtual time past the stale window.
        tokio::time::advance(HEALTH_STALE_AFTER + Duration::from_secs(1)).await;

        // ... but `Instant::now()` is real-clock-driven (sysinfo uses
        // it directly), not paused -- so we have to verify behavior
        // via direct construction of an old heartbeat.  Re-send with
        // an artificially aged tick.
        let aged = Heartbeat {
            last_tick: Instant::now() - Duration::from_secs(10),
            healthy: true,
            detail: "should be marked stale".into(),
            degraded_reason: None,
        };
        tx.send(aged).expect("send aged");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let view = snap.subsystems.get("stalled").expect("registered");
        assert!(view.stale, "expected stale: {view:?}");
        assert!(!view.healthy, "stale should imply unhealthy");
    }

    #[test]
    fn unhealthy_heartbeat_propagates() {
        let mon = StatusMonitor::new();
        let tx = mon.register("oncall").expect("register");
        tx.send(Heartbeat::unhealthy("rknn returned -1"))
            .expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("oncall").unwrap();
        assert!(!v.healthy);
        assert_eq!(v.detail, "rknn returned -1");
    }

    /// `snapshot()` on a fresh monitor (no
    /// `start_sampler` call) returns the default `MetricsSnapshot`
    /// (zeros).  The disk-for-workspace probe moved to the
    /// background sampler task; the request path no longer
    /// touches the filesystem.  This pins the contract.
    #[tokio::test]
    async fn snapshot_without_sampler_returns_zero_metrics() {
        let mon = StatusMonitor::new();
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        assert_eq!(snap.cpu_pct, 0.0);
        assert_eq!(snap.mem_rss_kb, 0);
        assert_eq!(snap.disk_free_kb, 0);
    }

    /// `start_sampler` followed by a few hundred
    /// ms of wall time should produce a real (non-zero) RSS sample.
    /// Validates the wait-free read path end-to-end against the
    /// background sampler.  `tokio::test` so the task runs.
    #[tokio::test]
    async fn sampler_publishes_real_rss_within_first_tick() {
        let mon = StatusMonitor::new();
        // 50 ms cadence so the test doesn't have to wait the full
        // production 500 ms; the first interval tick fires
        // immediately so even sub-tick sleeps below would be enough,
        // but small slack keeps the test deterministic across CI.
        mon.start_sampler(None, Duration::from_millis(50));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        assert!(
            snap.mem_rss_kb > 0,
            "sampler should have published a real RSS sample by now; \
             got {snap:?}"
        );
    }

    /// `register` is fail-fast: a second `register(name)` for
    /// the same name returns `Err(AlreadyRegistered)` so a
    /// daemon-wiring bug surfaces at boot.
    #[test]
    fn re_register_returns_already_registered() {
        let mon = StatusMonitor::new();
        let tx1 = mon.register("flappy").expect("first register");
        let err = mon
            .register("flappy")
            .expect_err("second register must fail");
        match err {
            StatusError::AlreadyRegistered { name } => assert_eq!(name, "flappy"),
        }
        // The original sender still works -- first registration is
        // not poisoned by the failed second.
        tx1.send(Heartbeat::ok("still alive")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("flappy").expect("registered");
        assert_eq!(v.detail, "still alive");
    }

    /// `register` accepts owned `String` (e.g. a runtime-derived
    /// name), zero-allocating only the watch channel itself, and
    /// round-trips into the `subsystems` map keyed by String.
    #[test]
    fn register_accepts_owned_string_name() {
        let mon = StatusMonitor::new();
        let dynamic_name = format!("inference#{}", 7);
        let tx = mon.register(dynamic_name.clone()).expect("register owned");
        tx.send(Heartbeat::ok("gen-7 up")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap
            .subsystems
            .get(&dynamic_name)
            .expect("registered under owned name");
        assert_eq!(v.detail, "gen-7 up");
    }

    /// `Heartbeat::degraded` carries a non-fatal
    /// reason that surfaces on the wire as an optional
    /// `degraded_reason` field.  The `healthy` flag stays `true`
    /// (degraded != unhealthy); the wire JSON gains the new field
    /// only when `Some`, so pre-B.3 clients ignoring unknown
    /// fields still parse the rest of the snapshot byte-for-byte.
    #[test]
    fn heartbeat_degraded_surfaces_reason_on_wire_and_keeps_healthy_true() {
        let mon = StatusMonitor::new();
        let tx = mon.register("inference").expect("register");
        tx.send(Heartbeat::degraded(
            "running but stale",
            "rknn frame interval > 200ms (target 250ms)",
        ))
        .expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("inference").expect("registered");
        assert!(v.healthy, "degraded does not flip healthy off");
        assert_eq!(
            v.degraded_reason.as_deref(),
            Some("rknn frame interval > 200ms (target 250ms)"),
        );

        // Wire-shape gate: serialized JSON contains `degraded_reason`
        // exactly when it's `Some`.  Old clients see the new key;
        // they're expected to ignore unknown fields.  The complement
        // (Heartbeat::ok -> no key in JSON at all) is verified below.
        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            json.contains("\"degraded_reason\":\"rknn frame interval > 200ms (target 250ms)\""),
            "expected degraded_reason in wire JSON: {json}",
        );
    }

    /// Wire-compatibility gate: `Heartbeat::ok` MUST NOT add a new
    /// JSON field to the response.  The pre-B.3 baseline shape
    /// (`{healthy, detail, age_ms, stale}`) is preserved byte-for-
    /// byte for any subsystem that hasn't opted into reporting
    /// degradation.  Defends the acceptance criterion that
    /// HTTP wire bytes for `GET /api/v1/status` are byte-identical
    /// to baseline for the existing producers.
    ///
    /// The substring assertion below quotes the JSON key -- looking
    /// for `"degraded_reason"` (with quotes) rather than the bare
    /// substring -- so that a `detail` field happening to contain
    /// the literal text `degraded_reason` doesn't accidentally
    /// satisfy the gate.
    #[test]
    fn heartbeat_ok_omits_degraded_reason_field_from_wire() {
        let mon = StatusMonitor::new();
        let tx = mon.register("audio_io").expect("register");
        tx.send(Heartbeat::ok("running")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("audio_io").expect("registered");
        assert_eq!(v.degraded_reason, None);
        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            !json.contains("\"degraded_reason\""),
            "ok-path heartbeat must not emit degraded_reason field: {json}",
        );
    }

    /// Audit follow-up -- pin the deliberate
    /// stale-carry-forward behaviour.  When a producer stops
    /// sending heartbeats but its last send was a
    /// `Heartbeat::degraded`, the snapshot path preserves the
    /// degradation reason past the staleness window: the operator
    /// dashboard sees BOTH `stale: true` AND the prior
    /// `degraded_reason`.  The intent is "you can see what the
    /// subsystem last reported before it went silent" -- the
    /// `stale` flag carries the temporal context (these aren't
    /// fresh assertions).  If a future requirement reverses this
    /// (e.g. "degraded_reason should be cleared on staleness"),
    /// flip both this test and the snapshot mapping at
    /// `StatusMonitor::snapshot` together so the contract stays
    /// pinned by the test rather than by accident.
    #[test]
    fn stale_degraded_heartbeat_preserves_reason_on_wire() {
        let mon = StatusMonitor::new();
        let tx = mon.register("inference").expect("register");
        // Construct an aged degraded heartbeat directly -- same
        // shape `Heartbeat::degraded` produces, but with a
        // `last_tick` shifted into the past so it lands stale on
        // the next snapshot.  The aged-construction shape mirrors
        // `stale_heartbeat_is_unhealthy` above so any future change
        // to the `Heartbeat` field set surfaces as a single test
        // touchup.
        let aged = Heartbeat {
            last_tick: Instant::now() - Duration::from_secs(10),
            healthy: true,
            detail: "rknn slow".into(),
            degraded_reason: Some(Cow::Borrowed("frame > 250ms target")),
        };
        tx.send(aged).expect("send aged degraded");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("inference").expect("registered");
        assert!(v.stale, "expected stale: {v:?}");
        assert_eq!(
            v.degraded_reason.as_deref(),
            Some("frame > 250ms target"),
            "stale heartbeat should carry the prior degraded_reason forward",
        );
        // Healthy is now false because the snapshot path AND-folds
        // `hb.healthy && !stale`; the operator sees the staleness
        // signal first, the prior degradation reason second.
        assert!(!v.healthy, "stale entry shows healthy=false");
        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            json.contains("\"degraded_reason\":\"frame > 250ms target\""),
            "stale-but-was-degraded heartbeat must preserve the field: {json}",
        );
    }

    /// Replacing the sampler aborts the prior task.  Without
    /// this, the predecessor's `JoinHandle` would be dropped
    /// (which detaches but does NOT abort), so the old loop
    /// would keep publishing into the shared `ArcSwap` until
    /// the runtime shut down -- a leak across every config
    /// reload.  `start_sampler` retains an
    /// `AbortHandle` and aborts the predecessor explicitly.
    ///
    /// Verified end-to-end: install a sampler that publishes a
    /// sentinel `cpu_pct` value (we cheat by stamping `cpu_pct`
    /// from the raw sysinfo read, then replacing the sampler with
    /// a fresh one and waiting long enough that any leaked
    /// predecessor would have ticked at least twice).  After
    /// replacement we wait one fresh tick, capture the
    /// `captured_at` instant, then watch for `captured_at` to
    /// stop advancing for a generous bounded poll window: a
    /// surviving leaked sampler would keep bumping `captured_at`.
    #[tokio::test]
    async fn replacing_sampler_aborts_predecessor() {
        let mon = StatusMonitor::new();
        // Tight cadence so a leaked sampler would tick repeatedly
        // within the bounded poll window below.
        let period = Duration::from_millis(20);
        mon.start_sampler(None, period);
        // Let the first sampler tick a couple of times so it has
        // demonstrably been running.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let before = mon
            .inner
            .metrics
            .load_full()
            .captured_at
            .expect("first sampler must publish at least one sample");

        // Replace the sampler.  After this returns, the prior
        // sampler's `AbortHandle::abort()` has been invoked.  The
        // task may still execute one final iteration before
        // yielding (Tokio aborts at await points), but it cannot
        // execute many more.
        mon.start_sampler(None, period);

        // Drop the monitor entirely -- this exercises the
        // `Drop for Inner` path on the *replacement* sampler.  If
        // either abort were ineffective, the task would keep
        // bumping `captured_at` on the `Arc<ArcSwap<_>>` we still
        // hold below.
        let metrics = Arc::clone(&mon.inner.metrics);
        drop(mon);

        // Bounded poll budget: after both samplers are aborted,
        // `captured_at` must stabilize.  We wait long enough that
        // a still-running sampler at 20 ms cadence would have
        // ticked many times.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let stable_marker = metrics
            .load_full()
            .captured_at
            .expect("snapshot retains captured_at across abort");
        // Wait a further window equal to many sampler periods; if
        // any task is still running, `captured_at` will have
        // advanced.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let after = metrics
            .load_full()
            .captured_at
            .expect("snapshot retains captured_at after second wait");
        assert_eq!(
            stable_marker, after,
            "captured_at advanced after both samplers were aborted -- a sampler leaked: \
             before_replace={before:?} stable={stable_marker:?} after={after:?}",
        );
    }

    /// `disk_free_kb` recovers when the workspace directory is
    /// created *after* `start_sampler`.  The sampler re-
    /// canonicalizes per tick (a single `canonicalize` at init
    /// would leave `disk_state` stuck at `None` forever), so
    /// the daemon's "start_sampler before create_dir_all"
    /// wiring recovers within one or two ticks of the
    /// directory becoming visible.
    #[tokio::test]
    async fn disk_free_kb_recovers_after_workspace_created() {
        // Use a fresh temp path that does NOT yet exist.  std's
        // `temp_dir()` is process-wide; suffix with a uuid-like
        // counter so parallel test runs don't collide.
        let base = std::env::temp_dir();
        let unique = format!(
            "acoustics_lab_status_cold_boot_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        let workspace = base.join(unique);
        // Pre-condition: the path must NOT exist.  If it does
        // (extremely unlikely with the nanos suffix), the test
        // would silently degrade into the easy case.
        assert!(
            !workspace.exists(),
            "test setup expects a non-existent workspace path: {workspace:?}",
        );

        let mon = StatusMonitor::new();
        // Tight cadence so the recovery test completes in well
        // under a second of wall time.
        let period = Duration::from_millis(50);
        mon.start_sampler(Some(workspace.clone()), period);

        // Give the sampler a couple of ticks while the directory
        // is still missing.  `disk_free_kb` MUST be 0 in this
        // window (nothing to canonicalize against).
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            mon.snapshot(BroadcastLagSnapshot::default()).disk_free_kb,
            0,
            "disk_free_kb should be 0 while workspace is missing",
        );

        // Now create the directory.  Any reasonable temp-dir
        // mount has free space; the next sampler tick should pick
        // up a non-zero value.
        std::fs::create_dir_all(&workspace).expect("create workspace");

        // Bounded poll: up to 10 ticks (~500 ms) for the
        // sampler to observe the new directory.  Two ticks is
        // the documented budget; the extra slack covers CI
        // scheduler jitter.
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut got_nonzero = 0u64;
        while Instant::now() < deadline {
            tokio::time::sleep(period).await;
            let snap = mon.snapshot(BroadcastLagSnapshot::default());
            if snap.disk_free_kb > 0 {
                got_nonzero = snap.disk_free_kb;
                break;
            }
        }
        // Best-effort cleanup; ignore errors (the test process is
        // about to exit anyway and parallel tests have unique
        // suffixes).
        let _ = std::fs::remove_dir_all(&workspace);

        assert!(
            got_nonzero > 0,
            "disk_free_kb did not recover within bounded poll window after \
             workspace creation; sampler did not re-canonicalize",
        );
    }

    /// `start_sampler(_, Duration::ZERO)` must not panic the
    /// runtime: `tokio::time::interval` panics on a zero period,
    /// so the public API clamps to `MIN_SAMPLER_PERIOD`.
    #[tokio::test]
    async fn start_sampler_clamps_zero_period() {
        let mon = StatusMonitor::new();
        // Pre-fix this would panic synchronously inside
        // `tokio::time::interval`.  Post-fix we get a sampler
        // ticking at the floor.
        mon.start_sampler(None, Duration::ZERO);
        // Sleep > MIN_SAMPLER_PERIOD so the clamped sampler has
        // had time to publish.
        tokio::time::sleep(MIN_SAMPLER_PERIOD * 4).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        // If the sampler clamped and ran, `mem_rss_kb` will be
        // populated (this test process always has a non-trivial
        // RSS).  If the period had not been clamped, the panic
        // would have aborted this test at the call above.
        assert!(
            snap.mem_rss_kb > 0,
            "clamped-period sampler should have published a real RSS value: {snap:?}",
        );
    }
}
