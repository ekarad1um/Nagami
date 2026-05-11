//! REST API for the Acoustics Lab daemon.
//!
//! All routes are under `/api/v1/*`.  The `Router` returned by
//! [`router`] threads `AppState` through every handler.
//!
//! ## Audio & Inference
//!
//! * `GET /api/v1/health` -- liveness ping.
//! * `GET /api/v1/mic` / `POST /api/v1/mic/policy` -- mic policy.
//! * `GET/POST /api/v1/inference` -- inference cadence.
//! * `GET /api/v1/active` / `POST /api/v1/active` -- read or
//!   activate a workspace head or bundled default; the active
//!   manifest carries `runtime_head_id` + `n_classes` so callers
//!   no longer need a dedicated runtime-head route.
//!
//! ## Workspace & Assets
//!
//! `POST/GET/DELETE /api/v1/workspace[/:id[/...]]`, plus the
//! unified asset surface
//! `GET/PUT/DELETE /api/v1/workspace/{id}/assets/{*path}` with a
//! 256 MiB body limit on `PUT` so users can upload full
//! classifier bundles.  Training endpoints under the workspace
//! start/list/status/cancel in-process fine-tune jobs; converter
//! endpoints under the workspace extract heads from uploaded
//! TFJS bundles.
//!
//! ## Converter
//!
//! `POST /api/v1/workspace/{id}/convert` extracts the head from an
//! operator-uploaded TFJS Layers-Model bundle under the workspace's
//! `converters/` tree, writes a Burn `head.mpk` + `labels.txt`, and
//! registers them in that workspace's metadata.  Backbone/preproc are
//! NOT converted; the daemon ships with a deployment-bundled backbone.
//!
//! ## Status
//!
//! `GET /api/v1/status` returns a daemon-wide snapshot.  Daemon
//! logs are intentionally not aggregated through the API; reach
//! them via the host's standard log facilities (`journalctl`,
//! tail of `acousticsd.log` under the workspace logs dir).

#![warn(missing_debug_implementations)]

mod error;
// Tier T3 per `docs/ARCH_BOUNDARIES.md`: only `api::routes::*`
// + `api::tests` consume the extractor wrappers.  No
// out-of-`api` consumer; `pub(crate)` keeps in-module access
// working while preventing accidental promotion to public-API
// surface.
pub(crate) mod extract;
mod routes;
#[cfg(test)]
mod tests;

pub use error::ApiError;

use std::sync::Arc;

use crate::common::traits::head_store::HeadStore;
use crate::common::traits::lag_source::LagSource;
use crate::config::{ConfigHandle, MicSettingsHandle};
use crate::file_mgr::{FsService, JobRegistry};
use crate::inference::InferenceCfg;
use crate::status::StatusReporter;
use crate::training::TrainingRegistry;
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::FromRef;
use serde::Deserialize;

/// Threaded into every handler.  Cheap to clone (each field is Arc-
/// or Sender-shaped).  Constructed by the daemon before mounting the
/// router.
#[derive(Clone)]
pub struct AppState {
    /// Object-safe `ConfigHandle` trait.  Production
    /// wires `Arc<crate::config::ConfigCell>` (the file-backed impl) which
    /// dyn-coerces here; tests substitute in-memory mocks satisfying
    /// the trait surface without spinning up the disk-backed path.
    pub config: Arc<dyn ConfigHandle>,
    /// Read+swap surface for the active classifier head,
    /// surfaced as a `Arc<dyn HeadStore>` so test mocks
    /// substitute without rebuilding inference.  Production wires
    /// `crate::inference::HotHead` (whose `VersionedSwap<HeadInner>`
    /// gives reads + writer-mutex-serialised mutations + a
    /// monotonic `ResourceVersion` ).
    pub head: Arc<dyn HeadStore>,
    /// Read+write surface for the live mic settings
    /// (immutable launch catalogue + hot-swappable user-pref
    /// policy).  Production wires `crate::config::MicSettingsCell`
    /// (whose `MicSettingsHandle` impl runs validation +
    /// `VersionedSwap`-serialised in-memory swap +
    /// `ConfigHandle::mutate`-based persistence).  The arbitrator
    /// holds the same cell as `Arc<dyn MicSettingsStore>` (the
    /// read-only super-trait) for wait-free reads.  The catalogue
    /// is set once at boot and never changes; the policy is
    /// updated atomically here on `POST /mic/policy` and on
    /// user-config hot-reload.
    pub mic_settings: Arc<dyn MicSettingsHandle>,
    /// Live inference cadence, watched by `crate::inference::InferenceEngine`.
    pub inference_cfg: Arc<ArcSwap<InferenceCfg>>,
    /// Workspace + asset filesystem service.  Production
    /// wires `crate::file_mgr::FsServiceImpl` (a thin facade over the
    /// in-tree `WorkspaceMgr` that the API crate no longer
    /// imports concretely).  Tests substitute mocks satisfying
    /// the trait surface without touching the real disk.
    pub files: Arc<dyn FsService>,
    /// Daemon-wide status reporter.  Production wires
    /// `crate::status::StatusMonitor` (which also owns the boot-time
    /// `register()` surface that the daemon uses to give each
    /// subsystem a heartbeat sender).  Tests substitute mocks
    /// returning canned snapshots without spinning up sysinfo.
    pub monitor: Arc<dyn StatusReporter>,
    /// In-process training jobs registry.  Production
    /// wires `crate::training::JobRegistry`; tests substitute mocks
    /// without spinning up the in-process job machinery.
    pub training: Arc<dyn TrainingRegistry>,
    /// Read-side handle on `stream_io`'s per-stream broadcast-lag
    /// counters, surfaced as a snapshot on each `/api/v1/status`
    /// call.  Production wires `stream_io::BroadcastLagCounters`;
    /// tests substitute a canned-snapshot mock.  The trait-object
    /// shape keeps the `api` crate independent of `stream_io`.
    pub broadcast_lag_reader: Arc<dyn LagSource>,
    /// Global active-head mutex.  The `active/` tree is shared
    /// across workspaces; activations serialize through this
    /// mutex, held end-to-end across read+stage+publish+install
    /// +prune so two concurrent `POST /active` requests cannot
    /// interleave (publish and install must stay atomic per
    /// request, and prune must not run while a peer can publish
    /// a generation outside this request's keep-list).
    /// Per-workspace mutexes are taken AFTER this one (active →
    /// workspace lock order).  Sync `parking_lot::Mutex` because
    /// the entire critical section runs on one `spawn_blocking`
    /// worker -- the guard never crosses `.await`.
    pub active_mutex: Arc<parking_lot::Mutex<()>>,
    /// Optional deployment-bundled default head file pair.  Sourced
    /// on `POST /active {default: true}`.  Resolved once at daemon
    /// boot from the launch config.
    pub default_head: Option<crate::config::DefaultHeadRef>,
    /// Path to the trainer's Burn-format backbone artefact,
    /// resolved at daemon boot from the first
    /// `[[backbone.candidates]]` entry whose `kind = "burn"` in
    /// the launch TOML.  `POST /workspace/{id}/train` reads
    /// these bytes into RAM at job-start; the file is never
    /// modified by the daemon.  `None` when the launch
    /// catalogue contains no Burn candidate, in which case
    /// `POST /train` 404s with a "no Burn backbone configured"
    /// diagnostic.  Sharing this path with the inference
    /// engine's catalogue ([`crate::inference::BackboneCatalogue`])
    /// means a single launch-TOML edit hot-swaps both consumers
    /// at their next boot / next job start.
    pub training_backbone_path: Option<std::path::PathBuf>,
    /// Cross-cutting in-process job registry.  Owns the admission
    /// gate (per-`JobType` concurrency caps plus reference-overlap
    /// detection), the per-job event ring for SSE replay, the
    /// bounded recent-history surface that `GET /jobs` reads, and
    /// the broadcast channel SSE subscribers fan out from.
    /// Constructed at daemon boot and shared with
    /// `WorkspaceMgr::with_admission_and_jobs` so the admission
    /// paths register against the same instance the api routes
    /// read.
    pub jobs: Arc<JobRegistry>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("config_path", &self.config.path())
            .field("inference_cfg", &self.inference_cfg.load())
            .finish_non_exhaustive()
    }
}

// MARK: FromRef<AppState> Impls
//
// Each handler picks the trait subset it needs via
// `State<Arc<dyn ...>>`; pulling an unwired trait is a compile
// error.  Hand-written rather than `#[derive(FromRef)]` because
// the API crate does not depend on `axum_macros`.

impl FromRef<AppState> for Arc<dyn ConfigHandle> {
    fn from_ref(state: &AppState) -> Self {
        state.config.clone()
    }
}

impl FromRef<AppState> for Arc<dyn HeadStore> {
    fn from_ref(state: &AppState) -> Self {
        state.head.clone()
    }
}

impl FromRef<AppState> for Arc<dyn MicSettingsHandle> {
    fn from_ref(state: &AppState) -> Self {
        state.mic_settings.clone()
    }
}

impl FromRef<AppState> for Arc<ArcSwap<InferenceCfg>> {
    fn from_ref(state: &AppState) -> Self {
        state.inference_cfg.clone()
    }
}

impl FromRef<AppState> for Arc<dyn FsService> {
    fn from_ref(state: &AppState) -> Self {
        state.files.clone()
    }
}

impl FromRef<AppState> for Arc<dyn StatusReporter> {
    fn from_ref(state: &AppState) -> Self {
        state.monitor.clone()
    }
}

impl FromRef<AppState> for Arc<dyn TrainingRegistry> {
    fn from_ref(state: &AppState) -> Self {
        state.training.clone()
    }
}

impl FromRef<AppState> for Arc<dyn LagSource> {
    fn from_ref(state: &AppState) -> Self {
        state.broadcast_lag_reader.clone()
    }
}

impl FromRef<AppState> for Arc<JobRegistry> {
    fn from_ref(state: &AppState) -> Self {
        state.jobs.clone()
    }
}

/// Build the API router.  Daemon mounts at `/api/v1` (see
/// [`router_v1_nested`] for the convenience wrapper).  Each
/// per-domain router builder lives in its own private
/// `routes::*` module so adding a new endpoint only touches
/// one file plus the merge list below.
///
/// ## Fallback handlers
///
/// `Router::fallback` catches "no route matched" -> 404 wrapped
/// in our `{error, code}` envelope (vs axum's default plain-text
/// "Not Found").  `method_not_allowed_fallback` catches "path
/// matched, method didn't" -> 405 in the same envelope.  Both
/// surface through `ApiError`'s `IntoResponse`, keeping the
/// wire-shape uniform across success and failure responses for
/// every reachable path.
///
/// ## Trust posture
///
/// The daemon does NOT terminate auth.  Production deployments
/// front this router with an Nginx (or equivalent) reverse proxy
/// that handles bearer / mTLS / IP allow-listing; the daemon
/// trusts every request that reaches it.  Operators who expose
/// the daemon directly must understand the trust model is "open
/// on the bound interface".
pub fn router(state: AppState) -> Router {
    Router::new()
        .merge(routes::health::router())
        .merge(routes::mic::router())
        .merge(routes::inference::router())
        .merge(routes::active::router())
        .merge(routes::workspace::router())
        .merge(routes::dataset::router())
        .merge(routes::heads::router())
        .merge(routes::training::router())
        .merge(routes::status::router())
        .merge(routes::converter::router())
        .merge(routes::jobs::router())
        .fallback(fallback_404)
        .method_not_allowed_fallback(fallback_405)
        .with_state(state)
}

/// Catch-all for unmatched paths.  Surfaces the request method +
/// URI so an operator's curl typo is self-diagnosing.
async fn fallback_404(req: axum::http::Request<axum::body::Body>) -> error::ApiError {
    error::ApiError::NotFound(format!("no route matched: {} {}", req.method(), req.uri()))
}

/// Catch-all for path-matched-but-method-mismatched (e.g. `GET
/// /converter` when only `POST` is registered).  Maps through the
/// `ApiError::MethodNotAllowed` per-variant override -> 405.
async fn fallback_405(req: axum::http::Request<axum::body::Body>) -> error::ApiError {
    error::ApiError::MethodNotAllowed {
        method: req.method().to_string(),
        path: req.uri().path().to_string(),
    }
}

/// Build the API mounted at `/api/v1`.  The daemon prefers this
/// because it's a single line vs nesting the bare `router(...)`
/// under a parent `Router::new().nest("/api/v1", ...)`.
pub fn router_v1_nested(state: AppState) -> Router {
    Router::new().nest("/api/v1", router(state))
}

// Shared read-your-writes helpers.  Lives at this layer rather
// than inside one route module so any future endpoint that adopts
// `?min_version=N` reaches the same `VersionQuery` extractor +
// `check_min_version` predicate; today the sole consumer is
// `routes::mic`.

/// Query string for any read endpoint that supports
/// read-your-writes via `?min_version=N`.  Empty `None` accepts
/// the current snapshot regardless of version.
#[derive(Deserialize)]
pub(crate) struct VersionQuery {
    #[serde(default)]
    pub(crate) min_version: Option<u64>,
}

/// Gate the current snapshot on `min_version`.  Returns
/// `Ok(())` when the caller's expectation is met (or
/// omitted); otherwise [`ApiError::TooEarly`] (HTTP 425)
/// without blocking -- callers retry after their write's
/// [`crate::common::version::SwapReceipt`] settles.
pub(crate) fn check_min_version(
    current: crate::common::version::ResourceVersion,
    requested: Option<u64>,
) -> Result<(), ApiError> {
    let cur: u64 = current.get();
    if let Some(req) = requested
        && cur < req
    {
        return Err(ApiError::TooEarly {
            requested: req,
            current: cur,
        });
    }
    Ok(())
}
