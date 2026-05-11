// Each `tests/*.rs` binary that pulls in `mod api_fixtures;`
// sees every `pub` item flagged dead unless it happens to use
// it.  Since the module is intentionally a la-carte (one binary
// uses `fresh_app_state`, another only uses `json_body`),
// allow-dead-code at the module top so the shared surface does
// not flag in the binaries that do not consume every helper.
#![allow(dead_code)]

//! Shared `AppState` fixture for the integration-test binaries.
//!
//! Five `tests/*.rs` files (and `modules/api/tests.rs`) used to
//! carry near-identical 70–95 LOC `fresh_state(&Path) -> AppState`
//! helpers.  Every difference between them was either an irrelevant
//! comment drift or a one-field override (`default_head`,
//! the wrapper struct name).  Centralising here drops each
//! consumer to ~10 LOC and removes the drift surface — when the
//! `AppState` shape changes, the fixture changes once.
//!
//! Pull in via `mod api_fixtures;` in any test binary and call
//! `api_fixtures::fresh_app_state(tempdir.path())`.  Tests that
//! need a non-default `default_head` override the field
//! after construction (it's `pub`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, header};
use axum::response::Response;
use tower::ServiceExt;

use acoustics_lab::api::AppState;
use acoustics_lab::common::traits::head_store::HeadStore;
use acoustics_lab::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use acoustics_lab::config::{
    Config, ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell, MicSettingsHandle,
};
use acoustics_lab::file_mgr::{FsService, FsServiceImpl};
use acoustics_lab::inference::{HeadInner, HotHead};
use acoustics_lab::status::{StatusMonitor, StatusReporter};
use acoustics_lab::training::{JobRegistry as TrainingJobRegistry, TrainingRegistry};
use arc_swap::ArcSwap;

/// Drain `resp` and parse it as JSON into `T`.  The 4 MiB body
/// cap is well above any response shape exercised by the
/// integration suite; tests asserting on truncation behaviour
/// should call `to_bytes` directly with their own bound.
pub async fn json_body<T: serde::de::DeserializeOwned>(resp: Response) -> T {
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.expect("body");
    serde_json::from_slice(&bytes).expect("parse json")
}

/// Drive `router` with one HTTP request via
/// `tower::ServiceExt::oneshot` and return the response.
/// Passing `Some(s)` as `body` sends `s` as
/// `application/json`; passing `None` sends an empty body
/// (used by GET / DELETE call sites).  The router is
/// `clone()`d before `oneshot` so callers can issue further
/// requests against the same router.
pub async fn call(router: &Router, method: Method, path: &str, body: Option<&str>) -> Response {
    let mut req = Request::builder().method(method).uri(path);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let body = body.map(|s| Body::from(s.to_string())).unwrap_or_default();
    let req = req.body(body).expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

/// `PUT /api/v1/workspace/{ws}/assets/{path}` with the raw bytes
/// as the request body.  `path` is the workspace-relative target
/// appended to the URL via the route's `{*path}` wildcard; each
/// component is percent-encoded so characters that are invalid in
/// a URI (backslash, NUL, CR/LF, non-ASCII) survive the
/// `Request::builder().uri()` parse and reach the daemon for the
/// `AssetPath::parse` rejection that the path-traversal test
/// suite asserts on.  `/` separators between components are
/// preserved verbatim so the wildcard captures the multi-segment
/// tail.
///
/// The asset surface is unified under `/assets/{*path}` for read
/// (GET), write (PUT), and delete (DELETE) so this helper drives
/// the same URI family as the sibling [`call`]-based GET / DELETE
/// helpers.
pub async fn upload(router: &Router, ws: &str, path: &str, payload: &[u8]) -> Response {
    let encoded: String = path
        .split('/')
        .map(|seg| urlencoding::encode(seg).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    let req = Request::builder()
        .method(Method::PUT)
        .uri(format!("/api/v1/workspace/{ws}/assets/{encoded}"))
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from(payload.to_vec()))
        .expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

/// `POST /api/v1/workspace` with a `{"name":"<name>"}` body and
/// return the new workspace's id (the `id` field of the JSON
/// response).  Asserts a 200; tests that need to exercise the
/// 4xx surfaces of the create-workspace endpoint should drive
/// [`call`] directly.
pub async fn create_workspace(router: &Router, name: &str) -> String {
    let resp = call(
        router,
        Method::POST,
        "/api/v1/workspace",
        Some(&format!("{{\"name\":\"{name}\"}}")),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    v["id"].as_str().expect("id").to_string()
}

/// `LagSource` stub that always reports zero drops.  Substitutes
/// the production WS fan-out counters so the API can be tested in
/// isolation; tests asserting on lag counts plug in their own
/// source.
#[derive(Debug, Clone, Copy, Default)]
pub struct StubLagSource(pub BroadcastLagSnapshot);

impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// Resolve the on-disk per-workspace directory under the
/// fixture's `FsServiceImpl` root.  [`fresh_app_state`] roots
/// the FsService at `<dir>/workspaces/`, so the canonical
/// per-workspace dir lands at `<dir>/workspaces/workspaces/<id>/`
/// (one `workspaces` for the FsService root, one for the
/// `WORKSPACES_DIR_NAME` inside it).  Centralising the doubled
/// join here insulates tests from the fixture's internal
/// rooting choice -- changing the FsService root becomes a
/// single-site edit instead of a sweep across every consumer.
pub fn fixture_workspace_dir(dir: &Path, ws_id: impl AsRef<Path>) -> PathBuf {
    dir.join("workspaces").join("workspaces").join(ws_id)
}

/// Build an `AppState` rooted at `dir` with the canonical test
/// wiring: a freshly-persisted default `Config`, the launch
/// catalogue's `default-mock` candidate, a
/// 2-class synthetic `HotHead`, an `FsServiceImpl` over
/// `dir/workspaces/`, fresh job + status + training registries,
/// and a stub `LagSource`.  `default_head` defaults to an absent
/// file pair under `dir.join("bundled_default")` so an accidental
/// `POST /active {default: true}` fails closed; tests that need a
/// real bundled fixture override the field after construction.
pub fn fresh_app_state(dir: &Path) -> AppState {
    let cfg_path = dir.join("config.toml");
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");
    let cfg = Config::default_for();
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    // Mirror the daemon's boot wiring: launch catalogue ships a
    // single `default-mock` candidate so `Fixed { id: "default-mock" }`
    // requests pass the cell's cross-check.
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    let head: Arc<dyn HeadStore> = Arc::new(HotHead::from_inner(HeadInner {
        weight: vec![0.0; acoustics_lab::common::dims::BACKBONE_FEATURE_DIM * 2],
        bias: vec![0.0; 2],
        labels: vec!["a".into(), "b".into()],
        head_id: acoustics_lab::common::ids::HeadId::new(),
        n_classes: 2,
    }));
    let jobs = Arc::new(acoustics_lab::file_mgr::JobRegistry::new(
        acoustics_lab::file_mgr::JobRegistryCfg::default(),
    ));
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_and_jobs(
        workspace_root,
        Default::default(),
        jobs.clone(),
    ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(TrainingJobRegistry::new());
    AppState {
        config,
        head,
        mic_settings,
        inference_cfg,
        files,
        monitor,
        training,
        broadcast_lag_reader: Arc::new(StubLagSource::default()),
        active_mutex: Arc::new(parking_lot::Mutex::new(())),
        default_head: Some(DefaultHeadRef {
            path: dir.join("bundled_default/head.mpk"),
            labels_path: dir.join("bundled_default/labels.txt"),
        }),
        jobs,
    }
}
