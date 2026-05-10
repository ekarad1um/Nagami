//! API integration + unit tests.

#![cfg(test)]
// Tests stage tempfile fixtures with `std::fs::write`; the production
// constraint in `clippy.toml` (writes through file_mgr) does not
// apply to test setup helpers.
#![allow(clippy::disallowed_methods)]
use super::*;
use crate::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use crate::config::Config;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode};
use axum::response::Response;
use tower::ServiceExt;

/// `LagSource` stub: substitutes the production WS fan-out counters
/// so the API can be tested in isolation.
#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

// Active-head endpoint tests live in `tests/active_head_endpoints.rs`
// and exercise the real `POST /active` end-to-end via `HotHead`.

fn fresh_state(dir: &std::path::Path) -> AppState {
    use crate::config::{ConfigCell, LaunchConfig, MicSettingsCell};
    let cfg_path = dir.join("config.toml");
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");
    let mut cfg = Config::default_for(workspace_root.clone());
    // `Config::default_for` ships `/run/acoustics_lab.sock`, which
    // `StreamCfg::validate_uds_path` rejects on hosts without `/run`
    // (macOS dev, sandboxed CI).  Patch to a tempdir-relative path so
    // the fixture validates everywhere without weakening the
    // production default.
    cfg.stream.uds_path = dir.join("test.sock");
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    // Mirror the daemon's boot wiring: launch catalogue ships a
    // single `default-mock` candidate so `Fixed { id: "default-mock" }`
    // requests pass the cell's cross-check.
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn crate::config::MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    // Synthetic 2-class HotHead populates the head slot so the
    // active-head extractor wiring resolves; tests asserting
    // head-related wire shapes use `tests/active_head_endpoints.rs`.
    let head: Arc<dyn crate::common::traits::head_store::HeadStore> = Arc::new(
        crate::inference::HotHead::from_inner(crate::inference::HeadInner {
            weight: vec![0.0; crate::common::dims::BackboneFeatureDim::USIZE * 2],
            bias: vec![0.0; 2],
            labels: vec!["a".into(), "b".into()],
            head_id: crate::common::ids::HeadId::new(),
            n_classes: 2,
        }),
    );
    let jobs = Arc::new(crate::file_mgr::JobRegistry::new(
        crate::file_mgr::JobRegistryCfg::default(),
    ));
    let files: Arc<dyn FsService> =
        Arc::new(crate::file_mgr::FsServiceImpl::with_admission_and_jobs(
            workspace_root,
            Default::default(),
            jobs.clone(),
        ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(crate::status::StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(crate::training::JobRegistry::new());
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
        // Tests that exercise `POST /active` override this via their
        // own fixture; the default points at an absent tempdir path
        // so an accidental bundled-default activation fails closed
        // rather than touching a checked-in fixture.
        bundled_default_dir: dir.join("bundled_default"),
        jobs,
    }
}

async fn json_body<T: serde::de::DeserializeOwned>(resp: Response) -> T {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    serde_json::from_slice(&bytes).expect("parse json")
}

async fn call(router: &Router, method: Method, path: &str, body: Option<&str>) -> Response {
    let mut req = Request::builder().method(method).uri(path);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let req = req
        .body(
            body.map(|s| Body::from(s.to_string()))
                .unwrap_or(Body::empty()),
        )
        .expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

#[tokio::test]
async fn health_endpoint_ok() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/health", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["status"], "ok");
}

/// 404 must return the `{error, code}` body shape, not axum's
/// default plain-text, so a client matching on `code == "not_found"`
/// behaves the same on legitimate misses and typo'd paths.
#[tokio::test]
async fn fallback_404_uses_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/this_path_does_not_exist", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");
    assert!(
        v["error"].as_str().unwrap().contains("no route matched"),
        "envelope error must surface the unmatched method+path; got {v}",
    );
}

/// 405 path-method mismatch must use the same `{error, code}`
/// envelope.
#[tokio::test]
async fn fallback_405_uses_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    // `/inference` is registered as GET / POST only; PUT should 405.
    let resp = call(&r, Method::PUT, "/inference", None).await;
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "method_not_allowed");
    assert!(
        v["error"].as_str().unwrap().contains("method not allowed"),
        "envelope error must surface the verb mismatch; got {v}",
    );
}

#[tokio::test]
async fn get_mic_returns_catalogue_and_policy_separately() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/mic", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // Default policy: FirstAvailable + Auto.
    assert_eq!(v["policy"]["mic"]["kind"], "first_available");
    assert_eq!(v["policy"]["channel"]["kind"], "auto");
    // Launch catalogue is returned read-only under its own top-level
    // key; the test fixture ships a single `default-mock` candidate.
    assert!(
        v["catalogue"]["candidates"].is_array(),
        "catalogue.candidates should be present in /mic response",
    );
    assert_eq!(
        v["catalogue"]["candidates"][0]["id"], "default-mock",
        "first-boot launch catalogue includes the synthetic mock candidate",
    );
}

#[tokio::test]
async fn post_mic_policy_persists_and_swaps() {
    use crate::audio_io::mic_arbitrator::{ChannelSelection, MicSelection};

    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let mic_settings_view = state.mic_settings.clone();
    let cfg_view = state.config.clone();
    let r = router(state);

    // Pin the `default-mock` candidate (provisioned by
    // `Config::default_for`) so the request passes the cell's
    // catalogue cross-check; `post_mic_policy_rejects_unknown_fixed_id`
    // is the negative case.
    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "default-mock" },
                "channel": { "kind": "fixed", "channel": 0 }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    if resp.status() != StatusCode::OK {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        panic!("status not ok; body={}", String::from_utf8_lossy(&bytes));
    }
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["policy"]["mic"]["kind"], "fixed");
    assert_eq!(v["policy"]["mic"]["id"], "default-mock");
    assert_eq!(v["policy"]["channel"]["kind"], "fixed");
    assert_eq!(v["policy"]["channel"]["channel"], 0);
    // Catalogue still present + unchanged.
    assert_eq!(v["catalogue"]["candidates"][0]["id"], "default-mock");

    let in_mem = (*mic_settings_view.snapshot()).clone();
    match &in_mem.policy.mic {
        MicSelection::Fixed { id } => assert_eq!(id.as_str(), "default-mock"),
        other => panic!("expected Fixed, got {other:?}"),
    }
    match &in_mem.policy.channel {
        ChannelSelection::Fixed { channel } => assert_eq!(*channel, 0),
        other => panic!("expected Fixed channel, got {other:?}"),
    }
    let on_disk = std::fs::read_to_string(cfg_view.path()).unwrap();
    assert!(
        on_disk.contains("default-mock"),
        "config not persisted: {on_disk}"
    );
}

/// Read-your-writes: callers feed the version they received from
/// POST back into `GET /mic?min_version=N` to confirm a prior write
/// has settled.
#[tokio::test]
async fn post_mic_policy_surfaces_version_and_get_honours_min_version() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);
    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "default-mock" },
                "channel": { "kind": "auto" }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let post_version = v["version"].as_u64().expect("version field on post");
    assert!(
        post_version >= 1,
        "first successful mutation must yield version >= 1, got {post_version}",
    );

    // GET with min_version == post_version must succeed
    // (current >= requested).
    let get = call(
        &r,
        Method::GET,
        &format!("/mic?min_version={post_version}"),
        None,
    )
    .await;
    assert_eq!(get.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(get).await;
    assert_eq!(v["version"].as_u64(), Some(post_version));

    // GET with min_version one past the current must
    // return 425 Too Early.
    let get = call(
        &r,
        Method::GET,
        &format!("/mic?min_version={}", post_version + 1),
        None,
    )
    .await;
    assert_eq!(get.status(), StatusCode::TOO_EARLY);
    let v: serde_json::Value = json_body(get).await;
    assert_eq!(v["code"], "too_early");
}

/// Without the catalogue cross-check, an unknown `Fixed { id }`
/// would leave the arbitrator silently inert (rate-limited warn, no
/// audio) until an operator noticed.
#[tokio::test]
async fn post_mic_policy_rejects_unknown_fixed_id() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let mic_settings_view = state.mic_settings.clone();
    let r = router(state);

    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "no-such-mic" },
                "channel": { "kind": "auto" }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "unknown Fixed id must yield 400",
    );
    // And the live policy must NOT have been swapped.
    let in_mem = (*mic_settings_view.snapshot()).clone();
    // Default policy is FirstAvailable; the rejected POST must
    // have left it that way.
    assert!(
        matches!(
            in_mem.policy.mic,
            crate::audio_io::mic_arbitrator::MicSelection::FirstAvailable,
        ),
        "live policy was mutated despite rejection: {:?}",
        in_mem.policy.mic,
    );
}

#[tokio::test]
async fn post_inference_validates_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    // top_k=0 -> 400
    let resp = call(&r, Method::POST, "/inference", Some(r#"{"top_k":0}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");

    // hop_samples=0 -> 400
    let resp = call(&r, Method::POST, "/inference", Some(r#"{"hop_samples":0}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_inference_swaps_cadence() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let cfg_view = state.inference_cfg.clone();
    let r = router(state);

    let body = r#"{"hop_samples":22050,"top_k":5}"#;
    let resp = call(&r, Method::POST, "/inference", Some(body)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["cfg"]["hop_samples"], 22050);
    assert_eq!(v["cfg"]["top_k"], 5);

    let live = **cfg_view.load();
    assert_eq!(live.hop_samples, 22050);
    assert_eq!(live.top_k, 5);
}

#[tokio::test]
async fn training_start_round_1_wrapper_body_returns_400() {
    // Train body is the flattened `TrainingCfg`; a wrapper body with
    // `dataset_path` + `training_cfg` parse-fails at deserialize
    // before any spawn.
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"train"}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["id"].as_str().unwrap();

    let body = serde_json::json!({
        "dataset_path": "missing",
        "training_cfg": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
        },
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/workspace/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "wrapper body must parse-fail",
    );
}

#[tokio::test]
async fn training_status_rejects_bad_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"train"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["id"].as_str().unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!("/workspace/{ws}/training/not-a-uuid"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

// The legacy `POST /converter` route (cross-workspace head extraction
// with `AssetKind::HeadMpk` / `HeadLabels`) is gone; end-to-end
// coverage of the new `POST /workspace/{id}/convert` producer lives
// in `tests/convert_pipeline.rs`.

#[tokio::test]
async fn status_endpoint_returns_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let monitor = crate::status::StatusMonitor::new();
    // 50 ms cadence + 150 ms wait = >=2 tick opportunities to publish
    // a real RSS sample.
    monitor.start_sampler(None, std::time::Duration::from_millis(50));
    let tx = monitor.register("test_alive").expect("register");
    tx.send(crate::status::Heartbeat::ok("running"))
        .expect("send");
    let mut state = fresh_state(dir.path());
    state.monitor = Arc::new(monitor);
    let r = router(state);
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let resp = call(&r, Method::GET, "/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // 0 would indicate the sampler task path is broken.
    assert!(v["mem_rss_kb"].as_u64().unwrap_or(0) > 0);
    assert_eq!(v["subsystems"]["test_alive"]["healthy"], true);
    assert_eq!(v["subsystems"]["test_alive"]["detail"], "running");
    // Pin field presence + numeric type so a schema change can't
    // silently drop them.
    assert_eq!(v["broadcast_audio_messages_dropped"].as_u64(), Some(0));
    assert_eq!(v["broadcast_inference_messages_dropped"].as_u64(), Some(0));
}

/// `/status` reflects whatever the `broadcast_lag_reader` returns;
/// production wires this from `stream_io::BroadcastLagCounters`.
#[tokio::test]
async fn status_endpoint_surfaces_broadcast_lags() {
    let dir = tempfile::tempdir().unwrap();
    let mut state = fresh_state(dir.path());
    state.broadcast_lag_reader = Arc::new(StubLagSource(BroadcastLagSnapshot {
        audio_messages_dropped: 17,
        inference_messages_dropped: 42,
    }));
    let r = router(state);
    let resp = call(&r, Method::GET, "/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["broadcast_audio_messages_dropped"].as_u64(), Some(17));
    assert_eq!(v["broadcast_inference_messages_dropped"].as_u64(), Some(42));
}

/// `metrics_age_ms` + `metrics_stale` let an operator distinguish a
/// wedged sampler from real zero metrics: pre-sampler state
/// (`captured_at = None`) reads `{age: 0, stale: true}`, post-tick
/// reads `{age: small, stale: false}`.
#[tokio::test]
async fn status_endpoint_surfaces_metrics_freshness() {
    // Case 1: pre-sampler -- captured_at = None.
    {
        let dir = tempfile::tempdir().unwrap();
        let r = router(fresh_state(dir.path()));
        let resp = call(&r, Method::GET, "/status", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(
            v["metrics_age_ms"].as_u64(),
            Some(0),
            "pre-sampler: metrics_age_ms must be 0; body={v}",
        );
        assert_eq!(
            v["metrics_stale"].as_bool(),
            Some(true),
            "pre-sampler: metrics_stale must be true (no sample yet); body={v}",
        );
    }

    // Case 2: sampler running -- captured_at = Some(recent).
    {
        let dir = tempfile::tempdir().unwrap();
        let monitor = crate::status::StatusMonitor::new();
        monitor.start_sampler(None, std::time::Duration::from_millis(50));
        let mut state = fresh_state(dir.path());
        state.monitor = Arc::new(monitor);
        let r = router(state);
        // 150 ms / 50 ms cadence = ~3 ticks; comfortably non-flaky.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let resp = call(&r, Method::GET, "/status", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(
            v["metrics_stale"].as_bool(),
            Some(false),
            "live sampler: metrics_stale must be false; body={v}",
        );
        // ~1 s slack covers scheduler jitter on busy CI.
        let age = v["metrics_age_ms"]
            .as_u64()
            .expect("metrics_age_ms must be u64");
        assert!(
            age < 1_000,
            "live sampler: metrics_age_ms must be < 1 s, got {age}",
        );
    }
}

/// Path-traversal attempts via the workspace-id parameter MUST be
/// rejected before reaching the filesystem layer; `WorkspaceId::parse`
/// enforces strict UUID-v4 format.
#[tokio::test]
async fn workspace_id_rejects_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    for nasty in [
        "../etc",
        "..%2Fetc", // pre URL-decode by axum
        "/etc/passwd",
        "etc/passwd",
        "abc",                                     // not a UUID
        "00000000-0000-0000-0000-00000000000",     // 35 chars
        "../00000000-0000-4000-8000-000000000000", // UUID-v4 with traversal prefix
        "00000000_0000_4000_8000_000000000000",    // wrong separators
    ] {
        // axum URL-decodes the path; pre-decode here to mirror what
        // arrives at the handler.
        let decoded = nasty.replace("%2F", "/");
        let resp = call(
            &r,
            Method::DELETE,
            &format!("/workspace/{}", urlencoding::encode(&decoded)),
            None,
        )
        .await;
        // 400 (parse rejected), 404 (axum routing miss), or 405 are
        // all acceptable; 200 OK and any 5xx are not.
        assert_ne!(
            resp.status(),
            StatusCode::OK,
            "path traversal accepted for {nasty:?}"
        );
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::NOT_FOUND
                || resp.status() == StatusCode::METHOD_NOT_ALLOWED,
            "unexpected status {} for {nasty:?}",
            resp.status()
        );
    }
}

#[tokio::test]
async fn workspace_create_list_delete_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::GET, "/workspace", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["workspaces"].as_array().unwrap().is_empty());

    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"first"}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    assert_eq!(v["name"], "first");

    let resp = call(&r, Method::GET, "/workspace", None).await;
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["workspaces"].as_array().unwrap();
    assert_eq!(ws.len(), 1);
    assert_eq!(ws[0]["id"], id);

    // Duplicate name -> 409 Conflict.
    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"first"}"#)).await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);

    let resp = call(&r, Method::DELETE, &format!("/workspace/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Delete again -> 404.
    let resp = call(&r, Method::DELETE, &format!("/workspace/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn workspace_upload_via_multipart() {
    use axum::http::header;
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    // Create workspace.
    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"u"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    // Multipart `{path, file}` where `path` is workspace-rooted
    // (`datasets/<...>` or `converters/<...>`); leading `/` is also
    // accepted, canonical form drops it.
    let boundary = "----acousticslabtestbnd";
    let body = format!(
        "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"path\"\r\n\r\n\
             datasets/cls/demo.bin\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"demo.bin\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n\
             {payload}\r\n\
             --{boundary}--\r\n",
        payload = "DEMO-MPK-PAYLOAD"
    );
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/workspace/{id}/upload"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("build req");
    let resp = r.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK, "upload status");
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "datasets/cls/demo.bin");
    assert_eq!(v["size_bytes"], "DEMO-MPK-PAYLOAD".len());
    assert!(v["sha256"].as_str().unwrap().len() == 64);
    assert_eq!(v["workspace_revision_id"], 1);

    // List at the workspace root: daemon subdirs (`datasets/`,
    // `converters/`, `heads/`, ...) + metadata files are visible;
    // `.tmp/` is excluded.
    let resp = call(&r, Method::GET, &format!("/workspace/{id}/assets"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let names: Vec<&str> = v["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert!(names.contains(&"datasets"), "datasets/ in {names:?}");
    assert!(!names.contains(&".tmp"), ".tmp excluded: {names:?}");
    let resp = call(
        &r,
        Method::GET,
        &format!("/workspace/{id}/assets/datasets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    // First-level child of `datasets/` is the class folder, not
    // the leaf file (the trainer keys class labels off this dir).
    assert_eq!(entries[0]["name"], "cls");
}

/// 5 MiB body verifies the chunked reader works end-to-end and
/// doesn't depend on `field.bytes()`; pins the streaming path so a
/// regression to a buffered read shows up as a test failure (we
/// can't observe peak RSS in a unit test).
#[tokio::test]
async fn workspace_upload_streams_large_payload() {
    use axum::http::header;
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"big"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    // Deterministic 5 MiB payload (cycled bytes of `id`) so we can
    // sha256 it independently without using a constant pattern.
    let mut payload = Vec::with_capacity(5 * 1024 * 1024);
    let pat = id.as_bytes();
    for i in 0..(5 * 1024 * 1024) {
        payload.push(pat[i % pat.len()]);
    }
    let expected_sha = crate::api::routes::converter::sha256_hex(&payload);

    let boundary = "----acousticslab-large";
    let head = format!(
        "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"path\"\r\n\r\n\
             datasets/cls/big.bin\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"big.bin\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
    );
    let tail = format!("\r\n--{boundary}--\r\n");
    let mut body = Vec::with_capacity(head.len() + payload.len() + tail.len());
    body.extend_from_slice(head.as_bytes());
    body.extend_from_slice(&payload);
    body.extend_from_slice(tail.as_bytes());

    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/workspace/{id}/upload"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("build req");
    let resp = r.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["size_bytes"].as_u64(), Some(payload.len() as u64));
    assert_eq!(v["sha256"].as_str().unwrap(), expected_sha);
}

/// Legacy `kind` field is rejected unconditionally; old clients get
/// a 400 explaining the `{path, file}` shape.
#[tokio::test]
async fn workspace_upload_rejects_legacy_kind_field() {
    use axum::http::header;
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"k"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    let boundary = "----al-bigkind";
    let body = format!(
        "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"kind\"\r\n\r\n\
             head_mpk\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"x.mpk\"\r\n\r\n\
             DATA\r\n\
             --{boundary}--\r\n"
    );
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/workspace/{id}/upload"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("build req");
    let resp = r.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// `file` arriving before `path` is rejected with 400; the streaming
/// write needs the validated destination before allocating a tempfile.
#[tokio::test]
async fn workspace_upload_rejects_file_before_path() {
    use axum::http::header;
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::POST, "/workspace", Some(r#"{"name":"o"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    let boundary = "----al-orderbug";
    let body = format!(
        "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"x.bin\"\r\n\r\n\
             DATA\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"path\"\r\n\r\n\
             foo.bin\r\n\
             --{boundary}--\r\n"
    );
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/workspace/{id}/upload"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("build req");
    let resp = r.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn router_v1_nested_mounts_under_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/api/v1/health", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    // Without prefix -> 404.
    let resp = call(&r, Method::GET, "/health", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Uploading to a valid UUID that is NOT a workspace must return 404
/// with the envelope and leave no orphan directory.  Pre-fix the
/// route created `<root>/<uuid>/.tmp/` before the existence check,
/// leaving residue that later workspace listings would stumble over.
#[tokio::test]
async fn upload_to_nonexistent_workspace_is_404_no_orphan_dir() {
    use axum::http::header;

    let dir = tempfile::tempdir().unwrap();
    // `fresh_state` passes `dir/workspaces` as the FsService root;
    // per-id dirs nest under `<root>/workspaces/<id>/`.
    let workspace_root = dir.path().join("workspaces").join("workspaces");
    let r = router(fresh_state(dir.path()));

    const PHANTOM_ID: &str = "00000000-0000-4000-8000-000000000777";
    let phantom_dir = workspace_root.join(PHANTOM_ID);
    assert!(
        !phantom_dir.exists(),
        "phantom workspace dir must not exist before the upload",
    );

    let boundary = "----acousticslab-orphan";
    let body = format!(
        "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"path\"\r\n\r\n\
             datasets/cls/x.bin\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"x.bin\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n\
             DATA\r\n\
             --{boundary}--\r\n",
    );
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/workspace/{PHANTOM_ID}/upload"))
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let resp = r.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");

    assert!(
        !phantom_dir.exists(),
        "upload to nonexistent workspace left an orphan dir at {}",
        phantom_dir.display(),
    );
}

/// Malformed JSON body must surface as `{error, code: "bad_request"}`
/// instead of axum's plain-text 400; the wrapper extractors map
/// deserialization rejections into `ApiError::Bad`.
#[tokio::test]
async fn bad_json_body_returns_envelope_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/inference", Some("not-json {")).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    assert!(
        v["error"].as_str().unwrap().contains("invalid JSON body"),
        "envelope must surface the diagnosis; got {v}",
    );
}

/// Bad query-parameter types also envelope-wrap; `ApiQuery` maps
/// `Query<T>` rejections to `ApiError::Bad`.
#[tokio::test]
async fn bad_query_string_returns_envelope_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    // `min_version` is `Option<u64>`; "abc" can't deserialize.
    // `GET /mic` is the surviving consumer of `VersionQuery`.
    let resp = call(&r, Method::GET, "/mic?min_version=abc", None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}
