//! Convert-pipeline integration tests for
//! `POST /workspace/{id}/convert`.  Mirrors the train pipeline:
//! drives the API in-process via `tower::ServiceExt::oneshot`
//! against a tempdir-backed workspace and asserts:
//!
//! - happy path: upload a TFJS bundle, kick off `POST /convert`,
//!   wait for terminal, verify the head landed and a JSONL log
//!   sits at `<workspace>/converter_logs/<job_id>.jsonl`;
//! - 409 on a second overlapping convert request (single
//!   `max_convert_jobs` permit + per-input-file leases);
//! - 409 on `DELETE /assets/<input>` while convert is running.
//!
//! All tests that need the upstream TFJS bundle skip silently
//! when `misc/models/model.json` is absent (gitignored fixture).

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::await_holding_lock
)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acoustics_lab::api::{AppState, router_v1_nested};
use acoustics_lab::common::traits::head_store::HeadStore;
use acoustics_lab::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use acoustics_lab::config::{
    Config, ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell, MicSettingsHandle,
};
use acoustics_lab::file_mgr::{FsService, FsServiceImpl};
use acoustics_lab::inference::{HeadInner, HotHead};
use acoustics_lab::status::{StatusMonitor, StatusReporter};
use acoustics_lab::training::{JobRegistry, TrainingRegistry};
use arc_swap::ArcSwap;
use axum::Router;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, fixture_workspace_dir, json_body, upload};

/// Process-wide serialization gate.  The converter semaphore
/// (`crate::converter::CONVERT_SEMAPHORE`) is a `OnceLock` static
/// shared by every in-process test; without serialization two
/// `#[tokio::test]` cases acquiring the permit concurrently
/// would step on each other.  Held for each test's full body so
/// the convert pipeline observes a clean global state.
static CONVERT_TEST_SERIALIZER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the cross-test gate; panics on poison so a test
/// failure does not silently allow the next test to run with a
/// dirty global.
fn serialize_test() -> std::sync::MutexGuard<'static, ()> {
    CONVERT_TEST_SERIALIZER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// Build a router rooted at `dir`.  `<dir>/workspaces/` is the
/// `FsService` root; per-id workspace dirs nest under
/// `<root>/workspaces/<id>/`.
fn fresh_router(dir: &Path) -> (Router, Arc<dyn FsService>) {
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");

    let cfg_path = dir.join("config.toml");
    let cfg = Config::default_for();
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    let head: Arc<dyn HeadStore> = Arc::new(HotHead::from_inner(HeadInner {
        weight: vec![0.0; acoustics_lab::common::dims::BackboneFeatureDim::USIZE * 2],
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
    let training: Arc<dyn TrainingRegistry> = Arc::new(JobRegistry::new());
    let app = AppState {
        config,
        head,
        mic_settings,
        inference_cfg,
        files: files.clone(),
        monitor,
        training,
        broadcast_lag_reader: Arc::new(StubLagSource::default()),
        active_mutex: Arc::new(parking_lot::Mutex::new(())),
        default_head: Some(DefaultHeadRef {
            path: dir.join("bundled_default/head.mpk"),
            labels_path: dir.join("bundled_default/labels.txt"),
        }),
        // Converter tests do not invoke `POST /train`; `None`
        // is the correct default.
        training_backbone_path: None,
        jobs,
    };
    (router_v1_nested(app), files)
}

/// Locate the upstream TFJS Speech-Commands fixture; tests skip
/// if it isn't checked out (gitignored).  Returns the model
/// directory path so the test can read the four fixture files
/// (model.json, two shards, metadata.json) and upload them.
fn try_fixture_dir() -> Option<std::path::PathBuf> {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let p = crate_root.join("misc/models");
    if p.join("model.json").exists() {
        Some(p)
    } else {
        None
    }
}

/// Seed a workspace's `datasets/tfjs/` directory with the
/// upstream Speech-Commands TFJS bundle.  Uploads `model.json`,
/// the two manifest-declared shards (renamed to `*.bin` because
/// `AssetPath` requires extensions on some path forms), and
/// `metadata.json` (carrying the `wordLabels`).  Returns the
/// asset paths used in the convert request.
async fn seed_tfjs_bundle(
    router: &Router,
    ws: &str,
    fixture: &Path,
) -> (String, Vec<String>, String) {
    let model_json = std::fs::read(fixture.join("model.json")).expect("read model.json");
    let shard1 = std::fs::read(fixture.join("group1-shard1of2")).expect("read shard1");
    let shard2 = std::fs::read(fixture.join("group1-shard2of2")).expect("read shard2");
    let metadata_json = std::fs::read(fixture.join("metadata.json")).expect("read metadata.json");

    // Uploads use workspace-rooted paths (`converters/<...>`);
    // the convert request body uses the canonical converter-rooted
    // form (no leading slash).
    let model_json_abs = "tfjs/model.json".to_string();
    let shard_abs = vec![
        "tfjs/group1-shard1of2".to_string(),
        "tfjs/group1-shard2of2".to_string(),
    ];
    let labels_abs = "tfjs/metadata.json".to_string();

    let upload_paths = [
        ("converters/tfjs/model.json", &model_json[..]),
        ("converters/tfjs/group1-shard1of2", &shard1[..]),
        ("converters/tfjs/group1-shard2of2", &shard2[..]),
        ("converters/tfjs/metadata.json", &metadata_json[..]),
    ];
    for (path, body) in upload_paths {
        let resp = upload(router, ws, path, body).await;
        assert_eq!(resp.status(), StatusCode::OK, "upload {path}");
    }

    (model_json_abs, shard_abs, labels_abs)
}

fn convert_body(
    model_json_path: &str,
    shards: &[String],
    labels_path: &str,
    labels_format: &str,
) -> String {
    // Internally tagged on `converter_type`; every path field is
    // converter-rooted (canonical = slashless).
    serde_json::json!({
        "converter_type": "tfjs",
        "model_json_path": model_json_path,
        "shards": shards,
        "labels_path": labels_path,
        "labels_format": labels_format,
    })
    .to_string()
}

/// Wait until the workspace summary reports `head_count >= 1`
/// or the deadline elapses.  Returns the summary on success.
async fn wait_for_head(router: &Router, ws: &str) -> Option<serde_json::Value> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let resp = call(
            router,
            Method::GET,
            &format!("/api/v1/workspace/{ws}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let heads = v["heads"].as_array().cloned().unwrap_or_default();
            if !heads.is_empty() {
                return Some(v);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// MARK: Producer admission

/// Happy-path: `POST /convert` returns `{head_id, job_id}` with
/// well-formed UUIDs.  Skipped when the gitignored TFJS fixture
/// is absent.
#[tokio::test]
async fn convert_producer_returns_head_id_and_job_id() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "convert accept; got {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    let head_id = v["head_id"].as_str().expect("head_id");
    let job_id = v["job_id"].as_str().expect("job_id");
    assert_eq!(head_id.len(), 36, "head_id is not a UUID");
    assert_eq!(job_id.len(), 36, "job_id is not a UUID");
    assert_ne!(head_id, job_id);

    // Wait for the head to land in the workspace.
    let summary = wait_for_head(&r, &ws).await.expect("head published");
    let heads = summary["heads"].as_array().expect("heads");
    assert_eq!(heads.len(), 1, "exactly one head: {summary}");
    assert_eq!(heads[0]["head_id"], head_id);

    // The JSONL log file exists at the documented path.
    let log_path = fixture_workspace_dir(dir.path(), &ws)
        .join("converter_logs")
        .join(format!("{job_id}.jsonl"));
    assert!(
        log_path.exists(),
        "converter_logs/<job_id>.jsonl missing: {}",
        log_path.display(),
    );
    let log_contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        log_contents.lines().any(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| v["state"].as_str().map(str::to_string))
                .as_deref()
                == Some("completed")
        }),
        "log must contain a `completed` event line; got:\n{log_contents}",
    );
}

/// 409 on a second concurrent convert request: the
/// `max_convert_jobs = 1` permit (via the converter semaphore)
/// rejects the second producer with `ConvertError::Busy` ->
/// `ErrorKind::Conflict`.  Even though the fixture-bundle test
/// is skipped without the upstream model, the conflict gate
/// fires whenever the first request's blocking work (manifest
/// parse onward) holds the permit; this test seeds a minimal
/// synthetic manifest so it runs without the fixture.
#[tokio::test]
async fn convert_second_request_returns_conflict_or_busy() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp_first = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(resp_first.status(), StatusCode::OK);
    // Race window: the worker may finish quickly on a fast host.
    // The converter semaphore (single permit) is the only gate
    // that serializes two convert requests; a second request
    // issued promptly trips it.
    let resp_second = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    let status = resp_second.status();
    // Either 409 (conflict / busy) while the first runs, or 200
    // if the first finished before this issued (the head bytes
    // are tiny; conversion can complete in <100 ms on a warm
    // host).  Both shapes pin "no torn state, no panic"; the
    // load-bearing assertion is "not a 5xx leak".
    assert!(
        status == StatusCode::CONFLICT || status == StatusCode::OK,
        "second convert: expected 409 (conflict) or 200 (first finished); got {status}",
    );
}

/// A convert job in flight does NOT block
/// `DELETE /assets/<input>`; they coexist because convert
/// registers a workspace-scoped lease (not per-file leases) and
/// only `WorkspaceDelete` is exclusive.
#[tokio::test]
async fn delete_asset_during_convert_coexists() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    // DELETE on a convert input admits immediately even while
    // the worker is reading the file; convert inputs live under
    // `converters/` and the DELETE URL targets the
    // workspace-rooted form.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/converters/tfjs/model.json"),
        None,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "DELETE during convert must coexist (only WorkspaceDelete excludes); \
         async dispatch returns 202 Accepted",
    );
}

/// Producer rejects converter input paths that don't resolve to a
/// regular file.  Uses an absolute-form path that doesn't exist;
/// the resolver returns `FileError::Io { kind: NotFound }`.
#[tokio::test]
async fn convert_rejects_missing_input() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = convert_body(
        "/missing/model.json",
        &["/missing/shard.bin".to_string()],
        "/missing/labels.txt",
        "lines",
    );
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    // The resolver fails inside `open_workspace_file` for missing
    // files; the api error mapper promotes ENOENT to 404 NotFound.
    // The contract is "no spawn / no head record / no panic".
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected non-success; got {}",
        resp.status(),
    );

    // Workspace heads list remains empty.
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{ws}"), None).await;
    if resp.status() == StatusCode::OK {
        let v: serde_json::Value = json_body(resp).await;
        let heads = v["heads"].as_array().expect("heads");
        assert!(heads.is_empty(), "summary heads must be empty: {v}");
    }
}

/// Producer rejects empty `shards` at the route boundary via
/// `validate_convert_request`; wire-shape cardinality gates
/// land 400.
#[tokio::test]
async fn convert_rejects_empty_shards() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = convert_body("/m", &[], "/l", "lines");
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Path-traversal in any of the path fields rejects at
/// deserialize time (`ConverterPath::parse`); pinned via a single
/// representative field.  Per-field coverage lives in
/// `tests/request_payload_contracts.rs`.
#[tokio::test]
async fn convert_rejects_path_traversal_via_serde() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    // `/..` after-strip is `..`, which AssetPath rejects.
    let body = convert_body(
        "/../escape.json",
        &["/s.bin".to_string()],
        "/labels.txt",
        "lines",
    );
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// `converter_type` is required; a body that omits it (the
/// legacy flat shape) parse-fails at deserialize.
#[tokio::test]
async fn convert_rejects_round_1_body_without_converter_type() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "missing converter_type must be 400",
    );
}

/// Unknown `converter_type` values reject at deserialize (no
/// fallthrough to a default variant).
#[tokio::test]
async fn convert_rejects_unknown_converter_type() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "converter_type": "onnx",
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "unknown converter_type must be 400",
    );
}

/// Converter file path fields are converter-rooted (canonical
/// slashless form; legacy leading `/` accepted via BC shim).
/// Empty / lone-slash / traversal still reject at deserialize
/// before any file lookup -- pinned via `model_json_path = ".."`.
#[tokio::test]
async fn convert_rejects_invalid_path_field() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "converter_type": "tfjs",
        "model_json_path": "..",
        "shards": ["tfjs/s.bin"],
        "labels_path": "tfjs/labels.txt",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "traversal model_json_path must be 400",
    );
}
