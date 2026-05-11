//! Train pipeline integration tests for
//! `POST /workspace/{id}/train`.
//!
//! Asserts:
//! - The route accepts the flattened `TrainingCfg` body and
//!   returns `{head_id, job_id}` with well-formed UUIDs.
//! - The single-train-job invariant surfaces as HTTP 409 with
//!   `another_train_running` on a second concurrent request.
//! - The job holds only a workspace-scoped reference, so an
//!   overlapping `DELETE /assets/<...>` coexists; only a
//!   `WorkspaceDelete` would 409.
//! - Wrapper-shape bodies and out-of-range numerics reject
//!   before any job spawns.
//!
//! The end-to-end head-publish path is not exercised here -- it
//! requires a Burn-trainable wav fixture and adds tens of seconds
//! per run.  The post-train rotation is covered by
//! `tests/trained_head_rotation.rs` and the unit tests in
//! `modules/training.rs`.
//!
//! All routes are driven via `tower::ServiceExt::oneshot` against
//! a `tempfile::tempdir()`-backed workspace root, mirroring the
//! pattern in `tests/dataset_endpoints.rs`.

#![allow(clippy::disallowed_methods)]

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
use api_fixtures::{call, create_workspace, json_body, upload};

#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// Build a router rooted at `dir`, with a synthetic backbone
/// stub at `<root>/backbone/backbone.mpk`.  The stub is opaque
/// bytes -- the trainer fails at backbone-load time with the
/// producer-allocated job entering `JobState::Failed`, which is
/// sufficient for the producer-admission assertions in this
/// file.  No head record is committed on failure.
fn fresh_router(dir: &Path) -> Router {
    // `FsServiceImpl` roots at `dir` (not `dir/workspaces`) so
    // the producer's backbone resolution can find the stub at
    // `dir/backbone/backbone.mpk` below.  Per-workspace dirs
    // land at `dir/workspaces/<id>/`.  Production paths
    // (uploader, training, converter, head_rotation, staging)
    // self-mkdir the per-workspace subtree on first write, so
    // pre-creating `dir/workspaces/` here is purely defensive
    // -- the test still passes without it.
    std::fs::create_dir_all(dir.join("workspaces")).expect("workspace root");

    // Stub backbone fixture under <root>/backbone/.  The
    // producer admission gate stats it; the trainer's
    // `Backbone::load_mpk` will fail with a malformed-mpk
    // error, but that surfaces as `JobState::Failed` AFTER the
    // producer returns -- the admission contract is the unit
    // under test here.
    let backbone_dir = dir.join("backbone");
    std::fs::create_dir_all(&backbone_dir).expect("backbone dir");
    std::fs::write(backbone_dir.join("backbone.mpk"), b"stub-backbone-bytes")
        .expect("write backbone stub");

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
    // FsServiceImpl roots at `dir` so the producer's backbone
    // resolution (`files.root().join("backbone").join("backbone.mpk")`)
    // hits the stub above.  `workspaces/` is the per-workspace
    // subdirectory under the same root.
    let jobs = Arc::new(acoustics_lab::file_mgr::JobRegistry::new(
        acoustics_lab::file_mgr::JobRegistryCfg::default(),
    ));
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_and_jobs(
        dir.to_path_buf(),
        Default::default(),
        jobs.clone(),
    ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(JobRegistry::new());
    router_v1_nested(AppState {
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
    })
}

/// Seed a workspace with a two-class dataset.  The trainer walks
/// `<workspace>/datasets/` directly, so class folders live as
/// immediate children of `datasets/`.
async fn seed_dataset(router: &Router, ws: &str) {
    for cls in ["cat", "dog"] {
        for i in 0..2 {
            let resp = upload(
                router,
                ws,
                &format!("datasets/{cls}/sample{i}.bin"),
                b"stub-audio-bytes",
            )
            .await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "upload {cls}/sample{i}.bin failed: {}",
                resp.status(),
            );
        }
    }
}

/// The flattened [`TrainingCfg`] body: no wrapper, no
/// `dataset_path`.
fn train_body() -> String {
    serde_json::json!({
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
    })
    .to_string()
}

// MARK: Producer admission

/// Happy-path producer admission: a valid request returns
/// `{head_id, job_id}` with well-formed UUIDs.  The training
/// task itself will fail (the stub backbone won't load), but
/// the admission contract is satisfied at this point and the
/// API client has the head identifier it will use to query the
/// job.
#[tokio::test]
async fn train_producer_returns_head_id_and_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "train accept; got {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    let head_id = v["head_id"].as_str().expect("head_id");
    let job_id = v["job_id"].as_str().expect("job_id");
    // UUID-v4 shape: 36 chars `xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`.
    assert_eq!(head_id.len(), 36, "head_id is not a UUID");
    assert_eq!(job_id.len(), 36, "job_id is not a UUID");
    assert_ne!(head_id, job_id, "head_id and job_id must differ");
}

/// `max_train_jobs = 1`: a second train request while the first
/// is unfinished returns 409 with `another_train_running`.
#[tokio::test]
async fn train_second_request_rejects_another_train_running() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp_first = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp_first.status(),
        StatusCode::OK,
        "first train accept; got {}",
        resp_first.status(),
    );

    // Race window: the first train is queued on the daemon but
    // its `spawn_blocking` may not have woken yet.  Either way
    // the in-flight permit is held for the whole job lifetime,
    // so a second request issued promptly hits the gate.
    let resp_second = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp_second.status(), StatusCode::CONFLICT);
    let v: serde_json::Value = json_body(resp_second).await;
    assert_eq!(
        v["code"], "another_train_running",
        "expected `another_train_running`; body={v}",
    );
}

/// A train job in flight does NOT block a dataset
/// `DELETE /assets/<path>` in the same workspace; both register
/// only a workspace-scoped reference and `WorkspaceDelete` is the
/// only exclusive admission shape.
#[tokio::test]
async fn train_does_not_block_dataset_delete_in_same_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let _ = v["head_id"].as_str().expect("head_id");

    // DELETE on a path under `datasets/` admits even while a
    // train job is in flight; the async delete returns 200 with
    // a `{job_id}` body.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cat"),
        None,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "DELETE during train must coexist (only WorkspaceDelete excludes); \
         async dispatch returns 202 Accepted",
    );
    let v: serde_json::Value = json_body(resp).await;
    assert!(
        v["job_id"].as_str().is_some(),
        "DELETE response must carry job_id; body={v}"
    );
}

/// The train body is the flattened [`TrainingCfg`] (no wrapper,
/// no `dataset_path`); a wrapper shape parse-fails at
/// deserialize before any spawn.
#[tokio::test]
async fn train_rejects_round_1_wrapper_body() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let body = serde_json::json!({
        "dataset_path": "audio",
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
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "wrapper body must reject as 400",
    );

    // Workspace heads list remains empty.
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{ws}"), None).await;
    if resp.status() == StatusCode::OK {
        let v: serde_json::Value = json_body(resp).await;
        let heads = v["heads"].as_array().expect("heads array in summary");
        assert!(heads.is_empty(), "summary heads must be empty: {v}");
    }
}

/// Producer rejects an out-of-range `learning_rate` at the
/// route boundary, before any spawn; numeric gates land 400 with
/// `Categorized::UserInput`.
#[tokio::test]
async fn train_rejects_invalid_learning_rate() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let body = serde_json::json!({
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0,
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Once the in-flight job terminates (the stub backbone load
/// fails fast), a follow-up train request is admitted again.
/// Pins the "permit auto-releases on terminal state" contract.
#[tokio::test]
async fn train_admission_recycles_after_terminal_state() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().unwrap().to_string();

    // Poll the job state until it terminates.  The stub
    // backbone is opaque bytes; `Backbone::load_mpk` returns
    // an error promptly so the job transitions to `Failed`
    // within a few hundred ms.  Bound at 30 s to keep the test
    // resilient on slow hosts.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspace/{ws}/training/{job_id}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let state = v["state"].as_str().unwrap_or("");
            if state == "failed" || state == "completed" || state == "cancelled" {
                break;
            }
        }
        if Instant::now() >= deadline {
            panic!("training job did not terminate within 30 s");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Permit released; a second request is admitted.
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "permit not recycled");
}

/// The trainer writes a `started` event then a terminal event
/// (`failed` against the stub backbone here) into
/// `<workspace>/training_logs/<job_id>.jsonl`, readable through
/// `GET /assets/training_logs/<job_id>.jsonl`.
#[tokio::test]
async fn train_emits_started_then_terminal_jsonl_event() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspace/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().unwrap().to_string();

    // Wait for the stub backbone to fail so the terminal event
    // is on disk.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspace/{ws}/training/{job_id}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let state = v["state"].as_str().unwrap_or("");
            if state == "failed" || state == "completed" || state == "cancelled" {
                break;
            }
        }
        if Instant::now() >= deadline {
            panic!("training job did not terminate within 30 s");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Page the JSONL through the unified surface.  We assert
    // the producer wrote BOTH a `started` event and a terminal
    // event; a regression that only writes one (or neither) is
    // the failure mode this test pins.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/training_logs/{job_id}.jsonl?limit=10",),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().expect("events array");
    let states: Vec<&str> = events.iter().filter_map(|e| e["state"].as_str()).collect();
    // For diagnostic clarity if the assertion fails: dump the
    // on-disk JSONL.  fresh_router roots the FsService at
    // `dir`, so the workspace dir is `dir/workspaces/<id>/`.
    let log_path = dir
        .path()
        .join("workspaces")
        .join(
            acoustics_lab::common::ids::WorkspaceId::parse(&ws)
                .unwrap()
                .to_string(),
        )
        .join("training_logs")
        .join(format!("{job_id}.jsonl"));
    let body = std::fs::read_to_string(&log_path).unwrap_or_default();
    assert!(
        states.contains(&"started"),
        "expected `started` event; got {states:?}; on-disk body=\n{body}",
    );
    assert!(
        states
            .iter()
            .any(|s| matches!(*s, "completed" | "failed" | "cancelled")),
        "expected a terminal event; got {states:?}; on-disk body=\n{body}",
    );
}
