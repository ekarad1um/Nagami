//! Integration tests for the active-head endpoints
//! (`POST /active`, `GET /active`).
//!
//! Drives the API in-process via `tower::ServiceExt::oneshot`
//! against a tempdir-rooted workspace.  The bundled-default
//! fixture is a hand-rolled `head.mpk` (with the ACSTHEAD
//! header) + `labels.txt` written into a per-test directory so
//! the daemon's `bundled_default_dir` can point at it without
//! depending on the in-tree `misc/heads/00000000-default/`
//! checked-in fixture.
//!
//! Coverage:
//! - `POST /active {default: true}` from a clean root: writes
//!   the bundled-default generation, runtime head id matches.
//! - `POST /active {default: true}` is idempotent (re-issuing
//!   produces a fresh activation_id but consistent body).
//! - `POST /active {workspace_id, head_id}` activates a trained
//!   head; `GET /active` reflects the head provenance.
//! - Deleting the source workspace surfaces
//!   `source_workspace_alive: false` while the active runtime
//!   keeps serving (independent bytes).
//! - `POST /active {default: true}` after a head activation
//!   resets the runtime.
//! - 404 when `head_id` is not in the workspace.
//! - Only current + previous generations remain after multiple
//!   activations.

#![allow(clippy::disallowed_methods)]

use std::path::Path;

use acoustics_lab::api::{AppState, router};
use acoustics_lab::common::ids::{HeadId, WorkspaceId};
use acoustics_lab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
};
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, fresh_app_state, json_body};

/// Build an API state pointing at a tempdir.  The
/// `bundled_default_dir` is staged with a real `head.mpk` (via
/// the in-tree Burn recorder + ACSTHEAD header) so
/// `POST /active {default: true}` exercises the full pipeline.
/// The remaining wiring is shared with the other integration
/// binaries via [`fresh_app_state`].
fn fresh_state(dir: &Path) -> AppState {
    let mut state = fresh_app_state(dir);
    state.bundled_default_dir = stage_bundled_default(dir);
    state
}

/// Stage a deployment-bundled default fixture under
/// `<dir>/bundled_default/`.  The `head.mpk` is real (Burn
/// recorder + ACSTHEAD header) so the runtime preload via
/// `HotHead::load` succeeds.
fn stage_bundled_default(dir: &Path) -> std::path::PathBuf {
    use acoustics_lab::common::head_header::write_with_payload;
    use acoustics_lab::model::Head;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    let bundled = dir.join("bundled_default");
    std::fs::create_dir_all(&bundled).unwrap();
    // Burn's recorder appends `.mpk` to the path stem.
    let raw_stem = bundled.join("raw");
    let raw_mpk = bundled.join("raw.mpk");
    let device: burn::tensor::Device<NdArray<f32>> = Default::default();
    let head = Head::<NdArray<f32>>::new(2, &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder.record(head.into_record(), raw_stem).unwrap();
    let payload = std::fs::read(&raw_mpk).unwrap();
    let mut composed = std::fs::File::create(bundled.join("head.mpk")).unwrap();
    write_with_payload(
        &mut composed,
        acoustics_lab::common::dims::BackboneFeatureDim::USIZE as u32,
        2,
        &payload,
    )
    .unwrap();
    drop(composed);
    std::fs::write(bundled.join("labels.txt"), "alpha\nbeta\n").unwrap();
    // Cleanup: drop the raw bytes so the fixture only exposes
    // `head.mpk` + `labels.txt`.
    let _ = std::fs::remove_file(&raw_mpk);
    bundled
}

/// Stage a real trained head into a hand-rolled workspace
/// directory.  This bypasses `FsService::create` so the
/// FsService's eager cache lazy-loads from the just-written
/// `workspace.json` + `heads.json` on the activation route's
/// first `summary` call -- the create-then-publish flow would
/// pin a stale cell since `publish_trained_head` updates a
/// caller-supplied [`WorkspaceCacheCell`] separate from the
/// FsService's internal map.
fn publish_one_trained_head(
    state: &AppState,
    workspace_id: acoustics_lab::common::ids::WorkspaceId,
) -> HeadId {
    use acoustics_lab::common::head_header::write_with_payload;
    use acoustics_lab::file_mgr::schema::{
        head_artifact_path, head_manifest_path, heads_dir, workspace_dir_for, write_head_index,
        write_head_manifest, write_workspace_core,
    };
    use acoustics_lab::model::Head;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    let workspace_dir = workspace_dir_for(state.files.root(), &workspace_id);
    std::fs::create_dir_all(&workspace_dir).unwrap();
    std::fs::create_dir_all(workspace_dir.join(".tmp")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("datasets")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("training_logs")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("converter_logs")).unwrap();
    std::fs::create_dir_all(heads_dir(&workspace_dir)).unwrap();

    // Build a real head.mpk with ACSTHEAD header.
    let raw_stem = workspace_dir.join(".tmp").join("raw");
    let raw_mpk = workspace_dir.join(".tmp").join("raw.mpk");
    let device: burn::tensor::Device<NdArray<f32>> = Default::default();
    let head = Head::<NdArray<f32>>::new(3, &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder.record(head.into_record(), raw_stem).unwrap();
    let payload = std::fs::read(&raw_mpk).unwrap();
    std::fs::remove_file(&raw_mpk).ok();

    let head_id: HeadId = HeadId::new();
    let mpk_path = head_artifact_path(&workspace_dir, head_id);
    let mut composed = std::fs::File::create(&mpk_path).unwrap();
    write_with_payload(
        &mut composed,
        acoustics_lab::common::dims::BackboneFeatureDim::USIZE as u32,
        3,
        &payload,
    )
    .unwrap();
    drop(composed);
    let mpk_bytes = std::fs::read(&mpk_path).unwrap();
    let sha256 = sha256_hex(&mpk_bytes);

    let workspace_revision = WorkspaceRevision {
        id: 0,
        at: "2026-05-07T12:00:00Z".to_string(),
    };
    let manifest = HeadManifest {
        head_id,
        workspace_id,
        workspace_revision: workspace_revision.clone(),
        sha256: sha256.clone(),
        n_classes: 3,
        size_bytes: mpk_bytes.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
    };
    write_head_manifest(&workspace_dir, &manifest).unwrap();
    // Sanity: the manifest writer wrote the file exactly where
    // the activation primitive will read it.
    assert!(head_manifest_path(&workspace_dir, head_id).is_file());

    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id,
        workspace_revision: workspace_revision.clone(),
        sha256,
        n_classes: 3,
        size_bytes: mpk_bytes.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    write_head_index(&workspace_dir, &idx).unwrap();

    let core = WorkspaceCore {
        id: workspace_id,
        name: "main".to_string(),
        tags: Vec::new(),
        created_at: "2026-05-07T12:34:56Z".to_string(),
        workspace_revision,
        head_count: 1,
    };
    write_workspace_core(&workspace_dir, &core).unwrap();
    head_id
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; d.len() * 2];
    for (i, &b) in d.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    String::from_utf8(out).unwrap()
}

/// `POST /active {default: true}` writes the bundled-default
/// generation; the response body matches the manifest shape.
#[tokio::test]
async fn post_active_default_writes_bundled_generation() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "status was {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");
    assert_eq!(
        v["runtime_head_id"],
        acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR,
    );
    assert_eq!(v["n_classes"], 2);
    assert!(v["sha256"].as_str().unwrap().len() == 64);
    assert!(v["activation_id"].is_string());
}

/// `POST /active {default: true}` is idempotent (force-set):
/// re-issuing produces a fresh `activation_id` but consistent
/// body fields (origin / runtime_head_id / n_classes match).
#[tokio::test]
async fn post_active_default_is_idempotent_force_set() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let first: serde_json::Value = {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        json_body(resp).await
    };
    let second: serde_json::Value = {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        json_body(resp).await
    };
    assert_ne!(first["activation_id"], second["activation_id"]);
    assert_eq!(first["runtime_head_id"], second["runtime_head_id"]);
    assert_eq!(first["origin"], second["origin"]);
    assert_eq!(first["n_classes"], second["n_classes"]);
}

/// `POST /active {workspace_id, head_id}` activates a workspace
/// head; `GET /active` reflects the provenance.
#[tokio::test]
async fn post_active_head_then_get_active_returns_head_origin() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "status was {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_id"], ws_id.to_string());
    assert_eq!(v["n_classes"], 3);

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_alive"], true);
}

/// After the source workspace is deleted, `GET /active` still
/// serves the previously-activated head (independent bytes) and
/// surfaces `source_workspace_alive: false`.
#[tokio::test]
async fn get_active_reports_source_workspace_alive_false_after_workspace_delete() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555541").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let workspace_dir =
        acoustics_lab::file_mgr::schema::workspace_dir_for(state.files.root(), &ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Simulate workspace delete by removing the dir directly.
    // The `FsService::start_delete_workspace` async path would
    // have done this in batches; the active generation owns
    // independent bytes so the runtime is unaffected.
    std::fs::remove_dir_all(&workspace_dir).unwrap();

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_alive"], false);
}

/// `POST /active {default: true}` after a head activation
/// rolls the runtime back to the bundled default.
#[tokio::test]
async fn post_active_default_after_head_resets_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");

    let resp = call(&r, Method::GET, "/active", None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");
    assert_eq!(
        v["runtime_head_id"],
        acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR,
    );
}

/// Activating a head id that's not in the workspace's
/// `heads.json` returns 404.
#[tokio::test]
async fn post_active_head_404_when_head_missing_from_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let _head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let unknown = "00000000-0000-4000-8000-000000000099";
    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": unknown,
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");
}

/// `POST /active` retains only the current + previous
/// generations after multiple activations.
#[tokio::test]
async fn post_active_prunes_to_two_generations() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    for _ in 0..4 {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
    let generations_dir = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations");
    let count = std::fs::read_dir(&generations_dir).unwrap().count();
    assert!(
        count <= 2,
        "expected at most 2 retained generations, observed {count} under {}",
        generations_dir.display()
    );
}

/// `POST /active` rejects an unknown body shape with 422 (the
/// `ApiJson` extractor's `deny_unknown_fields` posture).  The
/// `untagged` enum tries each arm; if neither parses, the
/// extractor surfaces the error.
#[tokio::test]
async fn post_active_rejects_unknown_body_shape() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(
        &r,
        Method::POST,
        "/active",
        Some(r#"{"unexpected_field": "x"}"#),
    )
    .await;
    assert!(
        resp.status() == StatusCode::UNPROCESSABLE_ENTITY
            || resp.status() == StatusCode::BAD_REQUEST,
        "expected 4xx for unknown body shape, got {}",
        resp.status()
    );
}

/// `POST /active` rejects `default: false` with 400 (the
/// untagged Default arm parses but the route refuses the false
/// value with an explicit Bad).
#[tokio::test]
async fn post_active_rejects_default_false() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": false}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

// MARK: active-head shape pins
//
// Lock the wire shape end-to-end -- on disk
// (`generations/<id>/manifest.json`) and on the wire
// (`POST /active` + `GET /active` responses) -- so a future
// cascade re-adding the legacy `source_dataset_revision` alias
// parse-fails at the test boundary.

/// The on-disk active manifest for a Head-origin activation
/// carries exactly `origin`, `source_workspace_id`,
/// `source_head_id`, `workspace_revision`, `runtime_head_id`,
/// `sha256`, `labels_sha256`, `n_classes`, `labels`,
/// `activated_at` -- never the legacy `source_dataset_revision`.
#[tokio::test]
async fn active_head_manifest_on_disk_carries_field_set() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555548").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let activation_id = v["activation_id"].as_str().expect("activation_id");

    // Read the on-disk manifest as an opaque map -- going through
    // `read_active_manifest` would be a tautology against the same
    // typed struct that produced the bytes; the on-disk wire shape
    // is the actual contract.
    let manifest_path = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations")
        .join(activation_id)
        .join("manifest.json");
    assert!(
        manifest_path.is_file(),
        "active generation manifest must land on disk at {}",
        manifest_path.display(),
    );
    let bytes = std::fs::read(&manifest_path).unwrap();
    let m: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = m.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "origin",
        "source_workspace_id",
        "source_head_id",
        "workspace_revision",
        "runtime_head_id",
        "sha256",
        "labels_sha256",
        "n_classes",
        "labels",
        "activated_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "active Head-origin manifest must carry exactly the field set; got {actual:?}",
    );
    for forbidden in [
        "source_dataset_revision",
        "dataset_revision_at_train",
        "dataset_path",
        "training_cfg",
        "training_cfg_sha256",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "legacy field {forbidden:?} must not appear",
        );
    }
    // The `workspace_revision` sub-object carries exactly `id` and `at`.
    let rev = obj["workspace_revision"]
        .as_object()
        .expect("workspace_revision is a sub-object");
    let rev_keys: std::collections::BTreeSet<&str> = rev.keys().map(String::as_str).collect();
    let expected_rev: std::collections::BTreeSet<&str> = ["id", "at"].into_iter().collect();
    assert_eq!(
        rev_keys, expected_rev,
        "active manifest's workspace_revision sub-object must carry exactly id + at",
    );
    assert_eq!(obj["origin"].as_str(), Some("head"));
    // Provenance fields point at the source.
    assert_eq!(
        obj["source_workspace_id"].as_str(),
        Some(ws_id.to_string().as_str())
    );
    assert_eq!(
        obj["source_head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
    assert_eq!(
        obj["runtime_head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
}

/// The on-disk active manifest for a Default-origin activation
/// carries no `source_*` fields and no `workspace_revision`.
#[tokio::test]
async fn active_default_manifest_on_disk_carries_field_subset() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let activation_id = v["activation_id"].as_str().expect("activation_id");

    let manifest_path = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations")
        .join(activation_id)
        .join("manifest.json");
    let bytes = std::fs::read(&manifest_path).unwrap();
    let m: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = m.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    // Default origin omits the source-* fields and
    // `workspace_revision`.
    let expected: std::collections::BTreeSet<&str> = [
        "origin",
        "runtime_head_id",
        "sha256",
        "labels_sha256",
        "n_classes",
        "labels",
        "activated_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "active Default-origin manifest must carry exactly the field subset; got {actual:?}",
    );
    for forbidden in [
        "source_workspace_id",
        "source_head_id",
        "workspace_revision",
        "source_dataset_revision",
        "dataset_revision_at_train",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "Default-origin manifest must not carry {forbidden:?}",
        );
    }
    assert_eq!(obj["origin"].as_str(), Some("default"));
    assert_eq!(
        obj["runtime_head_id"].as_str(),
        Some(acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR),
    );
}

/// `POST /active` and `GET /active` Head-origin response bodies
/// carry `workspace_revision` and never the legacy
/// `source_dataset_revision` alias.
#[tokio::test]
async fn active_response_carries_workspace_revision_field() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555549").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    // POST response.
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(
        v["workspace_revision"].is_object(),
        "POST /active Head-origin response must carry `workspace_revision`; body={v}",
    );
    let rev = v["workspace_revision"].as_object().unwrap();
    assert!(rev.contains_key("id"), "workspace_revision must carry `id`");
    assert!(rev.contains_key("at"), "workspace_revision must carry `at`");
    assert!(
        v.get("source_dataset_revision").is_none(),
        "legacy `source_dataset_revision` must not appear; body={v}",
    );

    // GET response.
    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["workspace_revision"].is_object());
    assert!(
        v.get("source_dataset_revision").is_none(),
        "GET /active must not surface legacy alias; body={v}",
    );
    assert_eq!(v["source_workspace_alive"], true);
}

/// A hand-staged manifest with the legacy `source_dataset_revision`
/// (instead of `workspace_revision`) parse-fails through
/// `read_active_manifest`: the legacy alias deserializes as an
/// unknown key (silently dropped) and the required
/// `workspace_revision` is absent, so serde surfaces the missing
/// field.
#[tokio::test]
async fn legacy_active_manifest_shape_parse_fails_on_read() {
    use acoustics_lab::file_mgr::schema::read_active_manifest;
    let dir = tempfile::tempdir().unwrap();
    // Stage a fresh active layout WITHOUT going through the
    // route, then drop a legacy-shape manifest under it and ask
    // the schema reader to consume it.  Both `GET /active` and
    // boot recovery consume this reader; both must fail closed
    // on a legacy body.
    let root = dir.path().join("workspaces");
    std::fs::create_dir_all(&root).unwrap();
    let activation_id = "11111111-2222-4333-8444-555555555560";
    let gen_dir = root.join("active").join("generations").join(activation_id);
    std::fs::create_dir_all(&gen_dir).unwrap();

    let round_1_body = serde_json::json!({
        "origin": "head",
        "source_workspace_id": "11111111-2222-4333-8444-555555555548",
        "source_head_id": "11111111-2222-4333-8444-555555555540",
        // Legacy alias; the schema now requires
        // `workspace_revision`.
        "source_dataset_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
        "runtime_head_id": "11111111-2222-4333-8444-555555555540",
        "sha256": "deadbeef",
        "labels_sha256": "cafef00d",
        "n_classes": 3,
        "labels": ["a", "b", "c"],
        "activated_at": "2026-05-07T12:34:56Z",
    });
    std::fs::write(
        gen_dir.join("manifest.json"),
        serde_json::to_vec(&round_1_body).unwrap(),
    )
    .unwrap();

    let res = read_active_manifest(&root, activation_id);
    assert!(
        res.is_err(),
        "legacy active manifest body must parse-fail (missing required `workspace_revision`); got {res:?}",
    );
}

/// `GET /active` on a fresh root (no `current.json`) returns 404.
#[tokio::test]
async fn get_active_returns_404_on_fresh_root() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Concurrent `POST /active` requests: every commit must be
/// atomic and the on-disk surviving generations must include
/// the directory `current.json` points at.  Two invariants
/// are at stake -- both span the whole activation worker
/// (read → publish → install → prune):
///
/// 1. publish + install must not reorder (otherwise
///    `current.json` and the runtime `HotHead` diverge);
/// 2. prune must not run while a peer can publish a fresh
///    generation (otherwise the peer's directory is outside
///    this request's `keep` and gets deleted, leaving
///    `current.json` dangling).
///
/// Asserts: all requests succeed, activation_ids are unique,
/// `GET /active` resolves a present generation, on-disk
/// retention is at most {current, previous}.
///
/// This is a defensive guard, not a deterministic catcher --
/// the in-process `oneshot` race window is narrow enough that
/// a buggy build can pass.  A failing run is a real regression.
/// Loom would be the deterministic tool.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_active_default_concurrent_requests_all_succeed() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let root = state.files.root().to_path_buf();
    let r = router(state);

    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let r = r.clone();
        handles.push(tokio::spawn(async move {
            let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "concurrent activation rejected (status {})",
                resp.status(),
            );
            let v: serde_json::Value = json_body(resp).await;
            v["activation_id"].as_str().unwrap().to_string()
        }));
    }
    let mut activation_ids = Vec::with_capacity(N);
    for h in handles {
        activation_ids.push(h.await.expect("join"));
    }

    // (b) Every request returned a unique activation_id (staging
    // generates a fresh UUID per request).
    let mut sorted = activation_ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        activation_ids.len(),
        "duplicate activation_ids returned across concurrent requests: {activation_ids:?}",
    );

    // (c) `GET /active` succeeds -- the current generation
    // directory still exists on disk (i.e. a peer prune did not
    // delete the dir `current.json` points at).
    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /active 404'd after concurrent activations -- current generation \
         directory missing on disk",
    );
    let v: serde_json::Value = json_body(resp).await;
    let final_id = v["activation_id"].as_str().unwrap().to_string();
    assert!(
        activation_ids.contains(&final_id),
        "final current.json activation_id {final_id:?} not in any of the {} \
         concurrent responses {activation_ids:?}",
        activation_ids.len(),
    );

    // (d) Prune contract: retain set is {current, previous}, so
    // the generations dir holds at most 2 entries after the
    // dust settles.  Larger = prune did not run; smaller and
    // missing-current = dangling pointer (caught by (c)).
    let generations = root.join("active").join("generations");
    let surviving: Vec<String> = std::fs::read_dir(&generations)
        .expect("read_dir generations")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    assert!(
        surviving.len() <= 2,
        "after {N} concurrent activations the generations dir has {} entries; \
         expected <=2 (current + previous): {surviving:?}",
        surviving.len(),
    );
    assert!(
        surviving.contains(&final_id),
        "current generation {final_id:?} missing from disk after concurrent prune; \
         on-disk surviving = {surviving:?}",
    );
}
