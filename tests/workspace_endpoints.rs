//! Integration tests for the workspace lifecycle routes:
//! `POST /workspace`, `GET /workspace`, `GET /workspace/{id}`,
//! `DELETE /workspace/{id}`.
//!
//! Boots the API router in-process (no daemon binary) over a
//! tempdir, drives the four routes via
//! `tower::ServiceExt::oneshot`, and asserts the wire shape +
//! post-delete state (404 on follow-up GET, eventual cleanup of
//! `<root>/.tmp/delete-workspace-*`).

#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use acoustics_lab::api::router_v1_nested;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, fresh_app_state, json_body};

/// End-to-end happy path:
/// `POST /workspace` -> id, `GET /workspace` -> contains id,
/// `GET /workspace/{id}` -> summary (id + name + created_at +
/// `workspace_revision { id: 0, .. }` + `heads: []`),
/// `DELETE /workspace/{id}` -> `{job_id}`, follow-up
/// `GET /workspace/{id}` -> 404, `<root>/.tmp/delete-workspace-*`
/// eventually swept clean.
#[tokio::test]
async fn workspace_lifecycle_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    // POST /workspace -> id
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"test"}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /workspace must succeed"
    );
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().expect("id is string").to_string();
    assert!(!id.is_empty(), "create response carries an id");
    assert_eq!(v["name"], "test");
    assert_eq!(
        v["workspace_revision"]["id"], 0,
        "fresh workspace has revision 0; body={v}"
    );

    // GET /workspace -> contains id
    let resp = call(&r, Method::GET, "/api/v1/workspace", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["workspaces"].as_array().expect("workspaces array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], id);
    assert_eq!(entries[0]["name"], "test");

    // GET /workspace/{id} -> summary shape
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["id"], id);
    assert_eq!(v["name"], "test");
    assert!(v["created_at"].as_str().is_some(), "created_at present");
    assert_eq!(v["workspace_revision"]["id"], 0);
    let heads = v["heads"].as_array().expect("heads array");
    assert!(heads.is_empty(), "fresh workspace has no heads");
    // Defence against accidental asset leak: the summary returns
    // the lifecycle shape only (no `assets` key).
    assert!(v.get("assets").is_none(), "summary must not echo assets");

    // DELETE /workspace/{id} -> {job_id}
    let resp = call(&r, Method::DELETE, &format!("/api/v1/workspace/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().expect("job_id is string");
    assert!(!job_id.is_empty(), "delete response carries job_id");

    // Follow-up GET /workspace/{id} -> 404 (the cache and the
    // workspace dir entry are both gone immediately; the drain
    // runs in the background).
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{id}"), None).await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "deleted workspace must surface 404 on follow-up GET",
    );

    // `<root>/.tmp/delete-workspace-*` is eventually cleaned up.
    // The drain runs on the tokio blocking pool; poll up to ~5 s
    // (a fresh workspace has no datasets, so the drain finishes
    // in milliseconds).
    let staging = root.join("workspaces").join(".tmp");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let any_residue = if staging.exists() {
            std::fs::read_dir(&staging)
                .map(|d| {
                    d.filter_map(Result::ok).any(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with("delete-workspace-")
                    })
                })
                .unwrap_or(false)
        } else {
            false
        };
        if !any_residue {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "workspace delete drain did not converge within 5 s; staging={}",
                staging.display()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// MARK: create_with_tags + PATCH

#[tokio::test]
async fn create_workspace_accepts_optional_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"scoped","tags":["  field-recordings ","pet-noises"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // Tags are trimmed before persist.
    assert_eq!(v["tags"][0], "field-recordings");
    assert_eq!(v["tags"][1], "pet-noises");
    assert_eq!(v["tags"].as_array().map(|a| a.len()), Some(2));
}

#[tokio::test]
async fn create_workspace_rejects_invalid_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"scoped","tags":["a/b"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn patch_workspace_renames_and_retags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    // Create with one tag.
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"orig","tags":["a"]}"#),
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    let prev_revision = v["workspace_revision"]["id"].as_u64().unwrap();
    // PATCH name + tags.
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspace/{id}"),
        Some(r#"{"name":"renamed","tags":["b","c"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["name"], "renamed");
    assert_eq!(v["tags"][0], "b");
    assert_eq!(v["tags"][1], "c");
    // Revision unchanged: name + tag edits do not bump it.
    assert_eq!(v["workspace_revision"]["id"].as_u64(), Some(prev_revision));
    // Follow-up GET reflects the new state.
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{id}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["name"], "renamed");
}

#[tokio::test]
async fn patch_workspace_rejects_empty_body() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"x"}"#),
    )
    .await;
    let id = json_body::<serde_json::Value>(resp).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    // Empty body -> 400.
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspace/{id}"),
        Some(r#"{}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn patch_workspace_rejects_name_collision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    // Create two workspaces.
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"taken"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"free"}"#),
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    // Attempt to rename `free` to `Taken` (case-insensitive
    // collision) -- 409.
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspace/{id}"),
        Some(r#"{"name":"Taken"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn patch_workspace_returns_404_for_unknown_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    // Use a valid-looking UUID-v4 that doesn't correspond to any
    // workspace.
    let phantom = "11111111-2222-4333-8444-555555555556";
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspace/{phantom}"),
        Some(r#"{"name":"ghost"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Names are case-insensitive under Unicode case folding via
/// `str::to_lowercase`, not just ASCII.  Pre-relax behavior used
/// `to_ascii_lowercase` which left non-ASCII letter-pairs
/// distinct (`Café` and `café` would have both been allowed).
/// Pin the relaxed behavior so a future regression that drops
/// back to ASCII folding fails this test loudly.
#[tokio::test]
async fn create_workspace_rejects_unicode_case_collision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    // Latin-extended pair: é (U+00E9) vs É (U+00C9).
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"Café"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspace",
        Some(r#"{"name":"café"}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "lowercase variant of an existing Latin-extended name must collide \
         (Unicode case folding, not ASCII-only)",
    );
}
