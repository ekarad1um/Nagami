//! Integration tests for the dataset routes:
//!
//! - `GET    /workspace/{id}/assets` (paginated direct-child listing)
//! - `GET    /workspace/{id}/assets/{*path}` (file stream, dir
//!   listing, JSONL page, or byte-range slice)
//! - `PUT    /workspace/{id}/assets/{*path}` (raw-body upload)
//! - `DELETE /workspace/{id}/assets/{*path}` (always-async tombstone+
//!   stage+drain across all four mutable trees)
//!
//! Boots the API router in-process via `tower::ServiceExt::oneshot`,
//! drives the four routes against a tempdir-backed workspace, and
//! asserts the wire shape + the path-traversal negatives.

#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use acoustics_lab::api::router_v1_nested;
use axum::body::to_bytes;
use axum::http::{Method, StatusCode, header};
use axum::response::Response;

mod api_fixtures;
use api_fixtures::{
    call, create_workspace, fixture_workspace_dir, fresh_app_state, json_body, upload,
};

async fn body_bytes(resp: Response) -> Vec<u8> {
    to_bytes(resp.into_body(), 1 << 22)
        .await
        .expect("body")
        .to_vec()
}

// MARK: Happy-path

#[tokio::test]
async fn dataset_happy_path_upload_list_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    let ws = create_workspace(&r, "main").await;
    let bytes = b"audio body";
    // Upload to a workspace-rooted nested path.
    let resp = upload(&r, &ws, "datasets/audio_dataset/cat/sample.wav", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "datasets/audio_dataset/cat/sample.wav");
    assert_eq!(v["size_bytes"], bytes.len());
    assert_eq!(v["workspace_revision_id"], 1);
    assert_eq!(v["sha256"].as_str().unwrap().len(), 64);

    // GET /assets lists workspace-root direct children.  A fresh
    // workspace has only `workspace.json` + `heads.json` on disk;
    // every leaf subdir (`datasets/`, `converters/`, `heads/`,
    // `training_logs/`, `converter_logs/`, `.tmp/`) is created
    // lazily by the writer that touches it.  After the upload
    // above, `datasets/` exists with the sample inside; the
    // converter / log dirs remain absent until a producer runs.
    // `.tmp/` is excluded by the listing regardless.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    let names: Vec<&str> = entries.iter().filter_map(|e| e["name"].as_str()).collect();
    assert!(names.contains(&"datasets"), "datasets/ in {names:?}");
    assert!(
        !names.contains(&"converters"),
        "converters/ must be absent until a converter has run: {names:?}",
    );
    assert!(
        !names.contains(&".tmp"),
        ".tmp/ must be excluded from listings: {names:?}"
    );

    // GET /assets/datasets/audio_dataset -> directory listing.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/audio_dataset"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["name"], "cat");

    // GET /assets/datasets/audio_dataset/cat/sample.wav ->
    // file stream.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/audio_dataset/cat/sample.wav"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "audio/wav");
    let received = body_bytes(resp).await;
    assert_eq!(received, bytes);

    // DELETE /assets/datasets/audio_dataset/cat -> 202 Accepted
    // + {job_id}.  Async dispatch: the rename + tombstone are
    // durable but the staged drain runs in the background.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/datasets/audio_dataset/cat"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["job_id"].as_str().is_some(), "job_id present; body={v}");

    // Eventual 404 on the deleted file.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspace/{ws}/assets/datasets/audio_dataset/cat/sample.wav"),
            None,
        )
        .await;
        if resp.status() == StatusCode::NOT_FOUND {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "deleted asset still observable after 5 s; status={}",
                resp.status(),
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The `converters/` tree accepts the same upload + delete
/// dispatcher as `datasets/`; the only differences are the
/// tombstone variant (`Converter` vs `Dataset`), the `JobType`
/// (`ConverterDelete` vs `DatasetDelete`), and the on-disk
/// subdirectory.
#[tokio::test]
async fn converters_happy_path_upload_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let bytes = br#"{"format":"tfjs","weights":[]}"#;
    let resp = upload(&r, &ws, "converters/tfjs/model.json", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "converters/tfjs/model.json");
    assert_eq!(v["size_bytes"], bytes.len());

    // GET /assets/converters/tfjs/model.json streams the file.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/converters/tfjs/model.json"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "application/json");
    let received = body_bytes(resp).await;
    assert_eq!(received, bytes);

    // DELETE on the converters tree returns 202 Accepted +
    // {job_id} (async dispatch); the file is eventually 404.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/converters/tfjs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspace/{ws}/assets/converters/tfjs/model.json"),
            None,
        )
        .await;
        if resp.status() == StatusCode::NOT_FOUND {
            break;
        }
        if Instant::now() >= deadline {
            panic!("converter delete drain did not converge");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Paths whose top-level isn't `datasets/` or `converters/`
/// reject with HTTP 400 by the upload + delete dispatchers.
#[tokio::test]
async fn upload_rejects_non_mutable_top_level() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for bad in [
        "heads/x.mpk",
        "training_logs/log.jsonl",
        "scratch/file.bin",
        "datasets",
        "converters",
    ] {
        let resp = upload(&r, &ws, bad, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "upload `{bad}` must reject; got {}",
            resp.status()
        );
    }
}

// MARK: path-traversal negatives

#[tokio::test]
async fn upload_rejects_path_traversal_variants() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // The empty-path case (`""`) is omitted from this list because
    // it now matches a different surface: with the URL-wildcard
    // shape `PUT /assets/{*path}`, an empty path produces the URL
    // `/assets/` which doesn't match the wildcard route; the
    // helper-built URL collapses to `/assets`, which has only `GET`
    // registered, so the wire surface is 405 (method not allowed),
    // not 400.  See the `workspace_upload_without_path_returns_405`
    // unit test.
    for (label, bad_path) in [
        (".. literal", ".."),
        (".. with subpath", "../etc/passwd"),
        ("absolute", "/abs"),
        ("interior dotdot", "foo/../bar"),
        ("url-encoded ..", "%2E%2E%2Fetc"),
        ("backslash", "foo\\bar"),
        ("trailing slash", "foo/"),
        ("double slash", "foo//bar"),
        ("nul byte", "foo\0bar"),
        ("control byte", "foo\nbar"),
        ("non-ascii", "caf\u{00e9}/foo"),
    ] {
        let resp = upload(&r, &ws, bad_path, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "{label} ({bad_path:?}) must reject with 400; got {}",
            resp.status(),
        );
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(v["code"], "bad_request", "{label} envelope code; body={v}");
    }
}

#[tokio::test]
async fn get_and_delete_reject_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // GET /assets/.. -> 400.  axum 0.8's wildcard URL-decodes
    // before the handler sees it; the parser then catches the
    // literal `..` via `LeadingDot`.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/.."),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    // DELETE /assets/.. -> 400.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/.."),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// MARK: Revision bump

#[tokio::test]
async fn upload_bumps_workspace_revision() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // GET summary before upload -> revision 0.
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{ws}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["workspace_revision"]["id"], 0);
    // Upload bumps to 1.
    let resp = upload(&r, &ws, "datasets/cls/a.json", br#"{"k":1}"#).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(&r, Method::GET, &format!("/api/v1/workspace/{ws}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["workspace_revision"]["id"], 1);
}

// MARK: MIME table pin

#[tokio::test]
async fn mime_types_match_redesign_section_7() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for (filename, expected_ct) in [
        ("audio.wav", "audio/wav"),
        ("manifest.json", "application/json"),
        ("labels.txt", "text/plain; charset=utf-8"),
        ("blob.bin", "application/octet-stream"),
    ] {
        let upload_path = format!("datasets/cls/{filename}");
        let resp = upload(&r, &ws, &upload_path, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "upload {upload_path} succeeded"
        );
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspace/{ws}/assets/{upload_path}"),
            None,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert_eq!(ct, expected_ct, "MIME for {filename}");
    }
}

/// `GET /assets[?path=]` carries `mtime` (RFC3339, UTC) for every
/// listed entry — files and directories alike.  Pinned because
/// clients use mtime to sort durable JSONL log files by recency
/// once the unified surface absorbs the dedicated log routes;
/// this test surfaces a regression if a future refactor strips
/// the field from `DatasetEntry` or stops populating it.
#[tokio::test]
async fn list_assets_carries_mtime_per_entry() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // One real upload so the `datasets/` subdir contains a file
    // (file branch) AND so a non-empty subtree exists for the
    // directory branch below.
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Workspace-root listing: every direct child reports mtime.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().expect("entries array");
    assert!(!entries.is_empty(), "fresh workspace has direct children");
    for entry in entries {
        let name = entry["name"].as_str().unwrap_or("<no-name>");
        let mtime = entry["mtime"]
            .as_str()
            .unwrap_or_else(|| panic!("entry {name:?} missing mtime; entry={entry}"));
        // RFC3339 surface check: 4-digit year + `T` + `Z` suffix
        // is enough to catch a future regression that ships an
        // epoch-millis or locale-formatted timestamp.
        assert!(
            mtime.len() >= 20 && mtime.chars().nth(4) == Some('-') && mtime.ends_with('Z'),
            "entry {name:?} mtime {mtime:?} is not RFC3339 UTC",
        );
    }

    // Sub-listing on a populated directory via the unified
    // wildcard form (the previous `?path=` query on the root
    // listing was dropped as redundant with `/assets/{*path}`).
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().expect("entries array");
    let sample = entries
        .iter()
        .find(|e| e["name"].as_str() == Some("sample.bin"))
        .expect("sample.bin in listing");
    assert!(
        sample["mtime"].as_str().is_some(),
        "file entry must carry mtime; got {sample}",
    );
}

/// `GET /assets/{*path}?after_seq=&limit=` returns a JSONL page
/// when the resolved file ends in `.jsonl`.  Pinned because the
/// unified surface absorbs the dedicated /training_logs and
/// /converter_logs reads; clients should reach a `.jsonl` file
/// via `/assets` and get the same page shape (`{ events,
/// next_after_seq }`).
#[tokio::test]
async fn assets_jsonl_page_round_trips_on_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    // Synthesise a converter_logs JSONL directly on disk.  The
    // converter producer writes here in production
    // (`ConvertJobLog` in modules/converter.rs); for the wire
    // shape pin we don't need a real run.
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    let lines = [
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"first"}"#,
        r#"{"seq":2,"at":"2026-05-07T12:00:01Z","message":"second"}"#,
        r#"{"seq":3,"at":"2026-05-07T12:00:02Z","message":"third"}"#,
    ];
    std::fs::write(&log_path, lines.join("\n")).unwrap();

    // Page 1: first two events.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/converter_logs/{job_id}.jsonl?limit=2"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().expect("events array");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["seq"], 1);
    assert_eq!(events[1]["seq"], 2);
    assert_eq!(v["next_after_seq"].as_u64(), Some(2));

    // Page 2: continuation.
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/converter_logs/{job_id}.jsonl?after_seq=2&limit=10",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["seq"], 3);
    assert_eq!(v["next_after_seq"].as_u64(), Some(3));
}

/// `?after_seq= / ?limit=` on a non-`.jsonl` file 400s with
/// `bad_request`.  The byte-stream surface stays unambiguous:
/// the only way to reach the JSON-page response is to ask for
/// JSONL paging on a JSONL file.
#[tokio::test]
async fn assets_jsonl_page_rejects_on_binary_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"audio").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/sample.bin?after_seq=0&limit=5"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains(".jsonl"),
        "diagnostic must name the .jsonl gate; got {err}",
    );
}

/// Without the paging query params, `GET /assets/{*path}` keeps
/// streaming bytes.  Pinned because the JSONL paging branch
/// must not affect the existing byte-stream path -- a regression
/// here would silently change the wire for every binary asset.
#[tokio::test]
async fn assets_byte_stream_unchanged_when_no_query_params() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body = b"raw audio";
    let resp = upload(&r, &ws, "datasets/cls/clip.wav", body).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/clip.wav"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "audio/wav", "byte-stream branch keeps MIME mapping");
    let received = body_bytes(resp).await;
    assert_eq!(received, body);
}

/// `?byte_offset= / ?byte_limit=` slice a regular file to the
/// requested byte range.  Pinned because the slice surface is
/// the path random-access binary clients (e.g. WAV seek) rely
/// on; a regression here breaks every player that loads
/// fragments instead of the whole file.
#[tokio::test]
async fn assets_byte_range_returns_slice_with_offset_and_limit() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..256u32).map(|i| (i & 0xff) as u8).collect();
    let resp = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Slice [10, 30) -> 20 bytes.
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_offset=10&byte_limit=20",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(20), "Content-Length reflects slice size");
    let received = body_bytes(resp).await;
    assert_eq!(received, body[10..30]);
}

/// `?byte_offset=` alone streams from the offset to EOF.  Mirror
/// of the `?byte_limit=`-only test below; both must work without
/// the partner param.
#[tokio::test]
async fn assets_byte_range_offset_only_streams_to_eof() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..100u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_offset=70"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, body[70..]);
}

/// `?byte_limit=` alone streams the first N bytes (offset
/// defaults to 0).
#[tokio::test]
async fn assets_byte_range_limit_only_streams_from_zero() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..100u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_limit=15"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, body[..15]);
}

/// `?byte_limit=0` is a valid degenerate slice -- the client
/// asks for zero bytes and gets a 200 OK with `Content-Length:
/// 0`.  Pinned because the boundary `take(0)` must terminate
/// the stream cleanly without burning a full file open + EOF
/// read; a regression here would either 400 (over-strict) or
/// stream the full file (under-strict).
#[tokio::test]
async fn assets_byte_range_limit_zero_returns_empty_slice() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..50u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_limit=0"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(0), "byte_limit=0 advertises Content-Length: 0");
    let received = body_bytes(resp).await;
    assert!(received.is_empty(), "byte_limit=0 yields zero bytes");
}

/// `byte_offset + byte_limit` past EOF clamps silently to the
/// remainder; the response carries however many bytes existed.
/// The slice is "best-effort": the client gets the bytes it asked
/// for or fewer, never an error.
#[tokio::test]
async fn assets_byte_range_clamps_oversized_limit_to_eof() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..50u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    // Ask for 1000 bytes starting at offset 40 -- file is 50
    // bytes long, so the slice clamps to the last 10.
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_offset=40&byte_limit=1000",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(10), "clamped Content-Length");
    let received = body_bytes(resp).await;
    assert_eq!(received, body[40..]);
}

/// `byte_offset == file_size` is a valid, edge-case slice
/// requesting zero bytes from EOF.  Pinned because the boundary
/// is tighter than `byte_offset > file_size` (the latter is
/// 400); operators that calculate offsets and arrive at EOF
/// exactly should not get a spurious error.
#[tokio::test]
async fn assets_byte_range_offset_at_eof_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body = b"hello";
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_offset={}",
            body.len(),
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert!(received.is_empty(), "offset at EOF yields zero bytes");
}

/// `byte_offset > file_size` returns 400 `bad_request` -- the
/// slice surface refuses to silently yield nothing for an offset
/// the operator clearly mis-computed.
#[tokio::test]
async fn assets_byte_range_offset_past_eof_returns_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", b"hi").await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls/payload.bin?byte_offset=999"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("byte_offset"),
        "diagnostic must name the gate; got {err}",
    );
}

/// Byte-range slicing also works on `.jsonl` files.  The file
/// extension does not gate the byte-stream surface; the gate is
/// on the query namespace (raw bytes vs JSONL events).  Operators
/// that want the parsed event shape ask for `?after_seq=` / `?limit=`
/// instead; clients that need a raw chunk of the JSONL backstop
/// (e.g. for log forwarding) ask for `?byte_offset=` / `?byte_limit=`.
#[tokio::test]
async fn assets_byte_range_works_on_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    let raw = b"{\"seq\":1}\n{\"seq\":2}\n{\"seq\":3}\n";
    std::fs::write(&log_path, raw).unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/converter_logs/{job_id}.jsonl?byte_offset=10&byte_limit=10",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, &raw[10..20]);
}

/// Mixing the byte-range and JSONL-page query namespaces returns
/// 400.  The diagnostic must name both axes so the client can
/// see which combination it triggered.
#[tokio::test]
async fn assets_byte_range_rejects_with_jsonl_paging_params() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    std::fs::write(
        log_dir.join(format!("{job_id}.jsonl")),
        b"{\"seq\":1}\n{\"seq\":2}\n",
    )
    .unwrap();

    // byte_offset + after_seq -> 400.
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/converter_logs/{job_id}.jsonl?byte_offset=0&after_seq=0",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("byte_offset") && err.contains("after_seq"),
        "diagnostic must name both axes; got {err}",
    );

    // byte_limit + limit -> 400 (limit's JSONL-paging meaning
    // collides with byte-slice ceiling).
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspace/{ws}/assets/converter_logs/{job_id}.jsonl?byte_limit=4&limit=2",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Byte-range params on a directory return 400 -- the slice
/// surface is file-only.  Mirrors the `?after_seq=` rejection
/// shape so dir-listing requests with an accidental byte param
/// fail loudly rather than silently fall through.
#[tokio::test]
async fn assets_byte_range_rejects_on_directory() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls?byte_offset=0&byte_limit=1"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

/// `?after_seq=` on a directory 400s with the same diagnostic
/// pattern the `.jsonl` gate uses.  Without this branch a
/// directory-listing call with an accidental `?after_seq=` would
/// silently fall through to the listing branch (returning a
/// listing rather than the page the client expected).
#[tokio::test]
async fn assets_jsonl_page_rejects_on_directory() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspace/{ws}/assets/datasets/cls?after_seq=0"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

/// `DELETE /assets/training_logs/<id>.jsonl` routes through the
/// async tombstone+stage+drain machinery -- the response is
/// `202 Accepted` + `{ job_id }`, the file is staged into
/// `<workspace>/.tmp/delete-training-logs-<job_id>/payload`,
/// and the background drain unlinks the staged bytes.  Per-file
/// log delete is now consistent with the dataset/converter
/// async path; no more "200 + removed: N" sync shape.
#[tokio::test]
async fn delete_assets_training_log_file_returns_async_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("training_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    std::fs::write(
        &log_path,
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"hi"}"#,
    )
    .unwrap();

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/training_logs/{job_id}.jsonl"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"].as_str().expect("job_id present on async wipe");
    assert!(!returned_job.is_empty());
    assert!(
        v.get("removed").is_none(),
        "async log delete carries no `removed` field",
    );

    // Eventual: poll the per-job tombstone JSON under `.tmp/`
    // (cleared by `finalize_staged_delete` after drain
    // completes).  Polling the renamed-away `log_path` would
    // be a no-op -- the synchronous rename under the workspace
    // mutex removes it BEFORE the 202 response returns, so the
    // path is already absent by the time the test starts
    // polling; only the per-job tombstone reflects drain
    // completion.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-training-logs-{returned_job}.json"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while tombstone_path.exists() {
        if Instant::now() > deadline {
            panic!(
                "training-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        !log_path.exists(),
        "single-file rename moved the jsonl out of log_dir",
    );
}

/// `DELETE /assets/converter_logs` (whole-dir) routes through
/// the async path: 202 + job_id + background drain.  The empty
/// `converter_logs/` dir is recreated so subsequent producer
/// runs find the canonical structural shape; staged jsonls are
/// drained off-mutex.
#[tokio::test]
async fn delete_assets_converter_logs_whole_dir_returns_async_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let mut jsonl_paths = Vec::new();
    for n in 0..3 {
        let job_id = acoustics_lab::common::ids::JobId::new();
        let p = log_dir.join(format!("{job_id}.jsonl"));
        std::fs::write(&p, format!(r#"{{"seq":{n},"at":"2026-05-07T12:00:00Z"}}"#)).unwrap();
        jsonl_paths.push(p);
    }

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/converter_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"].as_str().expect("job_id on whole-tree wipe");
    assert!(!returned_job.is_empty());

    // Eventual: poll the per-job tombstone JSON for absence
    // (cleared by `finalize_staged_delete` after drain
    // completes).  Polling the original `jsonl_paths` is a
    // no-op: the whole-tree rename moved the entire log dir
    // under staging BEFORE the 202 response returned, so the
    // original child paths are already absent.  The tombstone
    // alone reflects drain progress.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-converter-logs-{returned_job}.json"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while tombstone_path.exists() {
        if Instant::now() > deadline {
            panic!(
                "converter-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Whole-tree wipe recreates the empty log dir for the
    // canonical workspace structural shape; the original
    // children are gone.
    assert!(log_dir.exists(), "empty converter_logs/ recreated");
    assert!(log_dir.is_dir());
    for p in &jsonl_paths {
        assert!(!p.exists(), "{} drained", p.display());
    }
}

/// Whole-tree wipe of an existing-but-empty `training_logs/`
/// returns 202 + job_id.  Per-workspace log dirs are created
/// lazily by the first producer run; this test materializes
/// the empty dir explicitly to exercise the "exists but empty"
/// idempotent-clear path.  The "never created" case is covered
/// by the doc note on `start_async_log_delete` (404, since the
/// target doesn't exist) and `delete_assets_training_logs_absent_returns_404`.
#[tokio::test]
async fn delete_assets_training_logs_whole_dir_succeeds_on_empty() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    // Materialize the empty dir to mirror the post-producer
    // shape; lazy mkdir is otherwise the default.
    let ws_path = fixture_workspace_dir(dir.path(), &ws).join("training_logs");
    std::fs::create_dir_all(&ws_path).expect("mkdir training_logs");

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["job_id"].as_str().is_some());
}

/// Whole-tree wipe of a never-created `training_logs/` returns
/// 404.  Per-workspace log dirs are lazy: the first
/// train/convert producer materializes them, and "clear logs"
/// on a workspace that has never trained surfaces as
/// NOT_FOUND so an operator's idempotent clear pattern can
/// distinguish "no logs to clear" from a successful purge.
#[tokio::test]
async fn delete_assets_training_logs_absent_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Single-file log delete of a missing `.jsonl` returns 404.
/// Mirror of the dataset/converter sub-path semantic; with the
/// async migration we no longer have a sync "removed: 0"
/// fast-path, so operators that need idempotent "remove this
/// log file" check 404 explicitly.
#[tokio::test]
async fn delete_assets_training_log_file_missing_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let phantom = acoustics_lab::common::ids::JobId::new();

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/training_logs/{phantom}.jsonl"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Whole-tree wipe of `datasets/` via `DELETE /assets/datasets`
/// returns 202 Accepted + JobId: the rename is durable under the
/// per-workspace mutex but the staged drain runs in the
/// background, so the response is "queued" rather than "done".
/// The empty `datasets/` dir is recreated so the workspace still
/// has its canonical structural shape.
#[tokio::test]
async fn delete_assets_datasets_whole_tree_returns_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspace/{ws}/assets/datasets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().expect("job_id present on async wipe");
    assert!(!job_id.is_empty());
    assert!(v.get("removed").is_none(), "async wipe carries no removed");
}

/// Uploads to the log trees are rejected at the validator
/// (logs are daemon-produced; clients should never write here).
#[tokio::test]
async fn upload_rejects_log_tree_paths() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for path in ["training_logs/manual.jsonl", "converter_logs/manual.jsonl"] {
        let resp = upload(&r, &ws, path, b"forbidden").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "upload to {path} must be rejected; got {}",
            resp.status(),
        );
    }
}

// MARK: Conflict semantics
//
// Path-overlap no longer gates upload-during-delete: uploads and
// file-deletes never ask for path leases and never conflict with
// train / convert jobs.  Only an in-flight `WorkspaceDelete` for
// this workspace blocks an upload.  The canonical
// upload-blocked-by-active-workspace-delete contract is pinned
// at the lib level by
// `modules/file_mgr/dataset.rs::tests::upload_blocked_by_active_workspace_delete`
// and `modules/file_mgr/job_registry.rs::tests::upload_lease_blocked_by_active_workspace_delete`;
// no HTTP-level integration test lives here because the
// workspace-delete drain is too short-lived on real filesystems
// to catch reliably without a synthetic-handle hook.
