//! Integration tests for the workspace-side counter surface
//! published through `GET /api/v1/status`.
//!
//! Boots the API router in-process, installs a fresh
//! `WorkspaceMetrics` global, then drives:
//!
//! 1. The empty-state snapshot exposes the new
//!    `workspace.*` counter block as zeros.
//! 2. A successful dataset upload increments
//!    `assets_uploaded_total`, `bytes_uploaded_total`, and
//!    the `workspace_core_writes_total` counter (the upload
//!    flow rewrites `workspace.json` for the revision bump).
//! 3. Recording boot-recovery orphans on the metrics handle
//!    surfaces on `boot_orphans_swept_total` (the daemon
//!    boot path is exercised in `tests/boot_recovery.rs`;
//!    here we hit the metrics primitive directly because
//!    the router does not own the recovery sweep).
//!
//! The `WorkspaceMetrics` global is process-wide; the test
//! file uses an explicit `Arc<WorkspaceMetrics>` constructed
//! per test plus `install_for_tests`, so concurrent test
//! binaries do not step on each other.

#![allow(clippy::disallowed_methods)]

use std::sync::Arc;
use std::time::Duration;

use acoustics_lab::api::router_v1_nested;
use acoustics_lab::status::WorkspaceMetrics;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, fresh_app_state, json_body, upload};

/// Install the file_mgr metrics hooks against `metrics`.
/// Idempotent (the underlying `OnceLock`s are single-shot);
/// subsequent calls in the same test binary are no-ops.
fn install_metrics_hooks(metrics: Arc<WorkspaceMetrics>) {
    let m = Arc::clone(&metrics);
    acoustics_lab::file_mgr::metrics_hooks::install_workspace_core_write_hook(move |d| {
        m.record_workspace_core_write(d);
    });
    let m = Arc::clone(&metrics);
    acoustics_lab::file_mgr::metrics_hooks::install_head_index_write_hook(move |d| {
        m.record_head_index_write(d);
    });
    let m = Arc::clone(&metrics);
    acoustics_lab::file_mgr::metrics_hooks::install_upload_hook(move |bytes| {
        m.record_upload(bytes);
    });
    let m = Arc::clone(&metrics);
    acoustics_lab::file_mgr::metrics_hooks::install_dataset_mutation_rejected_hook(move || {
        m.record_dataset_mutation_rejected();
    });
    let m = Arc::clone(&metrics);
    acoustics_lab::file_mgr::metrics_hooks::install_job_events_dropped_hook(move |n| {
        m.record_job_events_dropped(n);
    });
}

/// `GET /api/v1/status` exposes the `workspace.*` counter
/// block; every counter is present as a numeric field.  Values
/// are cumulative since process start; a fixture that has not
/// driven any uploads / writes still sees the keys and a
/// non-negative numeric body.
/// (Process-shared `OnceLock<WorkspaceMetrics>` means we
/// assert presence + numeric type, not absolute zero, since
/// other tests in the same binary may have already
/// incremented counters.)
#[tokio::test]
async fn status_includes_workspace_counter_block() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    let resp = call(&r, Method::GET, "/api/v1/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let workspace = v.get("workspace").expect("workspace counter block present");
    for key in [
        "assets_uploaded_total",
        "bytes_uploaded_total",
        "workspace_core_writes_total",
        "head_index_writes_total",
        "dataset_mutations_rejected_total",
        // Converter rejections live on a separate counter.
        "converter_mutations_rejected_total",
        "workspace_core_write_p99_us",
        "head_index_write_p99_us",
        "job_events_dropped_total",
        "sse_clients_current",
        "boot_orphans_swept_total",
        // Per-workspace recovery failure aggregate so operators
        // can spot `workspaces_scanned < expected` symptoms.
        "boot_workspace_recovery_failures_total",
    ] {
        let val = workspace
            .get(key)
            .unwrap_or_else(|| panic!("workspace.{key} present"));
        // sse_clients_current is signed (i64); the rest are
        // unsigned (u64).  Accept either numeric shape.
        assert!(
            val.as_u64().is_some() || val.as_i64().is_some(),
            "workspace.{key} must serialize as a number; got {val:?}",
        );
    }
}

/// A successful dataset upload increments the upload + core
/// write counters on the installed metrics global.  The
/// status route reads the same global, so the wire snapshot
/// reflects the increments end-to-end.
#[tokio::test]
async fn upload_increments_counters_on_status_wire() {
    let dir = tempfile::tempdir().unwrap();
    // Install a fresh metrics handle.  `install_for_tests`
    // is a no-op when a prior test has already installed
    // one (OnceLock), so we keep a direct `Arc` to the
    // pinned handle.  The schema / dataset hot paths read
    // the global; if a prior test in this binary already
    // installed one, the increments still apply to that
    // handle and the assertions still hold (the counters
    // are cumulative; we measure deltas).
    let metrics = Arc::new(WorkspaceMetrics::new());
    acoustics_lab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    // Use whichever handle the global ended up with -- our
    // local `metrics` if `install_for_tests` succeeded, or
    // the prior-installed one if a sibling test got there
    // first.  Either way, the file_mgr hooks installed below
    // forward all counter events onto that handle.
    let metrics_for_assert: Arc<WorkspaceMetrics> =
        match acoustics_lab::status::workspace_metrics::global() {
            Some(g) => Arc::clone(g),
            None => metrics, // unreachable in practice; install_for_tests would have set it.
        };
    install_metrics_hooks(Arc::clone(&metrics_for_assert));
    let before = metrics_for_assert.snapshot();

    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let bytes = b"audio body of known length";
    let resp = upload(&r, &ws, "datasets/audio_dataset/cat/sample.wav", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Give the metrics global a beat (no async work pending,
    // but a yield is cheap defence against scheduler skew).
    tokio::task::yield_now().await;

    let after = metrics_for_assert.snapshot();
    assert_eq!(
        after.assets_uploaded_total - before.assets_uploaded_total,
        1,
        "assets_uploaded_total bumped by exactly one upload",
    );
    assert_eq!(
        after.bytes_uploaded_total - before.bytes_uploaded_total,
        bytes.len() as u64,
        "bytes_uploaded_total bumped by the upload size",
    );
    assert!(
        after.workspace_core_writes_total > before.workspace_core_writes_total,
        "workspace_core_writes_total advances on the dataset_revision bump (before={} after={})",
        before.workspace_core_writes_total,
        after.workspace_core_writes_total,
    );

    // Wire-shape gate: `GET /api/v1/status` reflects the
    // post-upload counter values.
    let resp = call(&r, Method::GET, "/api/v1/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let ws_block = v.get("workspace").expect("workspace block");
    assert!(
        ws_block["assets_uploaded_total"].as_u64().expect("u64") >= after.assets_uploaded_total,
        "wire counter >= snapshot counter (counters are cumulative): {ws_block}",
    );
}

/// Boot-recovery orphan counts surface on
/// `boot_orphans_swept_total` after the daemon's wakeup
/// records them.  Direct primitive test; the daemon-side
/// wakeup path is exercised in `tests/boot_recovery.rs`
/// + `tests/daemon_lifecycle_*.rs`.
#[tokio::test]
async fn boot_orphans_swept_records_on_metrics_global() {
    let metrics = Arc::new(WorkspaceMetrics::new());
    acoustics_lab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    let pinned: Arc<WorkspaceMetrics> = match acoustics_lab::status::workspace_metrics::global() {
        Some(g) => Arc::clone(g),
        None => metrics,
    };
    let before = pinned.snapshot().boot_orphans_swept_total;

    pinned.record_boot_orphans_swept(5);
    pinned.record_boot_orphans_swept(2);

    let after = pinned.snapshot().boot_orphans_swept_total;
    assert_eq!(after - before, 7);

    // Also exercise the bounded p99 and the SSE guard, so
    // downstream tests can't regress them silently.
    pinned.record_workspace_core_write(Duration::from_millis(2));
    pinned.record_head_index_write(Duration::from_micros(500));
    let snap = pinned.snapshot();
    assert!(snap.workspace_core_write_p99_us > 0);
    assert!(snap.head_index_write_p99_us > 0);

    // sse_client_guard increments + decrements via Drop.
    let before_clients = pinned.snapshot().sse_clients_current;
    {
        let _g = pinned.sse_client_guard();
        let _g2 = pinned.sse_client_guard();
        assert_eq!(pinned.snapshot().sse_clients_current - before_clients, 2,);
    }
    assert_eq!(pinned.snapshot().sse_clients_current, before_clients);
}

/// Dataset and converter rejections dispatch into separate
/// per-tree counters: the upload + delete admission paths capture
/// the `AssetTree` from `validate_mutable_subpath` and emit the
/// matching counter on rejection.
#[tokio::test]
async fn mutation_rejections_dispatch_per_tree() {
    let metrics = Arc::new(WorkspaceMetrics::new());
    acoustics_lab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    let pinned: Arc<WorkspaceMetrics> = match acoustics_lab::status::workspace_metrics::global() {
        Some(g) => Arc::clone(g),
        None => metrics,
    };
    install_metrics_hooks(Arc::clone(&pinned));

    let before = pinned.snapshot();

    // Direct emission test: the `emit_mutation_rejected` helper
    // in `dataset.rs` is private, so we exercise the per-tree
    // counters via the `WorkspaceMetrics::record_*` methods
    // directly.  The wire-shape pin (status_includes_workspace_counter_block)
    // confirms both counters are present in the snapshot.
    pinned.record_dataset_mutation_rejected();
    pinned.record_dataset_mutation_rejected();
    pinned.record_converter_mutation_rejected();

    let after = pinned.snapshot();
    assert_eq!(
        after.dataset_mutations_rejected_total - before.dataset_mutations_rejected_total,
        2,
        "dataset rejections increment dataset counter only",
    );
    assert_eq!(
        after.converter_mutations_rejected_total - before.converter_mutations_rejected_total,
        1,
        "converter rejections increment converter counter only",
    );
}
