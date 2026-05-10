//! Integration tests for boot-time recovery.
//!
//! Drives `file_mgr::recover_all` end-to-end against tempdir
//! roots.  Each test seeds an active generation (hand-rolled
//! `head.mpk` with ACSTHEAD header so the bundled-default
//! fallback's runtime preload would work against a
//! `HotHead::load` loader), mutates on-disk state to reproduce a
//! crash mode, runs the recovery sweep, and asserts:
//!
//! - the active-result variant matches the failure mode,
//! - the surviving on-disk state is the post-recovery shape
//!   the daemon's first `boot_inference` call expects,
//! - the report counters reflect the work done.
//!
//! The synthetic loader returns an empty `()` candidate so the
//! recovery primitive runs without depending on the inference
//! crate's runtime preload (the daemon's actual boot path uses
//! the real `HotHead::load` loader; the on-disk shape is
//! identical).

#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use acoustics_lab::common::ids::{HeadId, JobId, WorkspaceId};
use acoustics_lab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
};
use acoustics_lab::file_mgr::active_head_writer::{
    ActivationOriginInput, HeadInnerLoader, PendingActivation, publish_active_generation,
    stage_and_validate_activation, staging_path_for,
};
use acoustics_lab::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, active_current_path, active_generation_dir, head_artifact_path,
    head_manifest_path, heads_dir, read_active_current, read_workspace_core, workspace_dir_for,
    workspaces_dir, write_head_index, write_head_manifest, write_workspace_core,
};
use acoustics_lab::file_mgr::staging::{DeleteTombstone, stage_payload, write_tombstone};
use acoustics_lab::file_mgr::time_util::now_rfc3339;
use acoustics_lab::file_mgr::{RecoveryActiveResult, WorkspaceCacheCell, recover_all};
use sha2::{Digest, Sha256};

// MARK: shared fixtures

fn synth_loader() -> Box<HeadInnerLoader> {
    Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
        Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
    })
}

fn ws_id(byte: u8) -> WorkspaceId {
    let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
    WorkspaceId::parse(&s).unwrap()
}

fn head_id(byte: u8) -> HeadId {
    let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
    HeadId::parse(&s).unwrap()
}

fn rev(id: u64) -> WorkspaceRevision {
    WorkspaceRevision {
        id,
        at: "2026-05-07T12:00:00Z".to_string(),
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
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

/// Stage a deployment-bundled default fixture under
/// `<dir>/bundled_default/`.  The mpk content is a synthetic
/// blob; the synthetic loader doesn't parse it, so a real Burn
/// recorder is unnecessary here.  Recovery's bundled-default
/// activation pipeline only requires the bytes + the labels
/// list.
fn fresh_bundled_default(dir: &Path, mpk: &[u8], labels_text: &str) -> PathBuf {
    let bundled = dir.join("bundled_default");
    std::fs::create_dir_all(&bundled).unwrap();
    std::fs::write(bundled.join("head.mpk"), mpk).unwrap();
    std::fs::write(bundled.join("labels.txt"), labels_text).unwrap();
    bundled
}

/// Activate a fresh bundled-default generation so the test
/// starts with a known-good current pointer.  Returns the
/// `(bundled_dir, activation_id)` pair.
fn seed_active_generation(root: &Path, mpk: &[u8], labels_text: &str) -> (PathBuf, String) {
    std::fs::create_dir_all(root).unwrap();
    let bundled = fresh_bundled_default(root, mpk, labels_text);
    let pending = PendingActivation {
        root,
        origin_input: ActivationOriginInput::Default,
        bundled_default_dir: &bundled,
        now_rfc3339: now_rfc3339(),
    };
    let result = stage_and_validate_activation(pending, &*synth_loader()).unwrap();
    let staging = staging_path_for(root, &result.activation_id);
    publish_active_generation(root, &staging, &result.manifest, &result.activation_id).unwrap();
    (bundled, result.activation_id)
}

/// Seed a workspace dir with a single trained head.  Mirrors the
/// daemon's `WorkspaceMgr::create` + `publish_trained_head`
/// outcome for tests that exercise the per-workspace recovery
/// sweep.
fn fresh_workspace_with_head(root: &Path, ws: WorkspaceId, head: HeadId) -> PathBuf {
    let ws_dir = workspace_dir_for(root, &ws);
    std::fs::create_dir_all(heads_dir(&ws_dir)).unwrap();
    std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
    let mpk = b"MPK-CONTENT";
    let manifest = HeadManifest {
        head_id: head,
        workspace_id: ws,
        workspace_revision: rev(5),
        sha256: hex_sha256(mpk),
        n_classes: 2,
        size_bytes: mpk.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["alpha".to_string(), "beta".to_string()],
    };
    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id: head,
        workspace_revision: manifest.workspace_revision.clone(),
        sha256: manifest.sha256.clone(),
        n_classes: manifest.n_classes,
        size_bytes: manifest.size_bytes,
        created_at: manifest.created_at.clone(),
    });
    write_head_index(&ws_dir, &idx).unwrap();
    write_head_manifest(&ws_dir, &manifest).unwrap();
    std::fs::write(head_artifact_path(&ws_dir, head), mpk).unwrap();
    write_workspace_core(
        &ws_dir,
        &WorkspaceCore {
            id: ws,
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        },
    )
    .unwrap();
    ws_dir
}

fn fresh_caches() -> Arc<dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>> {
    Arc::new(dashmap::DashMap::new())
}

// MARK: corrupt active head -> previous generation

#[test]
fn corrupt_active_head_falls_back_to_previous_generation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Pre-seed two generations so recovery has a previous to
    // promote.  The first activation publishes; we sleep then
    // publish a second, which becomes "current".
    let (bundled1, gen1) = seed_active_generation(root, b"DEFAULT-MPK-A", "cat\ndog\n");
    std::thread::sleep(std::time::Duration::from_millis(50));
    let pending2 = PendingActivation {
        root,
        origin_input: ActivationOriginInput::Default,
        bundled_default_dir: &bundled1,
        now_rfc3339: now_rfc3339(),
    };
    let r2 = stage_and_validate_activation(pending2, &*synth_loader()).unwrap();
    publish_active_generation(
        root,
        &staging_path_for(root, &r2.activation_id),
        &r2.manifest,
        &r2.activation_id,
    )
    .unwrap();
    let current_id = r2.activation_id.clone();
    // Corrupt the current generation's head.mpk so the
    // streaming-hash verify fails; recovery should promote the
    // previous (gen1).
    let head_path = active_generation_dir(root, &current_id).join(ACTIVE_HEAD_FILENAME);
    std::fs::write(&head_path, b"TAMPERED").unwrap();
    let caches = fresh_caches();
    let report = recover_all(root, &bundled1, &caches, &*synth_loader()).unwrap();
    match &report.active {
        RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
            assert_eq!(*activation_id, gen1);
        }
        other => panic!("expected PromotedPrevious, got {other:?}"),
    }
    // current.json was rewritten to point at gen1.
    let pointer = read_active_current(root).unwrap();
    assert_eq!(pointer.activation_id, gen1);
}

// MARK: stage-residue boot sweep

#[test]
fn workspace_delete_tombstone_resumes_at_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Lay down the root layout + a current active generation so
    // recovery has a healthy active head to verify alongside the
    // staging sweep.
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    // Stage a half-completed workspace-delete: write the
    // tombstone + rename a victim workspace tree under root
    // `.tmp/`.  Recovery should drain the payload, finalize, and
    // evict the cache cell.
    let ws = ws_id(0xAA);
    let staging = root.join(".tmp");
    std::fs::create_dir_all(&staging).unwrap();
    let tombstone = DeleteTombstone::Workspace {
        job_id: JobId::new(),
        workspace_id: ws,
        created_at: now_rfc3339(),
    };
    let staged = write_tombstone(&staging, &tombstone).unwrap();
    let victim_dir = root.join("victim");
    std::fs::create_dir_all(victim_dir.join("heads")).unwrap();
    std::fs::write(victim_dir.join("workspace.json"), b"{}").unwrap();
    std::fs::write(victim_dir.join("heads/inner"), b"x").unwrap();
    stage_payload(&victim_dir, &staged).unwrap();
    // Pre-seed a stale cache cell for the victim workspace;
    // recovery's eviction hook should drop it.
    let caches = fresh_caches();
    caches.insert(
        ws,
        Arc::new(WorkspaceCacheCell::new(
            WorkspaceCore {
                id: ws,
                name: "victim".to_string(),
                tags: Vec::new(),
                created_at: "2026-05-07T12:34:56Z".to_string(),
                workspace_revision: rev(0),
                head_count: 0,
            },
            HeadIndex::default(),
        )),
    );
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();
    assert_eq!(report.root_staging.workspace_tombstones_completed, 1);
    assert!(!staged.tombstone.exists());
    assert!(!staged.stage_dir.exists());
    assert!(caches.get(&ws).is_none(), "cache cell evicted post-resume");
}

// MARK: daemon-owned head orphans

#[test]
fn daemon_owned_head_orphans_swept_on_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();
    let ws = ws_id(0xBB);
    let real_head = head_id(0xCC);
    let ws_dir = fresh_workspace_with_head(root, ws, real_head);
    // Drop two unreferenced files into heads/ -- the boot sweep
    // must remove both because `<head_id>` is not in
    // `heads.json.heads[]`.
    let orphan = head_id(0xDD);
    let orphan_mpk = head_artifact_path(&ws_dir, orphan);
    let orphan_json = head_manifest_path(&ws_dir, orphan);
    std::fs::write(&orphan_mpk, b"ORPHAN-MPK").unwrap();
    std::fs::write(&orphan_json, b"{}").unwrap();
    let caches = fresh_caches();
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();
    assert_eq!(report.workspaces.workspaces_scanned, 1);
    // Two files removed (mpk + json).
    assert_eq!(report.workspaces.head_orphans_swept, 2);
    assert!(!orphan_mpk.exists());
    assert!(!orphan_json.exists());
    // The legitimate head still resolves.
    assert!(head_artifact_path(&ws_dir, real_head).is_file());
    assert!(head_manifest_path(&ws_dir, real_head).is_file());
}

// MARK: head_count drift

#[test]
fn head_count_drift_repaired_on_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();
    let ws = ws_id(0xEE);
    let real_head = head_id(0xFA);
    let ws_dir = fresh_workspace_with_head(root, ws, real_head);
    // Add a second trained head record so heads.json has 2
    // entries, then tamper workspace.json to claim head_count=0.
    let head2 = head_id(0xFB);
    let mpk2 = b"MPK-2";
    let manifest2 = HeadManifest {
        head_id: head2,
        workspace_id: ws,
        workspace_revision: rev(5),
        sha256: hex_sha256(mpk2),
        n_classes: 2,
        size_bytes: mpk2.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["alpha".to_string(), "beta".to_string()],
    };
    write_head_manifest(&ws_dir, &manifest2).unwrap();
    std::fs::write(head_artifact_path(&ws_dir, head2), mpk2).unwrap();
    let mut idx = HeadIndex::default();
    for hid in [real_head, head2] {
        idx.heads.push(HeadRecord {
            head_id: hid,
            workspace_revision: rev(5),
            sha256: hex_sha256(if hid == real_head {
                b"MPK-CONTENT"
            } else {
                mpk2
            }),
            n_classes: 2,
            size_bytes: 11,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        });
    }
    write_head_index(&ws_dir, &idx).unwrap();
    let mut core = read_workspace_core(&ws_dir).unwrap();
    core.head_count = 0;
    write_workspace_core(&ws_dir, &core).unwrap();
    let caches = fresh_caches();
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();
    assert_eq!(report.workspaces.head_count_repaired, 1);
    let core = read_workspace_core(&ws_dir).unwrap();
    assert_eq!(core.head_count, 2);
    // Active stays valid (Current variant) since we only touched
    // workspace state.
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
    assert!(active_current_path(root).is_file());
}

// MARK: boot recovery commit pins

/// A hand-staged manifest with the legacy `source_dataset_revision`
/// (instead of `workspace_revision`) parse-fails at
/// `read_active_manifest`; recovery treats this as a corrupt
/// current generation and falls back to the bundled default.
#[test]
fn legacy_active_manifest_falls_back_to_bundled_default() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    // Stage a fresh active layout with a known-good `current.json`
    // but a hand-rolled legacy-shape manifest under the pointed
    // generation.
    let bundled = fresh_bundled_default(root, b"DEFAULT-MPK", "alpha\n");
    let activation_id = "11111111-2222-4333-8444-555555555560".to_string();
    let gen_dir = active_generation_dir(root, &activation_id);
    std::fs::create_dir_all(&gen_dir).unwrap();
    // The mpk + labels can be anything -- recovery's verify_generation
    // returns `Failed` at the manifest-parse step before reaching
    // the streaming-hash gates.
    std::fs::write(gen_dir.join(ACTIVE_HEAD_FILENAME), b"any").unwrap();
    std::fs::write(
        gen_dir.join(acoustics_lab::file_mgr::schema::ACTIVE_LABELS_FILENAME),
        "alpha\n",
    )
    .unwrap();
    let round_1_manifest = serde_json::json!({
        "origin": "head",
        "source_workspace_id": "11111111-2222-4333-8444-555555555548",
        "source_head_id": "11111111-2222-4333-8444-555555555540",
        // Legacy alias; the schema now requires
        // `workspace_revision`, so this missing-field shape
        // parse-fails.
        "source_dataset_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
        "runtime_head_id": "11111111-2222-4333-8444-555555555540",
        "sha256": "deadbeef",
        "labels_sha256": "cafef00d",
        "n_classes": 1,
        "labels": ["alpha"],
        "activated_at": "2026-05-07T12:34:56Z",
    });
    std::fs::write(
        gen_dir.join(acoustics_lab::file_mgr::schema::ACTIVE_MANIFEST_FILENAME),
        serde_json::to_vec(&round_1_manifest).unwrap(),
    )
    .unwrap();
    // Point current.json at the corrupt generation.
    acoustics_lab::file_mgr::schema::write_active_current(
        root,
        &acoustics_lab::file_mgr::schema::ActiveCurrentPointer {
            activation_id: activation_id.clone(),
        },
    )
    .unwrap();

    let caches = fresh_caches();
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();
    match &report.active {
        RecoveryActiveResult::DefaultedFromBundle {
            activation_id: new_id,
            ..
        } => {
            assert_ne!(*new_id, activation_id, "fresh bundled default published");
        }
        other => panic!("expected DefaultedFromBundle on legacy manifest, got {other:?}"),
    }
    // current.json was rewritten away from the corrupt generation.
    let pointer = read_active_current(root).unwrap();
    assert_ne!(pointer.activation_id, activation_id);
}

/// One sweep handles multi-failure residue -- workspace
/// tombstone + per-workspace head orphans + head_count drift +
/// incomplete create -- in dependency order.  Pins that
/// `recover_all` walks the sub-sweeps without short-circuiting
/// on partial failure.
#[test]
fn recover_all_aggregates_multi_failure_residue() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    // (a) Healthy workspace + head_count drift + head orphan.
    let ws_a = ws_id(0xA0);
    let head_a = head_id(0xA1);
    let ws_dir_a = fresh_workspace_with_head(root, ws_a, head_a);
    let mut core_a = read_workspace_core(&ws_dir_a).unwrap();
    core_a.head_count = 0; // claim 0 even though heads.json has 1
    write_workspace_core(&ws_dir_a, &core_a).unwrap();
    let orphan_a = head_id(0xA2);
    std::fs::write(head_artifact_path(&ws_dir_a, orphan_a), b"orphan-mpk").unwrap();
    std::fs::write(head_manifest_path(&ws_dir_a, orphan_a), b"{}").unwrap();

    // (b) Incomplete-create workspace dir (no workspace.json).
    let ws_b = ws_id(0xB0);
    let ws_dir_b = workspace_dir_for(root, &ws_b);
    std::fs::create_dir_all(ws_dir_b.join("heads")).unwrap();
    std::fs::create_dir_all(ws_dir_b.join(".tmp")).unwrap();

    // (c) Root-level workspace tombstone awaiting drain.
    let ws_c = ws_id(0xC0);
    let staging = root.join(".tmp");
    std::fs::create_dir_all(&staging).unwrap();
    let tombstone = DeleteTombstone::Workspace {
        job_id: JobId::new(),
        workspace_id: ws_c,
        created_at: now_rfc3339(),
    };
    let staged = write_tombstone(&staging, &tombstone).unwrap();
    let victim = root.join("victim_c");
    std::fs::create_dir_all(victim.join("heads")).unwrap();
    std::fs::write(victim.join("workspace.json"), b"{}").unwrap();
    stage_payload(&victim, &staged).unwrap();

    // Pre-seed a stale cache cell for ws_c so root-staging
    // recovery's eviction hook is exercised.
    let caches = fresh_caches();
    caches.insert(
        ws_c,
        Arc::new(WorkspaceCacheCell::new(
            WorkspaceCore {
                id: ws_c,
                name: "victim".to_string(),
                tags: Vec::new(),
                created_at: "2026-05-07T12:34:56Z".to_string(),
                workspace_revision: rev(0),
                head_count: 0,
            },
            HeadIndex::default(),
        )),
    );

    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();

    // Per-workspace counters.
    assert_eq!(
        report.workspaces.workspaces_scanned, 1,
        "exactly one valid workspace (ws_a); ws_b is incomplete-create",
    );
    assert_eq!(report.workspaces.head_count_repaired, 1, "ws_a repaired");
    assert_eq!(
        report.workspaces.head_orphans_swept, 2,
        "two orphan files removed (mpk + json)",
    );
    assert_eq!(
        report.workspaces.incomplete_creates_removed, 1,
        "ws_b incomplete-create removed",
    );
    assert!(!ws_dir_b.exists(), "ws_b directory removed");
    // Repair landed.
    let core_a = read_workspace_core(&ws_dir_a).unwrap();
    assert_eq!(core_a.head_count, 1);
    // Orphan files gone.
    assert!(!head_artifact_path(&ws_dir_a, orphan_a).exists());
    assert!(!head_manifest_path(&ws_dir_a, orphan_a).exists());

    // Root-staging counters.
    assert_eq!(report.root_staging.workspace_tombstones_completed, 1);
    assert!(!staged.tombstone.exists());
    assert!(caches.get(&ws_c).is_none(), "ws_c cache cell evicted");

    // Active stays valid (Current variant): the active generation
    // was untouched in this fixture, only workspace + root state
    // had residue.
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
}

/// A per-workspace recovery failure (e.g. corrupt `heads.json`)
/// increments `workspace_recovery_failures` AND is excluded from
/// `workspaces_scanned`; the orchestrator keeps walking the
/// remaining workspaces.
#[test]
fn per_workspace_recovery_failure_counted() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    // Healthy workspace ws_a -- scanned successfully.
    let ws_a = ws_id(0xE0);
    let head_a = head_id(0xE1);
    let _ws_dir_a = fresh_workspace_with_head(root, ws_a, head_a);

    // Broken workspace ws_b -- has workspace.json (so it's not
    // an incomplete-create) but heads.json is corrupt JSON
    // (parse-fail through `read_head_index`).  The per-workspace
    // sweep returns Err; the orchestrator logs + continues +
    // bumps `workspace_recovery_failures`.
    let ws_b = ws_id(0xE2);
    let ws_dir_b = workspace_dir_for(root, &ws_b);
    std::fs::create_dir_all(ws_dir_b.join("heads")).unwrap();
    std::fs::create_dir_all(ws_dir_b.join(".tmp")).unwrap();
    write_workspace_core(
        &ws_dir_b,
        &WorkspaceCore {
            id: ws_b,
            name: "broken".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-08T12:00:00Z".to_string(),
            workspace_revision: rev(0),
            head_count: 0,
        },
    )
    .unwrap();
    // `HeadIndex` carries `deny_unknown_fields`; this hand-rolled
    // body parses as malformed JSON entirely.
    std::fs::write(ws_dir_b.join("heads.json"), b"{not json}").unwrap();

    let caches = fresh_caches();
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();

    assert_eq!(
        report.workspaces.workspace_recovery_failures, 1,
        "ws_b's parse failure must surface on the typed counter",
    );
    assert_eq!(
        report.workspaces.workspaces_scanned, 1,
        "only ws_a was scanned; ws_b's failure does not count",
    );
    // ws_a was repaired normally.
    assert!(matches!(
        report.active,
        RecoveryActiveResult::Current { .. }
    ));
    // ws_b's directory remains on disk; the orchestrator does
    // NOT auto-delete a broken workspace, operator action is
    // required.
    assert!(ws_dir_b.exists());
}

/// The converter-tombstone resume path fires alongside the
/// dataset path in a single `recover_workspaces` sweep; dispatch
/// is by tombstone filename prefix.  Pins that the converter
/// sweep is reachable from the orchestrator's `recover_all`
/// entry point.
#[test]
fn recover_all_drains_dataset_and_converter_tombstones_together() {
    use acoustics_lab::common::asset_path::AssetPath;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (bundled, _) = seed_active_generation(root, b"DEFAULT-MPK", "cat\n");
    std::fs::create_dir_all(workspaces_dir(root)).unwrap();

    let ws = ws_id(0xD0);
    let head = head_id(0xD1);
    let ws_dir = fresh_workspace_with_head(root, ws, head);

    // Stage one dataset-delete tombstone + one converter-delete
    // tombstone in the same workspace's .tmp/.  Filename prefix
    // dispatches the variant:
    //   delete-assets-<job_id>/   -> Dataset
    //   delete-converters-<job_id>/ -> Converter
    let staging = ws_dir.join(".tmp");
    let dataset_tombstone = DeleteTombstone::Dataset {
        job_id: JobId::new(),
        workspace_id: ws,
        path: Some(AssetPath::parse("audio/cat").unwrap()),
        workspace_revision_id: 7,
        created_at: now_rfc3339(),
    };
    let dataset_staged = write_tombstone(&staging, &dataset_tombstone).unwrap();
    let dataset_target = root.join("dataset_payload");
    std::fs::write(&dataset_target, b"data-bytes").unwrap();
    stage_payload(&dataset_target, &dataset_staged).unwrap();

    let converter_tombstone = DeleteTombstone::Converter {
        job_id: JobId::new(),
        workspace_id: ws,
        path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
        workspace_revision_id: 8,
        created_at: now_rfc3339(),
    };
    let converter_staged = write_tombstone(&staging, &converter_tombstone).unwrap();
    let converter_target = root.join("converter_payload");
    std::fs::write(&converter_target, b"manifest-bytes").unwrap();
    stage_payload(&converter_target, &converter_staged).unwrap();

    let caches = fresh_caches();
    let report = recover_all(root, &bundled, &caches, &*synth_loader()).unwrap();

    assert_eq!(report.workspaces.dataset_tombstones_completed, 1);
    assert_eq!(report.workspaces.converter_tombstones_completed, 1);
    assert!(!dataset_staged.tombstone.exists());
    assert!(!dataset_staged.stage_dir.exists());
    assert!(!converter_staged.tombstone.exists());
    assert!(!converter_staged.stage_dir.exists());
}
