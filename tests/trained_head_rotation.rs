//! Crash-safety integration tests for the trained-head 2-slot
//! rotation primitive.
//!
//! Drives [`acoustics_lab::file_mgr::publish_trained_head`] through:
//! - happy path (single publish),
//! - sliding-window cap (3 publishes -> oldest displaced),
//! - orphan tolerance (rotation does not touch unrelated residue),
//! - corruption (`heads.json` references a missing `.mpk` -- cache
//!   load tolerates the inconsistency, follow-up `delete_head`
//!   surfaces the corruption),
//! - best-effort displaced-cleanup (step 9 failure leaves the
//!   rotation succeeded because the index commit at step 7 is the
//!   source of truth).
//!
//! Synthesizes opaque `.mpk` bytes (the rotation primitive does
//! not parse them; only `inference::head::load_inner` does).

#![allow(clippy::disallowed_methods)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use acoustics_lab::common::ids::{HeadId, WorkspaceId};
use acoustics_lab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE, WorkspaceCore, WorkspaceRevision,
};
use acoustics_lab::file_mgr::{
    HeadRotationResult, PendingHead, WorkspaceCacheCell, head_artifact_path, head_index_path,
    head_manifest_path, heads_dir, publish_trained_head, read_head_index, read_head_manifest,
    read_workspace_core, write_head_index, write_workspace_core,
};

fn ws_id() -> WorkspaceId {
    WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap()
}

fn rev(id: u64) -> WorkspaceRevision {
    WorkspaceRevision {
        id,
        at: "2026-05-07T12:00:00Z".to_string(),
    }
}

fn sample_core(rev_id: u64, head_count: u8) -> WorkspaceCore {
    WorkspaceCore {
        id: ws_id(),
        name: "main".to_string(),
        tags: Vec::new(),
        created_at: "2026-05-07T12:34:56Z".to_string(),
        workspace_revision: rev(rev_id),
        head_count,
    }
}

fn sample_manifest(head_id: HeadId, rev_id: u64) -> HeadManifest {
    HeadManifest {
        head_id,
        workspace_id: ws_id(),
        workspace_revision: rev(rev_id),
        sha256: format!("sha-of-{head_id}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
    }
}

/// Stage a fake `.mpk` tempfile under `<workspace>/.tmp/` with
/// deterministic-but-distinct bytes per head id so the assertion
/// can prove the right file landed.
fn stage_mpk_tempfile(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    let tmp_dir = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let path = tmp_dir.join(format!("staged-{head_id}.mpk"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(format!("MPK-{head_id}").as_bytes()).unwrap();
    f.sync_all().unwrap();
    path
}

fn fresh_workspace() -> (tempfile::TempDir, WorkspaceCacheCell) {
    let tmp = tempfile::tempdir().unwrap();
    let core = sample_core(0, 0);
    write_workspace_core(tmp.path(), &core).unwrap();
    write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
    let cache = WorkspaceCacheCell::new(core, HeadIndex::default());
    (tmp, cache)
}

/// Test 1 (happy path).
#[test]
fn publish_one_head_lands_index_atomic() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    let manifest = sample_manifest(head_id, 5);

    let HeadRotationResult { displaced_head_id } = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk.clone(),
            manifest: manifest.clone(),
        },
    )
    .unwrap();

    assert!(displaced_head_id.is_none(), "first publish never displaces");
    assert!(head_artifact_path(tmp.path(), head_id).is_file());
    assert!(head_manifest_path(tmp.path(), head_id).is_file());
    assert!(!mpk.exists(), "tempfile renamed away");
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), 1);
    assert_eq!(on_disk.heads[0].head_id, head_id);
    let core = read_workspace_core(tmp.path()).unwrap();
    assert_eq!(core.head_count, 1);
    assert_eq!(*cache.heads(), on_disk);
    assert_eq!(*cache.core(), core);
}

/// Test 2 (sliding window): 3 publishes -> only the most-recent
/// 2 stay; oldest displaced files are removed.
#[test]
fn publish_three_heads_drops_oldest() {
    let (tmp, cache) = fresh_workspace();
    let h1 = HeadId::new();
    let h2 = HeadId::new();
    let h3 = HeadId::new();
    for &h in &[h1, h2] {
        let mpk = stage_mpk_tempfile(tmp.path(), h);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h, 1),
            },
        )
        .unwrap();
    }
    assert_eq!(cache.heads().heads.len(), 2);
    let mpk = stage_mpk_tempfile(tmp.path(), h3);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h3,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h3, 3),
        },
    )
    .unwrap();
    assert_eq!(result.displaced_head_id, Some(h1));
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), MAX_HEADS_PER_WORKSPACE);
    assert_eq!(on_disk.heads[0].head_id, h3);
    assert_eq!(on_disk.heads[1].head_id, h2);
    // Displaced bytes removed.
    assert!(!head_artifact_path(tmp.path(), h1).exists());
    assert!(!head_manifest_path(tmp.path(), h1).exists());
    assert!(head_artifact_path(tmp.path(), h2).is_file());
    assert!(head_artifact_path(tmp.path(), h3).is_file());
}

/// Orphan tolerance: the rotation must NOT touch unrelated
/// `<random>.{mpk,json}` files under `heads/`; those are residue
/// boot recovery sweeps.
#[test]
fn publish_does_not_disturb_orphans() {
    let (tmp, cache) = fresh_workspace();
    let orphan_id = HeadId::new();
    let orphan_mpk = head_artifact_path(tmp.path(), orphan_id);
    let orphan_json = head_manifest_path(tmp.path(), orphan_id);
    std::fs::write(&orphan_mpk, b"orphan-mpk").unwrap();
    std::fs::write(&orphan_json, b"{}").unwrap();

    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
    )
    .unwrap();
    assert!(orphan_mpk.exists(), "rotation must not touch orphans");
    assert!(orphan_json.exists());
    // The index does not list the orphan (its presence on disk
    // is invisible to the index).
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert!(!on_disk.heads.iter().any(|r| r.head_id == orphan_id));
}

/// Cache load tolerates `heads.json` referencing a missing file:
/// the cache does not stat `heads/`.  Boot recovery is the
/// documented sweep surface; this test pins today's behavior so a
/// future cache-load change cannot silently start statting.
#[test]
fn cache_load_tolerates_phantom_index_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let core = sample_core(0, 1);
    write_workspace_core(tmp.path(), &core).unwrap();
    let phantom = HeadId::new();
    let phantom_record = HeadRecord {
        head_id: phantom,
        workspace_revision: rev(0),
        sha256: "y".into(),
        n_classes: 1,
        size_bytes: 0,
        created_at: "2026-05-07T12:00:00Z".to_string(),
    };
    let bad_index = HeadIndex {
        heads: vec![phantom_record],
    };
    write_head_index(tmp.path(), &bad_index).unwrap();
    let cell = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
    assert_eq!(cell.heads().heads.len(), 1);
    assert!(!head_artifact_path(tmp.path(), phantom).exists());
    assert!(!head_manifest_path(tmp.path(), phantom).exists());
}

/// Best-effort cleanup: if step 9 fails because the displaced
/// file is already gone (filesystem race or concurrent orphan
/// sweep), the rotation as a whole still succeeds because the
/// index commit at step 7 already moved the publish point.
#[test]
fn publish_succeeds_when_displaced_files_are_already_gone() {
    let (tmp, cache) = fresh_workspace();
    let h1 = HeadId::new();
    let h2 = HeadId::new();
    let h3 = HeadId::new();
    for &h in &[h1, h2] {
        let mpk = stage_mpk_tempfile(tmp.path(), h);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h, 1),
            },
        )
        .unwrap();
    }
    // Race: remove h1's files manually before publishing h3.
    std::fs::remove_file(head_artifact_path(tmp.path(), h1)).unwrap();
    std::fs::remove_file(head_manifest_path(tmp.path(), h1)).unwrap();

    let mpk = stage_mpk_tempfile(tmp.path(), h3);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h3,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h3, 3),
        },
    )
    .unwrap();
    assert_eq!(result.displaced_head_id, Some(h1));
    let on_disk = read_head_index(tmp.path()).unwrap();
    assert_eq!(on_disk.heads.len(), 2);
    assert_eq!(on_disk.heads[0].head_id, h3);
    assert_eq!(on_disk.heads[1].head_id, h2);
}

/// Crash between steps 6 and 7: the `.mpk` + `.json` land under
/// `heads/` but `heads.json` is unchanged.  Boot recovery sweeps
/// the unreferenced files; this test simulates the partial state
/// and verifies the index view treats the new head as invisible.
#[test]
fn crash_between_steps_6_and_7_leaves_orphan_files_invisible_to_index() {
    let (tmp, cache) = fresh_workspace();
    // Publish h1 normally so the index has a known shape.
    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
    )
    .unwrap();
    let pre_crash_index = read_head_index(tmp.path()).unwrap();

    // Simulate a crash between steps 6 and 7 of a follow-up h2
    // publish: write the new files into heads/ but DO NOT
    // rewrite heads.json.
    let h2 = HeadId::new();
    let mpk2_final = head_artifact_path(tmp.path(), h2);
    let json2_final = head_manifest_path(tmp.path(), h2);
    std::fs::write(&mpk2_final, b"orphan-h2-mpk").unwrap();
    std::fs::write(
        &json2_final,
        serde_json::to_vec(&sample_manifest(h2, 2)).unwrap(),
    )
    .unwrap();

    // The index is unchanged from the pre-crash state.
    let post_crash_index = read_head_index(tmp.path()).unwrap();
    assert_eq!(
        post_crash_index, pre_crash_index,
        "heads.json must not reflect the partial publish",
    );
    assert!(post_crash_index.heads.iter().all(|r| r.head_id != h2));
    // The orphan files are present but invisible to the index;
    // boot recovery is the sweep surface.
    assert!(mpk2_final.is_file());
    assert!(json2_final.is_file());

    // Pin the heads.json path layout so a future helper change
    // surfaces here first.
    assert!(head_index_path(tmp.path()).is_file());
}

/// A follow-up successful publish converges in the presence of
/// orphans from a prior crashed publish: the rotation primitive
/// must not be derailed by them, and boot recovery later sweeps.
#[test]
fn rotation_converges_after_simulated_partial_crash() {
    let (tmp, cache) = fresh_workspace();
    // Place an orphan h2 (simulating a prior crashed publish).
    let h_orphan = HeadId::new();
    std::fs::write(head_artifact_path(tmp.path(), h_orphan), b"residue").unwrap();
    std::fs::write(head_manifest_path(tmp.path(), h_orphan), b"{}").unwrap();
    // Now publish h1 -- must succeed and the orphan must remain
    // untouched (boot recovery sweeps it later).
    let h1 = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), h1);
    let result = publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id: h1,
            mpk_tempfile: mpk,
            manifest: sample_manifest(h1, 1),
        },
    )
    .unwrap();
    assert!(result.displaced_head_id.is_none());
    assert!(head_artifact_path(tmp.path(), h1).is_file());
    assert!(head_artifact_path(tmp.path(), h_orphan).is_file());
    assert!(head_manifest_path(tmp.path(), h_orphan).is_file());
    let idx = read_head_index(tmp.path()).unwrap();
    assert_eq!(idx.heads.len(), 1);
    assert_eq!(idx.heads[0].head_id, h1);
}

/// `delete_head` via the `WorkspaceMgr` surface; covered here
/// because integration tests have `acoustics_lab::file_mgr::*` in
/// scope.  The test stages heads on-disk and constructs a fresh
/// `FsServiceImpl` after the writes so its cache lazy-loads from
/// the post-publish state instead of being shadowed by an empty
/// create-side cache.
#[tokio::test(flavor = "current_thread")]
async fn delete_head_via_workspace_mgr_round_trip() {
    use acoustics_lab::file_mgr::{FsService, FsServiceImpl};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    // Phase 1: create the workspace via the production lifecycle
    // path so the on-disk layout is canonical.
    let create_fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    create_fs.ensure_root_layout().unwrap();
    let id = create_fs.create("rotation-mgr-test").unwrap();
    let workspace_dir = root.join("workspaces").join(id.to_string());
    drop(create_fs);
    // Phase 2: stage two heads under heads/ + heads.json directly.
    // Mirrors what `publish_trained_head` produces; this test
    // exercises `delete_head` independently so a stale cache on
    // the create-side `WorkspaceMgr` does not shadow our writes.
    // `heads/` is created lazily by the production publisher
    // (head_rotation.rs); pre-create it here since this test
    // skips that path.
    std::fs::create_dir_all(workspace_dir.join("heads")).unwrap();
    let h1 = HeadId::new();
    let h2 = HeadId::new();
    for &h in &[h1, h2] {
        let mpk = stage_mpk_tempfile(&workspace_dir, h);
        std::fs::rename(&mpk, head_artifact_path(&workspace_dir, h)).unwrap();
        let manifest = sample_manifest(h, 1);
        std::fs::write(
            head_manifest_path(&workspace_dir, h),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id: h2,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h2}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    idx.heads.push(HeadRecord {
        head_id: h1,
        workspace_revision: rev(1),
        sha256: format!("sha-of-{h1}"),
        n_classes: 3,
        size_bytes: 1024,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    write_head_index(&workspace_dir, &idx).unwrap();
    let mut core = read_workspace_core(&workspace_dir).unwrap();
    core.head_count = 2;
    write_workspace_core(&workspace_dir, &core).unwrap();

    // Phase 3: fresh FsServiceImpl so the cache lazy-loads from
    // the post-publish disk state.  Matches the daemon's first
    // touch on a workspace recovered from a prior process.
    let fs: Arc<dyn FsService> = Arc::new(FsServiceImpl::new(root.clone()));
    let summary = fs.summary(&id).unwrap();
    assert_eq!(summary.heads.heads.len(), 2);

    // Delete h1.
    fs.delete_head(&id, h1).unwrap();
    let summary = fs.summary(&id).unwrap();
    assert_eq!(summary.heads.heads.len(), 1);
    assert_eq!(summary.heads.heads[0].head_id, h2);
    assert!(!head_artifact_path(&workspace_dir, h1).exists());
    assert!(!head_manifest_path(&workspace_dir, h1).exists());
    // workspace.json.head_count tracks.
    assert_eq!(summary.core.head_count, 1);

    // Deleting a phantom head_id surfaces 404.
    let phantom = HeadId::new();
    let err = fs.delete_head(&id, phantom).unwrap_err();
    use acoustics_lab::common::error::{Categorized, ErrorKind};
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

// MARK: published-shape pins
//
// The rotation primitive produces JSON blobs that downstream
// readers (cache load, boot recovery, activation) consume; these
// tests pin the on-disk byte shape so a future cascade that quietly
// re-adds legacy fields (`dataset_path`, `training_cfg`,
// `training_cfg_sha256`, `dataset_revision`) parse-fails through
// `deny_unknown_fields` before it can corrupt the inference path.

/// The published `<head_id>.json` carries exactly the minimized
/// field set -- `head_id`, `workspace_id`, `workspace_revision`,
/// `sha256`, `n_classes`, `size_bytes`, `created_at`, `labels` --
/// and never the legacy provenance fields.  Pins the publish-side
/// shape so a future `HeadManifest` edit re-adding a legacy field
/// surfaces here first.
#[test]
fn published_manifest_carries_minimized_field_set_only() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk,
            manifest: sample_manifest(head_id, 7),
        },
    )
    .unwrap();

    // Read the on-disk JSON as an opaque map and inspect the keys.
    // Going through `serde_json::Value` rather than `read_head_manifest`
    // avoids a tautology -- `read_head_manifest` parses against the
    // typed `HeadManifest` and would happily ignore an unknown key
    // if `deny_unknown_fields` were dropped.  The `Value` map is the
    // wire shape on disk.
    let bytes = std::fs::read(head_manifest_path(tmp.path(), head_id)).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = v.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "head_id",
        "workspace_id",
        "workspace_revision",
        "sha256",
        "n_classes",
        "size_bytes",
        "created_at",
        "labels",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "published HeadManifest must carry exactly the minimized field set; got {actual:?}",
    );
    // Defence in depth: the BTreeSet equality above covers this,
    // but a future expected-set growth shouldn't quietly let
    // legacy fields back in.
    for forbidden in [
        "dataset_path",
        "training_cfg",
        "training_cfg_sha256",
        "dataset_revision",
        "dataset_revision_at_train",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "legacy field {forbidden:?} must not appear in the manifest",
        );
    }
    // The workspace_revision sub-object carries exactly `id` and `at`.
    let rev = obj["workspace_revision"]
        .as_object()
        .expect("workspace_revision is a sub-object");
    let rev_keys: std::collections::BTreeSet<&str> = rev.keys().map(String::as_str).collect();
    let expected_rev: std::collections::BTreeSet<&str> = ["id", "at"].into_iter().collect();
    assert_eq!(
        rev_keys, expected_rev,
        "workspace_revision sub-object must carry exactly id + at; got {rev_keys:?}",
    );
}

/// The published `heads.json` `HeadRecord` entries carry exactly
/// the minimized field set -- `head_id`, `workspace_revision`,
/// `sha256`, `n_classes`, `size_bytes`, `created_at` -- and never
/// the legacy provenance fields.
#[test]
fn published_index_carries_minimized_record_field_set_only() {
    let (tmp, cache) = fresh_workspace();
    let head_id = HeadId::new();
    let mpk = stage_mpk_tempfile(tmp.path(), head_id);
    publish_trained_head(
        tmp.path(),
        &cache,
        PendingHead {
            head_id,
            mpk_tempfile: mpk,
            manifest: sample_manifest(head_id, 11),
        },
    )
    .unwrap();

    let bytes = std::fs::read(head_index_path(tmp.path())).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let entries = v["heads"].as_array().expect("heads is an array");
    assert_eq!(entries.len(), 1, "exactly one published head");
    let rec = entries[0].as_object().expect("HeadRecord is a JSON object");
    let actual: std::collections::BTreeSet<&str> = rec.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "head_id",
        "workspace_revision",
        "sha256",
        "n_classes",
        "size_bytes",
        "created_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "published HeadRecord must carry exactly the field set; got {actual:?}",
    );
    for forbidden in [
        "dataset_path",
        "training_cfg_sha256",
        "dataset_revision",
        "dataset_revision_at_train",
        "labels",
        "workspace_id",
    ] {
        assert!(
            !rec.contains_key(forbidden),
            "legacy / non-record field {forbidden:?} must not appear in HeadRecord",
        );
    }
}

/// A hand-staged legacy `<head_id>.json` body (carrying
/// `dataset_path` / `training_cfg_sha256` / `training_cfg` plus
/// `dataset_revision_at_train`) parse-fails through
/// `read_head_manifest` thanks to `deny_unknown_fields`.  Defence
/// in depth at the read boundary: stale-binary boots or operator
/// tampering must fail closed instead of silently feeding legacy
/// metadata into the inference path.
#[test]
fn legacy_manifest_shape_parse_fails_on_read() {
    let tmp = tempfile::tempdir().unwrap();
    write_workspace_core(tmp.path(), &sample_core(0, 0)).unwrap();
    write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();

    let head_id = HeadId::new();
    let round_1_body = serde_json::json!({
        "head_id": head_id.to_string(),
        "workspace_id": ws_id().to_string(),
        // Legacy alias; the schema now requires
        // `workspace_revision`.
        "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T12:00:00Z" },
        "sha256": "abc",
        "n_classes": 3,
        "size_bytes": 1024,
        "created_at": "2026-05-07T12:34:56Z",
        "labels": ["a", "b", "c"],
        // Legacy fields that the current schema drops.
        "dataset_path": "audio/cat",
        "training_cfg_sha256": "deadbeef",
        "training_cfg": { "epochs": 4, "batch_size": 16, "learning_rate": 0.001 },
    });
    std::fs::write(
        head_manifest_path(tmp.path(), head_id),
        serde_json::to_vec(&round_1_body).unwrap(),
    )
    .unwrap();

    let res = read_head_manifest(tmp.path(), head_id);
    assert!(
        res.is_err(),
        "legacy manifest body on disk must parse-fail; got {res:?}",
    );
}

/// A hand-staged legacy `heads.json` (a `HeadRecord` with
/// `dataset_path` + `dataset_revision_at_train`) parse-fails
/// through `read_head_index`; same read-boundary defence as the
/// manifest test above.  Cache load and boot recovery both
/// consume `read_head_index`.
#[test]
fn legacy_head_index_shape_parse_fails_on_read() {
    let tmp = tempfile::tempdir().unwrap();
    write_workspace_core(tmp.path(), &sample_core(0, 0)).unwrap();
    std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();

    let head_id = HeadId::new();
    let round_1_index = serde_json::json!({
        "heads": [{
            "head_id": head_id.to_string(),
            "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T12:00:00Z" },
            "sha256": "abc",
            "n_classes": 3,
            "size_bytes": 1024,
            "created_at": "2026-05-07T12:34:56Z",
            "dataset_path": "audio/cat",
            "training_cfg_sha256": "deadbeef",
        }],
    });
    std::fs::write(
        head_index_path(tmp.path()),
        serde_json::to_vec(&round_1_index).unwrap(),
    )
    .unwrap();

    let res = read_head_index(tmp.path());
    assert!(
        res.is_err(),
        "legacy heads.json body on disk must parse-fail; got {res:?}",
    );
}
