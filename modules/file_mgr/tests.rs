//! `file_mgr` unit + integration tests.
//!
//! Test code writes asset fixtures with `std::fs::write` to
//! drive the registry / uploader paths.  The production
//! constraint in `clippy.toml` (writes go through
//! [`crate::file_mgr::put_atomic`]) does not apply to test
//! setup.

#![allow(clippy::disallowed_methods)]

use super::*;

fn fresh_root() -> (tempfile::TempDir, WorkspaceMgr) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mgr = WorkspaceMgr::new(dir.path().to_path_buf());
    (dir, mgr)
}

fn fresh_fs_service() -> (tempfile::TempDir, FsServiceImpl) {
    let dir = tempfile::tempdir().expect("tempdir");
    let fs = FsServiceImpl::new(dir.path().to_path_buf());
    (dir, fs)
}

/// `WorkspaceMgr::create` no longer writes the legacy `weights/`,
/// `labels/`, or `metadata.json`.  These helpers rehydrate them so
/// the legacy `AssetKind`-shaped code paths (`upload`,
/// `install_from_path`, `read_metadata`, `with_metadata`,
/// `validate`, `list_assets`) keep round-trip coverage even though
/// no production caller writes the shape any more.
fn seed_legacy_layout(mgr: &WorkspaceMgr, id: &WorkspaceId) {
    let ws = mgr.workspace_dir(id);
    for sub in ["weights", "labels"] {
        std::fs::create_dir_all(ws.join(sub)).expect("seed legacy subdir");
    }
    let core = mgr.read_cached_core(id).expect("core");
    let metadata = WorkspaceMetadata::new(core.id, core.name.clone());
    mgr.write_metadata(id, &metadata)
        .expect("seed metadata.json");
}

/// Convenience: `create` + `seed_legacy_layout` in one call,
/// returning the workspace id.  Used by every legacy-surface
/// test to keep the boilerplate light.
fn create_legacy(mgr: &WorkspaceMgr, name: &str) -> WorkspaceId {
    let id = mgr.create(name).expect("create");
    seed_legacy_layout(mgr, &id);
    id
}

/// FsService variant of [`create_legacy`].  Reaches the
/// concrete `WorkspaceMgr` through `FsServiceImpl::mgr` (a
/// `pub(crate)` field is unavailable, so we re-derive the same
/// helper inline).
fn create_legacy_fs(fs: &FsServiceImpl, name: &str) -> WorkspaceId {
    let id = fs.create(name).expect("create");
    // Re-build a `WorkspaceMgr` view rooted at the same dir to
    // run the seed.  Cheap construction; matches the test fixture
    // shape (no real-world caller does this).
    let mgr = WorkspaceMgr::new(fs.root().to_path_buf());
    seed_legacy_layout(&mgr, &id);
    id
}

/// `MetadataGuard::commit` persists the mutated metadata
/// atomically; a subsequent read sees the change.
#[test]
fn metadata_guard_commit_persists_changes() {
    let (_dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    // Mutate via the guard, commit, re-read.
    let mut g = fs.metadata_mut(&id).expect("guard");
    g.metadata_mut().assets.push(AssetRecord {
        kind: AssetKind::HeadMpk,
        name: AssetId::parse("added.mpk").expect("valid AssetId"),
        sha256: "0".repeat(64),
        size_bytes: 0,
    });
    g.commit().expect("commit");

    let after = fs.read_metadata(&id).expect("read");
    assert_eq!(after.assets.len(), 1);
    assert_eq!(after.assets[0].name, "added.mpk");
}

/// Dropping the guard without committing rolls back: the
/// on-disk metadata stays at the pre-mutation state.  The
/// `Drop` impl logs a warn so accidental misuse is visible
/// in operator logs.
#[test]
fn metadata_guard_drop_without_commit_rolls_back() {
    let (_dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    let before = fs.read_metadata(&id).expect("read");
    assert!(before.assets.is_empty());

    {
        let mut g = fs.metadata_mut(&id).expect("guard");
        g.metadata_mut().assets.push(AssetRecord {
            kind: AssetKind::HeadMpk,
            name: AssetId::parse("uncommitted.mpk").expect("valid AssetId"),
            sha256: "0".repeat(64),
            size_bytes: 0,
        });
        // Drop without commit; should rollback.
    }

    let after = fs.read_metadata(&id).expect("read");
    assert!(
        after.assets.is_empty(),
        "uncommitted mutation persisted: {:?}",
        after.assets
    );
}

/// `install_from_path` installs a pre-staged tempfile via
/// atomic rename + metadata commit.  Mirrors `upload`'s
/// on-disk effect for the staged-bytes path.
#[test]
fn install_from_path_renames_and_commits() {
    let (dir, fs) = fresh_fs_service();
    let id = create_legacy_fs(&fs, "ws");

    // Stage a tempfile in the workspace's `.tmp/`.  The
    // production uploader self-mkdirs `.tmp/` on first write;
    // this test stages directly, so we materialize the dir
    // ourselves to mirror that contract.
    let tmp_dir = fs.workspace_tmpdir(&id);
    std::fs::create_dir_all(&tmp_dir).expect("mkdir workspace .tmp");
    let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir).expect("tempfile");
    use std::io::Write;
    tmp.write_all(b"hello").expect("write");
    tmp.as_file().sync_all().expect("sync");

    let receipt = fs
        .install_from_path(&id, AssetKind::HeadLabels, "greet.txt", tmp.path())
        .expect("install");

    // Asset on disk + metadata record both present.
    assert_eq!(receipt.size_bytes, 5);
    let on_disk = std::fs::read(&receipt.path).expect("read installed");
    assert_eq!(on_disk, b"hello");
    let meta = fs.read_metadata(&id).expect("read meta");
    assert_eq!(meta.assets.len(), 1);
    assert_eq!(meta.assets[0].kind, AssetKind::HeadLabels);
    assert_eq!(meta.assets[0].name, "greet.txt");
    assert_eq!(meta.assets[0].size_bytes, 5);
    // Receipt path lives under the workspace dir.
    assert!(receipt.path.starts_with(dir.path()));
}

#[test]
fn create_workspace_writes_redesign_layout() {
    // Workspace creation lays down `workspace.json` + an empty
    // `heads.json` under `<workspace>/`; every leaf subdirectory
    // (`datasets/`, `converters/`, `heads/`, `training_logs/`,
    // `converter_logs/`, `.tmp/`) is created lazily by the
    // first writer that touches it.  The legacy `weights/` /
    // `labels/` / `metadata.json` surface remains retired.
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("first").expect("create");
    let ws = mgr.root().join("workspaces").join(id.to_string());
    assert!(ws.is_dir(), "workspace dir itself must exist");
    assert!(ws.join("workspace.json").is_file());
    assert!(ws.join("heads.json").is_file());
    // No leaf subdir is materialized eagerly; the lazy mkdir
    // happens in the writer (uploader, training, converter,
    // head_rotation, staging).  Pin the empty-on-create shape
    // so a future regression of the lazy-mkdir contract surfaces
    // in CI rather than as a silent disk-layout shift.
    for sub in [
        "datasets",
        "converters",
        "training_logs",
        "converter_logs",
        ".tmp",
        "heads",
    ] {
        assert!(
            !ws.join(sub).exists(),
            "subdir {sub} must NOT exist before first writer; create_with_tags is lazy",
        );
    }
    // Legacy surfaces must NOT be created automatically.
    assert!(
        !ws.join("metadata.json").exists(),
        "metadata.json should not be created",
    );
    assert!(
        !ws.join("weights").exists(),
        "weights/ should not be created",
    );
    assert!(!ws.join("labels").exists(), "labels/ should not be created",);
}

/// `create` writes `workspace.json` + an empty `heads.json` and
/// seeds the eager cache; no legacy metadata stub.
#[test]
fn create_writes_workspace_core_and_heads_index() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ws-core").expect("create");
    let ws = mgr.root().join("workspaces").join(id.to_string());
    assert!(
        ws.join("workspace.json").is_file(),
        "workspace.json must be written"
    );
    assert!(
        ws.join("heads.json").is_file(),
        "heads.json must be written"
    );
    let summary = mgr.summary(&id).expect("summary");
    assert_eq!(summary.core.id, id);
    assert_eq!(summary.core.name, "ws-core");
    assert_eq!(summary.core.workspace_revision.id, 0);
    assert_eq!(summary.core.head_count, 0);
    assert!(summary.heads.heads.is_empty());
    assert!(summary.head_statuses.is_empty());
}

#[test]
fn create_rejects_duplicate_name() {
    let (_dir, mgr) = fresh_root();
    mgr.create("main").expect("first");
    let err = mgr.create("main").unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)));
}

#[test]
fn create_rejects_invalid_name() {
    let (_dir, mgr) = fresh_root();
    assert!(matches!(
        mgr.create("bad/name").unwrap_err(),
        FileError::InvalidName(_)
    ));
    assert!(matches!(
        mgr.create("").unwrap_err(),
        FileError::InvalidName(_)
    ));
}

// MARK: create_with_tags + patch_workspace

#[test]
fn create_with_tags_persists_normalized_tags() {
    let (_dir, mgr) = fresh_root();
    // ASCII whitespace surrounding each tag is trimmed; the
    // post-trim form lands on disk.  Order is preserved.
    let id = mgr
        .create_with_tags(
            "scoped",
            &["  field-recordings  ".to_string(), "pet-noises".to_string()],
        )
        .expect("create");
    let summary = mgr.summary(&id).expect("summary");
    assert_eq!(
        summary.core.tags,
        vec!["field-recordings".to_string(), "pet-noises".to_string()]
    );
}

#[test]
fn create_with_tags_rejects_empty_after_trim() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["   ".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_path_separator() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["a/b".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_case_insensitive_duplicates() {
    let (_dir, mgr) = fresh_root();
    let err = mgr
        .create_with_tags("scoped", &["Field".to_string(), "FIELD".to_string()])
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn create_with_tags_rejects_over_cap() {
    let (_dir, mgr) = fresh_root();
    let many: Vec<String> = (0..33).map(|i| format!("t{i}")).collect();
    let err = mgr.create_with_tags("scoped", &many).unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn patch_workspace_renames_and_retags_atomically() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &["a".into()]).unwrap();
    let revision_before = mgr.summary(&id).unwrap().core.workspace_revision.id;
    let patched = mgr
        .patch_workspace(&id, Some("renamed"), Some(&["b".into(), "c".into()]))
        .expect("patch");
    assert_eq!(patched.name, "renamed");
    assert_eq!(patched.tags, vec!["b".to_string(), "c".to_string()]);
    // Re-read to confirm the on-disk write committed.
    let summary = mgr.summary(&id).unwrap();
    assert_eq!(summary.core.name, "renamed");
    assert_eq!(summary.core.tags, vec!["b".to_string(), "c".to_string()]);
    // Name + tag edits do NOT bump the workspace revision.
    assert_eq!(summary.core.workspace_revision.id, revision_before);
}

#[test]
fn patch_workspace_name_only_preserves_tags() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &["pinned".into()]).unwrap();
    let patched = mgr
        .patch_workspace(&id, Some("renamed"), None)
        .expect("patch");
    assert_eq!(patched.name, "renamed");
    assert_eq!(patched.tags, vec!["pinned".to_string()]);
}

#[test]
fn patch_workspace_tags_only_preserves_name() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create_with_tags("orig", &[]).unwrap();
    let patched = mgr
        .patch_workspace(&id, None, Some(&["new".into()]))
        .expect("patch");
    assert_eq!(patched.name, "orig");
    assert_eq!(patched.tags, vec!["new".to_string()]);
}

#[test]
fn patch_workspace_self_rename_is_idempotent() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("solo").unwrap();
    // Renaming to the existing name (case-insensitive equality
    // with self) succeeds because the uniqueness check excludes
    // the workspace under edit.
    mgr.patch_workspace(&id, Some("solo"), None).expect("patch");
    mgr.patch_workspace(&id, Some("Solo"), None).expect("patch");
    assert_eq!(mgr.summary(&id).unwrap().core.name, "Solo");
}

#[test]
fn patch_workspace_rejects_name_collision_with_other() {
    let (_dir, mgr) = fresh_root();
    let _other = mgr.create("taken").unwrap();
    let id = mgr.create("free").unwrap();
    // Case-insensitive collision with `taken` is rejected
    // (`str::to_lowercase` covers ASCII and Unicode pairs alike).
    let err = mgr.patch_workspace(&id, Some("TAKEN"), None).unwrap_err();
    assert!(matches!(err, FileError::NameConflict(_)));
}

#[test]
fn patch_workspace_returns_not_found_for_missing_workspace() {
    let (_dir, mgr) = fresh_root();
    let phantom = WorkspaceId::new();
    let err = mgr
        .patch_workspace(&phantom, Some("ghost"), None)
        .unwrap_err();
    assert!(matches!(err, FileError::NotFound(_)));
}

#[test]
fn patch_workspace_rejects_invalid_name() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ok").unwrap();
    let err = mgr.patch_workspace(&id, Some(""), None).unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn patch_workspace_rejects_invalid_tags() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("ok").unwrap();
    let err = mgr
        .patch_workspace(&id, None, Some(&["a/b".into()]))
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

#[test]
fn list_workspaces_returns_created() {
    let (_dir, mgr) = fresh_root();
    let a = mgr.create("a").unwrap();
    let b = mgr.create("b").unwrap();
    let mut ids = mgr.list_workspaces().unwrap();
    ids.sort_by_key(|i| i.to_string());
    let mut expected = vec![a.to_string(), b.to_string()];
    expected.sort();
    let got: Vec<String> = ids.into_iter().map(|i| i.to_string()).collect();
    assert_eq!(got, expected);
}

#[test]
fn delete_removes_workspace() {
    let (_dir, mgr) = fresh_root();
    let id = mgr.create("doomed").unwrap();
    let ws = mgr.root().join("workspaces").join(id.to_string());
    assert!(ws.exists());
    mgr.delete(&id).unwrap();
    assert!(!ws.exists());
    assert!(matches!(
        mgr.delete(&id).unwrap_err(),
        FileError::NotFound(_)
    ));
}

#[tokio::test]
async fn upload_atomic_writes_with_sha256() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "upload-test");
    let payload = b"hello world\nthis is a head.mpk-flavored file";
    let receipt = mgr
        .upload(&id, AssetKind::HeadMpk, "demo.mpk", &payload[..])
        .await
        .expect("upload");

    assert_eq!(receipt.size_bytes, payload.len() as u64);
    let expected = hex_lowercase(&Sha256::digest(payload));
    assert_eq!(receipt.sha256, expected);
    assert!(receipt.path.exists());
    let on_disk = std::fs::read(&receipt.path).unwrap();
    assert_eq!(on_disk, payload);

    // Metadata reflects the upload.
    let meta = mgr.read_metadata(&id).unwrap();
    assert_eq!(meta.assets.len(), 1);
    assert_eq!(meta.assets[0].sha256, expected);
    assert_eq!(meta.assets[0].kind, AssetKind::HeadMpk);
}

#[tokio::test]
async fn upload_rejects_bad_extension() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "x");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "demo.txt", &b"x"[..])
        .await
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidExtension { .. }));
}

#[tokio::test]
async fn upload_rejects_path_traversal_name() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "x");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "../escape.mpk", &b"x"[..])
        .await
        .unwrap_err();
    assert!(matches!(err, FileError::InvalidName(_)));
}

/// Filenames that pass [`validate_asset_name`] may contain
/// multi-byte UTF-8 characters (Linux allows them).
/// [`validate_extension`] must not panic when the
/// suffix-match byte offset (`name.len() - (ext.len() + 1)`)
/// lands inside a codepoint.
///
/// Minimal trigger `("A\u{00e4}.mp", "mpk")`:
/// bytes `A 0xC3 0xA4 . m p` (len = 6); `need = 4`,
/// `offset = 2` -> the second byte of `\u{00e4}`.  An
/// `&str` slice at `name[2..]` would panic; the byte-level
/// comparison handles the offset correctly.
#[test]
fn validate_extension_does_not_panic_on_multibyte_utf8() {
    // Trigger case: without the byte-level slice this
    // would panic.
    let r = validate_extension("Aä.mp", &["mpk"]);
    assert!(
        matches!(r, Err(FileError::InvalidExtension { .. })),
        "got {r:?}"
    );

    // Other names that pass validate_asset_name and reach
    // validate_extension; all must produce a clean error rather
    // than panic.
    for bad in ["foo.äb", "foo.t\u{00e4}r", "naïve.bar", "Bä.x"] {
        let r = validate_extension(bad, &["mpk"]);
        assert!(
            matches!(r, Err(FileError::InvalidExtension { .. })),
            "expected InvalidExtension for {bad:?}, got {r:?}",
        );
    }
    // Multibyte stem with a recognised ASCII extension at
    // the tail; must succeed.
    validate_extension("naïve.mpk", &["mpk"]).expect("naïve.mpk");
    validate_extension("dataset-é.tar.gz", &["tar.gz", "zip"]).expect("dataset-é.tar.gz");
    // Case-insensitive happy path, ASCII only.
    validate_extension("ARCHIVE.TAR.GZ", &["tar.gz"]).expect("upper-case");
}

/// Asset names with embedded ASCII control characters
/// (newline, tab, etc.) are rejected.  Linux happily stores
/// them but they corrupt log lines, JSON metadata, and HTTP
/// receipt fields, and a name with `\n` would split into
/// two entries when a downstream tool tokenises log output.
#[test]
fn validate_asset_name_rejects_control_chars() {
    for bad in [
        "foo\nbar.mpk",
        "foo\tbar.mpk",
        "foo\x01bar.mpk",
        "x.mpk\x7f",
    ] {
        let err = validate_asset_name(bad).unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "control char {bad:?} not rejected"
        );
    }
    // Sanity: valid names still accepted.
    validate_asset_name("foo.mpk").unwrap();
    validate_asset_name("trained-09109000-3acb.labels.txt").unwrap();
}

/// [`WorkspaceMgr::with_admission`] enforces
/// [`AdmissionCfg::max_upload_bytes`].  Upload of more bytes
/// than the cap rejects mid-stream with `PayloadTooLarge`;
/// the tempfile drops without committing, so no orphan file
/// lands on disk and no metadata row is written.  The same
/// name + kind can be uploaded again after rejection.
#[tokio::test]
async fn admission_rejects_oversize_upload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = AdmissionCfg {
        max_upload_bytes: 100,
        max_concurrent_uploads: 4,
    };
    let mgr = WorkspaceMgr::with_admission(dir.path().to_path_buf(), cfg);
    let id = create_legacy(&mgr, "admit");

    let payload = [0u8; 200]; // 2x cap
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "big.mpk", &payload[..])
        .await
        .expect_err("oversize upload must reject");
    match err {
        FileError::PayloadTooLarge { observed, max } => {
            assert!(observed > 100, "observed must exceed cap: {observed}");
            assert_eq!(max, 100);
        }
        other => panic!("expected PayloadTooLarge, got {other:?}"),
    }

    // Metadata stays empty (no half-committed row).
    let meta = mgr.read_metadata(&id).expect("read meta");
    assert!(
        meta.assets.is_empty(),
        "rejected upload must not commit metadata: {:?}",
        meta.assets
    );

    // Same name + kind can be re-uploaded after rejection.
    mgr.upload(&id, AssetKind::HeadMpk, "big.mpk", &b"under cap"[..])
        .await
        .expect("re-upload under cap must succeed");
}

/// Concurrency cap rejects the (max+1)th in-flight upload
/// with `TooManyConcurrentUploads`.  Holds the only permit
/// (max = 1) directly via the admission state's semaphore;
/// the next `upload` call must reject without blocking --
/// the upload path uses `try_acquire_owned` (fail-fast) by
/// design.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_rejects_too_many_concurrent_uploads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = AdmissionCfg {
        max_upload_bytes: 1024 * 1024,
        max_concurrent_uploads: 1,
    };
    let mgr = WorkspaceMgr::with_admission(dir.path().to_path_buf(), cfg);
    let id = create_legacy(&mgr, "conc");

    let state = mgr
        .admission
        .as_ref()
        .expect("admission configured")
        .clone();
    let _hold = state
        .semaphore
        .clone()
        .try_acquire_owned()
        .expect("hold the only permit");

    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "x.mpk", &b"any"[..])
        .await
        .expect_err("upload must reject when no permits");
    match err {
        FileError::TooManyConcurrentUploads { active, max } => {
            assert_eq!(active, 1, "1 active permit (the test's hold)");
            assert_eq!(max, 1);
        }
        other => panic!("expected TooManyConcurrentUploads, got {other:?}"),
    }
}

/// [`WorkspaceMgr::new`] (no admission) accepts uploads of
/// any size.  Regression check that the cap path is opt-in
/// and the un-admitted ctor stays permissive.
#[tokio::test]
async fn no_admission_accepts_any_size() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "permissive");
    let payload = [0u8; 100 * 1024];
    mgr.upload(&id, AssetKind::HeadMpk, "big.mpk", &payload[..])
        .await
        .expect("no-admission ctor must accept any size");
}

/// [`WorkspaceMgr::read_metadata`] refuses a workspace
/// whose `schema_version` is newer than this build
/// understands.  Forward rejection is the load-bearing
/// case: a future v2 daemon writes the workspace, the
/// operator downgrades, the v1 daemon must not silently
/// lose v2 fields by reserialising the older shape.
#[test]
fn read_metadata_rejects_schema_too_new() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "future-schema");

    // Hand-edit `metadata.json` to claim a newer
    // `schema_version` than this build's `CURRENT`.  Reuses
    // the deserialise side of `WorkspaceMetadata` (the
    // struct field is `pub`) rather than emitting JSON by
    // hand; the serializer is the same path that produced
    // the on-disk file in the first place.
    let path = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join("metadata.json");
    let mut meta: WorkspaceMetadata =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    meta.schema_version = WorkspaceMetadata::CURRENT + 1;
    std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let err = mgr
        .read_metadata(&id)
        .expect_err("future schema must reject");
    match err {
        FileError::SchemaTooNew { found, max, .. } => {
            assert_eq!(found, WorkspaceMetadata::CURRENT + 1);
            assert_eq!(max, WorkspaceMetadata::CURRENT);
        }
        other => panic!("expected SchemaTooNew, got {other:?}"),
    }
}

/// [`WorkspaceMgr::read_metadata`] refuses a workspace
/// whose `schema_version` is below the floor.  Today this
/// is the hand-edited `0` case (no migration code); when a
/// future build raises `MIN_COMPATIBLE`, the same code path
/// activates for legitimately-aged workspaces.
#[test]
fn read_metadata_rejects_schema_too_old() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "ancient-schema");

    let path = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join("metadata.json");
    let mut meta: WorkspaceMetadata =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // 0 is below MIN_COMPATIBLE = 1 today.
    meta.schema_version = 0;
    std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let err = mgr.read_metadata(&id).expect_err("schema 0 must reject");
    match err {
        FileError::SchemaTooOld { found, min, .. } => {
            assert_eq!(found, 0);
            assert_eq!(min, WorkspaceMetadata::MIN_COMPATIBLE);
        }
        other => panic!("expected SchemaTooOld, got {other:?}"),
    }
}

/// Uploading `foo.mpk` after `Foo.mpk` rejects with
/// `NameConflict`.  Same-case re-upload (`Foo.mpk` after
/// `Foo.mpk`) is the existing overwrite path and stays
/// allowed (regression-checked here too).  Defends the
/// case-sensitive-everywhere policy on macOS HFS+ where
/// `tmp.persist` would silently overwrite.
#[tokio::test]
async fn upload_rejects_case_insensitive_collision() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "case-test");

    // Baseline: upload `Foo.mpk`.
    mgr.upload(&id, AssetKind::HeadMpk, "Foo.mpk", &b"first"[..])
        .await
        .expect("Foo.mpk");

    // Same-case re-upload is the existing overwrite path:
    // accepted, single metadata row, fresh sha.
    mgr.upload(&id, AssetKind::HeadMpk, "Foo.mpk", &b"second"[..])
        .await
        .expect("Foo.mpk re-upload");
    let meta = mgr.read_metadata(&id).unwrap();
    let foos: Vec<_> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .collect();
    assert_eq!(
        foos.len(),
        1,
        "same-case re-upload must overwrite, not duplicate: {foos:?}"
    );

    // Different-case (foo.mpk vs Foo.mpk); must reject.
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "foo.mpk", &b"different-case"[..])
        .await
        .expect_err("foo.mpk must collide with Foo.mpk");
    assert!(
        matches!(err, FileError::NameConflict(_)),
        "expected NameConflict, got {err:?}",
    );

    // Metadata still has exactly one HeadMpk record (the
    // `Foo.mpk` from the baseline).  The collision check
    // happened before the rename, so no orphan file or
    // half-committed metadata row.
    let meta = mgr.read_metadata(&id).unwrap();
    let foos: Vec<_> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .collect();
    assert_eq!(
        foos.len(),
        1,
        "rejected upload must not commit metadata: {foos:?}"
    );
    assert_eq!(foos[0].name, "Foo.mpk");
}

#[tokio::test]
async fn upload_overwrite_updates_sha_in_metadata() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "ov");
    let _ = mgr
        .upload(&id, AssetKind::HeadMpk, "h.mpk", &b"first"[..])
        .await
        .unwrap();
    let r2 = mgr
        .upload(&id, AssetKind::HeadMpk, "h.mpk", &b"second-revision"[..])
        .await
        .unwrap();
    let meta = mgr.read_metadata(&id).unwrap();
    assert_eq!(
        meta.assets.len(),
        1,
        "duplicate asset rows: {:?}",
        meta.assets
    );
    assert_eq!(meta.assets[0].sha256, r2.sha256);
    let on_disk = std::fs::read(&r2.path).unwrap();
    assert_eq!(on_disk, b"second-revision");
}

#[tokio::test]
async fn validate_detects_corruption_and_missing() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "v");
    let r1 = mgr
        .upload(&id, AssetKind::HeadMpk, "good.mpk", &b"abc"[..])
        .await
        .unwrap();
    let r2 = mgr
        .upload(&id, AssetKind::HeadMpk, "missing.mpk", &b"def"[..])
        .await
        .unwrap();
    // Corrupt one (overwrite with different bytes outside our API).
    std::fs::write(&r1.path, b"tampered").unwrap();
    // Delete the other.
    std::fs::remove_file(&r2.path).unwrap();
    // Add an extra file that's not in metadata.
    std::fs::write(
        mgr.root()
            .join("workspaces")
            .join(id.to_string())
            .join("weights/orphan.mpk"),
        b"orphan",
    )
    .unwrap();

    let report = mgr.validate(&id).unwrap();
    assert!(!report.ok);
    assert_eq!(report.corrupt.len(), 1);
    assert_eq!(report.corrupt[0].1, "good.mpk");
    assert_eq!(report.missing.len(), 1);
    assert_eq!(report.missing[0].1, "missing.mpk");
    assert_eq!(report.extra.len(), 1);
    assert_eq!(report.extra[0].1, "orphan.mpk");
}

/// Concurrent uploads to the same workspace must not lose
/// metadata records.  Without the per-workspace lock each
/// task does `read_metadata -> modify -> write_metadata`,
/// and tasks racing the read-modify-write step each see
/// the OLD `assets[]` and overwrite each other's appends.
/// [`WorkspaceMgr::with_metadata`]'s per-workspace lock
/// serializes the read-modify-write so all records
/// survive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_uploads_preserve_all_records() {
    let (dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "conc");
    let mgr = std::sync::Arc::new(mgr);

    const N: usize = 32;
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let mgr = mgr.clone();
        // WorkspaceId is Copy (wraps Uuid).
        handles.push(tokio::spawn(async move {
            let name = format!("file-{i:02}.mpk");
            let bytes = format!("payload-for-{i}").into_bytes();
            mgr.upload(&id, AssetKind::HeadMpk, &name, &bytes[..]).await
        }));
    }
    let mut ok = 0;
    for h in handles {
        h.await.expect("join").expect("upload");
        ok += 1;
    }
    assert_eq!(ok, N);

    // All N records must be present in metadata.json.
    let meta = mgr.read_metadata(&id).unwrap();
    let head_records: std::collections::HashSet<String> = meta
        .assets
        .iter()
        .filter(|a| a.kind == AssetKind::HeadMpk)
        .map(|a| a.name.as_str().to_string())
        .collect();
    assert_eq!(
        head_records.len(),
        N,
        "expected {N} unique head records, got {} (records were lost to a race)",
        head_records.len(),
    );
    for i in 0..N {
        let expected = format!("file-{i:02}.mpk");
        assert!(
            head_records.contains(&expected),
            "record {expected} missing from metadata",
        );
    }

    // Every on-disk file must also exist (the file writes
    // themselves are atomic via tempfile + rename and don't
    // need the metadata lock).
    for i in 0..N {
        let p = mgr
            .root()
            .join("workspaces")
            .join(id.to_string())
            .join("weights")
            .join(format!("file-{i:02}.mpk"));
        assert!(p.exists(), "missing on-disk file: {}", p.display());
    }
    let _ = dir; // keep alive
}

/// Unicode (multibyte UTF-8) filenames that the older
/// `validate_asset_name` accepted now reject pre-rename:
/// `validate_asset_name` delegates to [`AssetId::parse`]'s
/// ASCII allowlist.  Without this the upload path streamed
/// the body, renamed into place, then panicked at
/// `AssetId::parse(name).expect(...)` while constructing the
/// metadata record -- leaving an orphan file on disk.
#[tokio::test]
async fn upload_rejects_unicode_filename_pre_rename() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "uni");
    let err = mgr
        .upload(&id, AssetKind::HeadMpk, "naïve.mpk", &b"x"[..])
        .await
        .expect_err("Unicode filename must reject");
    assert!(
        matches!(err, FileError::InvalidName(_)),
        "expected InvalidName, got {err:?}"
    );
    // No orphan asset on disk, no metadata row.
    let weights = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join("weights");
    let entries: Vec<_> = std::fs::read_dir(&weights)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.is_empty(),
        "rejected upload must not leave files: {entries:?}"
    );
    let meta = mgr.read_metadata(&id).unwrap();
    assert!(meta.assets.is_empty());
}

/// Same reject as `upload_rejects_unicode_filename_pre_rename`
/// for the staging-then-install path.  The validator runs
/// before the rename, so no orphan can land in the asset
/// subdir.
#[test]
fn install_from_path_rejects_unicode_filename() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "uni-install");
    // `.tmp/` is created lazily by production writers; this test
    // stages directly, so mkdir to mirror the uploader contract.
    let staging = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join(".tmp");
    std::fs::create_dir_all(&staging).expect("mkdir workspace .tmp");
    let tmp = tempfile::NamedTempFile::new_in(&staging).expect("tempfile");
    let err = mgr
        .install_from_path(&id, AssetKind::HeadLabels, "naïve.txt", tmp.path())
        .expect_err("Unicode filename must reject");
    assert!(
        matches!(err, FileError::InvalidName(_)),
        "expected InvalidName, got {err:?}"
    );
}

/// Two concurrent `create("main")` calls must not both
/// succeed: the registry-level mutex serializes the
/// list-check-create sequence.  Without the lock both calls
/// can observe an empty registry and both commit distinct
/// UUIDs with the same human-readable name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_create_serializes_name_uniqueness() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mgr = std::sync::Arc::new(WorkspaceMgr::new(dir.path().to_path_buf()));

    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let mgr = mgr.clone();
        handles.push(tokio::task::spawn_blocking(move || mgr.create("main")));
    }
    let mut ok_count = 0;
    let mut conflict_count = 0;
    for h in handles {
        match h.await.expect("join") {
            Ok(_) => ok_count += 1,
            Err(FileError::NameConflict(_)) => conflict_count += 1,
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert_eq!(ok_count, 1, "exactly one creator should win");
    assert_eq!(conflict_count, N - 1, "the rest must see NameConflict");

    // Registry holds a single workspace, named "main".  The name
    // is read from `workspace.json` via the cached summary; the
    // legacy `metadata.json` is no longer written on `create`.
    let ids = mgr.list_workspaces().unwrap();
    assert_eq!(ids.len(), 1);
    let summary = mgr.summary(&ids[0]).unwrap();
    assert_eq!(summary.core.name, "main");
}

/// Release-build guard: `asset_path` panics when the caller
/// bypasses validation.  Regression check that the
/// `debug_assert!` was upgraded to a real assertion -- in the
/// old shape an unvalidated `..` name would silently produce
/// a path outside the workspace in release builds.
#[test]
#[should_panic(expected = "asset_path called with unvalidated name")]
fn asset_path_panics_on_unvalidated_name_in_release() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "guard");
    // Path-traversal `..` would otherwise produce
    // `<root>/<id>/weights/../escape.mpk` which canonicalises
    // outside the asset subdir.
    let _ = mgr.asset_path(&id, AssetKind::HeadMpk, "../escape.mpk");
}

/// `asset_path_typed` accepts a pre-validated [`AssetId`]
/// without re-checking; the returned path stays inside the
/// workspace + kind subdir.
#[test]
fn asset_path_typed_skips_validation() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "typed");
    let asset = AssetId::parse("foo.mpk").expect("valid");
    let p = mgr.asset_path_typed(&id, AssetKind::HeadMpk, &asset);
    let expected = mgr
        .root()
        .join("workspaces")
        .join(id.to_string())
        .join("weights")
        .join("foo.mpk");
    assert_eq!(p, expected);
}

#[tokio::test]
async fn list_assets_filters_by_kind() {
    let (_dir, mgr) = fresh_root();
    let id = create_legacy(&mgr, "list");
    mgr.upload(&id, AssetKind::HeadMpk, "h.mpk", &b"x"[..])
        .await
        .unwrap();
    mgr.upload(&id, AssetKind::HeadLabels, "h.txt", &b"a\nb\n"[..])
        .await
        .unwrap();
    let heads = mgr.list_assets(&id, AssetKind::HeadMpk).unwrap();
    let labels = mgr.list_assets(&id, AssetKind::HeadLabels).unwrap();
    assert_eq!(heads.len(), 1);
    assert_eq!(heads[0].name, "h.mpk");
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].name, "h.txt");
}
